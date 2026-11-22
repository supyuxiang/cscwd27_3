import os
from enum import Enum

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import json
import math
import sys
from pathlib import Path

import hydra
import numpy as np
import random
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# 布局参考过 verl 的 sft_trainer。loss、动态 padding 和续训状态是这份单卡脚本自己的。
_VERL_ROOT = Path(__file__).resolve().parents[1] / "verl"
sys.path.insert(0, str(_VERL_ROOT))

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_IGNORE_DISABLED_P2P", "1")
# This cluster exports ROCR_VISIBLE_DEVICES next to CUDA_VISIBLE_DEVICES.
# verl's Worker refuses to start when both are set. Keep the CUDA mask.
if os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

import swanlab

_LOSS_CHUNK = 32
_CKPT_VERSION = 1


class SwanlabLogger:
    def __init__(
            self,
            project_name,
            experiment_name,
            config,
    ):

        SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
        SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
        SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
        if SWANLAB_API_KEY:
            swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

        if config is None:
            config = {}

        swanlab.init(
            project=project_name,
            experiment_name=experiment_name,
            config={"FRAMEWORK": "verl", **config},
            logdir=SWANLAB_LOG_DIR,
            mode=SWANLAB_MODE,
        )

        self._finish = False

    def log(self, data, step):
        swanlab.log(
            data=data, step=step
        )

    def finish(self, exit_code: int = 0):
        if self._finish:
            return
        self._finish = True
        swanlab.finish()

    def __del__(self):
        self.finish()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not deterministic:
        return
    # cublas 的确定性工作区要在 CUDA 上下文使用该库之前设置。
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)


def my_hf_tokenizer(name_or_path, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"{name_or_path} has no eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"[data] pad_token missing; use eos id {tokenizer.eos_token_id}")
    tokenizer.padding_side = "right"
    return tokenizer


def _chat_template_ids(tokenizer, messages, **kwargs):
    out = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    if isinstance(out, torch.Tensor):
        return out.tolist()
    if hasattr(out, "input_ids"):
        return list(out.input_ids)
    return list(out)


def extract_system_prompt_and_generation(tokenizer, **apply_chat_template_kwargs):
    token1 = _chat_template_ids(
        tokenizer, [{"role": "user", "content": ""}], add_generation_prompt=False, **apply_chat_template_kwargs
    )
    token2 = _chat_template_ids(
        tokenizer,
        [{"role": "user", "content": ""}] * 2,
        add_generation_prompt=False,
        **apply_chat_template_kwargs,
    )
    # get system prompt tokens；delta==0 时 token1[:-0] 在 Python 里是空 list
    delta = len(token2) - len(token1)
    system_prompt = token1[:-delta] if delta > 0 else []
    token3 = _chat_template_ids(
        tokenizer, [{"role": "user", "content": ""}], add_generation_prompt=True, **apply_chat_template_kwargs
    )
    generate_prompt = token3[len(token1):]
    return system_prompt, generate_prompt


class DatasetPadMode(str, Enum):
    """Padding mode for dataset"""
    RIGHT = "right"   # only right for our experiments
    LEFT_RIGHT = "left_right"
    NO_PADDING = "no_padding"


def _as_plain_dict(kwargs) -> dict:
    if kwargs is None:
        return {}
    if isinstance(kwargs, DictConfig):
        return dict(OmegaConf.to_container(kwargs, resolve=True) or {})
    return dict(kwargs)


def _expand_path(path) -> str:
    return os.path.expanduser(str(path))


def _expand_files(files):
    if files is None:
        return None
    if isinstance(files, list | ListConfig):
        return [_expand_path(p) for p in files]
    return _expand_path(files)


def _as_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in mapping:
        known = ", ".join(sorted(mapping))
        raise ValueError(f"unknown dtype {name!r}; expected one of {known}")
    return mapping[key]


def _flash_unavailable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    keys = ("flashattention", "flash_attn", "flash attention", "flash_attention")
    return any(key in text for key in keys)


def masked_lm_loss(
    logits: torch.Tensor,  # bsz, seq_len, vocab_size
    input_ids: torch.Tensor,  # bsz, seq_len
    loss_mask: torch.Tensor,  # bsz, seq_len
) -> tuple[torch.Tensor, float, float]:
    """答案 token 的 NLL 总和、token 数、entropy 总和。

    NLL 用 logsumexp(z) - z_y，按时间切成小块，避免一次做整段 fp32 log_softmax。
    entropy 只写日志，放在 no_grad 里，不挂到 NLL 的计算图上。
    """
    logits_shift = logits[:, :-1]
    target = input_ids[:, 1:]
    mask = loss_mask[:, 1:].to(dtype=torch.float32)
    nll_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    ent_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    token_count = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, logits_shift.shape[1], _LOSS_CHUNK):
        chunk_logits = logits_shift[:, start:start + _LOSS_CHUNK].float()
        chunk_target = target[:, start:start + _LOSS_CHUNK]
        chunk_mask = mask[:, start:start + _LOSS_CHUNK]
        lse = torch.logsumexp(chunk_logits, dim=-1)
        chosen = chunk_logits.gather(-1, chunk_target.unsqueeze(-1)).squeeze(-1)
        nll = lse - chosen
        nll_sum = nll_sum + (nll * chunk_mask).sum()
        token_count = token_count + chunk_mask.sum()
        with torch.no_grad():
            logp = chunk_logits.detach() - lse.detach().unsqueeze(-1)
            entropy = -(logp.exp() * logp).sum(dim=-1)
            ent_sum = ent_sum + (entropy * chunk_mask).sum()
            del logp
        del chunk_logits
    n_tok = float(token_count.detach().item())
    if n_tok <= 0:
        raise ValueError("No supervised target tokens")
    return nll_sum, n_tok, float(ent_sum.detach().item())


def _answer_token_count(loss_mask: torch.Tensor) -> float:
    return float(loss_mask[:, 1:].sum().item())


def _window_ready(seen: int, buf_len: int, n_micro: int, grad_accum: int) -> bool:
    return buf_len == grad_accum or seen == n_micro


def _optimizer_state_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _resume_step(epoch: int, micro_offset: int, steps_per_epoch: int, grad_accum: int) -> int:
    if micro_offset % grad_accum != 0:
        raise RuntimeError(
            f"resume micro_offset={micro_offset} is not a multiple of grad_accum={grad_accum}"
        )
    return epoch * steps_per_epoch + micro_offset // grad_accum


###########    Dataset      #############


class Prompt_Base:
    def __init__(
        self,
        tokenizer,
        system_prompt: str = "You are a math expert.",
        user_prompt: str = (
            "Please solve the given math problem step by step and present the answer "
            'in the following format: "\\boxed{{X}}", where X is the answer.\n\n'
            "Question:\n{question}"
        ),
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.tokenizer = tokenizer

    def build_prompt(self, question: str):
        return (
            f"{self.system_prompt}\n"
            f"{self.user_prompt.format(question=question)}"
            "\n\nAnswer:\n"
        )


class Prompt_Instruct:
    def __init__(
        self,
        tokenizer,
        system_prompt: str = "You are a math expert.",
        user_prompt: str = (
            "Please solve the given math problem step by step and present the answer "
            'in the following format: "\\boxed{{X}}", where X is the answer.\n\n'
            "Question:\n{question}"
        ),
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.tokenizer = tokenizer

    def build_prompt(self, question: str, add_generation_prompt: bool = True):
        msg = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt.format(question=question)},
        ]
        return self.tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def _read_json_rows(json_files, max_samples: int, seed: int, shuffle: bool) -> list[dict]:
    data: list[dict] = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            rows = json.load(f)
        assert isinstance(rows, list) and rows, f"empty json: {json_file}"
        data.extend(rows)

    total = len(data)
    if max_samples > 0 and max_samples < total:
        rng = random.Random(seed)
        if shuffle:
            indices = rng.sample(range(total), max_samples)
        else:
            indices = list(range(max_samples))
        data = [data[i] for i in indices]
        print(f"[data] selected {len(data)} / {total}")
    else:
        print(f"[data] {total} from {list(json_files)}")
    return data


def _encode_example(tokenizer, prompt: str, golden_res: str, prompt_add_special: bool):
    if not str(golden_res).strip():
        raise ValueError("golden_res is empty")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is None")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=prompt_add_special)
    if not prompt_ids:
        raise ValueError("prompt encodes to no tokens")
    if prompt_ids[-1] == eos_id:
        raise ValueError("prompt encoding already ends with EOS; the answer would follow an end token")
    response_ids = tokenizer.encode(golden_res, add_special_tokens=False)
    if not response_ids:
        raise ValueError(f"golden_res encodes to no tokens: {golden_res[:80]!r}")
    if response_ids[-1] != eos_id:
        response_ids = response_ids + [eos_id]
    return prompt_ids + response_ids, len(prompt_ids)


def _filter_by_length(sequence_ls, max_length: int):
    kept = []
    dropped = 0
    max_seen = 0
    for full_ids, num_masked_tokens in sequence_ls:
        seq_len = len(full_ids)
        max_seen = max(max_seen, seq_len)
        if seq_len > max_length:
            dropped += 1
            continue
        if seq_len < 2 or num_masked_tokens >= seq_len:
            raise ValueError(
                f"sample has no next-token target: seq_len={seq_len} prompt_len={num_masked_tokens}"
            )
        kept.append((full_ids, num_masked_tokens))
    if not kept:
        raise AssertionError(
            f"all {len(sequence_ls)} samples exceed max_length={max_length} (max_seen={max_seen})"
        )
    if dropped:
        print(
            f"[data] filter length>{max_length}: "
            f"kept={len(kept)} dropped={dropped} max_seen={max_seen}"
        )
    else:
        print(f"[data] length ok: n={len(kept)} max_seen={max_seen} <= {max_length}")
    return kept


def _print_protocol(name: str, tokenizer, sequence_ls):
    full_ids, nmask = sequence_ls[0]
    boundary = full_ids[max(0, nmask - 3):nmask + 3]
    print(
        f"[data] {name} sample0 prompt_len={nmask} seq_len={len(full_ids)} "
        f"bos={tokenizer.bos_token_id} eos={tokenizer.eos_token_id} "
        f"pad={tokenizer.pad_token_id} last_id={full_ids[-1]} boundary_ids={boundary}"
    )
    print(f"[data] {name} prompt_tail={tokenizer.decode(full_ids[max(0, nmask - 12):nmask])!r}")
    print(f"[data] {name} response_head={tokenizer.decode(full_ids[nmask:nmask + 12])!r}")
    print(f"[data] {name} response_tail={tokenizer.decode(full_ids[-8:])!r}")


def _pack_example(full_ids, num_masked_tokens: int) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(full_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.zeros(input_ids.shape[0], dtype=torch.float32)
    loss_mask[num_masked_tokens:] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
    }


def _build_kept(rows, tokenizer, prompt_of, max_length: int, prompt_add_special: bool, name: str):
    sequence_ls = []
    for item in rows:
        prompt = prompt_of(str(item["question"]))
        full_ids, num_masked_tokens = _encode_example(
            tokenizer, prompt, str(item["golden_res"]), prompt_add_special
        )
        sequence_ls.append((full_ids, num_masked_tokens))
    kept = _filter_by_length(sequence_ls, max_length)
    _print_protocol(name, tokenizer, kept)
    return kept


class MySFTDataset_Base(Dataset):
    def __init__(
        self,
        json_files: str | list[str],
        tokenizer_path: str | Path,
        config: DictConfig,
        truncation="error",
        max_samples: int = -1,
    ):
        self.pad_mode = DatasetPadMode.RIGHT
        self.truncation = truncation
        self.max_length = int(config.get("max_length", 4096))
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = int(config.get("seed", 42))
        self.max_samples = max_samples
        if self.max_length < 2:
            raise ValueError(f"max_length={self.max_length} needs at least 2 tokens")

        if not isinstance(json_files, list | ListConfig):
            json_files = [json_files]
        self.json_files = [_expand_path(p) for p in json_files]
        self.tokenizer = my_hf_tokenizer(str(tokenizer_path))
        self.prompt = Prompt_Base(self.tokenizer)
        rows = _read_json_rows(self.json_files, self.max_samples, self.seed, self.shuffle)
        self.sequence_ls = _build_kept(
            rows,
            self.tokenizer,
            self.prompt.build_prompt,
            self.max_length,
            prompt_add_special=True,
            name="base",
        )
        self.lengths = [len(ids) for ids, _ in self.sequence_ls]

    def __len__(self) -> int:
        return len(self.sequence_ls)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        full_ids, num_masked_tokens = self.sequence_ls[idx]
        return _pack_example(full_ids, num_masked_tokens)


class MySFTDataset_Instruct(Dataset):
    def __init__(
        self,
        json_files: str | list[str],
        tokenizer_path: str | Path,
        config: DictConfig,
        truncation="error",
        max_samples: int = -1,
    ):
        self.pad_mode = DatasetPadMode.RIGHT
        self.truncation = truncation
        self.max_length = int(config.get("max_length", 4096))
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = int(config.get("seed", 42))
        self.max_samples = max_samples
        if self.max_length < 2:
            raise ValueError(f"max_length={self.max_length} needs at least 2 tokens")

        if not isinstance(json_files, list | ListConfig):
            json_files = [json_files]
        self.json_files = [_expand_path(p) for p in json_files]
        self.tokenizer = my_hf_tokenizer(str(tokenizer_path))
        self.prompt = Prompt_Instruct(self.tokenizer)
        rows = _read_json_rows(self.json_files, self.max_samples, self.seed, self.shuffle)
        # 生成提示停在 assistant 开头，模板不会补 <|im_end|>，编码时再接 eos。
        self.sequence_ls = _build_kept(
            rows,
            self.tokenizer,
            self.prompt.build_prompt,
            self.max_length,
            prompt_add_special=False,
            name="instruct",
        )
        self.lengths = [len(ids) for ids, _ in self.sequence_ls]

    def __len__(self) -> int:
        return len(self.sequence_ls)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        full_ids, num_masked_tokens = self.sequence_ls[idx]
        return _pack_example(full_ids, num_masked_tokens)


class LengthBucketBatchSampler(Sampler[list[int]]):
    """先打乱，再在局部按长度排序后切 batch，最后打乱 batch 顺序。

    bucket_batches * batch_size 能被 batch_size 整除，所以一个 epoch 的 batch 数
    等于 ceil(n / batch_size)，只有最后一个 batch 可能不满。
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_batches: int,
        seed: int,
        shuffle: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size={batch_size}")
        if bucket_batches <= 0:
            raise ValueError(f"bucket_batches={bucket_batches}")
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.bucket_span = int(batch_size) * int(bucket_batches)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.skip = 0
        n = len(self.lengths)
        self._n_batches = math.ceil(n / self.batch_size) if n else 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._n_batches

    def _batches_for_epoch(self, epoch: int) -> list[list[int]]:
        rng = random.Random(self.seed + epoch * 1_000_003)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_span):
            chunk = indices[start:start + self.bucket_span]
            chunk.sort(key=lambda i: (self.lengths[i], i))
            for b in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b:b + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        epoch = getattr(self, "epoch", 0)
        batches = self._batches_for_epoch(epoch)
        if len(batches) != self._n_batches:
            raise RuntimeError(f"sampler produced {len(batches)} batches, expected {self._n_batches}")
        skip = self.skip
        if skip < 0 or skip > len(batches):
            raise ValueError(f"sampler skip={skip} outside 0..{len(batches)}")
        yield from batches[skip:]


class SFTTensorCollator:
    """把不等长样本右补到这个 batch 的最长长度，再对齐到 pad_to_multiple。"""

    def __init__(self, pad_token_id: int, pad_to_multiple: int = 8):
        if pad_token_id is None:
            raise ValueError("pad_token_id is None")
        self.pad_token_id = int(pad_token_id)
        self.pad_to_multiple = int(pad_to_multiple)
        if self.pad_to_multiple < 1:
            raise ValueError(f"pad_to_multiple={pad_to_multiple}")

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        longest = max(int(item["input_ids"].shape[0]) for item in batch)
        multiple = self.pad_to_multiple
        if multiple > 1:
            longest = ((longest + multiple - 1) // multiple) * multiple
        ids, attn, loss, pos = [], [], [], []
        for item in batch:
            pad = longest - int(item["input_ids"].shape[0])
            ids.append(F.pad(item["input_ids"], (0, pad), value=self.pad_token_id))
            attn.append(F.pad(item["attention_mask"], (0, pad), value=0))
            loss.append(F.pad(item["loss_mask"], (0, pad), value=0))
            pos.append(torch.arange(longest, dtype=torch.long))
        return {
            "input_ids": torch.stack(ids, dim=0),
            "attention_mask": torch.stack(attn, dim=0),
            "loss_mask": torch.stack(loss, dim=0),
            "position_ids": torch.stack(pos, dim=0),
        }


#################################


class SFTTrainer:
    """单卡 HuggingFace SFT。"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")
        self.deterministic = bool(config.trainer.get("deterministic", False))
        seed = int(config.trainer.get("seed", 1))
        set_seed(seed, self.deterministic)
        print(f"[train] seed={seed} deterministic={self.deterministic}")

        self._build_logger()
        self._build_dataset()
        self.tokenizer = self.train_dataset.tokenizer
        self._build_dataloader()
        self.resume_path = self._resolve_resume_path()
        self._build_model()
        self._build_optim()
        self._load_trainer_state()
        self._prepare_probe_samples()
        self._logged_state_dtype = False
        self._last_saved_step = -1

    def _build_logger(self):
        self.swanlab_logger = SwanlabLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def _build_dataset(self):
        tok_path = _expand_path(self.config.model.get("tokenizer_path") or self.config.model.path)
        self.train_dataset = MySFTDataset_Base(
            json_files=_expand_files(self.config.data.train_files),
            tokenizer_path=tok_path,
            config=self.config.data,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        if self.config.data.val_files:
            self.val_dataset = MySFTDataset_Base(
                json_files=_expand_files(self.config.data.val_files),
                tokenizer_path=tok_path,
                config=self.config.data,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        else:
            self.val_dataset = None

    def _build_dataloader(self):
        micro = int(self.config.data.micro_batch_size_per_gpu)
        global_bsz = int(self.config.data.train_batch_size)
        epochs = int(self.config.trainer.total_epochs)
        if micro <= 0 or global_bsz <= 0 or epochs <= 0:
            raise ValueError(f"micro={micro} global_batch={global_bsz} epochs={epochs} must be positive")
        if global_bsz % micro != 0:
            raise ValueError(f"train_batch_size={global_bsz} 必须能被 micro_batch_size_per_gpu={micro} 整除")
        self.micro_batch_size = micro
        self.global_batch_size = global_bsz
        self.grad_accum = global_bsz // micro
        bucket_batches = int(self.config.data.get("bucket_batches", 32) or 32)
        pad_to = int(self.config.data.get("pad_to_multiple", 8) or 1)
        if bucket_batches <= 0 or pad_to <= 0:
            raise ValueError(f"bucket_batches={bucket_batches} pad_to_multiple={pad_to}")

        num_workers = int(self.config.data.get("num_workers", 0))
        pin_memory = self.device.type == "cuda"
        self.collate_fn = SFTTensorCollator(self.tokenizer.pad_token_id, pad_to)
        seed = int(self.config.trainer.get("seed", 1))
        self.train_batch_sampler = LengthBucketBatchSampler(
            lengths=self.train_dataset.lengths,
            batch_size=micro,
            bucket_batches=bucket_batches,
            seed=seed,
            shuffle=True,
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.n_micro = len(self.train_dataloader)
        self.steps_per_epoch = math.ceil(self.n_micro / self.grad_accum)
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch < 1")

        if self.val_dataset is not None:
            self.val_batch_sampler = LengthBucketBatchSampler(
                lengths=self.val_dataset.lengths,
                batch_size=micro,
                bucket_batches=bucket_batches,
                seed=seed,
                shuffle=False,
            )
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_sampler=self.val_batch_sampler,
                collate_fn=self.collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            self.val_dataloader = None

        achievable = self.steps_per_epoch * epochs
        configured = self.config.trainer.total_training_steps
        if configured is None:
            self.total_training_steps = achievable
        else:
            configured = int(configured)
            if configured <= 0:
                raise ValueError(f"total_training_steps={configured}")
            if configured > achievable:
                print(
                    f"[train] total_training_steps {configured} exceeds the epoch loop "
                    f"({achievable}); stop at {achievable}"
                )
                self.total_training_steps = achievable
            else:
                self.total_training_steps = configured

        self.save_freq = self._as_freq(self.config.trainer.save_freq)
        self.test_freq = self._as_freq(self.config.trainer.test_freq)
        self.sample_freq = int(self.config.trainer.get("sample_freq", 100) or 0)
        self.n_probe_samples = int(self.config.trainer.get("n_probe_samples", 4) or 0)
        print(
            f"[train] device={self.device} samples={len(self.train_dataset)} "
            f"micro={self.micro_batch_size} accum={self.grad_accum} "
            f"micros/epoch={self.n_micro} steps/epoch={self.steps_per_epoch} "
            f"total_steps={self.total_training_steps}"
        )
        print(
            f"[data] dynamic right-pad align={pad_to} bucket_batches={bucket_batches}; "
            "short tail windows take one optimizer step on their own token mean"
        )
        print("[train] train/loss is answer-token mean NLL. ppl = exp(min(loss, 20)) on those tokens.")

    def _as_freq(self, value) -> int:
        if value is None:
            return 0
        if value == "after_each_epoch":
            return self.steps_per_epoch
        return int(value)

    def _resolve_resume_path(self) -> Path | None:
        explicit = self.config.trainer.get("resume_from_path", None)
        mode = str(self.config.trainer.get("resume_mode", "disable") or "disable").lower()
        if explicit:
            path = Path(_expand_path(explicit))
            print(f"[resume] path={path}")
            return path
        if mode != "auto":
            print(f"[resume] mode={mode}; start from scratch")
            return None
        root = Path(_expand_path(self.config.trainer.default_local_dir))
        found = []
        if root.is_dir():
            for path in root.glob("global_step_*"):
                if (path / "trainer_state.pt").is_file():
                    found.append(path)
        if not found:
            print(f"[resume] auto found no trainer_state under {root}")
            return None
        path = max(found, key=lambda p: int(p.name.rsplit("_", 1)[-1]))
        print(f"[resume] auto -> {path}")
        return path

    def _build_model(self):
        trust = bool(self.config.model.get("trust_remote_code", False))
        dtype = _as_dtype(self.config.engine.get("dtype", "bfloat16"))
        resume = self.resume_path
        adapter_resume = resume is not None and (resume / "adapter_config.json").is_file()
        if resume is not None and not adapter_resume:
            model_path = str(resume)
            print(f"[resume] load full weights from {resume}")
        else:
            model_path = _expand_path(self.config.model.path)
        load_kwargs = dict(torch_dtype=dtype, trust_remote_code=trust)
        attn = str(self.config.model.get("attn_implementation", "flash_attention_2"))
        if self.deterministic and attn == "flash_attention_2":
            print("[model] deterministic=True uses sdpa; flash_attention_2 is not the deterministic path")
            attn = "sdpa"
        self.model = self._load_model(model_path, attn, load_kwargs)
        self.model.to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.use_cache = False
        if self.config.model.get("enable_gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        lora_rank = int(self.config.model.get("lora_rank", 0) or 0)
        lora_adapter_path = self.config.model.get("lora_adapter_path", None)
        if adapter_resume:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(resume), is_trainable=True)
            self.model.print_trainable_parameters()
        elif lora_adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model, _expand_path(lora_adapter_path), is_trainable=True
            )
            self.model.print_trainable_parameters()
        elif lora_rank > 0 and resume is None:
            from peft import LoraConfig, TaskType, get_peft_model

            target = self.config.model.get("target_modules", "all-linear")
            if isinstance(target, list | ListConfig):
                target = list(target)
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=int(self.config.model.get("lora_alpha", 16) or 16),
                lora_dropout=self.config.model.get("lora_dropout", 0.05),
                target_modules=target,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()
        else:
            print("[model] full fine-tune")

        impl = getattr(self.model.config, "_attn_implementation", attn)
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        dtypes = sorted({str(p.dtype).replace("torch.", "") for p in self.trainable_params})
        n_param = sum(p.numel() for p in self.trainable_params)
        print(f"[model] attn={impl} dtype={dtype} trainable={n_param} trainable_dtype={dtypes}")

    def _load_model(self, model_path: str, attn: str, load_kwargs: dict):
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_path, attn_implementation=attn, **load_kwargs
            )
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            if attn != "flash_attention_2" or not _flash_unavailable(exc):
                raise
            print(f"[model] flash_attention_2 unavailable ({exc}); load sdpa")
            return AutoModelForCausalLM.from_pretrained(
                model_path, attn_implementation="sdpa", **load_kwargs
            )

    def _build_optim(self):
        lr = float(self.config.optim.lr)
        weight_decay = float(self.config.optim.get("weight_decay", 0.01))
        betas = tuple(float(x) for x in self.config.optim.get("betas", [0.9, 0.999]))
        self.clip_grad = float(self.config.optim.get("clip_grad", 1.0))
        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=lr, betas=betas, weight_decay=weight_decay
        )

        warmup = int(self.config.optim.get("lr_warmup_steps", -1))
        if warmup < 0:
            ratio = float(self.config.optim.get("lr_warmup_steps_ratio", 0.0))
            warmup = int(ratio * self.total_training_steps)
        sched_type = str(self.config.optim.get("lr_scheduler_type", "cosine"))
        if sched_type == "constant" and warmup > 0:
            sched_type = "constant_with_warmup"
        self.lr_scheduler = get_scheduler(
            name=sched_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=self.total_training_steps,
        )

    def _load_trainer_state(self):
        self.start_step = 0
        self.start_epoch = 0
        self.start_micro_offset = 0
        if self.resume_path is None:
            return
        state_path = self.resume_path / "trainer_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"{self.resume_path} has weights but no trainer_state.pt; "
                "refusing to restart the learning-rate schedule from step 0"
            )
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(state_path, map_location="cpu")
        if int(state.get("version", 0)) != _CKPT_VERSION:
            raise RuntimeError(f"unsupported trainer_state version: {state.get('version')}")
        if int(state["steps_per_epoch"]) != self.steps_per_epoch or int(state["grad_accum"]) != self.grad_accum:
            raise RuntimeError(
                "resume batch geometry differs from this run: "
                f"ckpt steps/epoch={state['steps_per_epoch']} accum={state['grad_accum']} "
                f"now steps/epoch={self.steps_per_epoch} accum={self.grad_accum}"
            )
        saved_files = state.get("train_files")
        current_files = [str(p) for p in self.train_dataset.json_files]
        if saved_files and list(saved_files) != current_files:
            print(f"[resume] train_files differ\n  ckpt={saved_files}\n  now={current_files}")
        self.optimizer.load_state_dict(state["optimizer"])
        _optimizer_state_to_device(self.optimizer, self.device)
        self.lr_scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"].get("cuda"):
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        self.start_step = int(state["global_step"])
        self.start_epoch = int(state["epoch"])
        self.start_micro_offset = int(state["micro_offset"])
        expected = _resume_step(
            self.start_epoch, self.start_micro_offset, self.steps_per_epoch, self.grad_accum
        )
        if expected != self.start_step:
            raise RuntimeError(f"ckpt step {self.start_step} does not match position step {expected}")
        epochs = int(self.config.trainer.total_epochs)
        if self.start_epoch > epochs or self.start_micro_offset > self.n_micro:
            raise RuntimeError(
                f"resume position epoch={self.start_epoch} offset={self.start_micro_offset} "
                f"is outside epochs={epochs} micros={self.n_micro}"
            )
        print(
            f"[resume] step={self.start_step} next_epoch={self.start_epoch} "
            f"skip_micro={self.start_micro_offset} lr={self.lr_scheduler.get_last_lr()[0]:.3e}"
        )

    def _capture_rng(self) -> dict:
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": None,
        }
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return rng

    def _save(self, step: int, epoch: int, micro_offset: int):
        path = Path(_expand_path(self.config.trainer.default_local_dir)) / f"global_step_{step}"
        path.mkdir(parents=True, exist_ok=True)
        try:
            from peft import PeftModel

            is_peft = isinstance(self.model, PeftModel)
        except Exception:
            is_peft = hasattr(self.model, "peft_config")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        state = {
            "version": _CKPT_VERSION,
            "global_step": step,
            "epoch": epoch,
            "micro_offset": micro_offset,
            "steps_per_epoch": self.steps_per_epoch,
            "grad_accum": self.grad_accum,
            "train_files": [str(p) for p in self.train_dataset.json_files],
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "rng": self._capture_rng(),
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        torch.save(state, path / "trainer_state.pt")
        self._last_saved_step = step
        kind = "adapter" if is_peft else "full"
        print(f"[ckpt] {kind} + trainer_state -> {path} next_epoch={epoch} skip_micro={micro_offset}")

    @staticmethod
    def _effective_rank(matrix: torch.Tensor, center: bool = False, eps: float = 1e-12) -> float:
        """Roy-Vetterli effective rank: exp(H(normalized singular values))."""
        if matrix is None or matrix.ndim != 2 or min(matrix.shape) < 2:
            return float("nan")
        x = matrix.detach().float()
        if center:
            x = x - x.mean(dim=0, keepdim=True)
        try:
            singular = torch.linalg.svdvals(x)
        except RuntimeError:
            return float("nan")
        singular = singular[singular > eps]
        if singular.numel() == 0:
            return 0.0
        probs = singular / singular.sum()
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
        return float(torch.exp(entropy).item())

    def _prepare_probe_samples(self):
        """固定几条已经编码好的样本，看 teacher-forcing 最后一层的谱。"""
        self.probe_samples = []
        if self.n_probe_samples <= 0:
            return
        dataset = self.val_dataset if self.val_dataset is not None else self.train_dataset
        n_take = min(self.n_probe_samples, len(dataset))
        for i in range(n_take):
            ids, prompt_len = dataset.sequence_ls[i]
            self.probe_samples.append({
                "index": i,
                "gold_ids": list(ids),
                "num_masked_tokens": prompt_len,
            })
        if self.sample_freq <= 0:
            print(f"[effrank] prepared {len(self.probe_samples)} samples; sample_freq=0 so they are not logged")
        else:
            print(
                f"[effrank] prepared {len(self.probe_samples)} samples; "
                f"log every {self.sample_freq} updates. "
                "hidden rank is the teacher-forced final layer, prompt and answer mixed"
            )

    def _compute_lora_effrank(self) -> dict:
        ranks_a, ranks_b = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.ndim != 2:
                continue
            lname = name.lower()
            if "lora_a" in lname:
                ranks_a.append(self._effective_rank(param.data, center=False))
            elif "lora_b" in lname:
                ranks_b.append(self._effective_rank(param.data, center=False))
        out = {}
        if ranks_a:
            out["train/effrank_lora_A"] = sum(ranks_a) / len(ranks_a)
        if ranks_b:
            out["train/effrank_lora_B"] = sum(ranks_b) / len(ranks_b)
        return out

    @torch.no_grad()
    def _compute_hidden_effrank(self) -> dict:
        if not self.probe_samples:
            return {}
        pad_id = self.tokenizer.pad_token_id
        max_len = min(2048, max(len(s["gold_ids"]) for s in self.probe_samples))
        rows = [s["gold_ids"][:max_len] for s in self.probe_samples]
        batch = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(batch)
        for i, ids in enumerate(rows):
            batch[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            mask[i, :len(ids)] = 1
        was_training = self.model.training
        self.model.eval()
        try:
            output = self.model(
                input_ids=batch,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = output.hidden_states[-1][mask.bool()]
            if hidden.shape[0] > 2048:
                hidden = hidden[:2048]
            return {"train/effrank_hidden": self._effective_rank(hidden, center=True)}
        except Exception as exc:
            print(f"[effrank] hidden failed: {type(exc).__name__}: {exc}")
            return {}
        finally:
            if was_training:
                self.model.train()

    @torch.no_grad()
    def _eval(self) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        try:
            tot, ntok, ent_tot = 0.0, 0.0, 0.0
            for data in self.val_dataloader:
                input_ids = data["input_ids"].to(self.device, non_blocking=True)
                attn = data["attention_mask"].to(self.device, non_blocking=True)
                pos = data["position_ids"].to(self.device, non_blocking=True)
                mask = data["loss_mask"].to(self.device, non_blocking=True)
                logits = self.model(
                    input_ids=input_ids, attention_mask=attn, position_ids=pos
                ).logits
                nll_sum, n_tok, ent_sum = masked_lm_loss(logits, input_ids, mask)
                tot += float(nll_sum.item())
                ent_tot += ent_sum
                ntok += n_tok
                del logits, nll_sum
            if ntok <= 0:
                raise RuntimeError("validation has no supervised tokens")
            mean = tot / ntok
            return {
                "val/loss": mean,
                "val/ppl": math.exp(min(mean, 20.0)),
                "val/entropy": ent_tot / ntok,
            }
        finally:
            if was_training:
                self.model.train()

    def _log_optim_dtype(self):
        if self._logged_state_dtype:
            return
        found = set()
        for param in self.trainable_params:
            state = self.optimizer.state.get(param)
            if not state:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                tensor = state.get(key)
                if torch.is_tensor(tensor):
                    found.add(f"{key}:{str(tensor.dtype).replace('torch.', '')}")
        print(f"[optim] state dtype={sorted(found) if found else 'empty'}")
        self._logged_state_dtype = True

    def _optimizer_step(self, batches: list[dict]) -> dict:
        counts = [_answer_token_count(batch["loss_mask"]) for batch in batches]
        if any(count <= 0 for count in counts):
            raise ValueError("micro-batch has no answer tokens")
        window_tokens = float(sum(counts))
        n_rows = sum(int(batch["input_ids"].shape[0]) for batch in batches)
        if n_rows != self.global_batch_size:
            print(
                f"[train] short window rows={n_rows}/{self.global_batch_size} "
                f"tokens={window_tokens:.0f}"
            )
        self.optimizer.zero_grad(set_to_none=True)
        nll_total = 0.0
        ent_total = 0.0
        for batch, counted in zip(batches, counts):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            position_ids = batch["position_ids"].to(self.device, non_blocking=True)
            loss_mask = batch["loss_mask"].to(self.device, non_blocking=True)
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            ).logits
            nll_sum, n_tok, ent_sum = masked_lm_loss(logits, input_ids, loss_mask)
            if abs(n_tok - counted) > 1e-3:
                raise RuntimeError(f"token count mismatch: loss={n_tok} mask={counted}")
            (nll_sum / window_tokens).backward()
            nll_total += float(nll_sum.detach().item())
            ent_total += ent_sum
            del logits, nll_sum
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.clip_grad)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._log_optim_dtype()
        mean_loss = nll_total / window_tokens
        return {
            "train/loss": mean_loss,
            "train/ppl": math.exp(min(mean_loss, 20.0)),
            "train/entropy": ent_total / window_tokens,
            "train/grad_norm": float(grad_norm),
            "train/lr": float(self.lr_scheduler.get_last_lr()[0]),
            "train/n_tok": window_tokens,
            "train/n_micro": len(batches),
        }

    def fit(self):
        epochs = int(self.config.trainer.total_epochs)
        print(f"Total training steps: {self.total_training_steps}")
        if self.start_step >= self.total_training_steps or self.start_epoch >= epochs:
            print(f"[resume] already finished at step {self.start_step}")
            return
        self.model.train()
        global_step = self.start_step
        last_valid_metric = None
        Path(_expand_path(self.config.trainer.default_local_dir)).mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, epochs):
            skip = self.start_micro_offset if epoch == self.start_epoch else 0
            self.train_batch_sampler.set_epoch(epoch)
            self.train_batch_sampler.skip = skip
            buf = []
            yielded = 0
            pbar = tqdm(
                self.train_dataloader,
                total=self.n_micro - skip,
                desc=f"Epoch {epoch + 1}/{epochs}",
            )
            for data in pbar:
                yielded += 1
                seen = skip + yielded
                buf.append(data)
                if not _window_ready(seen, len(buf), self.n_micro, self.grad_accum):
                    continue
                metrics = self._optimizer_step(buf)
                buf = []
                global_step += 1
                if seen == self.n_micro:
                    next_epoch, next_offset = epoch + 1, 0
                else:
                    next_epoch, next_offset = epoch, seen
                pbar.set_postfix(
                    loss=f"{metrics['train/loss']:.4f}",
                    ent=f"{metrics['train/entropy']:.3f}",
                    lr=f"{metrics['train/lr']:.2e}",
                    n=int(metrics["train/n_micro"]),
                )

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = self.test_freq > 0 and global_step % self.test_freq == 0
                is_save_step = self.save_freq > 0 and global_step % self.save_freq == 0
                is_sample_step = self.sample_freq > 0 and global_step % self.sample_freq == 0
                if is_last_step or is_sample_step:
                    metrics.update(self._compute_lora_effrank())
                    metrics.update(self._compute_hidden_effrank())
                self.swanlab_logger.log(data=metrics, step=global_step)
                if is_last_step or is_save_step:
                    self._save(global_step, next_epoch, next_offset)
                if self.val_dataloader is not None and (is_last_step or is_valid_step):
                    last_valid_metric = self._eval()
                    self.swanlab_logger.log(data=last_valid_metric, step=global_step)
                if is_last_step:
                    return
            if buf:
                raise RuntimeError("epoch ended with a partial window still buffered")

        if global_step > 0 and self._last_saved_step != global_step:
            self._save(global_step, epochs, 0)
        if last_valid_metric is not None:
            self.swanlab_logger.log(data=last_valid_metric, step=global_step)


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    trainer = SFTTrainer(config=config)
    try:
        trainer.fit()
    finally:
        trainer.swanlab_logger.finish()


if __name__ == "__main__":
    main()
