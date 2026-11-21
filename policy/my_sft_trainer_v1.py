import os
from enum import Enum

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import json
import math
import warnings
from typing import Any, List, Dict
from dataclasses import dataclass

import hydra
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig, ListConfig
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, get_scheduler

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# This file is a copy of verl/verl/trainer/sft_trainer.py. Imports resolve from the vendored tree.
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

        self._finish=False

    def log(self,data,step):
        swanlab.log(
            data=data,step=step
        )

    def finish(self, exit_code:int=0):
        if self._finish:
            return
        self._finish=True
        swanlab.finish()
    
    def __del__(self):
        self.finish()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)
    tokenizer.padding_side = "right"
    return tokenizer


######################     directly copy from verl     ######################
def extract_system_prompt_and_generation(tokenizer, **apply_chat_template_kwargs):
    token1 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True, **apply_chat_template_kwargs
        ).input_ids
    
    token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        ).input_ids
    
    # get system prompt tokens；delta==0 时 token1[:-0] 在 Python 里是空 list
    delta = len(token2) - len(token1)
    system_prompt = token1[:-delta] if delta > 0 else []
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True, **apply_chat_template_kwargs
        ).input_ids
    
    generate_prompt = token3[len(token1) :]

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
    return mapping.get(str(name).lower(), torch.bfloat16)




def masked_lm_loss(
    logits: torch.Tensor, # bsz, seq_len, vocab_size
    input_ids: torch.Tensor, # bsz, seq_len
    loss_mask: torch.Tensor # bsz, seq_len
) -> tuple[torch.Tensor, float, float]:
    logits_shift = logits[:, :-1].float() # bsz, seq_len-1, vocab_size
    logp = F.log_softmax(logits_shift, dim=-1) # bsz, seq_len-1, vocab_size
    tgt = input_ids[:, 1:].unsqueeze(-1) # bsz, seq_len-1, 1

    nll = -logp.gather(dim=-1, indx=tgt).squeeze(-1) # bsz, seq_len-1 -> bsz, seq_len-1, 1
    entropy = -(logp.exp() * logp).sum(dim=-1) # bsz, seq_len-1
    mask = loss_mask[:, 1:] # bsz, seq_len-1
    denom = mask.sum().clamp_min(1.0)
    loss = (nll * mask).sum() / denom
    mean_entropy = (entropy * mask).sum() / denom
    return loss, float(denom.item()), float(mean_entropy.detach().item())



###########    Dataet      #############


class Prompt_Base:
    def __init__(
        self,
        tokenizer,
        system_prompt: str = 'You are a math expert.',
        user_prompt: str = (
    "Please solve the given math problem step by step and present the answer "
    'in the following format: "\\boxed{{X}}", where X is the answer.\n\n'
    "Question:\n{question}"
)
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


class MySFTDataset_Base(Dataset):
    def __init__(
        self,
        json_files: str | list[str],
        tokenizer_path: str | Path,
        config: DictConfig,
        truncation='error',
        max_samples: int = -1,
    ):
        self.pad_mode = DatasetPadMode.RIGHT
        self.truncation = truncation # error, left, right
        self.max_length = int(config.get("max_length", 4096))
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed",42)
        self.max_samples = max_samples

        if not isinstance(json_files, list | ListConfig):
            json_files = [json_files]
        self.json_files = [
            _expand_path(p) for p in json_files
        ]

        self.tokenizer = my_hf_tokenizer(str(tokenizer_path))
        self.prompt = Prompt_Base(self.tokenizer)

        self._read_files_and_process() # get self.messages

    def _read_files_and_process(self):
        data: list[dict] = []
        for json_file in self.json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                rows = json.load(f)
            assert isinstance(rows, list) and rows, f"empty json: {json_file}"
            data.extend(rows)

        total = len(data)
        if self.max_samples > 0 and self.max_samples < total:
            rng = random.Random(self.seed)
            if self.shuffle:
                indices = rng.sample(range(total), self.max_samples)
            else:
                indices = list(range(self.max_samples))
            data = [data[i] for i in indices]
            print(f"[data] selected {len(data)} / {total}")
        else:
            print(f"[data] {total} from {self.json_files}")


        sequence_ls: list[tuple[str, int]] = []
        for item in data:
            question = str(item["question"])
            golden_res = str(item["golden_res"])
            prompt = self.prompt.build_prompt(question=question)

            # 分开编码再拼接。整段重编码时，Answer:\n 会和回答开头并成另一个 token。
            prompt_ids = self.tokenizer.encode(prompt)
            response_ids = self.tokenizer.encode(golden_res, add_special_tokens=False)
            full_ids = prompt_ids + response_ids
            num_masked_tokens = len(prompt_ids)
            sequence_ls.append(
                (
                    full_ids,
                    num_masked_tokens
                )
            )

        kept: list[tuple[str, int]] = []
        dropped = 0
        max_seen = 0
        for full_ids, num_masked_tokens in sequence_ls:
            seq_len = len(full_ids)
            max_seen = max(max_seen, seq_len)

            # 提前做了filter，故后面的truncation不会生效
            if seq_len > self.max_length:
                dropped += 1
                continue
            
            kept.append(
                (
                    full_ids,
                    num_masked_tokens
                )
            )
        assert kept, (
            f"all {len(sequence_ls)} samples exceed max_length={self.max_length} "
            f"(max_seen={max_seen})"
        )

        if dropped:
            print(
                f"[data] filter length>{self.max_length}: "
                f"kept={len(kept)} dropped={dropped} max_seen={max_seen}"
            )
        else:
            print(f"[data] length ok: n={len(kept)} max_seen={max_seen} <= {self.max_length}")
        self.sequence_ls = kept


    def __len__(self) -> int:
        return len(self.sequence_ls)


    def _tokenize_sequence(self, full_ids: torch.Tensor, num_masked_tokens:int):
        input_ids = torch.tensor(full_ids)
        loss_mask = torch.zeros_like(input_ids)
        attention_mask = torch.ones_like(input_ids)

        loss_mask[num_masked_tokens:] = 1

        assert input_ids.shape == loss_mask.shape == attention_mask.shape
        return input_ids, loss_mask, attention_mask


    def _pad_or_truncate(self, input_ids, loss_mask, attention_mask, position_ids):
        seq_len = input_ids.shape[0]
        if seq_len == self.max_length:
            return input_ids, loss_mask, attention_mask, position_ids

        if seq_len < self.max_length:
            pad = self.max_length - seq_len
            pad_id = self.tokenizer.pad_token_id
            
            input_ids = F.pad(input_ids, (0, pad), value=pad_id) #左边补0个，右边补pad个pad_id
            attention_mask = F.pad(attention_mask, (0, pad), value=0) #左边补0个，右边补pad个0
            loss_mask = F.pad(loss_mask, (0, pad), value=0) #左边补0个，右边补pad个0
            position_ids = F.pad(position_ids, (0, pad), value=0) #左边补0个，右边补pad个0
            return input_ids, loss_mask, attention_mask, position_ids

        # truncation
        if self.truncation == 'error':
            raise ValueError(f"{seq_len=} is larger than {self.max_length=}")
        elif self.truncation == 'right':
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            loss_mask = loss_mask[: self.max_length]
            position_ids = position_ids[: self.max_length]
        elif self.truncation == 'left':
            start = seq_len - self.max_length
            input_ids = input_ids[start:]
            attention_mask = attention_mask[start:]
            loss_mask = loss_mask[start:]
            position_ids = position_ids[start:]
        else:
            raise NotImplementedError
        return input_ids, loss_mask, attention_mask, position_ids

        
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        full_ids, num_masked_tokens = self.sequence_ls[idx]
        input_ids, loss_mask, attention_mask = self._tokenize_sequence(full_ids, num_masked_tokens)

        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)
        
        input_ids, loss_mask, attention_mask, position_ids = self._pad_or_truncate(
            input_ids, loss_mask, attention_mask, position_ids
        )
        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "position_ids": position_ids.long(),
            "loss_mask": loss_mask.float(),
        }





class Prompt_Instruct:
    def __init__(
        self,
        tokenizer,
        system_prompt: str = 'You are a math expert.',
        user_prompt: str = (
    "Please solve the given math problem step by step and present the answer "
    'in the following format: "\\boxed{{X}}", where X is the answer.\n\n'
    "Question:\n{question}"
)
    ):

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        self.tokenizer = tokenizer

    def build_prompt(self, question: str, add_generation_prompt: bool=True):
        msg = [
            {'role':'system', 'content':self.system_prompt},
            {'role':'user','content':self.user_prompt.format(question=question)}
        ]
        return self.tokenizer.apply_chat_template(
            msg,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )



class MySFTDataset_Instruct(Dataset):
    def __init__(
        self,
        json_files: str | list[str],
        tokenizer_path: str | Path,
        config: DictConfig,
        truncation='error',
        max_samples: int = -1,
    ):
        self.pad_mode = DatasetPadMode.RIGHT
        self.truncation = truncation # error, left, right
        self.max_length = int(config.get("max_length", 4096))
        self.shuffle = bool(config.get("shuffle", False))
        self.seed = config.get("seed",42)
        self.max_samples = max_samples

        if not isinstance(json_files, list | ListConfig):
            json_files = [json_files]
        self.json_files = [
            _expand_path(p) for p in json_files
        ]

        self.tokenizer = my_hf_tokenizer(str(tokenizer_path))
        self.prompt = Prompt_Instruct(self.tokenizer)

        self._read_files_and_process()

    def _read_files_and_process(self):
        data: list[dict] = []
        for json_file in self.json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                rows = json.load(f)
            assert isinstance(rows, list) and rows, f"empty json: {json_file}"
            data.extend(rows)

        total = len(data)
        if self.max_samples > 0 and self.max_samples < total:
            rng = random.Random(self.seed)
            if self.shuffle:
                indices = rng.sample(range(total), self.max_samples)
            else:
                indices = list(range(self.max_samples))
            data = [data[i] for i in indices]
            print(f"[data] selected {len(data)} / {total}")
        else:
            print(f"[data] {total} from {self.json_files}")

        sequence_ls = []
        for item in data:
            question = str(item["question"])
            golden_res = str(item["golden_res"])
            prompt = self.prompt.build_prompt(question=question)

            # 分开编码再拼接。整段重编码时，模板结尾会和回答开头并成另一个 token。
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = self.tokenizer.encode(golden_res, add_special_tokens=False)
            full_ids = prompt_ids + response_ids
            num_masked_tokens = len(prompt_ids)
            sequence_ls.append(
                (
                    full_ids,
                    num_masked_tokens
                )
            )

        kept = []
        dropped = 0
        max_seen = 0
        for full_ids, num_masked_tokens in sequence_ls:
            seq_len = len(full_ids)
            max_seen = max(max_seen, seq_len)

            if seq_len > self.max_length:
                dropped += 1
                continue

            kept.append(
                (
                    full_ids,
                    num_masked_tokens
                )
            )
        assert kept, (
            f"all {len(sequence_ls)} samples exceed max_length={self.max_length} "
            f"(max_seen={max_seen})"
        )

        if dropped:
            print(
                f"[data] filter length>{self.max_length}: "
                f"kept={len(kept)} dropped={dropped} max_seen={max_seen}"
            )
        else:
            print(f"[data] length ok: n={len(kept)} max_seen={max_seen} <= {self.max_length}")
        self.sequence_ls = kept

    def __len__(self) -> int:
        return len(self.sequence_ls)

    def _tokenize_sequence(self, full_ids, num_masked_tokens: int):
        input_ids = torch.tensor(full_ids)
        loss_mask = torch.zeros_like(input_ids)
        attention_mask = torch.ones_like(input_ids)

        loss_mask[num_masked_tokens:] = 1

        assert input_ids.shape == loss_mask.shape == attention_mask.shape
        return input_ids, loss_mask, attention_mask

    def _pad_or_truncate(self, input_ids, loss_mask, attention_mask, position_ids):
        seq_len = input_ids.shape[0]
        if seq_len == self.max_length:
            return input_ids, loss_mask, attention_mask, position_ids

        if seq_len < self.max_length:
            pad = self.max_length - seq_len
            pad_id = self.tokenizer.pad_token_id

            input_ids = F.pad(input_ids, (0, pad), value=pad_id)
            attention_mask = F.pad(attention_mask, (0, pad), value=0)
            loss_mask = F.pad(loss_mask, (0, pad), value=0)
            position_ids = F.pad(position_ids, (0, pad), value=0)
            return input_ids, loss_mask, attention_mask, position_ids

        if self.truncation == 'error':
            raise ValueError(f"{seq_len=} is larger than {self.max_length=}")
        elif self.truncation == 'right':
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            loss_mask = loss_mask[: self.max_length]
            position_ids = position_ids[: self.max_length]
        elif self.truncation == 'left':
            start = seq_len - self.max_length
            input_ids = input_ids[start:]
            attention_mask = attention_mask[start:]
            loss_mask = loss_mask[start:]
            position_ids = position_ids[start:]
        else:
            raise NotImplementedError
        return input_ids, loss_mask, attention_mask, position_ids

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        full_ids, num_masked_tokens = self.sequence_ls[idx]
        input_ids, loss_mask, attention_mask = self._tokenize_sequence(full_ids, num_masked_tokens)

        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        input_ids, loss_mask, attention_mask, position_ids = self._pad_or_truncate(
            input_ids, loss_mask, attention_mask, position_ids
        )
        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "position_ids": position_ids.long(),
            "loss_mask": loss_mask.float(),
        }


class SFTTensorCollator:
    """只 stack 成 [B, T] dict。"""

    def __init__(self, pad_mode: DatasetPadMode | str = DatasetPadMode.RIGHT):
        self.pad_mode = pad_mode

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        keys = ("input_ids", "attention_mask", "position_ids", "loss_mask")
        return {k: torch.stack([item[k] for item in batch], dim=0) for k in keys}


#################################


class SFTTrainer:
    """单卡 HuggingFace SFT"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda')

        seed = int(config.trainer.get("seed", 1))
        set_seed(seed)

        self._build_logger()

        self._build_dataset()
        self.tokenizer = self.train_dataset.tokenizer

        self._build_dataloader()

        self._build_model()

        self._build_optim()
        
        self._prepare_probe_samples()

    def _build_logger(self):
        self.swanlab_logger = SwanlabLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            config=OmegaConf.to_container(self.config, resolve=True)
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
                json_files= _expand_files(self.config.data.val_files),
                tokenizer_path=tok_path,
                config=self.config.data,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        else:
            self.val_dataset = None

    def _build_dataloader(self):
        micro = int(self.config.data.micro_batch_size_per_gpu)
        global_bsz = int(self.config.data.train_batch_size)
        if global_bsz % micro != 0:
            raise ValueError(f"train_batch_size={global_bsz} 必须能被 micro_batch_size_per_gpu={micro} 整除")
        self.micro_batch_size = micro
        self.global_batch_size = global_bsz
        self.grad_accum = global_bsz // micro

        num_workers = int(self.config.data.get("num_workers", 0))
        pin_memory = self.device.type == "cuda"
        self.collate_fn = SFTTensorCollator(self.config.data.get("pad_mode", DatasetPadMode.RIGHT))

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.micro_batch_size, # dataloader构造时用micro来作bsz。后续会设置accumulation
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        self.steps_per_epoch = len(self.train_dataloader) // self.grad_accum
        assert self.steps_per_epoch >= 1

        if self.val_dataset:
            self.val_dataloader = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.micro_batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
        else:
            self.val_dataloader = None

        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = int(self.config.trainer.total_training_steps)
        else: # 默认是走这个
            self.total_training_steps = self.steps_per_epoch * int(self.config.trainer.total_epochs)

        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch
        
        self.sample_freq = int(self.config.trainer.get("sample_freq", 1) or 0)
        self.n_probe_samples = int(self.config.trainer.get("n_probe_samples", 4) or 0)

        print(
            f"[train] device={self.device} samples={len(self.train_dataset)} "
            f"micro={self.micro_batch_size} accum={self.grad_accum} "
            f"steps/epoch={self.steps_per_epoch} total_steps={self.total_training_steps}"
        )

    def _build_model(self):
        model_path = _expand_path(self.config.model.path)
        trust = bool(self.config.model.get("trust_remote_code", False))
        dtype = _as_dtype(self.config.engine.get("dtype", "bfloat16"))
        load_kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=trust,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, attn_implementation="flash_attention_2", **load_kwargs
            )
        except Exception as exc:
            print(f"[model] flash_attention_2 unavailable ({exc}); fallback default attn")
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

        model.to(self.device)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.use_cache = False
        if self.config.model.get("enable_gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        lora_rank = int(self.config.model.get("lora_rank", 0) or 0)
        lora_adapter_path = self.config.model.get("lora_adapter_path", None)
        if lora_adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, _expand_path(lora_adapter_path), is_trainable=True)
            model.print_trainable_parameters()
        elif lora_rank > 0:
            from peft import LoraConfig, TaskType, get_peft_model

            target = self.config.model.get("target_modules", "all-linear")
            if isinstance(target, list | ListConfig):
                target = list(target)
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=int(self.config.model.get("lora_alpha", 16) or 16),
                lora_dropout=self.config.model.get('lora_dropout',0.05),
                target_modules=target,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        else:
            print("[model] full fine-tune")

        self.model = model
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

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

    def _save(self, step: int):
        path = Path(_expand_path(self.config.trainer.default_local_dir)) / f"global_step_{step}"
        path.mkdir(parents=True, exist_ok=True)
        # PeftModel.save_pretrained writes adapter only (no merge / no full base weights).
        try:
            from peft import PeftModel

            is_peft = isinstance(self.model, PeftModel)
        except Exception:
            is_peft = hasattr(self.model, "peft_config")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f'save lora-adapter: {path}' if is_peft else f'save full: {path}') 

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
        """Fix a few train/val conversations for hidden-state effrank."""
        self.probe_samples = []
        if self.n_probe_samples <= 0:
            return
        dataset = self.val_dataset if self.val_dataset is not None else self.train_dataset
        if dataset is None or not hasattr(dataset, "messages"):
            return
        n_take = min(self.n_probe_samples, len(dataset))
        for i in range(n_take):
            messages = dataset.messages[i]
            gold_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    tokenize=True,
                    **_as_plain_dict(self.config.data.get("apply_chat_template_kwargs", {})),
                ).input_ids

            self.probe_samples.append({"index": i, "gold_ids": gold_ids})
        print(f"[effrank] prepared {len(self.probe_samples)} probe samples")

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
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_len = min(2048, max(len(s["gold_ids"]) for s in self.probe_samples))
        rows = [s["gold_ids"][:max_len] for s in self.probe_samples]
        batch = torch.full((len(rows), max_len), pad_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(batch)
        for i, ids in enumerate(rows):
            batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            mask[i, : len(ids)] = 1
        try:
            was_training = self.model.training
            self.model.eval()
            try:
                output = self.model(
                    input_ids=batch,
                    attention_mask=mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            finally:
                if was_training:
                    self.model.train()
            hidden = output.hidden_states[-1][mask.bool()]
            if hidden.shape[0] > 2048:
                hidden = hidden[:2048]
            return {"train/effrank_hidden": self._effective_rank(hidden, center=True)}
        except Exception as exc:
            print(f"[effrank] hidden failed: {type(exc).__name__}: {exc}")
            return {}

    @torch.no_grad()
    def _eval(self) -> dict[str, float]:
        self.model.eval()
        tot, ntok, ent_tot = 0.0, 0.0, 0.0
        for data in self.val_dataloader:
            input_ids = data["input_ids"].to(self.device, non_blocking=True)
            attn = data["attention_mask"].to(self.device, non_blocking=True)
            mask = data["loss_mask"].to(self.device, non_blocking=True)
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            loss, n_tok, ent = masked_lm_loss(logits, input_ids, mask)
            tot += float(loss.item()) * n_tok
            ent_tot += ent * n_tok
            ntok += n_tok
        self.model.train()
        mean = tot / max(ntok, 1.0)
        mean_ent = ent_tot / max(ntok, 1.0)
        return {
            "val/loss": mean,
            "val/ppl": math.exp(min(mean, 20.0)),
            "val/entropy": mean_ent,
        }

    def fit(self):

        print(f"Total training steps: {self.total_training_steps}")
        self.model.train()
        global_step = 0
        last_valid_metric = None
        Path(_expand_path(self.config.trainer.default_local_dir)).mkdir(parents=True, exist_ok=True)

        for epoch in range(int(self.config.trainer.total_epochs)):
            accum = 0
            step_n_tok = 0.0
            step_loss_sum = 0.0
            step_ent_sum = 0.0
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            )
            for data in pbar:
                input_ids = data["input_ids"].to(self.device, non_blocking=True)
                attn = data["attention_mask"].to(self.device, non_blocking=True)
                mask = data["loss_mask"].to(self.device, non_blocking=True)
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
                loss, n_tok, entropy = masked_lm_loss(logits, input_ids, mask)
                step_n_tok += n_tok
                step_loss_sum += float(loss.detach()) * n_tok
                step_ent_sum += entropy * n_tok
                (loss / self.grad_accum).backward()
                accum += 1
                if accum < self.grad_accum:
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, self.clip_grad)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                accum = 0
                global_step += 1

                mean_loss = step_loss_sum / max(step_n_tok, 1.0)
                mean_entropy = step_ent_sum / max(step_n_tok, 1.0)
                metrics = {
                    "train/loss": mean_loss,
                    "train/ppl": math.exp(min(mean_loss, 20.0)),
                    "train/entropy": mean_entropy,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": float(self.lr_scheduler.get_last_lr()[0]),
                    "train/n_tok": step_n_tok,
                }
                step_n_tok = 0.0
                step_loss_sum = 0.0
                step_ent_sum = 0.0
                pbar.set_postfix(
                    loss=f"{metrics['train/loss']:.4f}",
                    ent=f"{metrics['train/entropy']:.3f}",
                    lr=f"{metrics['train/lr']:.2e}",
                )

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = self.test_freq > 0 and global_step % self.test_freq == 0
                is_save_step = self.save_freq > 0 and global_step % self.save_freq == 0
                is_sample_step = self.sample_freq > 0 and global_step % self.sample_freq == 0

                if is_last_step or is_sample_step:
                    metrics.update(self._compute_lora_effrank())
                    metrics.update(self._compute_hidden_effrank())
                    if "train/effrank_hidden" in metrics:
                        pbar.set_postfix(
                            loss=f"{metrics['train/loss']:.4f}",
                            ent=f"{metrics['train/entropy']:.3f}",
                            er=f"{metrics['train/effrank_hidden']:.1f}",
                            lr=f"{metrics['train/lr']:.2e}",
                        )

                self.swanlab_logger.log(data=metrics, step=global_step)

                if self.val_dataloader is not None and (is_last_step or is_valid_step):
                    last_valid_metric = self._eval()
                    self.swanlab_logger.log(data=last_valid_metric, step=global_step)

                if is_last_step or is_save_step:
                    self._save(global_step)

                if is_last_step:
                    return

            if accum > 0:
                self.optimizer.zero_grad(set_to_none=True)

        self._save(global_step)
        if last_valid_metric is not None:
            self.swanlab_logger.log(data=last_valid_metric, step=global_step)


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    trainer = SFTTrainer(config=config)
    trainer.fit()


if __name__ == "__main__":
    main()
