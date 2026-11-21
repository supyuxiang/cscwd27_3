# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
from functools import partial
from pathlib import Path

from tensordict.tensorclass import NonTensorData

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

import logging

import hydra
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.distributed import destroy_global_process_group
from verl.utils.logger import log_with_rank
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class SFTTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        log_gpu_memory_usage(f"rank {torch.distributed.get_rank()}: Before SFTTrainer init", logger=logger)

        self.rank = torch.distributed.get_rank()

        self._build_config()
        self._build_dataset()

        self._build_engine()

        self._build_dataloader()

        self._init_engine()

        self._build_ckpt_handler()

        # Initialize resume-related variables
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        if self.rank == 0:
            print(self.config)

        log_gpu_memory_usage(f"rank {self.rank}: After SFTTrainer init", logger=logger)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)
        lora_train_meta = self._get_lora_train_meta()

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
            lora_train_meta=lora_train_meta,
        )

    def _get_lora_train_meta(self):
        lora_adapter_path = self.config.model.get("lora_adapter_path", None)
        lora_rank = int(getattr(self.config.model, "lora_rank", 0) or 0)

        if lora_adapter_path is None and lora_rank <= 0:
            return None

        raw_lora_alpha = self.config.model.get("lora_alpha", None)
        if raw_lora_alpha is None:
            log_with_rank(
                "LoRA is enabled but `model.lora_alpha` is not set; fallback to 0 in checkpoint metadata.",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            lora_alpha = 0
        else:
            lora_alpha = int(raw_lora_alpha)
            if lora_alpha == 0:
                log_with_rank(
                    "LoRA is enabled but `model.lora_alpha` is 0; this may lead to ineffective LoRA scaling.",
                    logger=logger,
                    rank=self.rank,
                    level=logging.WARNING,
                    log_only_rank_0=True,
                )

        task_type = self.config.model.get("task_type", None)
        if task_type is None:
            task_type = "CAUSAL_LM"

        return {
            "r": lora_rank,
            "lora_alpha": int(lora_alpha or 0),
            "task_type": str(task_type),
        }

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)

        # check profile interval
        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()

    def _validate_profiler_interval(self):
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        self.training_client = TrainingWorker(config=config)
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        # Note that in SPMD world, this abstraction has to break
        self.engine = self.training_client.engine

    def _init_engine(self):
        # patch optimizer config
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

        self.training_client.reset()

    def _build_dataset(self):
        config = self.config
        self.model_config.tokenizer = _repair_tokenizer(self.model_config.tokenizer)
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
            min_rows=int(config.data.get("min_rows", 5000)),
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                max_samples=config.data.get("val_max_samples", -1),
                min_rows=0,
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

    def _get_batch_seqlens(self, data):
        # mean over dp group
        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        batch_seqlens = batch_seqlens.to(self.device_name)  # (global_bsz // dp)
        if self.engine.get_data_parallel_size() > 1:
            output_tensor = torch.empty(
                (batch_seqlens.shape[0] * self.engine.get_data_parallel_size(),),
                dtype=batch_seqlens.dtype,
                device=self.device_name,
            )  # (global_bsz,)

        dp_group = self.engine.get_data_parallel_group()
        dp_size = self.engine.get_data_parallel_size()

        if dp_size == 1 or dp_group is None:
            return batch_seqlens.tolist()

        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * dp_size,),
            dtype=batch_seqlens.dtype,
            device=self.device_name,
        )  # (global_bsz,)

        torch.distributed.all_gather_into_tensor(
            output_tensor=output_tensor,
            input_tensor=batch_seqlens,
            group=dp_group,
        )

        batch_seqlens = output_tensor.tolist()
        return batch_seqlens

    def fit(self):
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0

        # TODO: add a unified tracking
        self.tracking = None
        if is_logging:
            _prepare_swanlab(self.config)
            logged = OmegaConf.to_container(self.config, resolve=True)
            logged["data"]["kept_rows"] = len(self.train_dataset)
            logged["data"]["candidate_pairs"] = int(getattr(self.train_dataset, "n_pairs", len(self.train_dataset)))
            logged["data"]["rows_with_eos"] = int(getattr(self.train_dataset, "n_with_eos", 0))
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=logged,
            )
            self.tracking = tracking
            _print_swanlab_url(self.config)

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        train_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"rank {self.rank}: At start of epoch {epoch}", logger=logger)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=not is_logging,
                )
            ):
                global_step += 1

                # construct tensordict
                data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                batch_seqlens = self._get_batch_seqlens(data=data)
                # this is necessary. Otherwise, it is interpreted as NonTensorStack
                batch_seqlens_ntd = NonTensorData(batch_seqlens)

                tu.assign_non_tensor(data, update_lr_scheduler=True, global_token_num=batch_seqlens_ntd)

                # start profile in SPMD mode
                if global_step == self.start_profile_step:
                    self.training_client.start_profile()
                # train for on batch
                output = self.training_client.train_batch(data=data)
                # SFT has one train_batch per step (no PPO-style mini-batch loop), so advancing
                # the profiler here is the per-step boundary; it also drives a torch.profiler
                # schedule when one is configured (its unit is one such training step).
                self.training_client.step_profile()

                if global_step == self.end_profile_step:
                    self.training_client.stop_profile()

                if self.engine.is_mp_src_rank_with_outputs():
                    metrics = tu.get(output, "metrics")

                    # TODO: we can actual accumulate metrics for N steps and perform aggregate metrics
                    for k in ["loss", "grad_norm", "lr", "mfu"]:
                        if k in metrics.keys():
                            value = metrics.pop(k)
                            metrics[f"train/{k}"] = value

                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(batch_seqlens, device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9

                    if self.engine.get_data_parallel_rank() == 0:
                        tracking.log(data=metrics, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.test_freq == 0
                is_save_step = global_step % self.save_freq == 0

                # early exit or validation step
                if is_last_step and self.val_dataloader is not None or (self.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)

                        if self.engine.is_mp_src_rank_with_outputs():
                            metrics = tu.get(output, "metrics")
                            val_losses.append(metrics["loss"])

                    if self.engine.is_mp_src_rank_with_outputs():
                        val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))
                        # average over data parallel group
                        dp_group = self.engine.get_data_parallel_group()
                        if dp_group is not None:
                            torch.distributed.all_reduce(val_loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)

                    if is_logging:
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.save_freq > 0 and is_save_step):
                    aggressive_empty_cache(force_sync=True)
                    self.ckpt_handler.save_checkpoint(step=global_step)

                if is_last_step:
                    if is_logging:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    from verl.utils.distributed import initialize_global_process_group

    initialize_global_process_group()
    trainer = SFTTrainer(config=config)
    try:
        trainer.fit()
    finally:
        tracking = getattr(trainer, "tracking", None)
        if tracking is not None:
            tracking.finish()
    destroy_global_process_group()


def _prepare_swanlab(config) -> None:
    """Put the SwanLab key where verl's Tracking looks, without printing it.

    The key comes from SWANLAB_API_KEY or the same file policy/swan.py uses.
    verl passes mode "cloud", which this SwanLab build treats as online.
    """
    logger_names = [str(name) for name in config.trainer.logger]
    if "swanlab" not in logger_names:
        return
    key = os.environ.get("SWANLAB_API_KEY", "").strip()
    key_file = os.environ.get("SWANLAB_KEY_FILE", "/home/yxfeng/cscwd27_loopfuse/logs/swanlab_api_key")
    if not key and os.path.isfile(key_file):
        try:
            key = Path(key_file).read_text(encoding="utf-8").strip()
        except OSError:
            key = ""
    if not key:
        raise RuntimeError(f"SwanLab 已打开，但没有 API key。请设置 SWANLAB_API_KEY，或把 key 放在 {key_file}")
    os.environ["SWANLAB_API_KEY"] = key
    for var in ("SWANLAB_PROJECT", "SWANLAB_EXP", "SWANLAB_WORKSPACE"):
        os.environ.pop(var, None)
    mode = os.environ.get("SWANLAB_MODE", "").strip().lower()
    if mode not in ("online", "cloud"):
        os.environ["SWANLAB_MODE"] = "cloud"
    log_dir = Path(str(config.trainer.default_local_dir)) / "swanlog"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SWANLAB_LOG_DIR"] = str(log_dir)


def _print_swanlab_url(config) -> None:
    logger_names = [str(name) for name in config.trainer.logger]
    if "swanlab" not in logger_names:
        return
    try:
        import swanlab

        run = swanlab.get_run()
    except Exception as exc:  # noqa: BLE001
        print(f"[swanlab] run url unavailable: {type(exc).__name__}", flush=True)
        return
    url = getattr(run, "url", None)
    if url:
        print(f"[swanlab] {url}", flush=True)


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_sft(config)


def _repair_tokenizer(tokenizer):
    """DeepSeek-Math's tokenizer.json is not a Llama byte-level tokenizer.

    transformers' LlamaTokenizerFast ignores that pretokenizer and drops spaces.
    Reload the same tokenizer.json with PreTrainedTokenizerFast when a probe loses spaces.
    """
    probe = "The equation of the circle"
    probe_ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
    if " " in tokenizer.decode(probe_ids):
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    name_or_path = getattr(tokenizer, "name_or_path", None)
    tokenizer_file = Path(name_or_path) / "tokenizer.json" if name_or_path else None
    if tokenizer_file is None or not tokenizer_file.is_file():
        raise RuntimeError(f"分词器会丢掉空格，且找不到 tokenizer.json：{name_or_path}")

    from transformers import PreTrainedTokenizerFast

    fixed = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    if tokenizer.eos_token is not None:
        fixed.eos_token = tokenizer.eos_token
    if tokenizer.bos_token is not None:
        fixed.bos_token = tokenizer.bos_token
    if fixed.pad_token_id is None and fixed.eos_token is not None:
        fixed.pad_token = fixed.eos_token
    fixed.name_or_path = str(name_or_path)
    repaired = fixed(probe, add_special_tokens=False)["input_ids"]
    if " " not in fixed.decode(repaired):
        raise RuntimeError(f"修复后的分词器仍然丢掉空格：{tokenizer_file}")
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[openr1-sft] reloaded tokenizer from {tokenizer_file} so spaces are kept", flush=True)
    return fixed


class OpenR1ResponseDataset(torch.utils.data.Dataset):
    """Problem text plus one generation. No chat template is applied.

    A pair is kept when the question token count plus the response token count is
    at most ``max_length``. The trained sequence is those tokens, and the eos token
    is appended only when it still fits. Loss is on the response tokens and that eos.
    """

    def __init__(self, parquet_files, tokenizer, max_length=4096, min_rows=5000, max_samples=-1):
        import pandas as pd

        if isinstance(parquet_files, (str, Path)):
            files = [str(parquet_files)]
        else:
            files = [str(path) for path in parquet_files]
        if len(files) == 0:
            raise ValueError("parquet_files is empty")

        max_length = int(max_length)
        kept_ids = []
        prompt_lengths = []
        eos_id = tokenizer.eos_token_id
        n_pairs = 0
        n_with_eos = 0
        for path in files:
            df = pd.read_parquet(path, columns=["problem", "generations"])
            problem_ls = df["problem"].tolist()
            generation_ls = df["generations"].tolist()
            for problem, responses in zip(problem_ls, generation_ls):
                if not isinstance(problem, str) or problem.strip() == "":
                    continue
                question_ids = tokenizer.encode(problem, add_special_tokens=False, truncation=False)
                if responses is None:
                    continue
                for response in responses:
                    if not isinstance(response, str) or response.strip() == "":
                        continue
                    n_pairs += 1
                    response_ids = tokenizer.encode(response, add_special_tokens=False, truncation=False)
                    if len(question_ids) + len(response_ids) > max_length:
                        continue
                    sequence = list(question_ids) + list(response_ids)
                    added_eos = False
                    if eos_id is not None and len(sequence) < max_length and sequence[-1] != eos_id:
                        sequence.append(eos_id)
                        added_eos = True
                    if len(sequence) > max_length or len(question_ids) >= len(sequence):
                        continue
                    if added_eos:
                        n_with_eos += 1
                    kept_ids.append(torch.tensor(sequence, dtype=torch.long))
                    prompt_lengths.append(len(question_ids))

        n_filtered = len(kept_ids)
        if n_filtered < int(min_rows):
            raise RuntimeError(
                f"question+response tokens <= {max_length} 的样本只有 {n_filtered} 条，"
                f"少于要求的 {int(min_rows)} 条（候选 pair {n_pairs}）。"
            )
        if max_samples is not None and int(max_samples) >= 0:
            kept_ids = kept_ids[: int(max_samples)]
            prompt_lengths = prompt_lengths[: int(max_samples)]

        self.input_ids = kept_ids
        self.prompt_lengths = prompt_lengths
        self.n_pairs = n_pairs
        self.n_with_eos = n_with_eos
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[openr1-sft] pairs={n_pairs} kept={n_filtered} with_eos={n_with_eos} "
                f"max_length={max_length} using={len(kept_ids)} no_chat_template",
                flush=True,
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        loss_mask = torch.zeros(input_ids.shape[0], dtype=torch.long)
        loss_mask[self.prompt_lengths[index] :] = 1
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "position_ids": torch.arange(input_ids.shape[0], dtype=torch.long),
        }


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1, min_rows=5000):
    """Build the plain question/response dataset. Chat templates are not used."""
    del processor
    if data_config.custom_cls.get("path", None):
        raise ValueError("policy/sft.py does not apply a chat template and does not load custom_cls")
    return OpenR1ResponseDataset(
        parquet_files=data_paths,
        tokenizer=tokenizer,
        max_length=int(data_config.get("max_length", 4096)),
        min_rows=int(min_rows),
        max_samples=int(max_samples),
    )


def _launch_four_gpus():
    """Re-exec under torchrun when this process is not already distributed."""
    if os.environ.get("RANK") is not None or os.environ.get("WORLD_SIZE") not in (None, "1"):
        return
    visible = torch.cuda.device_count()
    if visible < 4:
        raise SystemExit(
            f"默认四卡全量微调，当前可见 GPU 只有 {visible} 张。请在至少 4 张卡的分配上运行本脚本。"
        )
    script = str(Path(__file__).resolve())
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            "4",
            script,
            *sys.argv[1:],
        ],
    )


'''
python -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node 4 \
    /home/yxfeng/cscwd27_ranker/policy/sft.py
'''


if __name__ == "__main__":
    _launch_four_gpus()
    main()
