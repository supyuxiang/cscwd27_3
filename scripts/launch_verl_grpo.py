"""Launch verl GRPO with NCCL fallbacks forced onto every Ray worker.

Ray workers do not reliably inherit the shell environment, and
``get_ppo_ray_runtime_env`` drops variables that are already set on the
driver. node5's 4090s are not all peer-connected, so NCCL aborts with
CUDA 217 unless these are present inside the worker.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

NCCL_ENV = {
    "NCCL_P2P_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
    "NCCL_IGNORE_DISABLED_P2P": "1",
    "NCCL_SHM_USE_CUDA_MEMCPY": "0",
    "NCCL_P2P_USE_CUDA_MEMCPY": "0",
    "NCCL_CUMEM_ENABLE": "0",
}
os.environ.update(NCCL_ENV)

# verl's SwanLab init runs inside a Ray actor, which does not inherit the shell.
# Without a key it tries an interactive login and the batch job dies.
_key = os.environ.get("SWANLAB_API_KEY", "").strip()
if not _key:
    _key_file = os.environ.get("SWANLAB_KEY_FILE", "/home/yxfeng/cscwd27_loopfuse/logs/swanlab_api_key")
    try:
        _key = Path(_key_file).read_text(encoding="utf-8").strip()
    except OSError:
        _key = ""
if _key:
    os.environ["SWANLAB_API_KEY"] = _key
    NCCL_ENV["SWANLAB_API_KEY"] = _key

import verl.trainer.constants_ppo as constants  # noqa: E402

_orig = constants.get_ppo_ray_runtime_env


def _with_nccl(config=None):
    runtime_env = _orig(config)
    runtime_env.setdefault("env_vars", {}).update(NCCL_ENV)
    return runtime_env


constants.get_ppo_ray_runtime_env = _with_nccl
constants.PPO_RAY_RUNTIME_ENV["env_vars"].update(NCCL_ENV)

runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
