"""Thin SwanLab wrapper. Only rank 0 logs, and a missing key or network error
degrades to console logging instead of killing the run."""

from __future__ import annotations

import os


class SwanLog:
    def __init__(self, project: str, name: str, config: dict | None = None, enabled: bool = True):
        self.run = None
        if not enabled:
            return
        key = os.environ.get("SWANLAB_API_KEY", "").strip()
        key_file = os.environ.get("SWANLAB_KEY_FILE", "/home/yxfeng/cscwd27_loopfuse/logs/swanlab_api_key")
        if not key and os.path.isfile(key_file):
            try:
                with open(key_file, encoding="utf-8") as handle:
                    key = handle.read().strip()
            except OSError:
                key = ""
        if not key:
            print("[swanlab] no API key, logging to console only", flush=True)
            return
        try:
            import swanlab

            # These must not leak into Settings as raw strings (project is parsed as JSON).
            for var in ("SWANLAB_PROJECT", "SWANLAB_EXP", "SWANLAB_MODE", "SWANLAB_WORKSPACE"):
                os.environ.pop(var, None)
            swanlab.login(api_key=key, save=False)
            self.run = swanlab.init(
                project=project,
                name=name,
                mode="cloud",
                config=config or {},
            )
            print(f"[swanlab] run {getattr(self.run, 'url', '(no url)')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[swanlab] disabled after error: {type(exc).__name__}: {exc}", flush=True)
            self.run = None

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.run is None:
            return
        try:
            if step is None:
                self.run.log(metrics)
            else:
                self.run.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"[swanlab] log failed: {type(exc).__name__}: {exc}", flush=True)

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            pass
        self.run = None
