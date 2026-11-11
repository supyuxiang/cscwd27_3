"""Run one DPO-Ranker configuration."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpo_ranker.pipeline import run


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/run.py configs/math_qwen7b.yaml")
    with Path(sys.argv[1]).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run(config)


if __name__ == "__main__":
    main()
