"""Write the IRAKer train json as verl parquet.

The prompt is the plain base-model string from ``build_prompt`` (no chat
template). It is stored as a single user turn; the GRPO launcher replaces the
tokenizer chat template with an identity template so that turn is emitted as-is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Dataset

from ranker.core import build_prompt

SRC = Path("/home/yxfeng/datasets/sft_qwen2.5-3B/en/train_sft_Qwen_math7b_instruct_correct.json")
DEST = Path("/data3/yxfeng/cscwd27_iraker/run1/verl_data")


def main() -> None:
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    records = []
    for index, row in enumerate(rows):
        question = str(row["question"])
        records.append(
            {
                "data_source": "iraker_math",
                "prompt": [{"role": "user", "content": build_prompt(question)}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": str(row["answer"])},
                "extra_info": {"index": index, "split": "train", "question": question},
            }
        )
    DEST.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(records).to_parquet(DEST / "train.parquet")
    Dataset.from_list(records[:32]).to_parquet(DEST / "val.parquet")
    print(f"wrote {len(records)} train rows and 32 val rows to {DEST}")


if __name__ == "__main__":
    main()
