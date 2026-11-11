"""Greedy exact-match evaluation for a saved policy or a base model."""

import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpo_ranker.sample import format_prompt
from dpo_ranker.verifier import is_correct


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python scripts/eval.py CONFIG TEST_JSON")
    with Path(sys.argv[1]).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    rows = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    model_path = Path(config["output_dir"]) / "policy"
    name = str(model_path if model_path.is_dir() else config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(config.get("device", "cuda"))
    model.eval()
    hits = 0
    with torch.no_grad():
        for row in rows:
            prompt = format_prompt(tokenizer, row["question"])
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**encoded, max_new_tokens=int(config["max_new_tokens"]), do_sample=False)
            text = tokenizer.decode(output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True)
            hits += int(is_correct(text, str(row["answer"])))
    print(json.dumps({"n": len(rows), "exact_match": hits / max(len(rows), 1)}))


if __name__ == "__main__":
    main()
