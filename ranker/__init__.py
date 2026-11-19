"""IRAKer: iterative ranker-as-reward training for math reasoning.

Pipeline
  1. sample   : vLLM samples K candidates per training question from the base model
  2. featurize: replay prompt+candidate, cache probe-layer instruction/response features
  3. ranker   : train a <0.5M parameter listwise/pointwise ranker on correct/incorrect labels
  4. sft      : rejection sampling -> keep the ranker's favourite, train SFT
  5. grpo     : keep sampling, but replace the group-relative advantage with the
                z-scored ranker score
  6. eval     : math500 / aime24 / aime25 / aime26 / amc23, avg@n with math_verify
"""

from eval.labeling import grade_text
from eval.math_ev import DEFAULT_SETS, evaluate_model
from generator.prompt import build_prompt
from generator.utils import probe_layer_index, set_seed
from .ranker_structure import ListwiseRanker, PointwiseRanker, load_ranker
from .ranker_train import train_ranker

__all__ = [
    "build_prompt",
    "grade_text",
    "probe_layer_index",
    "set_seed",
    "DEFAULT_SETS",
    "evaluate_model",
    "ListwiseRanker",
    "PointwiseRanker",
    "load_ranker",
    "train_ranker",
]
