"""Answer reward for verl GRPO.

Same grader as the rest of IRAKer: 1 when the response's last \\boxed{}
matches the reference answer, otherwise 0. verl z-scores this inside each
sampled group, which is the GRPO advantage.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ranker.core import grade_text


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    del data_source, extra_info, kwargs
    return float(grade_text(solution_str or "", str(ground_truth)))
