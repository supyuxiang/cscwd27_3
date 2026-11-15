# IRAKer

Reimplementing the **Language Ranker** idea ([arXiv:2510.21883](https://arxiv.org/abs/2510.21883))
for math reasoning, then using that ranker as the reward source for two different
post-training routes on **Qwen2.5-3B (base)**.

The paper's claim: the hidden states a model already computes contain enough
signal to *rerank* its own candidates, so a tiny learned head (<0.5M params)
beats a much larger reward model at a fraction of the cost. This repo turns that
head into a training signal.

---

## 1. The ranker

Given a question `q` and `K` sampled completions `{r_1..r_K}`:

* take the hidden state at layer `round(0.6 * (L-1))` (the paper finds ~60%
  depth best; for Qwen2.5-3B, `L=36` -> layer 21) at
  * the **last prompt token** -> instruction feature `i`
  * the **last token of each completion** -> response features `{r_k}`
* feed them to one of two heads:

| head | structure | params |
|---|---|---|
| pointwise | shared `Proj(2048→128)` + MLP, score = `scale · cos(ĩ, r̃_k)` | 394K |
| listwise | same projection, one transformer block over `[i, r_1..r_K]`, score = `scale · cos(ĩ, r̃_k)` | 460K |

Labels are 0/1 (the `\boxed{}` answer matches the reference). Pointwise uses BCE;
listwise uses KL between the softmax of scores and the normalised label
distribution over the group, which is the paper's `J_cls`.

## 2. Two ways to use it as a reward

**Route A - rejection-sampling SFT.** Score all `K` candidates per question,
keep the ranker's top-1, then SFT on `prompt + chosen` with the loss only on the
answer tokens.

**Route B - GRPO with the ranker replacing the advantage.** Standard GRPO uses
the z-scored verifier reward inside a group. Here the reward is the ranker score,
so `A_i = (s_i - mean(s)) / std(s)` where `s_i` is the ranker's score. The policy
is updated with the usual PPO-clipped surrogate plus a KL penalty to a frozen
reference.

## 3. Baselines

| run | data source | advantage / target |
|---|---|---|
| `baseline_sft` | teacher's `golden_res` from the training json | token-level CE |
| `baseline_grpo` | on-policy sampling | z-scored verifier 0/1 bit |
| `sft` (route A) | on-policy sampling, ranker top-1 | token-level CE |
| `grpo` (route B) | on-policy sampling | z-scored ranker score |

`baseline_sft` and `sft` differ only in *which* completion is imitated.
`baseline_grpo` and `grpo` differ only in *where the advantage comes from*.

## 4. Prompting

The base model is never given a chat template. Every stage uses exactly the
prompt the evaluator uses:

```
You are a math expert.
You must answer in English.

Question: {question} Please reason step by step and put your final answer within \boxed{}.
```

This matters: training on ChatML and evaluating on raw text measurably hurts a
base model, so tokenization is identical across sampling, SFT, GRPO and eval.

## 5. Pipeline

```
stage1_sample      vLLM, K candidates/question, grade with \boxed{}
      |
stage2_features    replay prompt+completion, cache layer-21 inst/resp features
      |
stage3_ranker      fit the <0.5M head on correct/incorrect labels
      |
      +--> analyse_pool   pass@k of the pool, ranker top-1 selection rate
      |
      +--> stage4_rejection_sft   keep ranker top-1  -> SFT -> eval
      |
      +--> stage5_grpo            ranker advantage  -> GRPO -> eval
      |
      +--> baseline_sft           teacher answers   -> SFT  -> eval
      |
      +--> baseline_grpo          verifier advantage-> GRPO -> eval
```

Run everything:

```bash
sbatch scripts/run_all.sbatch          # 2x 4090, node7
LIMIT=200 sbatch scripts/run_all.sbatch   # quick smoke test
```

Each stage is resumable: stage 1 keeps completed questions, stage 2 and 3 are
skipped when their output already covers every sampled question
(`FORCE_SAMPLE=1`, `FORCE_FEATURES=1`, `FORCE_RANKER=1` to override).

Evaluate any checkpoint:

```bash
python scripts/eval.py --model <ckpt> --n-sampling 4 --output eval.json
```

## 6. Evaluation

`math500`, `aime24`, `aime25`, `aime26`, `amc23` (English split), graded with
`math_verify` (LaTeX + expression extraction). Metric is `avg@4`: four sampled
completions per question at `T=0.7`, `top_p=0.95`, averaged over questions.

## 7. Files

```
iraker/
  core.py           prompt builder, \boxed grading, distributed helpers
  features.py       layer tap and instruction/response feature extraction
  policy.py         model loading, LoRA, DDP SFT trainer, generation
  sampling.py       vLLM candidate sampling + grading (resumable)
  ranker_model.py   pointwise / listwise heads, losses, save/load
  ranker_data.py    group splitting and padding
  ranker_train.py   ranker fitting loop with pairwise-accuracy validation
  rejection.py      ranker-scored selection helpers
  grpo.py           PPO-clipped loss with ranker or verifier advantage
  evaluate.py       math_verify evaluation over the five sets
scripts/
  stage1_sample.py  stage4_rejection_sft.py  baseline_sft.py
  stage2_features.py stage5_grpo.py          baseline_grpo.py
  stage3_ranker.py  analyse_pool.py          eval.py
  run_all.sbatch
tests/test_core.py  grading, ranker shapes/param counts, advantage math
```

## 8. Notes on the implementation

* **Features are replayed, not hooked during generation.** Caching the layer
  output mid-decode and replaying `prompt+completion` in one forward give the
  same vector, and the replay path batches properly and keeps vLLM free of
  custom hooks.
* **The ranker is frozen during RL.** Only the policy moves, so the two routes
  see the same reward signal.
* **Layer choice is a hyperparameter.** `--depth 0.6` matches the paper; it is
  stored in the ranker checkpoint so evaluation cannot silently use another
  layer.
* **Small-pool behaviour.** When a question has no correct sample, the ranker's
  top-1 is wrong by construction. `analysis_pool.json` reports that fraction
  (`all_wrong_questions`) next to the oracle ceiling, so route A's accuracy can
  be read against what the pool could even offer.
