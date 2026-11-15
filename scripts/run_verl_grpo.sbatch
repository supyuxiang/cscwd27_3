#!/bin/bash
#SBATCH --job-name=iraker-verl
#SBATCH --account=lumia
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:nvidia_geforce_rtx_4090:4
#SBATCH --mem=128G
#SBATCH --nodelist=node5
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/yxfeng/cscwd27_iraker/logs/%x-%j.out
#SBATCH --error=/home/yxfeng/cscwd27_iraker/logs/%x-%j.err

# GRPO via verl on 4x 4090. The actor, reference and vLLM rollout share the
# four cards (hybrid engine). Reward is the boxed-answer match, z-scored
# inside each group of 8.
set -euo pipefail
cd /home/yxfeng/cscwd27_iraker

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VERL_USE_UV=0
export PYTHONPATH=/home/yxfeng/cscwd27_iraker/verl:/home/yxfeng/cscwd27_iraker${PYTHONPATH:+:$PYTHONPATH}
# Ray sets CUDA_VISIBLE_DEVICES per worker. verl refuses to start if ROCR is also set.
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
# node5's 4090s are not all peer-connected. NCCL otherwise dies with
# "peer access is not supported between these two devices".
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

PY=/home/yxfeng/miniconda3/envs/verl1/bin/python
MODEL=/data3/yxfeng/models/Qwen2.5/Qwen2.5-3B
DATA=/data3/yxfeng/cscwd27_iraker/run1/verl_data
OUT=/data3/yxfeng/cscwd27_iraker/run1/verl_grpo

# Identity template: the stored user turn is already build_prompt(question).
TEMPLATE='{% for message in messages %}{{ message.content }}{% endfor %}'

mkdir -p logs "$OUT"

"$PY" scripts/prepare_verl_data.py

"$PY" scripts/launch_verl_grpo.py \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files="$DATA/train.parquet" \
    data.val_files="$DATA/val.parquet" \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.trust_remote_code=True \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    "actor_rollout_ref.model.custom_chat_template='${TEMPLATE}'" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    reward.custom_reward_function.path=/home/yxfeng/cscwd27_iraker/scripts/verl_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=iraker \
    trainer.experiment_name=grpo-verl \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=30 \
    trainer.default_local_dir="$OUT" \
    ray_kwargs.ray_init.num_cpus=32 \
    hydra.run.dir="$OUT/hydra/${SLURM_JOB_ID}" \
    "$@"
