#!/bin/bash
#SBATCH --job-name=distributed_quantize
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err



set -x  
# Set up environment variables:
export MASTER_PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))




quant_primitive=q3_K

model_path=$REF_MODEL_PATH
out_path=$OUT_PATH

train_data=$TRAIN_DATA
ref_model=$REF_MODEL_PATH

export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

OMP_NUM_THREADS=12 poetry run srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
     src/train.py \
    --model_path $model_path \
    --ref_model $ref_model \
    --out_path $out_path \
    --train_data $train_data \
    --hessian_corr 1e-1 \
    --hessian_train_seq 4096   \
    --total_train_steps 100 \
    --lr 1e-5 \
    --global_batch_size 512  \
    --seq_len 8192 \
    --micro_batch_size 1 \
    --checkpoint_iters 50 \
    --valid_seq 64 \
    --quant_strategy typewise_Q3_K_S \
    --use_checkpointing \
