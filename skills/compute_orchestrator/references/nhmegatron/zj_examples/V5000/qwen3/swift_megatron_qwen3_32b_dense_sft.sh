#!/bin/bash
set -exo pipefail

ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
    ipcrm -m $shmid
    echo "Deleted shared memory segment with ID: $shmid"
done

export XPU_USE_FAST_KERNEL=0
#export DISABLE_CAST_CACHE=1
#export XTE_DISABLE_FAST_BF16_CACHE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
source ../xpu_env.sh

ROOT_DIR=$(dirname -- "$(readlink -f -- "$0")")
ROOT_DIR=$(realpath ${ROOT_DIR}/../../)
echo $ROOT_DIR

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}

#export MEGATRON_LM_PATH=/mnt/lhycpfs/lhy/ludehui/${MCORE_VERSION:-core_r0_16_1}

OUTPUT_BASEPATH=${OUTPUT_DIR:-${ROOT_DIR}/output}

PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"/mnt/lhycpfs/lhy/model/Qwen3-32B"}

STORAGE_PATH=${STORAGE_PATH:-/mnt/lhycpfs/lhy/}
DATA_PATH=${DATASET_PATH:-${STORAGE_PATH}/dataset/qwen3_5/AI-ModelScope/alpaca-gpt4-data-en}

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
TP=${TP:-8}
PP=${PP:-2}
TRAIN_ITERS=${TRAIN_ITERS:-10}
SEQ_LEN=${SEQ_LEN:-32768}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
EVAL_INTERVAL=${EVAL_INTERVAL:-4000}

#RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
#RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-block}
#RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-20}

LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-6}

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export NODE_RANK=${RANK:-0}
export NNODES=${WORLD_SIZE:-1}
export MASTER_PORT=${MASTER_PORT:-9988}

export NPROC_PER_NODE=8
export LOCAL_WORLD_SIZE=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}

#export MAX_PIXELS=1003520
#export VIDEO_MAX_PIXELS=50176
#export FPS_MAX_FRAMES=12
#export IMAGE_MAX_TOKEN_NUM=1024
#export VIDEO_MAX_TOKEN_NUM=128
#export FPS_MAX_FRAMES=16

megatron sft \
    --model ${PRETRAIN_CHECKPOINT_PATH} \
    --save_safetensors true \
    --dataset ${DATA_PATH} \
    --load_from_cache_file true \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --packing true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr ${LR} \
    --lr_warmup_fraction 0.05 \
    --min_lr ${MIN_LR} \
    --save_step ${SAVE_INTERVAL} \
    --max_length ${SEQ_LEN} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --dataset_num_proc ${DATASET_NUM_PROC} \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --logging_steps 1 \
    --attention_softmax_in_fp32 false   \
    --moe_token_dispatcher_type allgather   \
    --overlap_grad_reduce true  \
    --overlap_param_gather true \
    --sequence_parallel true  \
    --overlap-grad-reduce   \
    --overlap-param-gather  \
    --moe_grouped_gemm true \
    --no-load-optim \
    --no-load-rng   \
    --train_iters ${TRAIN_ITERS}