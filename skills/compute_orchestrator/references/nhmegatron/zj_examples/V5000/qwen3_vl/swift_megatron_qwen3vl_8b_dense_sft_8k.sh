#!/bin/bash
set -exo pipefail

ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
    ipcrm -m $shmid
    echo "Deleted shared memory segment with ID: $shmid"
done

export XPU_USE_FAST_KERNEL=0
export DISABLE_CAST_CACHE=1
export XTE_DISABLE_FAST_BF16_CACHE=1

source ../xpu_env.sh
ROOT_DIR=$(dirname -- "$(readlink -f -- "$0")")
ROOT_DIR=$(realpath ${ROOT_DIR}/../../../)
echo $ROOT_DIR

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}

export MEGATRON_LM_PATH=${WORKSPACE_DIR}/Megatron-LM/${MCORE_VERSION:-core_r0_15_0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-"/mnt/lhycpfs/lhy/swift/ModelScope"}

OUTPUT_BASEPATH=${OUTPUT_DIR:-${ROOT_DIR}/output}

SWIFT_DATA_PATH=${SWIFT_DATA_PATH:-"/mnt/lhycpfs/lhy/swift"}

PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"/mnt/lhycpfs/lhy/model/Qwen3-VL-8B-Instruct"}
DATA_PATH=${DATA_PATH:-${SWIFT_DATA_PATH}/ModelScope/datasets/AI-ModelScope/LaTeX_OCR#110000}

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
TP=${TP:-4}
TRAIN_ITERS=${TRAIN_ITERS:-20}

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export NODE_RANK=${RANK:-0}
export NNODES=${WORLD_SIZE:-1}
export MASTER_PORT=${MASTER_PORT:-9988}

export NPROC_PER_NODE=8
export LOCAL_WORLD_SIZE=8

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    IMAGE_MAX_TOKEN_NUM=1024 \
    VIDEO_MAX_TOKEN_NUM=128 \
    FPS_MAX_FRAMES=16 \
    megatron sft \
    --model ${PRETRAIN_CHECKPOINT_PATH} \
    --save_safetensors true \
    --dataset ${DATA_PATH} \
    --load_from_cache_file true \
    --tensor_model_parallel_size ${TP} \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --recompute_granularity selective \
    --recompute_modules layernorm \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir ${OUTPUT_BASEPATH}/Qwen3-VL-8B-Instruct \
    --save_steps 200 \
    --vit_gradient_checkpointing false \
    --max_length 8192 \
    --dataloader_num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --logging_steps 1 \
    --model_kwargs '{"recompute_granularity": null}' \
    --dataset_num_proc 8 \
    --train_iters ${TRAIN_ITERS}
