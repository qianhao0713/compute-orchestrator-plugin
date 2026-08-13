#! /bin/bash
set -exo pipefail

export XPU_USE_FAST_KERNEL=0
export DISABLE_CAST_CACHE=1
export XTE_DISABLE_FAST_BF16_CACHE=1

source ../xpu_env.sh
# 关掉largeweight
export XTE_GROUPED_GEMM_LARGE_WEIGHT=0

ROOT_DIR=$(dirname -- "$(readlink -f -- "$0")")
ROOT_DIR=$(realpath ${ROOT_DIR}/../../../)
echo $ROOT_DIR

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}

export MEGATRON_LM_PATH=${WORKSPACE_DIR}/Megatron-LM/${MCORE_VERSION:-core_r0_15_0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-"/mnt/lhycpfs/lhy/swift/ModelScope"}

OUTPUT_BASEPATH=${OUTPUT_DIR:-${ROOT_DIR}/output}

SWIFT_DATA_PATH=${SWIFT_DATA_PATH:-"/mnt/lhycpfs/lhy/swift"}

PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"/mnt/lhycpfs/lhy/model/Qwen3-VL-30B-A3B-Instruct"}
DATA_PATH=${DATA_PATH:-${SWIFT_DATA_PATH}/ModelScope/datasets/AI-ModelScope/LaTeX_OCR#110000}

GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-40}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-5}

TP=${TP:-4}
EP=${EP:-4}
TRAIN_ITERS=${TRAIN_ITERS:-20}

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export NODE_RANK=${RANK:-0}
export NNODES=${WORLD_SIZE:-1}
export MASTER_PORT=${MASTER_PORT:-9988}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    OMP_NUM_THREADS=14 \
    NPROC_PER_NODE=8 \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    IMAGE_MAX_TOKEN_NUM=1024 \
    VIDEO_MAX_TOKEN_NUM=128 \
    FPS_MAX_FRAMES=16 \
    megatron sft \
    --model ${PRETRAIN_CHECKPOINT_PATH} \
    --save_safetensors true \
    --dataset ${DATA_PATH} \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion false \
    --tensor_model_parallel_size ${TP} \
    --expert_model_parallel_size ${EP} \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir ${OUTPUT_BASEPATH}/Qwen3-VL-30B-A3B-Instruct \
    --eval_steps 500 \
    --save_steps 500 \
    --max_length 8192 \
    --packing true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash \
    --logging-steps 1 \
    --train_iters ${TRAIN_ITERS}
