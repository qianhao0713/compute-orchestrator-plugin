#!/bin/bash
set -eox pipefail

ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
    ipcrm -m $shmid
    echo "Deleted shared memory segment with ID: $shmid"
done

ROOT_DIR=$(dirname -- "$(readlink -f -- "$0")")
ROOT_DIR=$(realpath ${ROOT_DIR}/../../)
echo $ROOT_DIR

source ../xpu_env.sh

### XPU ENV FOR MOE ###
export HYDRAX_USE_PROTEUS=0
export XMLIR_BATCH_PARALLEL=1
export XTE_DISABLE_MOE_DW_FUSION=0
export XTE_GROUPED_GEMM_LARGE_WEIGHT=1

export TINTER_RES=bf16
export XME_OUTPUT_HEAD_REDUCE_MEMORY=1
### XPU ENV FOR MOE ###

# --- 多机配置 ---
NPROC_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"13622"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

# NODE_RANK 和 MASTER_ADDR 必须由环境变量传入
if [ -z "$NODE_RANK" ]; then
    echo "ERROR: NODE_RANK not set. Use NODE_RANK=0 for master, NODE_RANK=1 for worker."
    exit 1
fi
if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: MASTER_ADDR not set. Set it to the IP of the master node (NODE_RANK=0)."
    exit 1
fi

### MODEL && DATASET ###
STORAGE_PATH=${STORAGE_PATH:-/mnt/lhycpfs/lhy/}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STORAGE_PATH}/model/Qwen3.5-35B-A3B}
export DATASET_PATH=${DATASET_PATH:-${STORAGE_PATH}/dataset/Chinese-DeepSeek-R1-Distill-data-110k-SFT#1048576}
export CACHED_DATA_PATH=${CACHED_DATA_PATH:-${STORAGE_PATH}/dataset/Chinese-DeepSeek-R1-Distill-data-110k-SFT-Cache1048576/train}
export CACHED_VAL_DATA_PATH=${CACHED_VAL_DATA_PATH:-${STORAGE_PATH}/dataset/Chinese-DeepSeek-R1-Distill-data-110k-SFT-Cache1048576/val}

USE_CACHED_DATASET=${USE_CACHED_DATASET:-1}
if [[ ${USE_CACHED_DATASET} -eq 1 ]]; then
    DATASET_ARGS=(
        --cached_dataset "$CACHED_DATA_PATH"
        --cached_val_dataset "$CACHED_VAL_DATA_PATH"
    )
else
    DATASET_ARGS=(
        --dataset "$DATASET_PATH"
    )
fi
### MODEL && DATASET ###

### BASE CONFIG ###
BATCH_SIZE=${BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-5}
SEQ_LEN=${SEQ_LEN:-8192}
PAD_LEN=${SEQ_LEN}
SAVE_STEPS=${SAVE_STEPS:-4000}
EVAL_STEPS=${EVAL_STEPS:-4000}
EVAL_ITERS=${EVAL_ITERS:-0}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}
export SWIFT_USE_MCORE_GDN=${SWIFT_USE_MCORE_GDN:-1}
### BASE CONFIG ###

### PARALLEL ###
if [[ $SEQ_LEN == "8192" ]]; then
    TP=${TP:-1}
    PP=${PP:-1}
    EP=${EP:-4}
elif [[ $SEQ_LEN == "32768" ]]; then
    TP=${TP:-2}
    PP=${PP:-2}
    EP=${EP:-2}
fi

if [[ ${EP} -ne 1 ]]; then
    if [[ ${EP} -eq ${TP} ]]; then
        MOE_BACKEND_ARGS=(
            --moe_token_dispatcher_type allgather
        )
    else
        MOE_BACKEND_ARGS=(
            --moe_token_dispatcher_type flex
            --moe_enable_deepep true
        )
        # xpu env for deepep
        export BKCL_RDMA_VERBS=1
        export XSHMEM_MODE=1
        export XSHMEM_QP_NUM_PER_RANK=64
    fi
else
    MOE_BACKEND_ARGS=(
        --moe_token_dispatcher_type alltoall
    )
fi
### PARALLEL ###

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=$NPROC_PER_NODE \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
megatron sft \
    --model ${CHECKPOINT_PATH} \
    --logging_steps 1 \
    --save_safetensors true \
    ${DATASET_ARGS[@]} \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --target_modules all-linear \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --expert_model_parallel_size ${EP} \
    --sequence_parallel true \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap false \
    --moe_aux_loss_coeff 1e-3 \
    ${MOE_BACKEND_ARGS[@]} \
    --micro_batch_size ${BATCH_SIZE} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --recompute_granularity ${RECOMPUTE_GRANULARITY} \
    --recompute_method ${RECOMPUTE_METHOD} \
    --recompute_num_layers ${RECOMPUTE_NUM_LAYERS} \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --use_distributed_optimizer true \
    --lr ${LR}  \
    --lr_warmup_fraction 0.05 \
    --min_lr ${MIN_LR}  \
    --output_dir ${OUTPUT_DIR} \
    --eval_iters ${EVAL_ITERS} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_length ${SEQ_LEN} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --dataset_num_proc ${DATASET_NUM_PROC} \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --packing true \
    --group_by_length false \
    --model_author swift \
    --model_name swift-robot
