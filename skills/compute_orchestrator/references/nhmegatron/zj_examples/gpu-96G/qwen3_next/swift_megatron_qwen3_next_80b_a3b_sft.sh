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

export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-true}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SWIFT_USE_MCORE_GDN=${SWIFT_USE_MCORE_GDN:-1}
export HYDRAX_USE_PROTEUS=${HYDRAX_USE_PROTEUS:-0}

unset XFA_GEMM_TYPE
unset XFA_BWD_USE_DS_SCALE
unset XTE_GROUPED_GEMM_LARGE_WEIGHT
unset XMLIR_BATCH_PARALLEL

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

STORAGE_PATH=${STORAGE_PATH:-/mnt/lhycpfs/lhy/}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STORAGE_PATH}/model/Qwen3-Next-80B-A3B-Instruct}
DATASET_PATH=${DATASET_PATH:-${STORAGE_PATH}/dataset/qwen3_5/AI-ModelScope/alpaca-gpt4-data-en}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/output/Qwen3-Next-80B-A3B}

### BASE CONFIG ###
MODEL_SIZE=80B-A3B

BATCH_SIZE=${BATCH_SIZE:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-5}
SEQ_LEN=${SEQ_LEN:-65536}
SAVE_STEPS=${SAVE_STEPS:-4000}
EVAL_STEPS=${EVAL_STEPS:-4000}
TRAIN_ITERS=${TRAIN_ITERS:-100}
RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-full}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-uniform}
RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}
### BASE CONFIG ###

### PARALLEL ###
TP=${TP:-4}
PP=${PP:-4}
EP=${EP:-4}
CP=${CP:-1}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,garbage_collection_threshold:0.8' \
NPROC_PER_NODE=$NPROC_PER_NODE \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron sft \
    --model ${CHECKPOINT_PATH} \
    --dataset ${DATASET_PATH} \
    --logging_steps 1 \
    --finetune true \
    --model_type qwen3_next \
    --template qwen3_thinking \
    --save_safetensors true \
    --add_version false \
    --split_dataset_ratio 0 \
    --load_from_cache_file true \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --expert_model_parallel_size ${EP} \
    --context_parallel_size ${CP} \
    --micro_batch_size ${BATCH_SIZE} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --recompute_granularity ${RECOMPUTE_GRANULARITY} \
    --recompute_method ${RECOMPUTE_METHOD} \
    --recompute_num_layers ${RECOMPUTE_NUM_LAYERS} \
    --train_iters ${TRAIN_ITERS} \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr ${LR} \
    --lr_warmup_fraction 0.05 \
    --min_lr ${MIN_LR} \
    --output_dir ${OUTPUT_DIR} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_length ${SEQ_LEN} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --dataset_num_proc ${DATASET_NUM_PROC} \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --moe_permute_fusion true \
    --moe_token_dispatcher_type allgather \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap false \
    --moe_router_load_balancing_type aux_loss \
    --moe_aux_loss_coeff 1e-6 \
    --attention_softmax_in_fp32 false \
    --packing true
