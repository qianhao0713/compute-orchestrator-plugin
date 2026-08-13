#!/bin/bash
set -exo pipefail

ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
    ipcrm -m $shmid
    echo "Deleted shared memory segment with ID: $shmid"
done

ROOT_DIR=$(dirname -- "$(readlink -f -- "$0")")
ROOT_DIR=$(realpath ${ROOT_DIR}/../../)
echo $ROOT_DIR

source ../xpu_env.sh

export XMLIR_ENABLE_FAST_FC=true
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

unset XMLIR_FA_GEMM_TYPE
unset XFA_GEMM_TYPE
unset XFA_BWD_USE_DS_SCALE

export XFA_GEMM_TYPE=float16
export XFA_BWD_USE_DS_SCALE=1
unset XTE_GROUPED_GEMM_LARGE_WEIGHT

# --- 多机配置 ---
GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
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
CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STORAGE_PATH}/model/021-32B-A4B/021-32b-a4b-lcpt-1208-hf-dsv2}
DATASET_PATH=${DATASET_PATH:-${STORAGE_PATH}/dataset/sft-cot-10w10w/train}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/output/021-32B}

if [ "${USE_CACHED_DATASET:-1}" = "1" ]; then
   DS_FLAG=(--cached_dataset)
else
   DS_FLAG=(--dataset)
fi

IFS=':' read -r -a DATASET_ARR <<EOF
${DATASET_PATH}
EOF

### PARALLEL ###
SEQ_LEN_MODE=${SEQ_LEN_MODE:-32k}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}

if [[ $SEQ_LEN_MODE == '128k' || $SEQ_LEN -eq 131072 ]]; then
    export XMLIR_BATCH_PARALLEL=false
    export SEQ_LEN=131072
    export TP=${TP:-4}
    export PP=${PP:-1}
    export EP=${EP:-8}
    export CP=${CP:-2}

elif [[ $SEQ_LEN_MODE == '64k' || $SEQ_LEN -eq 65536 ]]; then
    export SEQ_LEN=65536
    export TP=${TP:-4}
    export PP=${PP:-1}
    export EP=${EP:-8}
    export CP=${CP:-2}
else
    export SEQ_LEN=${SEQ_LEN:-32768}
    export TP=${TP:-4}
    export PP=${PP:-4}
    export EP=${EP:-4}
    export CP=${CP:-1}
fi

MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
SAVE_STEPS=${SAVE_STEPS:-4000}
EVAL_STEPS=${EVAL_STEPS:-4000}
TRAIN_ITERS=${TRAIN_ITERS:-1000}

export RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-"full"}
export RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-"uniform"}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}
export MOE_AUX_LOSS=${MOE_AUX_LOSS:-1e-6}

### for avoid runtime check_in_cluster_bf16 to raise core trap excp
export XMLIR_DIST_CHECK_INF_NAN=false
sed -i "s/kwargs\['check_for_nan_in_grad'\] = True/kwargs['check_for_nan_in_grad'] = False/" "/root/miniconda/envs/python310_torch29_cuda/lib/python3.10/site-packages/swift/megatron/utils/megatron_lm_utils.py"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=$GPUS_PER_NODE \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
megatron pt \
    --model $CHECKPOINT_PATH \
    "${DS_FLAG[@]}" "${DATASET_ARR[@]}" \
    --logging_steps 1 \
    --finetune true \
    --save_safetensors true \
    --add_version false \
    --model_type deepseek_v2 \
    --template qwen3_thinking \
    --split_dataset_ratio 0 \
    --load_from_cache_file true \
    --tensor_model_parallel_size ${TP} \
    --pipeline_model_parallel_size ${PP} \
    --expert_model_parallel_size ${EP} \
    --context_parallel_size ${CP} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --global_batch_size ${GLOBAL_BATCH_SIZE} \
    --recompute_granularity $RECOMPUTE_GRANULARITY \
    --recompute_method $RECOMPUTE_METHOD \
    --recompute_num_layers $RECOMPUTE_NUM_LAYERS \
    --train_iters ${TRAIN_ITERS} \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
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
    --moe_shared_expert_overlap true \
    --moe_permute_fusion false \
    --moe_grouped_gemm true \
    --moe_aux_loss_coeff ${MOE_AUX_LOSS} \
    --moe_router_dtype none \
    --overlap_grad_reduce true \
    --overlap_param_gather true \
    --overlap_p2p_comm true \
    --attention_softmax_in_fp32 false \
    --packing true
