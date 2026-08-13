#! /bin/bash
set -exo pipefail
source ../xpu_env.sh

ENV=dsw
ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
ROOT_DIR=$(realpath ${ROOT_DIR}/../../../)
echo $ROOT_DIR

export WORLD_SIZE=${WORLD_SIZE:-4}
SEQ_LEN_MODE=${SEQ_LEN_MODE:-32k}
PRESET_LOAD_OPTI=true
PRESET_AC="none"
PRESET_RECOMPUTE_METHOD="block"
PRESET_MP_AC_LAYERS=0
if [[ $SEQ_LEN_MODE == "32k" ]]; then
    # base config
    PRESET_GLOBAL_BATCH_SIZE=256
    PRESET_SEQ_LEN=32768
    # parallel config
    PRESET_TP=4
    PRESET_PP=1
elif [[ $SEQ_LEN_MODE == "8k" ]]; then
    # base config
    PRESET_GLOBAL_BATCH_SIZE=1024
    PRESET_SEQ_LEN=8192
    # parallel config
    PRESET_TP=1
    PRESET_PP=1
fi

#
# export CUDA_DEVICE_MAX_CONNECTIONS=1

WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace}
export PYTHONPATH=${PYTHONPATH}:${WORKSPACE_DIR}/Megatron-LM/${MCORE_VERSION:-core_r0_15_0}:${ROOT_DIR}/Megatron-LM/${MCORE_VERSION:-core_r0_15_0}

STORAGE_PATH=${STORAGE_PATH:-/mnt/lhycpfs/lhy/}
DATA_PATH=${DATASET_PATH:-${STORAGE_PATH}/dataset/pile-llama/pile-llama_text_document}
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-${STORAGE_PATH}/mbridge_ckpt/Qwen3-4B-mcore}
TOKENIZER_PATH=${TOKENIZER_PATH:-${PRETRAIN_CHECKPOINT_PATH}}
MODEL_SIZE=${MODEL_SIZE:-4B}

TP=${TP:-${PRESET_TP}}
PP=${PP:-${PRESET_PP}}
EP=${EP:-1}
ETP=${ETP:-1}
CP=${CP:-1}
SP=${SP:-true}
DO=${DO:-true}
DP_OVERLAP=${DP_OVERLAP:-true}
FL=${FL:-true}
SFT=${SFT:-false}

PR=${PR:-bf16}

LR=1e-5
MIN_LR=1e-6

SEQ_LEN=${SEQ_LEN:-${PRESET_SEQ_LEN}}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${PRESET_GLOBAL_BATCH_SIZE}}
TRAIN_ITERS=${TRAIN_ITERS:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-5000}
EVAL_ITERS=${EVAL_ITERS:-0}
EXIT_INTERVAL=${EXIT_INTERVAL:-5000}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
BATCH_SIZE=${BATCH_SIZE:-1}

AC=${AC:-${PRESET_AC}}
RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-${PRESET_RECOMPUTE_METHOD}}
MP_AC_LAYERS=${MP_AC_LAYERS:-${PRESET_MP_AC_LAYERS}}

OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}

CPT_CONTINUE=${CPT_CONTINUE:-false}
LOAD_OPTI=${LOAD_OPTI:-${PRESET_LOAD_OPTI}}
SAVE_CKPT=${SAVE_CKPT:-true}

OUTPUT_BASEPATH=${OUTPUT_DIR:-${ROOT_DIR}/output}

pkill -9 python || true
pkill -9 torchrun || true

GPUS_PER_NODE=`echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"13622"}
NNODES=${WORLD_SIZE:-"1"}
NODE_RANK=${RANK:-"0"}

DISTRIBUTED_ARGS=()
MODEL_ARGS=()
DATA_ARGS=()
TRAINING_ARGS=()


if [ $MODEL_SIZE = 0.6B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=1024
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=3072
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
elif [ $MODEL_SIZE = 1.7B ]; then
    NUM_LAYERS=28
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=6144
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
elif [ $MODEL_SIZE = 4B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=2560
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=9728
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
elif [ $MODEL_SIZE = 8B ]; then
    NUM_LAYERS=36
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=12288
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    MODEL_ARGS+=(--untie-embeddings-and-output-weights)
elif [ $MODEL_SIZE = 14B ]; then
    NUM_LAYERS=40
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=40
    INTERMEDIATE_SIZE=17408
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    MODEL_ARGS+=(--untie-embeddings-and-output-weights)
elif [ $MODEL_SIZE = 32B ]; then
    NUM_LAYERS=64
    HIDDEN_SIZE=5120
    NUM_ATTENTION_HEADS=64
    INTERMEDIATE_SIZE=25600
    NUM_KEY_VALUE_HEADS=8
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    ROPE_THETA=1000000
    RMS_NORM_EPS=1e-6
    MODEL_ARGS+=(--untie-embeddings-and-output-weights)
elif [ $MODEL_SIZE = A3B ]; then
    NUM_LAYERS=48
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=32
    INTERMEDIATE_SIZE=6144
    MOE_INTERMEDIATE_SIZE=768
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    MODEL_ARGS+=(--untie-embeddings-and-output-weights)

    if [ ${ENABLE_DEEPEP:-false} = true ]; then
        MODEL_ARGS+=(--moe-enable-deepep --moe-router-dtype=fp32)
        MOE_TOKEN_DISPATCHER_TYPE=flex
    fi

    MODEL_ARGS+=(
        --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER_TYPE:-allgather}
        --moe-router-topk ${ROUTER_TOPK}
        --num-experts ${NUM_EXPERTS}
        --expert-tensor-parallel-size ${ETP}
        --expert-model-parallel-size ${EP}
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE}
        --moe-router-load-balancing-type aux_loss
        --moe-aux-loss-coeff 0.001
        --moe-layer-freq "([1]*${NUM_LAYERS})"
    )

    if [ ${ENABLE_GROUPED_GEMM:-true} = true ]; then
        MODEL_ARGS+=(--moe-grouped-gemm)
        if [ ${MOE_USE_LEGACY_GROUPED_GEMM:-false} = true ]; then
            MODEL_ARGS+=(--moe-use-legacy-grouped-gemm)
        fi
    fi

elif [ $MODEL_SIZE = A22B ]; then
    NUM_LAYERS=94
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=64
    INTERMEDIATE_SIZE=12288
    MOE_INTERMEDIATE_SIZE=1536
    MAX_POSITION_EMBEDDINGS=40960
    EXTRA_VOCAB_SIZE=293
    NUM_KEY_VALUE_HEADS=4
    ROPE_THETA=1000000
    NUM_EXPERTS=128
    ROUTER_TOPK=8
    RMS_NORM_EPS=1e-6

    MODEL_ARGS+=(--untie-embeddings-and-output-weights)

    if [ ${ENABLE_DEEPEP:-false} = true ]; then
        MODEL_ARGS+=(--moe-enable-deepep --moe-router-dtype=fp32)
        MOE_TOKEN_DISPATCHER_TYPE=flex
    fi

    MODEL_ARGS+=(
        --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER_TYPE:-allgather}
        --moe-router-topk ${ROUTER_TOPK}
        --num-experts ${NUM_EXPERTS}
        --expert-tensor-parallel-size ${ETP}
        --expert-model-parallel-size ${EP}
        --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE}
        --moe-router-load-balancing-type aux_loss
        --moe-aux-loss-coeff 0.001
        --moe-layer-freq "([1]*${NUM_LAYERS})"
        --moe-router-pre-softmax
    )
    if [ ${ENABLE_GROUPED_GEMM:-true} = true ]; then
        MODEL_ARGS+=(--moe-grouped-gemm)
        if [ ${MOE_USE_LEGACY_GROUPED_GEMM:-false} = true ]; then
            MODEL_ARGS+=(--moe-use-legacy-grouped-gemm)
        fi
    fi

fi

DISTRIBUTED_ARGS+=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS+=(
    --transformer-impl transformer_engine
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --ffn-hidden-size $INTERMEDIATE_SIZE
    --seq-length $SEQ_LEN
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --log-interval 1
    --log-throughput
    --tensorboard-queue-size 1
    --num-workers 8
    --swiglu
    --normalization RMSNorm
    --norm-epsilon $RMS_NORM_EPS
    --position-embedding-type rope
    --disable-bias-linear
    --use-rotary-position-embeddings
    --rotary-base $ROPE_THETA
    --transformer-impl transformer_engine
    --cross-entropy-loss-fusion
    --qk-layernorm
    --kv-channels 128
    --use-cpu-initialization
    --use-mcore-models
    # --no-bias-swiglu-fusion
    --auto-detect-ckpt-format
    --group-query-attention
    --num-query-groups ${NUM_KEY_VALUE_HEADS}
    --no-rope-fusion 
)

DATA_ARGS+=(
    --tokenizer-type HuggingFaceTokenizer 
    --tokenizer-model $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 969,30,1
)

TRAINING_ARGS+=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --init-method-std 0.01
    --micro-batch-size $BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr $LR
    --min-lr $MIN_LR
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --weight-decay 0.1
    --clip-grad 1.0
    --seed 42
    --train-iters   $TRAIN_ITERS
    --eval-interval $EVAL_INTERVAL
    --eval-iters    $EVAL_ITERS
    --exit-interval $EXIT_INTERVAL
    --log-interval 1
    --log-throughput
    --use-cpu-initialization
    --initial-loss-scale 65536
    --attention-dropout 0.0
    --hidden-dropout 0.0
)

if [ $FL == true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
    MODEL_ARGS+=(--attention-backend flash)
    TRAINING_ARGS+=(--no-create-attention-mask-in-dataloader)
else
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
    MODEL_ARGS+=(--attention-backend fused)
fi

if [ $PR = fp16 ]; then
    MODEL_ARGS+=(
        --fp16
        --apply-query-key-layer-scaling
    )
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    MODEL_ARGS+=(--bf16)
elif [ $PR = fp8 ]; then
    MODEL_ARGS+=(
        --bf16
        --fp8-format hybrid
        --fp8-amax-compute-algo max
        --fp8-amax-history-len 1024
    )
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
       # exit -1
    fi
    TRAINING_ARGS+=(
        --recompute-method ${RECOMPUTE_METHOD}
        --recompute-num-layers ${MP_AC_LAYERS}
        --recompute-granularity full
    )
elif [ $AC = sel ]; then
    TRAINING_ARGS+=(--recompute-activations)
elif [ $AC = none ]; then
    TRAINING_ARGS+=()
elif [ $AC = offload ]; then
    TRAINING_ARGS+=(
        --cpu-offloading
        --cpu-offloading-num-layers ${MP_AC_LAYERS}
     )
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        TRAINING_ARGS+=(--tp-comm-overlap)
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        TRAINING_ARGS+=()
    fi
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    TRAINING_ARGS+=(--sequence-parallel)
elif [ $SP = false ]; then
    TRAINING_ARGS+=()
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    TRAINING_ARGS+=(--use-distributed-optimizer)
else
    TRAINING_ARGS+=()
fi

if [ $DP_OVERLAP = true ]; then
    TRAINING_ARGS+=(
        --overlap-grad-reduce
        --overlap-param-gather
    )
else
    TRAINING_ARGS+=()
fi

if [ $OPTIMIZER_OFFLOAD != false ]; then
    TRAINING_ARGS+=(
        --optimizer adam
        --optimizer-cpu-offload
        --optimizer-offload-policy static
        --overlap-cpu-optimizer-d2h-h2d
        --use-precision-aware-optimizer
        --optimizer-offload-fraction ${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
    )
fi

if [ $SFT = true ]; then
    TASK="sft"
    TRAINING_ARGS+=(--sft)
    DATA_ARGS+=(
        --eod-mask-loss
        --calculate-per-token-loss

        --dataset-type MMAP

        # MMAP format dataset, packing SFT
        --reset-position-ids
    )
else
    TASK="pretrain"
fi

if [ -z ${MP_PP0_LAYERS} ];then
    MODEL_ARGS+=()
elif [ ${PP} -gt 1 ]; then
    _check=$(( ( $NUM_LAYERS - ${MP_PP0_LAYERS} ) % ( ${PP} - 1 ) ))
    if [ $_check != 0 ]; then
        echo "With uneven pipelineing the left over layers must be divisible by left over stages."
        exit -1
    fi

    MODEL_ARGS+=(
        --decoder-first-pipeline-num-layers ${MP_PP0_LAYERS}
        --decoder-last-pipeline-num-layers ${MP_PP_LAST_LAYERS}
    )
else
    echo "uneven pipeline split must be used when PP > 1"
    exit -1
fi

if [ -z ${CPT_CONTINUE} ] || [ ${CPT_CONTINUE} = false ] || [ ${LOAD_OPTI} = false ]; then
    TRAINING_ARGS+=(
        --no-load-optim
        --no-load-rng
    )
elif [ ${CPT_CONTINUE} = true ];  then

    TRAINING_ARGS+=(--no-load-rng)
fi

TASK_NAME="mcore-qwen3-${MODEL_SIZE}-${TASK}"
SAVED_PRETRAIN_CHECKPOINT_PATH=${SAVED_PRETRAIN_CHECKPOINT_PATH:-${OUTPUT_BASEPATH}/checkpoint/${TASK_NAME}-TP${TP}-PP${PP}}
mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
if [ -e ${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt ]; then
    echo "${SAVED_PRETRAIN_CHECKPOINT_PATH}/latest_checkpointed_iteration.txt 文件存在"
    if [ ${CPT_CONTINUE} = true ];  then
        PRETRAIN_CHECKPOINT_PATH=${SAVED_PRETRAIN_CHECKPOINT_PATH}
    fi
else
    echo "${SAVED_PRETRAIN_CHECKPOINT_PATH} :文件夹为空"
fi

if [ ${PRETRAIN_CHECKPOINT_PATH} != none ]; then
    TRAINING_ARGS+=(
        --load $PRETRAIN_CHECKPOINT_PATH
        --auto-detect-ckpt-format
        --no-load-rng
    )
fi

if [ $SAVE_CKPT = true ]; then
    TRAINING_ARGS+=(
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH}
        --save-interval ${SAVE_INTERVAL}
        --ckpt-format torch_dist 
    )
fi

CURRENT_TIME=$(date +"%m-%d-%H-%M")
DETAIL_TASK_NAME="${TASK_NAME}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ep-${EP}-etp-${ETP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}/${CURRENT_TIME}_rc-${RECOMPUTE_METHOD}-${MP_AC_LAYERS}${LABEL}"
LOG_DIR=${OUTPUT_BASEPATH}/${DETAIL_TASK_NAME}
LOG_NAME="${NODE_RANK}.txt"
mkdir -p ${LOG_DIR}
if [[ -z ${LOG_FILE} ]];then
  LOG_FILE=${LOG_DIR}/${LOG_NAME}
fi
exec > >(tee -a "${LOG_FILE}") 2>&1

set -x
torchrun ${DISTRIBUTED_ARGS[@]} \
    -m xmegatron_ext.pretrain.pretrain_v0_15_0 \
    ${PARALLEL_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
