#! /bin/bash
set -x

# 训练节点数
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}

# 训练类型
export SFT=${SFT:-1}
export PACKING=${PACKING:-1}
export ENABLE_DEEPEP=${ENABLE_DEEPEP:-true}

# 多机配置
if [[ $WORLD_SIZE -eq 64 ]]; then
    # 模型层数
    export NUM_LAYERS=${NUM_LAYERS:-61}
    export DENSE_LAYERS=${DENSE_LAYERS:-3}
    export MOE_LAYERS=${MOE_LAYERS:-58}
    # 并行配置
    export TP=${TP:-2}
    export PP=${PP:-8}
    export EP=${EP:-64}
    export ETP=${ETP:-1}
    export CP=${CP:-1}
    export PIPELINE_LAYOUT=${PIPELINE_LAYOUT:-"Et*6|t*7(|t*8)*6L"}
    # 训练参数
    export SEQ_LEN=${SEQ_LEN:-32768}
    export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
    export AC=${AC:-full}
    export PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"/mnt/lhycpfs/lhy/mbridge_ckpt/DeepSeek-V3-mcore"}

elif [[ $WORLD_SIZE -eq 32 ]]; then
    # 模型层数
    export NUM_LAYERS=${NUM_LAYERS:-61}
    export DENSE_LAYERS=${DENSE_LAYERS:-3}
    export MOE_LAYERS=${MOE_LAYERS:-58}
    # 并行配置
    export TP=${TP:-4}
    export PP=${PP:-8}
    export EP=${EP:-32}
    export ETP=${ETP:-1}
    export CP=${CP:-1}
    export PIPELINE_LAYOUT=${PIPELINE_LAYOUT:-"Et*6|t*7(|t*8)*6L"}
    # 训练参数
    export SEQ_LEN=${SEQ_LEN:-32768}
    export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
    export AC=${AC:-full}
    export PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"/mnt/lhycpfs/lhy/mbridge_ckpt/DeepSeek-V3-mcore"}
    # 额外环境变量
    export XMLIR_BATCH_PARALLEL=0

elif [[ $WORLD_SIZE -eq 2 ]]; then
    # 模型层数
    export NUM_LAYERS=${NUM_LAYERS:-3}
    export DENSE_LAYERS=${DENSE_LAYERS:-1}
    export MOE_LAYERS=${MOE_LAYERS:-2}
    # 并行配置
    export TP=${TP:-2}
    export PP=${PP:-1}
    export EP=${EP:-16}
    export ETP=${ETP:-1}
    export CP=${CP:-1}
    # 训练参数
    export SEQ_LEN=${SEQ_LEN:-32768}
    export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
    export AC=${AC:-full}

elif [[ $WORLD_SIZE -eq 1 ]]; then
    # 模型层数
    export NUM_LAYERS=${NUM_LAYERS:-2}
    export DENSE_LAYERS=${DENSE_LAYERS:-1}
    export MOE_LAYERS=${MOE_LAYERS:-1}
    # 并行配置
    export TP=${TP:-2}
    export PP=${PP:-1}
    export EP=${EP:-8}
    export ETP=${ETP:-1}
    export CP=${CP:-1}
    # 训练参数
    export SEQ_LEN=${SEQ_LEN:-32768}
    export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
    export AC=${AC:-full}
fi

######################################以下部分同一模型内容相同######################################
################################################################################################

# 依赖库与数据集
export MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM/core_r0_15_0"}
export TRAINING_PATH=${TRAINING_PATH:-"/workspace/KLX-LLM"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"/mnt/lhycpfs/lhy/model/deepseek-v3-tokenizer"}
if [[ $SFT -ne 1 ]]; then
    export DATA_PATH=${DATA_PATH:-"/mnt/lhycpfs/lhy/dataset/deepseek-v3/deepseek_v3_dataset_text_document"}
elif [[ $PACKING -eq 1 ]]; then
    export DATA_PATH=${DATA_PATH:-"/mnt/lhycpfs/lhy/dataset/deepseek-v3/deepseek_v3_sft_packing_32k_text_document"}
else
    export XME_SFT_TOKENIZERS_ENCODING_CONVERT=true
    export DATA_PATH=${DATA_PATH:-"/mnt/lhycpfs/lhy/dataset/deepseek-v3/deepseek_v3_sft_padding.jsonl"}
fi

# 设置保存路径
MODEL_CONFIG_PATH="$TRAINING_PATH/examples/deepseek_v3"
OUTPUT_DIR=${OUTPUT_DIR:-"${TRAINING_PATH}/output"}
LOG_PATH="$OUTPUT_DIR/logs"
mkdir -p ${LOG_PATH}
[[ ${DISABLE_LOG_FILE:-0} -ne 1 ]] && exec > >(tee -a "${LOG_PATH}/${RANK}.log") 2>&1

SAVE_PATH="$OUTPUT_DIR/checkpoint/mcore-deepseek-v3-671B"
[[ $SFT -ne 1 ]] && SAVE_PATH+="-pretrain"
[[ $SFT -eq 1 ]] && { SAVE_PATH+="-sft"; [[ $PACKING -eq 1 ]] && SAVE_PATH+="-packing" || SAVE_PATH+="-padding"; }
SAVE_PATH+="-layer${NUM_LAYERS:=$((${DENSE_LAYERS:=1}+${MOE_LAYERS:=1}))}"
SAVE_PATH+="-dense${DENSE_LAYERS:=1}-moe${MOE_LAYERS:=1}"
SAVE_PATH+="-seqlen${SEQ_LEN:=1024}-gbs${GLOBAL_BATCH_SIZE:=256}"
SAVE_PATH+="-TP${TP:=1}-PP${PP:=1}-EP${EP:=8}-ETP${ETP:=1}"
SAVE_PATH=${SAVED_PRETRAIN_CHECKPOINT_PATH:-$SAVE_PATH}
mkdir -p ${SAVE_PATH}

TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$OUTPUT_DIR/tensorboard"}
WANDB_SAVE_DIR=${WANDB_SAVE_DIR:-"$OUTPUT_DIR/wandb"}
mkdir -p ${TENSORBOARD_DIR} ${WANDB_SAVE_DIR}

# 加载环境变量
source $TRAINING_PATH/examples/xpu_env.sh

DISTRIBUTED_ARGS=(
    --nproc_per_node `echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
    --nnodes ${WORLD_SIZE:-"1"}
    --node_rank ${RANK:-"0"}
    --master_addr ${MASTER_ADDR:-"localhost"}
    --master_port ${MASTER_PORT:-"43622"}
)

MODEL_ARGS=(
    --model-config deepseek_v3_config
    --moe-grouped-gemm
    --transformer-impl transformer_engine
    --attention-backend flash
    --num-layers ${NUM_LAYERS}
    --moe-layer-freq "[0]*${DENSE_LAYERS}+[1]*${MOE_LAYERS}"
    --no-rope-fusion
)

PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --expert-model-parallel-size ${EP}
    --expert-tensor-parallel-size ${ETP}
    --context-parallel-size ${CP}
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    $([[ $PIPELINE_LAYOUT != "" ]] && echo "--pipeline-model-parallel-layout $PIPELINE_LAYOUT")
    $([[ $ENABLE_DEEPEP == "true" ]] && echo "--moe-enable-deepep")
    --moe-token-dispatcher-type ${MOE_TOKEN_DISPATCHER_TYPE:-"alltoall"}
)

DATA_ARGS=(
    --tokenizer-model $TOKENIZER_PATH
    --data-path $DATA_PATH
    --split 969,30,1
    # Pretrain
    $([[ $SFT -ne 1 ]] && echo "--tokenizer-type HuggingFaceTokenizer")
    # SFT
    $([[ $SFT -eq 1 ]] && echo "--sft")
    $([[ $SFT -eq 1 ]] && echo "--eod-mask-loss")
    $([[ $SFT -eq 1 ]] && echo "--calculate-per-token-loss")
    # Packing SFT
    $([[ $SFT -eq 1 && $PACKING -eq 1 ]] && echo "--dataset-type MMAP")
    $([[ $SFT -eq 1 && $PACKING -eq 1 ]] && echo "--tokenizer-type HuggingFaceTokenizer")
    $([[ $SFT -eq 1 && $PACKING -eq 1 ]] && echo "--reset-position-ids")
    $([[ $SFT -eq 1 && $PACKING -eq 1 ]] && echo "--no-create-attention-mask-in-dataloader")
    # Padding SFT
    $([[ $SFT -eq 1 && $PACKING -ne 1 ]] && echo "--dataset-type JSONL")
    $([[ $SFT -eq 1 && $PACKING -ne 1 ]] && echo "--legacy-tokenizer")
    $([[ $SFT -eq 1 && $PACKING -ne 1 ]] && echo "--tokenizer-type SFTTokenizer")
    $([[ $SFT -eq 1 && $PACKING -ne 1 ]] && echo "--sft-tokenizer-prompt-format deepseek-v3")
)

TRAINING_ARGS=(
    --bf16
    --init-method-std 0.01
    --micro-batch-size ${MICRO_BATCH_SIZE:-1}
    --global-batch-size ${GLOBAL_BATCH_SIZE:-256}
    --seq-length ${SEQ_LEN:=4096}
    --max-position-embeddings $((SEQ_LEN>4096?SEQ_LEN:4096))
    --lr ${LR:-1e-5}
    --min-lr ${MIN_LR:-1e-6}
    --lr-decay-iters ${LR_DECAY_ITERS:-320000}
    --lr-decay-style ${LR_DECAY_STYLE:-cosine}
    --lr-warmup-fraction ${LR_WARMUP_FRACTION:-0.002}
    --weight-decay ${WEIGHT_DECAY:-0.1}
    --clip-grad 1.0
    --seed ${SEED:-42}
    --train-iters ${TRAIN_ITERS:-5000}
    --eval-iters ${EVAL_ITERS:-10}
    --save-interval ${SAVE_INTERVAL:-100000}
    --eval-interval ${EVAL_INTERVAL:-20}
    --exit-interval ${EXIT_INTERVAL:-2000}
    --log-interval 1
    --log-throughput
    --use-cpu-initialization
    --initial-loss-scale 65536
    --distributed-timeout-minutes 60
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
    --no-load-rng
    $([[ $CPT_CONTINUE != true ]] && echo "--no-load-optim")
    $([[ ${SAVE_CKPT:-1} -eq 1 ]] && echo "--save $SAVE_PATH")
    $([[ $CPT_CONTINUE == true || $PRETRAIN_CHECKPOINT_PATH != "" ]] && echo "--load ${PRETRAIN_CHECKPOINT_PATH:-$SAVE_PATH}")
    $([[ $AC == "full" ]] && echo "--recompute-granularity full")
    $([[ $AC == "full" ]] && echo "--recompute-method ${RECOMPUTE_METHOD:-uniform}")
    $([[ $AC == "full" ]] && echo "--recompute-num-layers ${MP_AC_LAYERS:-1}")
)

DEBUG_ARGS=(
    $([[ ${BENCHMARK_MOE_ROUTER_FORCE_LB:=1} -eq 1 ]] && echo "--moe-router-force-load-balancing")
    $([[ $PROFILER_DEBUG -eq 1 ]] && echo "--profile")
    $([[ $PROFILER_DEBUG -eq 1 ]] && echo "--use-pytorch-profiler")
    $([[ $PROFILER_DEBUG -eq 1 ]] && echo "--profile-step-start ${PROFILE_STEP_START:-3}")
    $([[ $PROFILER_DEBUG -eq 1 ]] && echo "--profile-step-end ${PROFILE_STEP_END:-4}")
)

WANDB_ARGS=(
    $([[ -n ${WANDB_API_KEY} ]] && echo "--wandb-project ${WANDB_PROJECT}")
    $([[ -n ${WANDB_API_KEY} ]] && echo "--wandb-exp-name ${WANDB_NAME}")
    $([[ -n ${WANDB_API_KEY} ]] && echo "--wandb-save-dir ${WANDB_SAVE_DIR}")
    $([[ -n ${WANDB_API_KEY} ]] && echo "--moe-per-layer-logging")
    $([[ -n ${WANDB_API_KEY} ]] && echo "--log-memory-to-tensorboard")
    $([[ -n ${WANDB_API_KEY} ]] && echo "--log-timers-to-tensorboard")
)

PYTHONPATH=$MEGATRON_PATH:$MODEL_CONFIG_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    -m xmegatron_ext.pretrain.pretrain_v0_15_0 \
    ${MODEL_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DEBUG_ARGS[@]} \
    ${WANDB_ARGS[@]}
