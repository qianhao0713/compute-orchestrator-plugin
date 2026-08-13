#! /bin/bash
set -eox pipefail

MODEL_SIZE=${MODEL_SIZE:-27B}
STORAGE_PATH=${STORAGE_PATH:-/mnt/lhycpfs/lhy/}
ROOT_DIR=$(dirname -- "$(readlink -f -- "$0")")
ROOT_DIR=$(realpath ${ROOT_DIR}/../../)
echo $ROOT_DIR

source ../xpu_env.sh

OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/output/Qwen3.5-$MODEL_SIZE}
LOG_PATH="$OUTPUT_DIR/logs"
mkdir -p ${LOG_PATH}
[[ ${DISABLE_LOG_FILE:-0} -ne 1 ]] && exec > >(tee -a "${LOG_PATH}/${RANK}.log") 2>&1

python -m torch_xmlir --doctor

export SEQ_LEN=${SEQ_LEN:-8192}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}

if [[ $SEQ_LEN == "8192" ]]; then
    export TP=${TP:-1}
    export PP=${PP:-2}
    export PP_LAYOUT=${PP_LAYOUT:-"Et*32|t*32L"}
elif [[ $SEQ_LEN == "32768" ]]; then
    if [[ $GLOBAL_BATCH_SIZE == "64" ]]; then
        export TP=${TP:-2}
        export PP=${PP:-2}
        export PP_LAYOUT=${PP_LAYOUT:-"Et*16|t*16|t*16|t*16L"}
    else
        export TP=${TP:-2}
        export PP=${PP:-2}
        export PP_LAYOUT=${PP_LAYOUT:-"Et*16|t*16|t*16|t*16L"}
    fi
    # For reducing memory of output head
    export XBLAS_GEMM_INFER_RES_TYPE=bf16
fi

export CP=${CP:-1}
export EP=${EP:-1}

export SAVE_STEPS=${SAVE_STEPS:-4000}
export EVAL_STEPS=${EVAL_STEPS:-4000}
export EVAL_ITERS=${EVAL_ITERS:-0}
export TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}

export CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STORAGE_PATH}/model/Qwen3.5-$MODEL_SIZE}
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

# XPU extra environment variables
export XMLIR_BATCH_PARALLEL=0

# Swift environment variables
export NPROC_PER_NODE=8
export NNODES=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"13622"}

export SWIFT_USE_MCORE_GDN=${SWIFT_USE_MCORE_GDN:-1}
export MAX_PIXELS=1003520
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12
export SKIP_MULTIMODAL_MTP_VALIDATION=1

megatron pt \
    --model $CHECKPOINT_PATH \
    --model_type qwen3_5 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --tuner_type full \
    --finetune true \
    --attention_backend flash \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --moe_permute_fusion false \
    --moe_grouped_gemm true \
    --use_distributed_optimizer true \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP \
    --pipeline_model_parallel_layout $PP_LAYOUT \
    --context_parallel_size $CP \
    --expert_model_parallel_size $EP \
    --sequence_parallel true \
    ${DATASET_ARGS[@]} \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --split_dataset_ratio 0.01 \
    --packing true \
    --padding_free false \
    --group_by_length false \
    --train_dataloader_shuffle false \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --num_train_epochs ${TRAIN_EPOCHS:-1} \
    --micro_batch_size 1 \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --max_length $SEQ_LEN \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --loss_scale ignore_empty_think \
    --use_cpu_initialization \
    --recompute_granularity ${RECOMPUTE_GRANULARITY:-"full"} \
    --recompute_method ${RECOMPUTE_METHOD:-"uniform"} \
    --recompute_num_layers ${RECOMPUTE_NUM_LAYERS:-1} \
    --eval_iters ${EVAL_ITERS} \
    --eval_steps ${EVAL_STEPS:-200} \
    --save_steps ${SAVE_STEPS:-200} \
    --save_safetensors true \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir $OUTPUT_DIR \
    --logging_steps 1 \
    --report_to ${REPORT_TO:-"tensorboard"} \
    --wandb_project ${WANDB_PROJECT:-"wandb-project-name"} \
    --wandb_exp_name ${WANDB_EXP_NAME:-"wandb-exp-name"}