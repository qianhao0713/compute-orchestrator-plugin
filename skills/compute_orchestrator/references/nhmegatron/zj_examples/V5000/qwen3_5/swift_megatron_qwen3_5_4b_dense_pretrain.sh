#! /bin/bash
set -eox pipefail

# 清理SHMEM
ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
    ipcrm -m $shmid
    echo "Deleted shared memory segment with ID: $shmid"
done

ROOT_DIR=$(dirname -- "$(readlink -f -- "$0")")
ROOT_DIR=$(realpath ${ROOT_DIR}/../../)
echo $ROOT_DIR
source ../xpu_env.sh

hostname -i
cat /proc/kunlun/version
python -m torch_xmlir --doctor

export SEQ_LEN=${SEQ_LEN:-8192}
if [[ $SEQ_LEN == "8192" ]]; then
    export TP=${TP:-1}
    export PP=${PP:-1}
    export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1024}
elif [[ $SEQ_LEN == "32768" ]]; then
    export TP=${TP:-2}
    export PP=${PP:-1}
    export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64}
fi

export RECOMPUTE_GRANULARITY=${RECOMPUTE_GRANULARITY:-"full"}
export RECOMPUTE_METHOD=${RECOMPUTE_METHOD:-"uniform"}
export RECOMPUTE_NUM_LAYERS=${RECOMPUTE_NUM_LAYERS:-1}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}

MODEL_SIZE=${MODEL_SIZE:-4B}
STORAGE_PATH=${STORAGE_PATH:-/mnt/lhycpfs/lhy/}
export CHECKPOINT_PATH=${CHECKPOINT_PATH:-${STORAGE_PATH}/model/Qwen3.5-$MODEL_SIZE}
export DATASET_PATH=${DATASET_PATH:-${STORAGE_PATH}/dataset/Chinese-DeepSeek-R1-Distill-data-110k-SFT#1048576}


export CACHED_DATA_PATH=/mnt/lhycpfs/lhy/dataset/Chinese-DeepSeek-R1-Distill-data-110k-SFT-Cache1048576/train
export CACHED_VAL_DATA_PATH=/mnt/lhycpfs/lhy/dataset/Chinese-DeepSeek-R1-Distill-data-110k-SFT-Cache1048576/val
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/output/Qwen3.5-$MODEL_SIZE}
LOG_PATH="$OUTPUT_DIR/logs"
mkdir -p ${LOG_PATH}
[[ ${DISABLE_LOG_FILE:-0} -ne 1 ]] && exec > >(tee -a "${LOG_PATH}/${RANK}.log") 2>&1


SAVE_STEPS=${SAVE_STEPS:-4000}
EVAL_STEPS=${EVAL_STEPS:-4000}
EVAL_ITERS=${EVAL_ITERS:-0}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}

export NPROC_PER_NODE=8
export NNODES=${WORLD_SIZE:-1}
export NODE_RANK=${RANK:-0}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}

export XMLIR_BATCH_PARALLEL=0

export CUDA_DEVICE_MAX_CONNECTIONS=1

export SWIFT_USE_MCORE_GDN=${SWIFT_USE_MCORE_GDN:-1}
export MAX_PIXELS=1003520
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12
export SKIP_MULTIMODAL_MTP_VALIDATION=1

MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"13622"}

megatron pt \
    --model $CHECKPOINT_PATH \
    --dataset $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --model_type qwen3_5 \
    --logging_steps 1 \
    --finetune true \
    --save_safetensors true \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --use_distributed_optimizer true \
    --tensor_model_parallel_size $TP \
    --pipeline_model_parallel_size $PP \
    --sequence_parallel true \
    --moe_permute_fusion false \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --recompute_granularity $RECOMPUTE_GRANULARITY \
    --recompute_method $RECOMPUTE_METHOD \
    --recompute_num_layers $RECOMPUTE_NUM_LAYERS \
    --num_train_epochs ${TRAIN_EPOCHS} \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --cross_entropy_fusion_impl te \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --eval_iters ${EVAL_ITERS} \
    --eval_steps ${EVAL_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_length $SEQ_LEN \
    --packing true \
    --padding_free false \
    --group_by_length false \
    --train_dataloader_shuffle false \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
    --dataset_num_proc ${DATASET_NUM_PROC} \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --use_cpu_initialization