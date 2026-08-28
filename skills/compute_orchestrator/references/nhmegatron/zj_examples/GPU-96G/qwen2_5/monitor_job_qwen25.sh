ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
ROOT_DIR=$(realpath ${ROOT_DIR}/../../../)
echo "ROOT_DIR: ${ROOT_DIR}"

POD_NAME=${POD_NAME:-'ji-aitrain-8488343569073627136-master-0'}
JOB_NAME=${JOB_NAME:-'ji-aitrain-8476268690702248877'}
echo "POD_NAME: ${POD_NAME}"

MODEL_NAME=${MODEL_NAME:-"qwen2_5-7b"}

if [[ -z ${OUTPUT_DIR} ]];
then
  OUTPUT_DIR=${ROOT_DIR}/output
fi

CURRENT_DAY=$(date "+%Y-%m-%d")
CURRENT_TIME=$(date "+%Y-%m-%d_%H:%M")
LOG_DIR="${OUTPUT_DIR}/log_${MODEL_NAME}"

NNODES=${WORLD_SIZE:-1}
RANK=${RANK:-0}

echo $CURRENT_TIME
dir="${LOG_DIR}/${CURRENT_TIME}_${NNODES}n_monitor"
if [ ! -d "$dir" ];then
    mkdir -p $dir
    chmod -R 777 $dir
fi
LOG_FILE="$dir/$RANK.log"

export LOG_FILE=$LOG_FILE
echo $LOG_FILE
echo -n "moniter robot login in" > $LOG_FILE

SCRIPT=${SCRIPT:-"xmegatron_qwen2.5_7b_pretrain.sh"}
nohup bash $SCRIPT &

POST_INTERVAL=${1:-10}
sleep 20s

python ../monitor_job.py \
        --log-path $LOG_FILE \
        --post-interval $POST_INTERVAL \
        --script $SCRIPT \
        --rank $RANK \
        --nnodes $NNODES \
        --model-name $MODEL_NAME \
        --pod-name $POD_NAME \
        --job-name $JOB_NAME \
        --output-path $LOG_DIR \
        --webhook https://oapi.dingtalk.com/robot/send?access_token=REPLACE_WITH_YOUR_DINGTALK_TOKEN # 替换为自己的钉钉群webhook

# 运行指令 bash monitor_job.sh 10
