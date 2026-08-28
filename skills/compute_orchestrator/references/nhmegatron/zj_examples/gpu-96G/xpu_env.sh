# 自动激活镜像中存在的 conda env（取第一个非 base 的 env）
_target_env=$(conda env list | awk '!/^#/ && NF>0 && $1!="base" {print $1; exit}')
echo "Activating conda env: $_target_env"
source activate "$_target_env"
unset _target_env

# GPU/XPU共享环境变量
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

if command -v nvidia-smi &>/dev/null; then
    # GPU环境变量
    export GLOO_SOCKET_IFNAME=eth0
    export NCCL_SOCKET_IFNAME=eth0
    export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export NCCL_ALGO=Ring
    return 0 2>/dev/null || exit 0
fi

# XPU环境变量
export XPU_FORCE_USERMODE_LAUNCH=1
export CUDA_DEVICE_ORDER=OAM_ID
export XPU_FORCE_SHARED_DEVICE_CONTEXT=1
export CUDA_DISABLE_PRINTF=${CUDA_DISABLE_PRINTF:-1}
export XPU_SSE_DEBUG_EXPORT_DISABLE=1

# BKCL环境变量
_rdma_nics=$(bash "$(dirname "${BASH_SOURCE[0]}")/get_rdma_config.sh" 2>/dev/null)
if [[ -n "$_rdma_nics" && "$_rdma_nics" != *"错误"* && "$_rdma_nics" != *"N/A"* ]]; then
    export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-$_rdma_nics}
elif ibdev2netdev | grep "eth" >/dev/null; then
    export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4}
else
    export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-bond2,bond2,bond3,bond3,bond4,bond4,bond5,bond5}
fi
unset _rdma_nics

# 清理SHMEM
ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
    ipcrm -m $shmid
    echo "Deleted shared memory segment with ID: $shmid"
done

export BKCL_TREE_THRESHOLD=1
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_FORCE_L3_RDMA=0
export BKCL_RDMA_VERBS=${BKCL_RDMA_VERBS:-1}
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_USE_RDMA=1
export BKCL_RDMA_PROXY_DISABLE=1
export BKCL_FLAT_RING=1
export BKCL_CCIX_BUFFER_GM=1
export BKCL_RING_BUFFER_SIZE=${BKCL_RING_BUFFER_SIZE:-2097152}
export BKCL_TIMEOUT=${BKCL_TIMEOUT:-1800}
export BKCL_GC_SIGNAL_MASK=SIGTERM,SIGQUIT

# XME环境变量(0.10与0.12)
export XME_USE_LOCAL_TE=true
export XME_USE_CUSTOM_TRAINING_LOG=true
export XME_CHECK_XPU_ENV=0
# 自动检测 TE 版本
TE_VERSION=$(python -c "import transformer_engine; print(transformer_engine.__version__)" 2>/dev/null)
if [[ "$TE_VERSION" == 2.12* ]]; then
    export XME_USE_TE_VERSION=2.12.0
else
    export XME_USE_TE_VERSION=1.13.0
fi

# 性能优化环境变量
export DIST_MULTI_STREAM=true
export FC_DW_MULTI_STREAM=1
export XMLIR_MEMCPY_RETRY_SYNC=true
export XTE_RECOMPUTE_LN_OUT_TOTAL=1
export HYDRAX_USE_PROTEUS=${HYDRAX_USE_PROTEUS:-1}

export XMLIR_ENABLE_FAST_FC=${XMLIR_ENABLE_FAST_FC:-true}
export XMLIR_PARALLEL_SAVE_MEMORY=${XMLIR_PARALLEL_SAVE_MEMORY:-false}
export XTE_GROUPED_GEMM_LARGE_WEIGHT=${XTE_GROUPED_GEMM_LARGE_WEIGHT:-1}
export XTE_DISABLE_MOE_DW_FUSION=${XTE_DISABLE_MOE_DW_FUSION:-0}
export XMLIR_BATCH_PARALLEL=${XMLIR_BATCH_PARALLEL:-true}

if [[ ${ENABLE_DEEPEP:-false} == "true" ]]; then
    export XSHMEM_MODE=1
    export XSHMEM_QP_NUM_PER_RANK=8
    export CUDA_ENABLE_P2P_NO_UVA=1
    export MOE_TOKEN_DISPATCHER_TYPE=flex
fi

# 性能加速环境变量
if [[ ${XPU_USE_FAST_KERNEL:=1} -eq 1 ]]; then
    export XMLIR_FC_BIAS_FUSION=true
    export XMLIR_FA_GEMM_TYPE=float16
    export XFA_GEMM_TYPE=float16
    export XFA_BWD_USE_DS_SCALE=1
    export XDNN_USE_FAST_SWISH=1
    export XDNN_USE_FAST_GELU=1
    export XDNN_USE_FAST_SCATTER=1
    export XDNN_FAST_DIV_SCALAR=1
    export XDNN_USE_FAST_SIGMOID=1
    export FAST_SWIGLU_ENABLE=1
    export XTE_FA_LOD_AT_CPU=1
fi

# Profiler环境变量
if [[ ${PROFILER_DEBUG:=0} -eq 1 ]]; then
    export XPU_ENABLE_PROFILER_TRACING=1
    export XPU_CUPTI_ENABLE_DEVICE=${XPU_CUPTI_ENABLE_DEVICE:-0}
    # For Swift Megatron Trainer
    export MEGATRON_PROFILE=1
    export MEGATRON_PROFILE_USE_PYTORCH=1
    export MEGATRON_PROFILE_STEP_START=${MEGATRON_PROFILE_STEP_START:-3}
    export MEGATRON_PROFILE_STEP_END=${MEGATRON_PROFILE_STEP_END:-4}
    export MEGATRON_PROFILE_RANKS=${MEGATRON_PROFILE_RANKS:-0}
    export MEGATRON_PROFILE_COLLECT_CHAKRA=1
    export MEGATRON_PROFILE_COLLECT_SHAPES=1
    export MEGATRON_PROFILE_COLLECT_CALLSTACK=1
    # Only enable memory profiling when needed
    export MEGATRON_PROFILE_COLLECT_MEMORY=${MEGATRON_PROFILE_COLLECT_MEMORY:-0}
    export MEGATRON_PROFILE_EXPORT_MEMORY_TIMELINE=${MEGATRON_PROFILE_EXPORT_MEMORY_TIMELINE:-0}
fi

# 检测硬件noc_idle timeout
if dmesg -T | grep -q "noc_idle" <(cat /dev/stdin); then
    echo "Error: noc timeout found in dmesg, please check node ${RANK} ${HOSTNAME} ${POD_IP}!"
    dmesg -T >>dmesg_${RANK}.log
    exit 1
fi

# 设置任务超时检测阈值为30分钟,默认20分钟
for dev_id in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
    if [ -f /proc/kunlun/dev$dev_id/task_timeout_detect_threshold_in_ms ]; then
        echo 1800000 >/proc/kunlun/dev$dev_id/task_timeout_detect_threshold_in_ms
        cat /proc/kunlun/dev$dev_id/task_timeout_detect_threshold_in_ms
    fi
done
