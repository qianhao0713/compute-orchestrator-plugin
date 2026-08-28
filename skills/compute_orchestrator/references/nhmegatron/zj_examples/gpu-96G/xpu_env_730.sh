# XPU相关参数
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export CUDA_DEVICE_ORDER=OAM_ID
#################################
export CUDART_DUMMY_REGISTER=1
export XPU_FORCE_USERMODE_LAUNCH=1
export XMLIR_DIST_SINGLETON_STREAM=true
# export DIST_MULTI_STREAM=${DIST_MULTI_STREAM:-false}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}
export XMLIR_FA_GEMM_TYPE=float
export XBLAS_FC_HBM_VERSION=40
export XMLIR_PARALLEL_SAVE_MEMORY=${XMLIR_PARALLEL_SAVE_MEMORY:-false}
export XMLIR_DISABLE_CUDA_ALLOCATOR=true

#################################
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=0
export XMLIR_ENABLE_FALLBACK_TO_CPU_BOOL=False
export XMLIR_DUMP_FALLBACK_OP_LIST_BOOL=true
export XMLIR_DIST_ASYNC_ISEND_IRECV=true
##################
# bf16类型专用(megatron相关变量参考<百舸megatron专用>)
##################
export USE_FAST_BF16_FC=true # 仅bf16下用到
# export XMLIR_BATCH_PARALLEL=false
# export XPU_FORCE_SHARED_DEVICE_CONTEXT=1
#################
# 通信通用
##################
##################
export BKCL_RDMA_PROXY_DISABLE=1
export BKCL_USE_AR=1
export BKCL_RING_OPT=1
export BKCL_FLAT_RING=1
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1
export BKCL_CCIX_BUFFER_GM=1
export BKCL_FORCE_L3_RDMA=0
export BKCL_RING_BUFFER_GM=1
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_FORCE_TREE=1
export BKCL_TREE_THRESHOLD=1
export BKCL_XLINK_D2D=0
export BKCL_XLINK_ETH=0
export BKCL_XLINK_C2C=1
export BKCL_TRANS_UNSUPPORTED_DATATYPE=1
export BKCL_KL3_TURBO_MODE=1
export BKCL_RING_BUFFER_SIZE=2097152
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=${BKCL_TIMEOUT:-360000}
unset BKCL_KL3_SYSCON_FLAG

# export CUDA_DISABLE_PRINTF=${CUDA_DISABLE_PRINTF:-1}
# export BKCL_RDMA_VERBS=${BKCL_RDMA_VERBS:-1}

# export BKCL_SOCKET_IFNAME=bond0
export BKCL_RDMA_NICS=bond2,bond2,bond3,bond3,bond4,bond4,bond5,bond5
# export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-"ens11f0np0,ens11f1np1,ens13f0np0,ens13f1np1,enP1s15f0np0,enP1s15f1np1,enP1s17f0np0,enP1s17f1np1"}

export ZCCL_BIND_NICS=mlx5_bond_2:1@0#mlx5_bond_2:1@1#mlx5_bond_3:1@2#mlx5_bond_3:1@3#mlx5_bond_4:1@4#mlx5_bond_4:1@5#mlx5_bond_5:1@6#mlx5_bond_5:1@7