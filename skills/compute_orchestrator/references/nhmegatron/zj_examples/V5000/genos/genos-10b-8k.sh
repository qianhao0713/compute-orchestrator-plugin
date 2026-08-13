#! /bin/bash

set -exo pipefail

ipcs -m | awk '$4 == 666 {print $2}' | while read shmid; do
    ipcrm -m $shmid
    echo "Deleted shared memory segment with ID: $shmid"
done 

source activate && conda activate python310_torch29_cuda


cd /workspace/Megatron-LM/core_r0_15_0/megatron/core/datasets && make

export MEGATRON_PATH=${MEGATRON_PATH:-"/workspace/Megatron-LM/core_r0_15_0"}



# GPU/XPU共享环境变量
export NCCL_IB_HCA=mlx5
export NCCL_IB_GID_INDEX=3
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
# export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# XPU环境变量
export XPU_FORCE_USERMODE_LAUNCH=1
export CUDA_DEVICE_ORDER=OAM_ID 
#export XPULINK_VISIBLE_DEVICES=2,3,0,1,5,4,7,6
#export XPULINK_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XPULINK_VISIBLE_DEVICES=0,2,1,3,5,7,4,6
export XBLAS_FC_HBM_VERSION=40
export XPU_FORCE_SHARED_DEVICE_CONTEXT=1
export CUDA_DISABLE_PRINTF=${CUDA_DISABLE_PRINTF:-1}

# BKCL环境变量
if ibdev2netdev | grep -q 'eth'; then
    export BKCL_RDMA_NICS=eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4
else
    export BKCL_RDMA_NICS=bond2,bond2,bond3,bond3,bond4,bond4,bond5,bond5
fi
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
export BKCL_RING_BUFFER_SIZE=2097152
export BKCL_TIMEOUT=${BKCL_TIMEOUT:-1800}
export XSHMEM_MODE=1
export XSHMEM_QP_NUM_PER_RANK=8
export CUDA_ENABLE_P2P_NO_UVA=1
export BKCL_GC_SIGNAL_MASK=SIGTERM,SIGQUIT

# XME环境变量
export XME_FORCE_SYNC_D2H_COPY=true

# 性能优化环境变量
export DIST_MULTI_STREAM=true
export FC_DW_MULTI_STREAM=1
export XMLIR_ENABLE_FAST_FC=true
export XMLIR_PARALLEL_SAVE_MEMORY=${XMLIR_PARALLEL_SAVE_MEMORY:-false}
export XTE_GROUPED_GEMM_LARGE_WEIGHT=${XTE_GROUPED_GEMM_LARGE_WEIGHT:-0}
# export XTE_GROUPED_GEMM_LARGE_WEIGHT=${XTE_GROUPED_GEMM_LARGE_WEIGHT:-0}
export XTE_DISABLE_MOE_DW_FUSION=${XTE_DISABLE_MOE_DW_FUSION:-0}
export XMLIR_BATCH_PARALLEL=${XMLIR_BATCH_PARALLEL:-0}



# 检测硬件noc_idle timeout
if dmesg -T |  grep -q "noc_idle" <(cat /dev/stdin); then
    echo "Error: noc timeout found in dmesg, please check node ${RANK} ${HOSTNAME} ${POD_IP}!"
    dmesg -T >> dmesg_${RANK}.log
    exit 1
fi

# 设置任务超时检测阈值为30分钟,默认20分钟
for dev_id in $(echo $CUDA_VISIBLE_DEVICES | tr ',' ' '); do
    if [ -f /proc/kunlun/dev$dev_id/task_timeout_detect_threshold_in_ms ]; then
        echo 1800000 > /proc/kunlun/dev$dev_id/task_timeout_detect_threshold_in_ms
        cat /proc/kunlun/dev$dev_id/task_timeout_detect_threshold_in_ms
    fi
done

DISTRIBUTED_ARGS=(
    --nproc_per_node `echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}'`
    --nnodes ${WORLD_SIZE:-"1"}
    --node_rank ${RANK:-"0"}
    --master_addr ${MASTER_ADDR:-"localhost"}
    --master_port ${MASTER_PORT:-"43622"}
)

MODEL_ARGS=(
    --num-layers 12
   # --moe-layer-freq [1]*12
    --hidden-size 4096
    --ffn-hidden-size 10240
    --num-attention-heads 16
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 10000
    --num-experts 8
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-3
    --moe-ffn-hidden-size 8192
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --transformer-impl transformer_engine
    --attention-backend flash
    --moe-router-dtype fp32
    --no-rope-fusion
    --disable-bias-linear
    --attention-softmax-in-fp32
    --accumulate-allreduce-grads-in-fp32
   # --disable-bf16-reduced-precision-matmul
    --moe-router-force-load-balancing
)


PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    #--moe-token-dispatcher-type flex 
   # --moe-token-dispatcher-type alltoall
    --moe-token-dispatcher-type allgather
   # --moe-enable-deepep
    # --pipeline-model-parallel-layout $PIPELINE_LAYOUT
   
)

DATA_ARGS=(
    --num-workers 32
    --dataloader-type cyclic
    --tokenizer-type SentencePieceTokenizer
    --tokenizer-model /mnt/lhycpfs/lhy/lhy/nhmegatron/zj_examples/V5000/genos/one_hot.bpe.model
    --data-path /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_000235385.1_SaiBol1.0_genomic_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_009663435.1_Callithrix_jacchus_cj1700_1.1_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02622.hap2_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_008086735.1_ASM808673v1_genomic_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_009764315.1_Tfra_2.0_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02630.hap2_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_014849445.1_KIZ_CMon_1.0_genomic_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_009828535.3_HMol_V3_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02984.hap1_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_021498455.1_ASM2149845v1_genomic_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_012559485.2_MFA1912RKSv2_genomic.clipped_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG03098.hap1_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_023764695.1_ASM2376469v1_genomic_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_024542745.1_ASM2454274v1_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG03540.hap1_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_023783065.1_ASM2378306v1_genomic.clipped_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_041146395.1_OSU_ERuf_1_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG03654.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_023783095.1_ASM2378309v1_genomic.clipped_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG00097.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG03688.hap2.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_023783475.1_ASM2378347v1_genomic.clipped_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG00320.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG03816.hap2.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_023783575.1_ASM2378357v1_genomic.clipped_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG00350.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG03874.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_023807365.1_ASM2380736v1_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG00558.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG03874.hap2.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_026956025.1_MCyc01_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG00658.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG03942.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_028878085.3_NHGRI_mSymSyn1-v2.0_alt_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG00733.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG04160.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_030128845.1_Clint_PTR_hifiasm-v0.15.2.alt_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG00735.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18565.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_030222135.1_86718_ANA_hifiasm-v0.15.2.pri_genomic.clipped_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG01109.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18565.hap2.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_031835075.1_ASM3183507v1_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG01243.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18747.hap2.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_040437455.1_PleCup_hybrid_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG01786.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18940.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_040869165.1_ASM4086916v1_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG01786.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18940.hap2.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_046862485.1_Cpen_1.0_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG01884.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18943.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_047047975.1_ASM4704797v1_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG01934.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18943.hap2.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_047496115.1_ASM4749611v1_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG01978.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18967.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_047655295.1_SemEnt_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02056.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18976.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_047834645.1_ASM4783464v1_genomic.clipped_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02080.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18976.hap2.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_048565385.1_mSaiBol1.pri_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02080.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA18982.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_049640505.1_ASM4964050v1_genomic.clipped_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02148.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA19043.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_947095605.1_mPonPyg1.1_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02148.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA19468.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCA_965153135.1_BT15_assembly_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02258.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA19468.hap2.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_000165445.2_Mmur_3.0_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02258.hap2_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA21093.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_003339765.1_Mmul_10_genomic.clipped_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02572.hap1_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/NA21106.hap1.eod_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_008122165.1_Kamilah_GGO_v0_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02572.hap2_text_document /mnt/zzbnew/Public/DataSet/nonhuman_primate/preprocess/nonhuman_primate_tokenization/nonhuman_primate_onehot_8k/GCF_008728515.1_Panubis1.0_genomic_text_document /mnt/zzbnew/Public/DataSet/human_genome_assemblies/preprocess/10B_data/one-hot/tokenization_5k_length_phase1_eod/HG02622.hap1_text_document
    # --data-cache-path /mnt/lhycpfs/lhy/lhy/genos_2/datacache
    --split 1000,0,0
    --no-create-attention-mask-in-dataloader
)

TRAINING_ARGS=(
    --bf16
    --init-method-std 0.01
    --micro-batch-size 1
    --global-batch-size 1024
    --seq-length 8192
    --max-position-embeddings 8192
    --lr 1e-4
    --min-lr 1e-5
    --lr-decay-samples 20053001
    --lr-decay-style cosine
    --lr-warmup-fraction 0.05
    --weight-decay 0.1
    --clip-grad 1.0
    --train-samples 25066252
    --eval-iters 0
    --eval-interval 50000000
    --log-interval 1
    --log-throughput
    # --use-cpu-initialization
    --distributed-timeout-minutes 60
    --ckpt-format torch_dist
    --auto-detect-ckpt-format
    --no-load-optim
    --no-load-rng
    --save-interval 10000
    #--no-save-optim
    # --no-save-rng
    --save /mnt/lhycpfs/lhy/lhy/genos_2/output
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1
)


PYTHONPATH=$MEGATRON_PATH:$MODEL_CONFIG_PATH:$PYTHONPATH \
    torchrun ${DISTRIBUTED_ARGS[@]} \
    -m xmegatron_ext.pretrain.pretrain_v0_15_0 \
    ${MODEL_ARGS[@]} \
    ${PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]}