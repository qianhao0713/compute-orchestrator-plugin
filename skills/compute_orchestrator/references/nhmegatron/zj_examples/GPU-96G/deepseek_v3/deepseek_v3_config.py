#!/usr/bin/env python3
"""DeepSeek model configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for DeepSeek V3 model."""

    # Core Architecture
    hidden_size: int = 7168
    make_vocab_size_divisible_by: int = 1280
    padded_vocab_size: int = 129280
    normalization: str = "RMSNorm"
    untie_embeddings_and_output_weights: bool = True
    add_bias_linear: bool = False

    # Attention Mechanism
    num_attention_heads: int = 128
    kv_channels: int = 128
    use_rotary_position_embeddings: bool = True
    rotary_percent: float = 1.0
    rotary_base: int = 10000
    rotary_scaling_factor: int = 40
    rotary_seq_len_interpolation_factor: int = 1
    multi_latent_attention: bool = True
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_head_dim: int = 128
    qk_layernorm: bool = True
    qk_pos_emb_head_dim: int = 64

    # FFN & MoE
    swiglu: bool = True
    ffn_hidden_size: int = 18432
    num_experts: int = 256
    moe_ffn_hidden_size: int = 2048
    moe_router_topk: int = 8
    moe_router_score_function: str = "sigmoid"
    moe_router_bias_update_rate: float = 0.001
    moe_router_enable_expert_bias: bool = True
    moe_router_pre_softmax: bool = True
    moe_router_topk_scaling_factor: float = 2.5
    moe_router_dtype: str = "fp32"
    moe_shared_expert_intermediate_size: int = 2048
    moe_aux_loss_coeff: float = 1e-3
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_token_drop_policy: str = "probs"
    moe_router_num_groups: int = 8
    moe_router_group_topk: int = 4

    # Initialization & Scaling
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    retro_encoder_attention_dropout: float = 0.0
    retro_encoder_hidden_dropout: float = 0.0
