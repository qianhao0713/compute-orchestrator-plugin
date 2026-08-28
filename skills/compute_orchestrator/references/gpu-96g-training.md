# gpu-96G model training

Use this path only for training supported Qwen, Llama, and DeepSeek families.
gpu-96G is a domestic accelerator cluster, and `nhmegatron` is its
domestic-accelerator training framework. gpu-96G has 96 GiB VRAM per card and accepts only these exact Portal tuples:

| GPUs | CPU | RAM GiB |
| ---: | ---: | ---: |
| 1 | 16 | 112 |
| 2 | 32 | 225 |
| 4 | 64 | 450 |

Send `resourceType: GPU`, `gpuType: gpu-96G`, and `workerNum: 1`. Select the
smallest tuple that supports the documented example and parallel strategy.

After migration to gpu-96G, run `xpu-smi` to enumerate and count accelerator
cards. Do not use `nvidia-smi` for gpu-96G card discovery, do not fall back to it,
and do not treat its failure or empty output as a missing-card condition. Compare
the `xpu-smi` device count with the requested Portal `gpuCount` before training.

## Local official-example snapshot

Search `references/nhmegatron/zj_examples/gpu-96G` relative to this Skill before
accessing GitLab. It contains the official gpu-96G text scripts and configurations
from `main` commit `49ebf44cee9fe35eff7cbf7937066dc0ee0e22d6`; see
`references/nhmegatron/SNAPSHOT.md` for provenance. Use it for template selection,
comparison, derivation, and resource estimation without repeated web requests.

The snapshot is not a complete training repository. Commands in its scripts may
refer to modules, binaries, data, or tokenizer assets that exist only in the
gpu-96G runtime or full upstream checkout. At execution time, run against the
platform's installed `nhmegatron` code and environment, not directly from the
plugin snapshot.

Use only the fixed Megatron/PyTorch environment and code already provided by the
gpu-96G platform. The canonical domestic-accelerator repository is
`https://gitlab.zhejianglab.com/nh-megatron/nhmegatron/` on branch `main`.
The example directory is relative to that repository root; it is not a path
under the user.s current project directory. Locate the platform.s checkout
first. If no checkout exists, use the plugin-local aliased snapshot for
reference; browsing or cloning the canonical GitLab repository is optional. Do
not use a fork, mirror, or alternative implementation. Use the corresponding
official 96 GiB accelerator examples, their environment script, and the
repository.s documented `toolkits` conversion flow.

## Official-example derivation rule

Do not implement gpu-96G training logic, Python entrypoints, distributed launchers,
or environment setup from scratch. Prefer an exact official script. If none
matches exactly, copy the closest official script under `zj_examples/gpu-96G` as
the sole template. Choose the same model family and architecture before using
parameter-count proximity; for example, derive Qwen3-1.7B from the official
Qwen3-8B example only after confirming architecture compatibility.

Limit changes to model structure values, tokenizer/checkpoint/data/output paths,
TP/PP/EP/CP/DP, micro/global batch sizes, sequence length, precision,
recomputation, optimizer settings, and documented feature switches. Preserve
the platform imports, training entrypoint, environment setup, launch mechanism,
and hardware-specific logic. Record the canonical source URL/path and revision,
derived script path, exact diff, and command in the handoff.

## Mandatory VRAM budget

Before selecting GPU count or running a derived script, estimate per-GPU peak
VRAM against the gpu-96G 96 GiB limit. Include model weights, gradients, optimizer
states, master weights, activations, attention and kernel workspaces,
communication buffers, allocator/CUDA/framework overhead, and any replicated
state implied by the chosen parallel strategy. Do not assume memory scales only
with parameter count. Account for sequence length, micro-batch size, precision,
activation checkpointing, TP/PP/EP/CP, and optimizer sharding.

Require reasonable safety headroom; target no more than about 80–85 GiB planned
peak per GPU unless an observed smoke test justifies a tighter bound. Reduce
micro-batch size, enable a documented recomputation strategy, or choose a larger
fixed GPU tier before exceeding the limit. Run a bounded smoke test and observe
peak memory before the full task. On OOM, stop, measure, and revise documented
parameters; do not blindly retry.

Supported examples include Qwen2.5 7B/72B, Qwen3 8B/32B/MoE 30B/MoE 235B,
Qwen3 VL 8B/32B, Qwen3.5/3.6, Llama3 8B/70B, DeepSeek V2 236B, and DeepSeek V3
671B. A missing exact size is not automatically unsupported: use the closest
same-family, same-architecture official example and the derivation rules above.

For the current image/algorithm-library choice and exact launch command, inspect
the canonical `nhmegatron` checkout and its examples after migration. Preserve the
example's TP/PP/EP/CP and recomputation constraints unless the requested model,
sequence length, and GPU count require a documented alternative. In particular,
do not combine SFT Packing with CP where the installed release marks that
combination unsupported.

Prepare stable datasets, checkpoints, converted model artifacts, configuration
values, output paths, and handoff state before expansion when possible. In the
continuation prompt, explicitly state that gpu-96G is a domestic accelerator
cluster, `nhmegatron` is its domestic-accelerator framework, and card
enumeration must use `xpu-smi` rather than `nvidia-smi`. Require the new
runtime to compare the `xpu-smi` count with the requested `gpuCount`. Also name
the official source example under `zj_examples/gpu-96G`, any minimally derived
script, its parameter diff, and the VRAM budget. Explicitly prohibit
from-scratch training or launcher code.
