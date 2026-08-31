#!/usr/bin/env python3
"""Add resource-assessment context to every user prompt."""

from __future__ import annotations

import json
import re
import sys
from typing import Any


MODEL_PATTERN = re.compile(r"(?<![a-z0-9])(?:qwen|llama|deepseek)[a-z0-9.\-]*", re.I)
TRAINING_PATTERN = re.compile(
    r"\b(?:sft|lora|qlora|pretrain(?:ing)?|fine[ -]?tun(?:e|ing)|"
    r"train(?:ing)?|megatron|deepspeed)\b|训练|微调|预训练|后训练",
    re.I,
)
GPU_PATTERN = re.compile(
    r"(?<![a-z0-9])(?:gpu|gpus|cuda|cudnn|nccl|nvidia|nvlink|vram|torchrun|"
    r"deepspeed|megatron|flashattention|flash-attention|GPU-32G|GPU-96G)"
    r"(?![a-z0-9])|"
    r"显卡|显存|多卡|单卡|卡间|GPU任务|GPU训练|GPU推理",
    re.I,
)


def build_hook_output(event: dict[str, Any]) -> dict[str, Any] | None:
    prompt = event.get("prompt")
    if not isinstance(prompt, str):
        return None
    model_training = MODEL_PATTERN.search(prompt) and TRAINING_PATTERN.search(prompt)
    explicit_gpu = bool(GPU_PATTERN.search(prompt) or model_training)

    if explicit_gpu:
        context = (
            "Compute policy: this request explicitly indicates a GPU workload. "
            "The compute-orchestrator skill applies and must be loaded before "
            "any setup, conversion, launcher, shell, torchrun, DeepSpeed, "
            "Megatron, or workload command is executed. The initial CPU "
            "runtime is not an approved default GPU workload target. First "
            "inspect the task and code, assess required resources, inspect "
            "current resources, query available clusters, select a compatible "
            "fixed specification, and follow the skill's provisioning and "
            "handoff workflow. Before selecting GPU-96G, classify the task "
            "as LLM or non-LLM. For an LLM task, use only the fixed installed "
            "environment and official nhmegatron templates. Search the plugin-local "
            "snapshot "
            "at skills/compute_orchestrator/references/nhmegatron/zj_examples/GPU-96G "
            "before making any web request. If needed, inspect "
            "https://gitlab.zhejianglab.com/nh-megatron/nhmegatron/ remotely; "
            "cloning is optional, and zj_examples/GPU-96G is relative to the "
            "nhmegatron repository root. Never implement GPU-96G training logic "
            "or launchers from scratch; minimally derive only from the closest "
            "same-family, same-architecture official example, calculate per-GPU "
            "peak VRAM with safety headroom under 96 GiB, and smoke-test. For a "
            "non-LLM task, inspect its executable path and all direct and "
            "transitive dependencies. Use GPU-96G only if the task neither "
            "introduces nor requires a separate GPU runtime, backend, library "
            "implementation, or binary extension beyond the environment-default "
            "torch. Never install, upgrade, downgrade, replace, or shadow "
            "that torch; reject GPU-96G if compatibility cannot "
            "be proven."
        )
    else:
        context = (
            "Compute policy: before executing setup, installation, launcher, "
            "shell, or workload commands for this request, first determine "
            "from the task and executable code whether GPU, CUDA, substantial "
            "VRAM, or multi-GPU compute may be required. If the assessment "
            "indicates GPU use or a meaningful possibility of GPU use, load "
            "the compute-orchestrator skill before executing those commands, "
            "then inspect current resources and query available clusters. Do "
            "not begin a GPU-suitable workload in the initial CPU runtime. "
            "Proceed without the skill only after positively classifying the "
            "task as CPU-only and suitable for current resources."
        )

    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    if not isinstance(event, dict):
        return 0
    output = build_hook_output(event)
    if output is not None:
        json.dump(output, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
