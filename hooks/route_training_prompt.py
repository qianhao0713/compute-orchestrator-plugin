#!/usr/bin/env python3
"""Inject a short, high-priority compute-policy route for every prompt."""

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
            "Mandatory compute gate: load the compute-orchestrator skill before "
            "setup, installation, shell, launcher, or workload commands. Follow its "
            "complete required state machine and all non-negotiable safety rules. "
            "Inspect code, estimate resources, and inspect the container. Reuse a "
            "compatible, sufficient current GPU; otherwise rank compatible clusters "
            "and call get_available_clusters with required_gpu_count. Select the "
            "fixed switch question from capacity plus fresh provisioning state: use "
            "the queued-replacement question only when every compatible cluster has "
            "known insufficient capacity and provisioning is true; when compatible "
            "capacity is available or unknown, use the normal switch question even if "
            "provisioning is true. Before every smoke test or workload call "
            "get_resource_status. If current resources suffice and provisioning is "
            "true, use the fixed existing-queue question: Continue runs this task, "
            "Cancel blocks it, and this branch never polls or calls ensure_resource. "
            "For GPU-96G read references/GPU-96G-runtime.md and classify LLM versus "
            "non-LLM; LLM also reads references/GPU-96G-training.md and uses official "
            "templates, while non-LLM enforces the named Conda environment, dependency "
            "boundary, and built-in operator checklist. Do not "
            "proceed until all applicable gates pass."
        )
    else:
        context = (
            "Compute classification gate: before setup, installation, shell, "
            "launcher, or workload commands, inspect the task and executable "
            "path for possible GPU or substantial compute use. If GPU use is "
            "required or meaningfully possible, load the compute-orchestrator "
            "skill and follow its workflow before executing. Check "
            "get_resource_status before workload launch. If resources suffice but "
            "provisioning is true, use the fixed queued-switch risk question; "
            "Continue allows the task without polling. Proceed directly "
            "only after positively classifying the task as CPU-only and fitting "
            "the current effective container resources."
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
