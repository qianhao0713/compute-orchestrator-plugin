---
name: compute-orchestrator
description: >
  Mandatory resource planning and provisioning for every GPU-related or
  compute-intensive task. Always use for GPU, CUDA, VRAM, GPU inference,
  training, multi-GPU, torchrun, NCCL, GPU-32G, GPU-96G, Qwen, Llama,
  DeepSeek, SFT, LoRA, pretraining, Megatron, DeepSpeed, paper reproduction,
  large preprocessing, substantial CPU/RAM work, resource failures, OOM, or
  whenever code inspection suggests GPU use even if the user did not mention
  it. Load before setup, installation, launchers, or workload commands.
---

# Compute Orchestrator

Safely choose, provision, and use the smallest sufficient compute environment.
Follow this state machine in order; do not skip a gate because the conversation
is long.

## Required workflow

1. Understand the task and inspect or produce its executable code/configuration.
2. Classify it as CPU or GPU from the actual execution path.
3. Estimate CPU, RAM, GPU count, and VRAM from code and input scale. Read
   [resource estimation](references/resource-estimation.md).
4. Call inspect_current_resources and compare effective container resources,
   not host resources, with the estimate.
5. If current resources suffice, smoke-test and execute exactly once.
6. If insufficient, finish only preparation that does not require the target
   resources. Do not install dependencies before switching. Persist reusable
   artifacts on stable storage.
7. Read [Portal provisioning](references/portal-provisioning.md), then execute
   its availability, status, confirmation, request, and handoff sequence.
8. After a real switch, re-inspect resources and persisted state, smoke-test,
   and execute exactly once. On NO_CHANGE, continue exactly once in the current
   runtime.
9. On failure, read [failure recovery](references/failure-recovery.md) before
   retrying or resizing.

## Resource envelope

Use only these public cluster names and exact (GPU, CPU, RAM GiB) tiers:

- GPU-32G, 32 GiB per GPU, architecture sm70:
  (1,8,64), (2,16,128), (4,32,256), (8,64,512).
- GPU-96G, 96 GiB per card, maximum 4 cards:
  (1,16,112), (2,32,225), (4,64,450).
- CPU-only: 1–32 CPU cores; request the smallest sufficient allocation.
- workerNum is 1 unless Portal explicitly supports another value.

Never invent another GPU type, count, or tuple. Do not request larger resources
merely for speed.

## GPU-96G routing

Whenever GPU-96G is a candidate or current runtime, read
[GPU-96G runtime](references/GPU-96G-runtime.md) before planning, installing, or
executing.

Classify the task first:

- LLM: follow the fixed nhmegatron and official-template workflow in
  [GPU-96G LLM training](references/GPU-96G-training.md). Never write LLM
  training code or launchers from scratch.
- non-LLM: allow only code whose GPU path uses the python310_torch29_cuda
  environment's torch without a separate GPU runtime, backend, library
  implementation, or binary extension. Apply the operator compatibility
  checklist in GPU-96G-runtime.md.

On GPU-96G, unconditionally use python310_torch29_cuda as the effective default
Conda environment. Activate it inside every Python/pip/torchrun/launcher shell
command or use conda run -n python310_torch29_cuda; never assume activation
persists across Claude Code calls.

## Mandatory safety gates

1. Before every resource expansion, call get_available_clusters. A GPU cluster
   absent from the latest result cannot be selected, submitted, waited for,
   polled, or queued until a later explicit result includes it.
2. Immediately before every ensure_resource, call get_resource_status and use
   its top-level boolean provisioning to select the fixed AskUserQuestion
   contract in portal-provisioning.md. Missing/non-boolean state, cancellation,
   free-form, or unknown answers forbid submission.
3. Complete resource-independent preparation before confirmation. After an
   accepted switch request, do no more work in the old container.
4. Never manually provide projectId or sessionId. The server resolves the
   project and the hook injects the authoritative Claude Code session. Never
   send clientMessageId.
5. handoffEnabled == false means no handoff preparation and an empty
   pendingRequest.message. When true, persist and verify stable artifacts and
   build the self-contained continuation defined in portal-provisioning.md.
6. HTTP success is insufficient; Portal business success requires code 00000.
   Accepted/reused is not ready. Preserve traceId.
7. Retry a transport timeout with the same requestId, session, and unchanged
   request. Use a new ID only for a genuinely new request or terminal failure.
8. Never put credentials, cookies, tokens, upload streams, or ephemeral paths in
   pendingRequest.
9. When provisioning == true, always treat a proposed request as a different
   target. Use queued-replacement confirmation; do not compare specifications or
   submit merely to accelerate the queue.
10. If current resources suffice, do not switch. On NO_CHANGE, do not wait for a
    continuation prompt.

## Training-script process safety

Before running any official or derived training script, inspect it for broad
process-killing commands including pkill -9 python, killall python, fuser -k,
and equivalents. If present, never modify or run it in place. Copy it to a
derived script, replace every broad kill with a training-only pattern such as
pkill -9 -f "torchrun|pretrain_v|xmegatron_ext", verify the copy contains no
broad kill, and run only the derived copy. Broad kills can terminate uvicorn and
compute-orchestrator.

## User-visible naming and decisions

In every Claude-facing plan, prompt, tool argument, status, and error, call the
clusters only GPU-32G and GPU-96G. Never disclose a GPU vendor, chip family,
product/device name, backend identifier, or legacy template directory component;
normalize it to the public cluster name.

Do not silently change experiment meaning. Ask only when cost, allocation, data
safety, or scientific meaning changes. For a known unsupported GPU-96G operator,
follow the notice/substitution rules in GPU-96G-runtime.md.

## MCP tools

- inspect_current_resources: effective CPU, total RAM, devices, VRAM, and runtime.
- get_available_clusters: authoritative enabled GPU cluster list.
- get_resource_status: authoritative provisioning and handoff state.
- ensure_resource: validated, idempotent Portal request.
- persist_handoff and verify_persistent_artifacts: stable continuation state.

Never access Portal directly with Bash, curl, wget, or a generic HTTP library.
Use only these MCP tools.

## Final checkpoint

Before execution or provisioning, verify:

- executable path and inputs were inspected;
- resource estimate and smallest supported tier are documented;
- current effective resources were inspected;
- selected GPU cluster appears in the latest availability result;
- GPU-96G branch rules and named Conda environment are active when applicable;
- training scripts contain no broad kill;
- the correct fixed confirmation was accepted immediately before provisioning;
- stable preparation/handoff requirements are satisfied;
- the workload will execute exactly once.
