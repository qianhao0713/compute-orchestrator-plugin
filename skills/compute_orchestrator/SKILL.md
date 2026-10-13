---
name: compute-orchestrator
description: >
  Mandatory resource planning and provisioning for any GPU-related,
  compute-intensive, or distributable CPU task, including inferred GPU use and
  multi-node CPU sharding. Use for GPU, CUDA, VRAM, inference, training,
  torchrun, NCCL, multi-GPU, Qwen, Llama, DeepSeek, SFT, LoRA, pretraining,
  Megatron, DeepSpeed, paper reproduction, large preprocessing, or substantial
  CPU/RAM work, K8s distributed CPU jobs, resource failures, or OOM. Load before
  setup, installation, smoke tests, launchers, or workload commands.
---

# Compute Orchestrator

Choose, provision, and use the smallest sufficient compatible environment.
Follow the state machine below in order even when GPU use was inferred rather
than requested or the conversation is long.

## Required state machine

1. Inspect the executable code/configuration and inputs. Classify the actual
   path as CPU or GPU. Read [resource estimation](references/resource-estimation.md)
   and estimate CPU, RAM, GPU count, and VRAM.
2. Call `inspect_current_resources`. Compare the estimate with effective
   container resources, never host resources.
3. Before every smoke test or workload launch, call `get_resource_status`.
   Missing or non-boolean `provisioning` fails closed.
4. If current resources suffice, keep the current compatible GPU cluster even
   if another is more suitable and execute normally regardless of
   `provisioning`. Do not poll, cancel, replace, or call `ensure_resource` for
   an existing queued request in this branch.
5. If resources are insufficient, complete resource-independent preparation
   first and persist reusable artifacts on stable storage. Do not install
   dependencies before switching.
6. Read [Portal provisioning](references/portal-provisioning.md) and follow its
   availability, selection, status, fixed confirmation, `ensure_resource`,
   handoff, response, and polling sequence exactly. After an accepted request,
   do no more work in the old container.
7. After a real switch, re-inspect resources and persisted state, install any
   required dependencies, smoke-test, and execute exactly once. On `NO_CHANGE`,
   continue exactly once in the current runtime.
8. On failure, read [failure recovery](references/failure-recovery.md) before
   retrying or resizing.

## Supported resources

Use only these public names and exact `(GPU, CPU, RAM GiB)` tiers:

- GPU-32G: 32 GiB per GPU, architecture `sm70`, maximum 8 cards:
  `(1,8,64)`, `(2,16,128)`, `(4,32,256)`, `(8,64,512)`.
- GPU-40G: 40 GiB per GPU, architecture `sm80`, maximum 8 cards:
  `(1,8,100)`, `(2,16,200)`, `(4,32,400)`, `(8,64,800)`.
- GPU-96G: 96 GiB per GPU, exactly 8 cards: `(8,128,900)`. Requests for
  1, 2, or 4 cards are invalid.
- CPU-only: 1–32 CPU cores; request the smallest sufficient allocation.
- `workerNum` is 1 unless Portal explicitly supports another value.

Never invent a GPU type, count, or tuple or request more resources merely for
speed.

## GPU selection

Do not switch when the current GPU cluster can run the task and its allocated
resources suffice. Otherwise rank compatible clusters by executable/runtime
support, operators, architecture, VRAM, and smallest sufficient tier.

Immediately before resource expansion, call
`get_available_clusters(required_gpu_count=estimated GPU count)` and use only
`capacityKnown`, `capacityAvailable`, and `capacitySufficient`:

1. GPU-96G is eligible only when the task genuinely requires 8 GPUs, or every
   other compatible GPU cluster in the latest result is either absent or has
   `capacityKnown == true` and `capacityAvailable == false`. Unknown capacity
   does not prove that a cluster has no cards. Never request 8 GPU-96G cards
   merely to run a task estimated for 1, 2, or 4 cards while another compatible
   GPU cluster still has any cards.
2. among eligible clusters, choose the most suitable compatible cluster if it
   has enough cards;
3. otherwise choose the highest-ranked eligible compatible cluster with enough
   cards;
4. if none has enough cards, choose the most suitable eligible compatible
   cluster and let Portal provisioning select the fixed queue confirmation from
   the fresh `provisioning` state.

A legacy string entry has unknown capacity, not zero capacity. A cluster absent
from the latest result cannot be selected, submitted, waited for, polled, or
queued until a later explicit result includes it. Never expose, infer, repeat,
or estimate an exact remaining-card count.

## GPU-96G routing

Whenever GPU-96G is current or a candidate, read
[GPU-96G runtime](references/GPU-96G-runtime.md) before planning, installing, or
executing. Unconditionally use its `python310_torch29_cuda` environment in every
Python, pip, torchrun, launcher, smoke-test, and workload shell command; never
assume activation persists across tool calls.

- LLM: read [GPU-96G model training](references/GPU-96G-training.md). Use the
  fixed `nhmegatron` environment/code and an official local template; never
  write training code or a launcher from scratch.
- Non-LLM: allow only a path whose direct and transitive GPU dependencies use
  that environment's existing `torch` without another GPU runtime, backend,
  implementation, or binary extension. Apply the runtime operator checklist.

## K8s distributed CPU jobs

When a task is CPU-only, independently shardable across workers (no collective
communication such as reduce, gather, scatter), and any of these apply:

- Single-machine resources are insufficient for the workload.
- The total computation is large enough to benefit from parallelism.
- The user explicitly requests distributed execution.

then submit it as a K8s Indexed Job instead of running locally.

### Required K8s workflow

1. Inspect the code and inputs. Confirm the task can be split by
   `JOB_COMPLETION_INDEX` into independent shards with no inter-shard
   communication. If not, this workflow does not apply.
2. Call `get_k8s_resources` to query per-node available CPU cores and memory.
   This is a request-capacity estimate, not real-time usage.
3. Estimate per-worker CPU and memory from the code and inputs. Choose values
   within the cluster's available capacity.
4. Choose `shard_count` (total shards) and `parallelism` (max concurrent
   workers). parallelism must not exceed shard_count. Keep both as small as
   the workload allows.
5. Select the container image. Prefer a Portal preset (`execution_profile`);
   otherwise specify `image` that the K8s cluster can pull. When no suitable
   image exists, install dependencies into a directory via conda before the
   main command, and adjust paths or code as needed.
6. Ensure all inputs and outputs are on the shared NAS mount. Inside the
   container, the shared PVC is at `/mnt/nas`. Both the Claude Code container
   and the K8s workers see the same directory. If paths are not already on the
   shared mount, adjust them to `/mnt/nas/...`.
7. Generate a unique output directory under the shared mount for this job.
8. Confirm with the user using this exact template:
   > 将向 K8s 提交 CPU 分布式任务：共 N 个分片，最大并发 M，每个 worker 使用
   > X 核 CPU 和 Y GiB 内存。该操作不会切换或中断当前 Claude Code 会话。
   > 是否确认提交？
   >
   > 确认提交 / 取消
9. Call `submit_k8s_job` with the chosen parameters. Save `jobId` from the
   response. On timeout, retry with the same `request_id` and identical
   parameters.
10. Poll `get_k8s_job_details` by `jobId` every 15-30 seconds. Monitor `state`,
    `succeededIndexesCount/total`, `runningPods`, and `reasonCode`. A task is
    fully done when state is SUCCEEDED/FAILED/CANCELLED and
    `resourcesReleased` is true.
11. For diagnostics, use `get_k8s_job_attempts` to locate a specific shard's
    `podUid`, then `get_k8s_job_logs` with `attemptId` to read its log.
12. To cancel, call `cancel_k8s_job` and continue polling until
    CANCELLED + resourcesReleased=true.

### K8s safety rules

- Only CPU tasks with independent shards. No collective communication
  (reduce, gather, scatter, all-reduce). No GPU work.
- `request_id` is the idempotency key. On timeout, retry with the same ID.
  Never generate a new one for the same task.
- Business success requires envelope `code == "00000"`; HTTP 200 alone is
  insufficient. Preserve `traceId`.
- Never supply `projectId` or `sessionId`. The server resolves the project
  from configuration.
- All I/O paths must be on the shared NAS (`/mnt/nas` inside the container).
  The script is responsible for reading inputs and organizing outputs.
- env keys must not include reserved names: JOB_ID, POD_UID,
  JOB_COMPLETION_INDEX. Values must be strings.
- Do not use `isVip` unless the user is explicitly in the VIP whitelist.
- `shard_count` max 5000, `parallelism` max 64, per-worker CPU max 32, memory
  max 128 GiB.
- Never access K8s Portal APIs directly with Bash, curl, wget, or HTTP
  libraries. Use only the MCP tools.

## Non-negotiable safety

- Immediately before every `ensure_resource`, obtain fresh
  `get_resource_status` state and use the exact language-matched
  `AskUserQuestion` selected by Portal provisioning. Cancellation, free-form,
  empty, multiple, unknown, or stale consent forbids submission. Use the
  queued-replacement question only when every available compatible cluster has
  known insufficient capacity and `provisioning == true`; available or unknown
  capacity uses the normal switch question even when a queued request exists.
- Never supply `projectId`, `sessionId`, or `clientMessageId`. The server resolves
  the project and the hook injects the authoritative Claude Code session.
- Follow `handoffEnabled` exactly: false means no continuation preparation and
  an empty message; true requires verified stable artifacts and the documented
  self-contained continuation.
- Portal business success requires code `00000`; HTTP success is insufficient.
  Preserve `traceId`. Reuse the same request ID and unchanged request only for a
  transport timeout.
- Never include credentials, cookies, tokens, upload streams, or ephemeral paths
  in a pending request.
- Before running any official or derived training script, inspect it for broad
  kills such as `pkill -9 python`, `killall python`, `fuser -k`, and equivalents.
  If found, do not modify or execute it in place. Copy it, replace every broad
  kill with a training-only pattern such as
  `pkill -9 -f "torchrun\|pretrain_v\|xmegatron_ext"`, verify, and run only the
  derived copy.

## User-visible privacy

In every plan, prompt, tool argument, status, and error, use only GPU-32G,
GPU-40G, and GPU-96G. Never disclose a vendor, chip family, physical
product/device name, backend identifier, UUID, serial number, or legacy template
directory name.

Never expose raw SMI output. Prefer `inspect_current_resources`. A direct
GPU-32G or GPU-40G query is allowed only as
`nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader,nounits`.
A direct GPU-96G query is allowed only as
`xpu-smi -q -d MEMORY,UTILIZATION,TEMPERATURE,CLOCK,PIDS`. Do not request or
display any other raw diagnostic table.

Do not silently change experiment meaning. Ask before changing cost, allocation,
data safety, or scientific meaning. For unsupported GPU-96G operators, follow
the notice, substitution, and user-decision rules in GPU-96G runtime.

## MCP boundary

Use only the following MCP tools for resource operations:

- Single-machine: `inspect_current_resources`, `get_resource_status`,
  `ensure_resource`, `persist_handoff`, `verify_persistent_artifacts`.
- GPU cluster selection: `get_available_clusters`.
- K8s distributed CPU: `get_k8s_resources`, `submit_k8s_job`,
  `get_k8s_job_details`, `list_k8s_jobs`, `get_k8s_job_attempts`,
  `get_k8s_job_logs`, `cancel_k8s_job`.

Never access Portal or K8s APIs directly with Bash, curl, wget, or a generic
HTTP library.
