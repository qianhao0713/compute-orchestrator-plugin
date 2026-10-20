---
name: compute-orchestrator
description: >
  Mandatory resource planning and provisioning for any GPU-related or
  compute-intensive task, including inferred GPU use. Use for GPU, CUDA, VRAM,
  inference, training, torchrun, NCCL, multi-GPU, Qwen, Llama, DeepSeek, SFT,
  LoRA, pretraining, Megatron, DeepSpeed, paper reproduction, large preprocessing,
  distributed CPU, multi-worker, parameter sweep, grid search, Indexed Job,
  batch/sharded processing, or substantial CPU/RAM work,
  resource failures, or OOM. Load before setup, installation, smoke tests,
  launchers, or workload commands.
---

# Compute Orchestrator

Choose, provision, and use the smallest sufficient compatible environment.
Follow the state machine below in order even when GPU use was inferred rather
than requested or the conversation is long.

## Required state machine

1. Inspect the executable code/configuration and inputs. Classify the actual
   path as CPU or GPU. Read [resource estimation](references/resource-estimation.md)
   and estimate CPU, RAM, GPU count, and VRAM.
   For independently shardable CPU work that needs multiple workers, or when
   the user explicitly requests distributed CPU execution, read
   [K8s CPU jobs](references/k8s-cpu-jobs.md) and follow that flow instead of
   container resource switching. Do not continue with steps 2–8 for that
   branch. Cross-worker collectives are unsupported.
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
- Interactive CPU-only: use the effective container resources when sufficient;
  otherwise request the smallest sufficient allocation from 1–32 CPU cores.
- K8s CPU jobs: independently shardable CPU-only work, submitted asynchronously
  without switching the Claude Code container; use the per-worker limits and
  workflow in [K8s CPU jobs](references/k8s-cpu-jobs.md).
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
executing. Use `/home/scientist/.venv/bin/python` for Python commands and install
only compatible additional packages into `/home/scientist/.venv`; do not use the
retired Conda environment.

- LLM: read [GPU-96G model training](references/GPU-96G-training.md). Use the
  fixed `nhmegatron` environment/code and an official local template; never
  write training code or a launcher from scratch.
- Non-LLM: allow only a path whose direct and transitive GPU dependencies use
  that environment's existing `torch` without another GPU runtime, backend,
  implementation, or binary extension. Apply the runtime operator checklist.

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
- Before every K8s CPU submission, run `prepare_cpu_job`, then
  `estimate_cpu_job_capacity`, show the exact language-matched fixed submission
  confirmation from [K8s CPU jobs](references/k8s-cpu-jobs.md), and submit only
  after affirmative consent. The model must determine and declare the complete
  script/configuration, input/output, required/optional environment, independent-
  shard, and worker-index contracts; do not rely on source-string or suffix
  heuristics. Preparation validates those declarations and path mappings under
  `AI_SCIENTIST_STABLE_WORKSPACE_ROOT`. Map `JOB_COMPLETION_INDEX` explicitly to
  user variables such as `RANK`. Before preparation, install any additional
  Python packages locally with
  `/home/scientist/.venv/bin/python -m pip install --ignore-installed ...` and
  pass every required distribution name (not import-module name) through
  `required_python_packages`. Preparation and submission must verify package
  metadata directly under the shared `.venv`; ordinary import success is not
  proof because it may resolve from system site-packages. K8s must run
  `/opt/conda/bin/python3`
  with its Pod-local `sitecustomize` bootstrap loading the shared `.venv`
  site-packages; never modify shared `.venv` links or install from a Worker.
  Never submit collective or live inter-worker
  communication code. Cancellation also requires its fixed confirmation.

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

Use only `inspect_current_resources`, `get_available_clusters`,
`get_resource_status`, `ensure_resource`, `persist_handoff`,
`verify_persistent_artifacts`, `prepare_cpu_job`,
`estimate_cpu_job_capacity`, `submit_cpu_job`, `get_cpu_job`,
`list_cpu_jobs`, `cancel_cpu_job`, `get_cpu_job_attempts`, and
`get_cpu_job_logs` for resource and K8s job operations. Never access Portal
directly with Bash, curl, wget, or a generic HTTP library.
