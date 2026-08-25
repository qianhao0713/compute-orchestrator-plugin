---
name: compute-orchestrator
description: >
  Mandatory resource planning and provisioning for every GPU-related task and
  other compute-intensive workloads. Always use whenever a request mentions or
  requires GPU, CUDA, NVIDIA, VRAM, GPU inference, GPU training, multi-GPU,
  torchrun, NCCL, Z1120, or V5000, even for a small command or smoke test. Also
  always use before executing model training or fine-tuning, including Qwen,
  Qwen2, Qwen2.5, Qwen3, Qwen3-VL, Llama, DeepSeek, SFT, LoRA, pretraining,
  continued pretraining, preference training, distributed training, Megatron,
  or DeepSpeed tasks. Also use for paper reproduction, evaluation, simulation,
  large preprocessing, substantial CPU/RAM/GPU/VRAM work, multi-GPU or
  multi-node work, and after OOM, CUDA OOM, process kills, or distributed
  failures. Load this skill and assess resources before running setup, launcher,
  torchrun, deepspeed, or training shell commands; never start such training
  directly in the initial CPU environment.
---

# Compute Orchestrator

## Purpose

Safely execute scientific computing tasks in the user's current environment.

The workflow must:

1. understand the scientific task;
2. determine whether CPU or GPU compute is required;
3. produce or inspect the actual executable code and configuration;
4. estimate CPU, RAM, GPU, VRAM, GPU count, and worker requirements from that code;
5. inspect the resources available in the current container;
6. execute locally when the current resources are sufficient;
7. otherwise request an appropriate environment from Portal;
8. resume execution after Portal has completed the resource operation;
9. reassess and retry safely after resource-related failures.

Do not request larger resources merely because they might be faster. Request the
smallest supported configuration that is reasonably likely to complete the task.

## Available resource envelope

Portal can provide:

- CPU-only environments with 1–32 CPU cores;
- Z1120 GPU environments with 32 GiB VRAM per GPU and CUDA architecture `sm70`;
- V5000 domestic-accelerator training environments with 96 GiB VRAM per card,
  represented by `gpuType: "V5000"`;
- GPU counts of 1, 2, 4, or 8;
- multi-GPU and, when supported by Portal, multi-node execution.

Never invent an unsupported GPU model or GPU count.

Use the Portal GPU type names `Z1120` and `V5000` consistently. Do not use
hardware aliases for Z1120.

Use only these exact Z1120 `(GPU, CPU, RAM GiB)` tiers: `(1,8,64)`,
`(2,16,128)`, `(4,32,256)`, and `(8,64,512)`.

Use V5000 only for training supported Qwen, Llama, and DeepSeek models. V5000
is a domestic accelerator cluster, and `nhmegatron` is its domestic-accelerator
training framework. Its environment, repository, conversion tools, examples,
and launch scripts are fixed. Do not write training code or launchers from scratch. A derived launcher
is allowed only by copying the closest official V5000 example and making minimal,
reviewable model/hyperparameter changes after a VRAM estimate. Read
[references/v5000-training.md](references/v5000-training.md) and search the local
snapshot at `references/nhmegatron/zj_examples/V5000` before planning or
continuing a V5000 task. After switching to V5000, use `xpu-smi`, not
`nvidia-smi`, to detect and count accelerator cards. Do not interpret a missing
or empty `nvidia-smi` result as evidence that V5000 cards are unavailable. Do
not fetch the same examples from the web when the local snapshot contains the
required template.

`workerNum` currently defaults to `1` unless the runtime and Portal explicitly
support a different value. Do not assume that requesting multiple GPUs implies
multiple workers or multiple nodes.

## Required MCP tools

### `get_available_clusters`

Call `GET /scientist/deployment/clusters/available` after deciding CPU versus
GPU and before every resource expansion. Treat returned identifiers
case-insensitively. Never select Z1120 when `z1120` is absent or V5000 when
`v5000` is absent. An included identifier means enabled, not guaranteed idle
capacity; provisioning may still queue. An empty list means no internal GPU
cluster may be selected, but it does not prohibit a CPU-only K8s expansion.

This skill expects the plugin MCP server to expose these logical tools:

### `get_resource_status`

Calls:

`GET /scientist/deployment/provisioning/current`

It must return the Portal response data, including at least:

- `provisioning`
- `requestId`
- `operationType`
- `status`
- `candidateStatus`
- `errorMsg`
- `resourceChanged`
- `currentResource`
- `targetProvider`
- `targetResourceId`
- `targetResource`
- `routeVersion`
- `routeSwitched`
- `switchStatus`
- `pendingPromptStatus`
- `runtimeSubmissionId`
- `traceId`, when available

### `ensure_resource`

Calls:

`POST /scientist/deployment/ensure`

Inputs:

- `requestId`
- `reason`
- `projectId` (resolved by the MCP server from its environment)
- `sessionId` (injected from the current Claude Code hook context)
- `pendingRequest.message`
- `pendingRequest.parts`
- `pendingRequest.model`
- `pendingRequest.workingDirectory`
- `pendingRequest.attachmentRefs`
- `resource.resourceType`
- `resource.cpu`
- `resource.memoryGiB`
- `resource.gpuType`
- `resource.gpuCount`
- `resource.workerNum`

The MCP server, not the model, should own HTTP authentication and Portal base URL
configuration.

`get_resource_status` also returns the server-side `handoffEnabled` flag. When it
is `false` (the default), do not prepare continuation text:
`ensure_resource` forces `pendingRequest.message` to `""`. When it is `true`,
construct a non-empty, self-contained continuation message as described below.


### Optional `inspect_current_resources`

Prefer a dedicated read-only MCP tool when available. Otherwise use local shell
commands to inspect the current container.

It should report:

- logical CPU count;
- total and available RAM;
- GPU model;
- GPU count;
- total and free VRAM per GPU;
- CUDA driver/runtime visibility;
- container limits when different from host resources.

## Non-negotiable rules

1. Generate or inspect the executable code before making the final memory or
   VRAM estimate.
2. Never rely only on the user's requested resource size.
3. Never treat HTTP success as business success. Portal success requires
   `code == "00000"`.
4. Treat `provisioning` as the authoritative indicator of an unfinished Portal
   operation.
5. A successful `ensure_resource` response means accepted or reused, not ready.
6. `currentResource` is the Portal-reported specification currently serving the
   user; `targetResource` is the requested destination during provisioning.
7. Every `ensure_resource` request must carry `projectId`, `sessionId`, and
   `pendingRequest`. When `handoffEnabled == false`, require
   `pendingRequest.message == ""`; when it is true, require a non-empty resumable
   continuation message.
   Never generate or supply `projectId`; the MCP server reads it from
   `PORTAL_PROJECT_ID` or extracts it from the stable workspace basename.
   Never generate or manually supply `sessionId`; the plugin hook injects the
   authoritative current Claude Code session ID and overwrites caller input.
   Do not send `clientMessageId`.
8. Never place Authorization headers, cookies, access tokens, secrets, temporary
   upload streams, or other ephemeral credentials inside `pendingRequest`.
9. Preserve `traceId` in error reports.
10. On a network timeout after submitting a request, retry with the same
    `requestId`, hook-injected `sessionId`, and unchanged `pendingRequest`.
11. Use a new `requestId` only for a genuinely new business request or a retry
    after a terminal failure.
12. Query available clusters before expansion and exclude unavailable GPU types.
13. Do not submit repeated requests merely to accelerate a queue.
14. Do not claim that an in-progress Portal operation was overwritten unless the
    backend exposes and successfully executes an explicit cancel/replace API.
15. Do not start a destructive, expensive, or long-running job until the code,
    inputs, outputs, and resource plan have been checked.
16. Ask the user only when a decision materially changes cost, resource
    allocation, data safety, or experiment meaning.
17. When Portal performs a machine switch, do not execute the workload again in
    the old container after the request is accepted; Portal will submit
    `pendingRequest` to Claude Code in the new runtime.
18. When Portal returns `NO_CHANGE`, no machine switch occurs and Portal does not
    submit `pendingRequest`; continue the task exactly once in the current
    container.
19. Complete all safe CPU-suitable preparation before provisioning. Use the
    target machine for environment installation, hardware validation, and the
    core compute workload, not for avoidable downloads or preparation.
20. Immediately before every new resource-switch request, use
    `AskUserQuestion` in the language the user is currently using. Explicitly
    warn that a successful resource switch will interrupt the user's other active
    sessions, ask whether to proceed, and explain that insufficient resources
    cause the request to queue before switching automatically after the queue
    succeeds. For example, ask Chinese users
    `成功切换资源会中断当前其他活跃的 session，请确认是否执行切换资源操作？如果当前资源不足会先进行排队，排队成功后会自动切换资源。`
    and English users
    `A successful resource switch will interrupt your other active sessions. Please confirm whether to proceed with the resource switch. If resources are currently insufficient, the request will be queued first, and resources will switch automatically once queuing succeeds.` Submit the request only
    after an explicit affirmative answer for the current target specification.
    Earlier general consent does not satisfy this final confirmation gate.

21. When deriving a training script from an official template, remove or scope
    down any blanket process-killing commands. Specifically, replace
    `pkill -9 python` with a targeted pattern that only matches training
    processes (for example,
    `pkill -9 -f "torchrun\|pretrain_v\|xmegatron_ext"`). Never copy
    `pkill -9 python`, `killall python`, `fuser -k`, or similar broad-spectrum
    kill commands into generated scripts, because they will kill the agent
    infrastructure (`uvicorn`, `compute-orchestrator`) running in the same
    container.

## End-to-end procedure

### Phase 1: Understand the task

Determine:

- the scientific objective;
- whether this is reproduction, training, inference, evaluation, simulation,
  preprocessing, compilation, or another workload;
- repository, paper, dataset, checkpoint, and environment requirements;
- expected input scale;
- expected outputs and success criteria;
- whether the workload can be reduced for a smoke test;
- whether distributed execution is actually supported by the code.

When reproducing a paper, distinguish:

- environment setup;
- data preparation;
- compilation;
- smoke test;
- full experiment;
- evaluation;
- result comparison.

Do not allocate full-scale resources for environment setup or a small smoke test
unless they are truly necessary.

### Phase 2: Decide CPU versus GPU

Classify the workload using the actual algorithm and libraries.

Use CPU when all important operations are CPU-bound or the code has no usable GPU
path. Examples include lightweight preprocessing, parsing, small classical
models, orchestration, downloading, compilation without CUDA kernels, and small
unit tests.

Use GPU when the implementation actually performs GPU-supported operations and
the expected workload benefits materially or cannot fit practical CPU execution.
Examples include deep-learning training, large-model inference, CUDA kernels,
large tensor workloads, and GPU-specific paper implementations.

Do not request a GPU only because PyTorch, TensorFlow, JAX, or CUDA appears in
dependencies. Verify that the execution path places meaningful work on a GPU.

### Phase 3: Generate or inspect the code

For a supported Qwen, Llama, or DeepSeek training task targeting V5000, do not
perform the generic environment/code generation steps below. Follow the fixed
repository, model conversion tools, environment, example configuration, and
launch script in [references/v5000-training.md](references/v5000-training.md).
Search the plugin-local snapshot at
`references/nhmegatron/zj_examples/V5000` first. It mirrors official examples
for analysis but is not a complete executable repository. Access the official
GitLab only when the snapshot lacks a needed file or the user requests an
upstream refresh. Select an exact official V5000 example when available.
Otherwise select the closest official example with the same model family and
architecture, copy it as the sole template, and change only necessary model
structure, parallelism, batch, sequence, precision, recomputation, and stable
path parameters. Never implement a new training loop or launcher from scratch.

Before the final resource estimate:

1. inspect the existing repository and configuration;
2. identify the executable entrypoint;
3. create or modify the minimum code needed for the requested task;
4. identify model size, precision, batch size, sequence length, image size,
   optimizer, activation checkpointing, data loader workers, compilation mode,
   number of processes, and distributed strategy;
5. validate that multi-GPU execution is implemented before requesting more than
   one GPU;
6. run syntax checks, imports, unit tests, or a tiny dry run when feasible.

Prefer making memory-reducing code changes before requesting more hardware when
they preserve the requested scientific result. Examples include:

- mixed precision;
- smaller micro-batches with gradient accumulation;
- activation checkpointing;
- memory-efficient attention;
- streaming or memory-mapped datasets;
- fewer data-loader workers;
- avoiding duplicate model or dataset copies;
- sharded loading;
- offloading, when scientifically and operationally acceptable.

Do not silently change hyperparameters that affect the scientific meaning of a
reproduction. Explain such changes and obtain user approval when needed.

### Phase 4: Estimate resources from the code

Produce an internal resource plan containing:

- `resourceType`: `CPU` or `GPU`;
- CPU cores;
- RAM in GiB;
- GPU hardware/type: Z1120 or, for supported training, V5000;
- GPU count: one of `1`, `2`, `4`, or `8`;
- worker count;
- confidence level;
- assumptions;
- headroom.

For supported model training, also assess V5000 and choose the smallest fixed
V5000 tier that fits: `(GPU, CPU, RAM GiB)` = `(1,16,112)`, `(2,32,225)`,
`(4,64,450)`, or `(8,128,900)`. Each GPU has 96 GiB VRAM. Do not submit a
non-matching V5000 tuple.

For Z1120 choose the smallest fitting fixed tier: `(GPU, CPU, RAM GiB)` =
`(1,8,64)`, `(2,16,128)`, `(4,32,256)`, or `(8,64,512)`. Do not submit a
non-matching Z1120 tuple.

#### CPU estimate

Consider:

- number of active processes;
- data-loader workers;
- compilation parallelism;
- preprocessing parallelism;
- BLAS/OpenMP thread counts;
- distributed launch processes;
- dataset decompression and caching.

Avoid assigning more CPU cores than the program can use.

#### RAM estimate

Include:

- model parameters held in host memory;
- optimizer and checkpoint loading buffers;
- dataset indexes and in-memory samples;
- preprocessing intermediates;
- data-loader prefetching;
- process duplication;
- compilation or linking memory;
- framework overhead;
- output buffers and checkpoints.

Use measured values from a small run when available. Add reasonable headroom,
normally at least 20%, and more when the estimate is uncertain or data-dependent.

#### VRAM estimate

For training, account for:

- parameters;
- gradients;
- optimizer states;
- master weights;
- activations;
- attention workspaces;
- temporary kernels;
- CUDA context and allocator overhead;
- per-process duplication;
- distributed communication buffers.

For inference, account for:

- model weights;
- KV cache;
- activations and temporary buffers;
- input batch size and sequence length;
- framework and CUDA overhead.

Do not decide GPU count by dividing estimated VRAM by per-GPU VRAM unless the
code supports a suitable distribution strategy. Data parallelism duplicates the
model on each GPU and does not solve model-fit OOM.

#### Z1120 compatibility

Before requesting Z1120, account for its 32 GiB VRAM per GPU and CUDA
architecture `sm70`, then check:

- required CUDA compute capability;
- dtype support;
- whether BF16 is assumed;
- FlashAttention or fused-kernel requirements;
- custom CUDA extension architecture flags;
- library versions that may require newer GPUs.

Adapt code to supported Z1120 execution where possible without changing the
scientific target. Otherwise explain that the workload is incompatible with the
available GPU fleet.

### Phase 5: Inspect current resources

First call `get_resource_status` and use `currentResource` as the Portal-level
description of the container currently serving the user. Then inspect the local
runtime to verify effective cgroup limits, free memory, accelerator memory,
cluster-appropriate device visibility, and transient resource pressure.

`currentResource` is preferred for comparing provisioned specifications across
machines. Local inspection is preferred for deciding whether the workload can
start safely at this moment. If they disagree, use the smaller effective value
and report the discrepancy.

Inspect the effective container limits, not only host hardware.

Use the common container checks for every cluster:

```bash
nproc
free -b
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
cat /sys/fs/cgroup/memory.max 2>/dev/null || true
```

Then use the probe that matches `currentResource.gpuType`.

For Z1120, use NVIDIA tooling:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
```

For V5000, use domestic-accelerator tooling:

```bash
xpu-smi
```

Count V5000 cards from the devices reported by `xpu-smi`. Never use
`nvidia-smi` to determine V5000 card count, and never fall back from
`xpu-smi` to `nvidia-smi` on V5000. Use the fixed V5000 environment and its
framework-specific probe only as an additional check; do not require generic
PyTorch CUDA discovery to identify V5000 cards.

Normalize the current resources and the required resources into comparable
values.

Current resources are sufficient only when all required dimensions fit:

- correct compute type;
- enough CPU;
- enough effective RAM;
- compatible GPU model;
- enough GPUs;
- enough free VRAM per process or shard;
- supported distributed topology;
- required cluster-specific runtime and accelerator compatibility.

Use headroom. Do not start a job that is estimated to consume effectively all
available RAM or VRAM.

Make the resource-switch decision only after completing this inspection:

- if the current container is sufficient, continue directly to Phase 6;
- if the current container is insufficient, continue to Phase 7 before
  requesting a resource switch;
- do not prepare data or install dependencies merely because Phase 4 estimated
  a target resource. Phase 5 inspection determines which branch applies.

### Phase 6: Execute locally when sufficient

When Phase 5 confirms that the current container is sufficient, execute the
task normally in the current container. Do not perform the pre-switch workflow
and do not request a resource switch.

Before the full run:

1. record the command and configuration;
2. run a bounded smoke test when possible;
3. observe peak RAM and VRAM;
4. verify outputs;
5. then launch the intended computation.

Capture logs and resource metrics. Use deterministic seeds and preserve the
environment/configuration needed for reproducibility.

### Phase 7: Complete pre-switch preparation when insufficient

Only when Phase 5 confirms that the current container is insufficient, complete
preparation work that does not depend on the target GPU, larger target machine,
or target runtime before submitting the provisioning request. The preparation
must fit safely within the current container and produce reusable artifacts that
reduce idle time after the switch.

Typical work includes downloading model weights and datasets, verifying
checksums, extracting archives, cloning repositories, checking out revisions,
generating configuration files, preparing manifests or splits, selecting or
adapting approved launch scripts, and persisting the resource plan.

Dependency installation is not pre-switch preparation. Do not install or
upgrade Python packages, system packages, CUDA libraries, framework
dependencies, or create/modify the execution environment in the old container.
Record all required dependencies in the handoff and install them only after the
new container is active. For V5000, continue to follow its fixed environment,
code, and script rules rather than creating a replacement environment.

Do not perform preparation in the current container when it itself requires the
target resource or would exceed current CPU/RAM limits.

Write all reusable artifacts to stable shared storage such as the project NAS
working directory. Do not rely on `/tmp`, container-local caches, shell history,
in-memory state, temporary upload URLs, or paths unavailable from the target
runtime.

Before requesting the switch, create or update a persistent handoff record
containing the task objective, repository path and revision, completed
preparation steps, model and dataset paths, checksums when available, generated
configuration and launch-script paths, dependencies to install in the new
container, the exact next command, expected outputs, and known assumptions.

Avoid duplicate downloads. Inspect stable destinations, partial files,
checksums, and metadata first, and use resumable downloads when supported.

Preferred sequence:

```text
estimate required resources
→ inspect current container resources
→ if sufficient: execute normally in the current container
→ if insufficient: finish only resource-independent data/model/repository preparation
→ persist and verify all artifacts and dependency requirements
→ use AskUserQuestion in the user's language to warn that other active sessions will be interrupted and ask whether to proceed
→ only after explicit confirmation, submit Portal resource request
→ install dependencies in the new container and execute subsequent steps
```

### Phase 8: Request resources when insufficient

Call `get_available_clusters` immediately before the status/ensure flow. For GPU
work, filter candidate cluster types by its result. If the preferred type is
absent, use another enabled compatible cluster only when it can preserve the
task's meaning and execution contract; otherwise stop and report that no
compatible cluster is enabled. Never silently move V5000 fixed-code training to
Z1120 or run arbitrary code on V5000.

#### 8.1 Query current Portal operation

Call `get_resource_status`.

First check the envelope:

- if `code != "00000"`, stop and report `msg` and `traceId`;
- otherwise inspect `data`.

Use `data.provisioning` as authoritative.

#### 8.2 No operation in progress

When `provisioning == false`:

1. confirm that all CPU-suitable pre-switch preparation has completed;
2. verify that model, dataset, repository, and configuration artifacts are
   present on stable shared storage; verify handoff artifacts when
   `handoffEnabled == true`;
3. generate a stable UUID-based `requestId`;
4. rely on the MCP server's environment-resolved project ID;
5. when `handoffEnabled == true`, construct a self-contained
   `pendingRequest.message` from persisted state; otherwise do not prepare the
   message and let the server send it as an empty string;
6. build the smallest sufficient supported resource specification;
7. write a concise `reason` tied to the task and estimate;
8. use `AskUserQuestion` in the language the user is currently using; state
   that a successful resource switch will interrupt the user's other active
   sessions, ask whether to proceed, and explain that insufficient resources
   cause the request to queue before switching automatically after the queue
   succeeds;
9. submit nothing unless the user explicitly confirms this target specification;
   if the user declines or does not answer affirmatively, stop without calling
   `ensure_resource`;
10. call `ensure_resource` immediately after confirmation, without doing more
    work in the old container;
11. verify `code == "00000"`;
12. compare returned `data.requestId` with the submitted value;
13. inspect `operationType` and `resourceChanged`;
14. if a switch/update is active, stop execution in the old runtime and poll;
15. if Portal returns `NO_CHANGE`, continue the task once in the current runtime.

The confirmation is valid only for the resource specification shown to the
user. Ask again if that specification changes. Once `ensure_resource` succeeds,
assume the current container may be reclaimed at any moment and perform no
further preparation or workload execution in it.

CPU request:

```json
{
  "requestId": "<stable-uuid>",
  "reason": "<task and sizing rationale>",
  "resource": {
    "resourceType": "CPU",
    "cpu": 1,
    "memoryGiB": 1,
    "gpuType": null,
    "gpuCount": 0,
    "workerNum": 1
  },
  "pendingRequest": {
    "message": "<self-contained continuation prompt>",
    "parts": [],
    "model": "<runtime-model-or-omit>",
    "workingDirectory": "<stable-NAS-working-directory>",
    "attachmentRefs": []
  }
}
```

Do not include `projectId`, `sessionId`, or `clientMessageId` in caller-supplied
MCP arguments. The server resolves `projectId`, and the hook injects `sessionId`.

GPU request:

```json
{
  "requestId": "<stable-uuid>",
  "reason": "<task and sizing rationale>",
  "resource": {
    "resourceType": "GPU",
    "cpu": 8,
    "memoryGiB": 64,
    "gpuType": "Z1120",
    "gpuCount": 1,
    "workerNum": 1
  },
  "pendingRequest": {
    "message": "<self-contained continuation prompt>",
    "parts": [],
    "model": "<runtime-model-or-omit>",
    "workingDirectory": "<stable-NAS-working-directory>",
    "attachmentRefs": []
  }
}
```

Do not include `projectId`, `sessionId`, or `clientMessageId` in caller-supplied
MCP arguments. The hook injects `sessionId`. Replace placeholder quantities with assessed values. For V5000 use
`gpuType: "V5000"` and one of its exact fixed tuples.

#### 8.3 Construct a resumable `pendingRequest` only when enabled

Apply this section only when `get_resource_status.handoffEnabled == true`. When
it is false, skip all continuation-message preparation; the server forces
`pendingRequest.message` to `""` even if the caller supplies text.

The new container starts a new Claude Code process. The prompt must therefore be
self-contained and operational, not a vague message such as "continue" or
"resume the previous task".

The `pendingRequest.message` should tell the new runtime not to repeat completed downloads or persisted preparation. It should begin by inspecting the handoff record and stable artifacts.

The `pendingRequest.message` should contain:

- the user's scientific objective;
- what has already been completed;
- the exact next action to perform;
- repository and stable working directory;
- relevant code/configuration paths;
- dataset/checkpoint references that survive container replacement;
- selected resource plan;
- completed model/data downloads and their stable paths;
- remaining environment dependencies to install on the target machine;
- commands already tested;
- constraints that must not change;
- expected outputs and success criteria;
- instructions to inspect persisted state before repeating work.

For V5000, it must explicitly state that V5000 is a domestic accelerator
cluster, `nhmegatron` is its domestic-accelerator training framework, and the
new runtime must use `xpu-smi` rather than `nvidia-smi` to enumerate and count
cards. It must require comparing the `xpu-smi` count with the requested
`gpuCount` before training. It must also state that `zj_examples/V5000` is
relative to the root
of the canonical `nhmegatron` repository from
`https://gitlab.zhejianglab.com/nh-megatron/nhmegatron/`, not relative to the
current project directory. If the repository is absent, it must instruct the
Runtime to consult the plugin-local snapshot first; remote GitLab inspection and
cloning are optional fallbacks. It must identify the exact official source template and any
derived script, enumerate every changed parameter, include the per-GPU VRAM
estimate and headroom, and forbid from-scratch training code or launchers.

A suitable continuation prompt resembles:

```text
Continue the paper reproduction task in the configured Portal project.
Work in <stable working directory>. The repository is at <path>.
Completed: <brief persisted progress>.
Next: inspect the persisted handoff record and verify the prepared model and data artifacts. V5000 is a domestic accelerator cluster, and nhmegatron is its domestic-accelerator training framework. On V5000, run xpu-smi to enumerate and count cards, compare the result with the requested gpuCount, and never use or fall back to nvidia-smi for V5000 card discovery. For V5000 training, locate the nhmegatron checkout or inspect https://gitlab.zhejianglab.com/nh-megatron/nhmegatron/ remotely; cloning is optional. Treat zj_examples/V5000 as relative to that repository root. Use an exact official example when available; otherwise copy the closest same-family, same-architecture official V5000 example as the sole template and modify only required model/hyperparameter/path values. Record the source template and diff. Before running, calculate per-GPU VRAM for weights, gradients, optimizer/master states, activations, temporary workspaces, communication buffers, and framework overhead; keep headroom below 96 GiB per GPU, then run a bounded smoke test. Never implement training logic or a launcher from scratch. Stop if no compatible official template exists or the memory plan does not fit. Then run <command/configuration> and save logs and outputs under stable paths. If a
resource-related failure occurs, collect measurements and re-run the resource
assessment workflow.
```

Use `parts` and `attachmentRefs` only for stable Runtime-native references that
will remain accessible after switching machines. Do not embed large binary
content or temporary local paths.

`workingDirectory` should point to a stable NAS-backed project directory. Before
requesting a switch, persist all generated code, configuration, checkpoints,
logs needed for continuation, and a concise progress record there.

The `pendingRequest` is a continuation envelope, not a place for authentication
or hidden secrets.

#### 8.4 Operation already in progress

When `provisioning == true`, compare the current `targetResource` with the newly
required resource.

Normalize missing `workerNum` to `1`. Keep `Z1120` and `V5000` distinct.
Compare at least:

- resource type;
- CPU;
- memory GiB;
- GPU type (`Z1120` or `V5000`);
- GPU count;
- worker count.

##### Equivalent request

If the resources are equivalent:

- treat the current operation as satisfying the request;
- normally do not create a new business request;
- poll the current operation;
- if the integration contract requires calling POST to retrieve the same
  operation, reuse the existing `data.requestId`, not a newly generated ID;
- verify that the returned request ID still identifies the existing operation.

##### Different request

If resources differ:

1. explain the active request and the newly required request;
2. ask whether the user wants to keep waiting for the active request or replace
   it;
3. clearly state that the current Portal API cannot replace an unfinished
   operation;
4. if the user chooses to keep it, continue polling;
5. if the user chooses replacement, do not treat that keep-versus-replace choice
   as the final switch confirmation and do not pretend to replace it:
   - if the plugin exposes an explicit cancel/replace MCP tool, use the same
     language-matched `AskUserQuestion` confirmation gate immediately before calling it;
   - otherwise wait for the current operation to reach a terminal state, then
     use the confirmation gate immediately before submitting the new request;
   - report this backend limitation.

Submitting a different `requestId` to `ensure_resource` is not a replacement
mechanism under the current API contract.

### Phase 9: Poll Portal state and handoff

Poll `get_resource_status` until `provisioning == false`.

Recommended intervals:

- 3–5 seconds for `ACCEPTED`, `ROLLING_UPDATING`, `PROVISIONING`, `READY`,
  `CREATING`, `STARTING`, and other active transition states;
- 10–30 seconds for `QUEUED`.

Interpret states as follows:

- `IDLE`: no operation; a new request may be submitted;
- `ACCEPTED`: request accepted, asynchronous work pending;
- `ROLLING_UPDATING`: CPU deployment update in progress;
- `QUEUED`: waiting for internal resources;
- `PROVISIONING`: target environment is being created;
- `READY`: candidate is ready but backend processing may still be incomplete;
- `CREATING`: initial environment is being created;
- `STARTING`: reclaimed initial environment is restarting;
- `SUCCEEDED`: operation completed;
- `FAILED`: terminal failure; report `errorMsg` and `traceId`;
- `CANCELLED`: terminal cancellation;
- `UNKNOWN`: protectively treat as in progress while `provisioning == true`.

After the candidate is ready, Portal may pass through `CHECKING_SESSION`,
`SWITCHING`, `DISPATCHING_PROMPT`, and `PROMPT_ACCEPTED`. Treat all of these as
in progress when `provisioning == true`.

Also inspect:

- `routeVersion` and `routeSwitched` to identify an actual route transition;
- `switchStatus` for `IDLE`, `SWITCHING`, `DRAINING`, or `ERROR`;
- `pendingPromptStatus` to track automatic continuation delivery;
- `runtimeSubmissionId` as evidence that the new Runtime accepted the resumed
  request.

Do not execute the target workload merely because `status == READY` if
`provisioning` remains true. Do not manually resend the continuation prompt when
Portal has already accepted or submitted it.

After `SUCCEEDED`, inspect current resources again. Do not assume the final
machine exactly matches the requested minimum.

If Portal starts or restarts the initial CPU container instead of applying the
recommended target specification, rerun the complete resource assessment after
the initial container becomes ready.

### Phase 10: Continue exactly once after the resource decision

There are two distinct outcomes.

#### Machine switched or updated

When Portal is performing a real resource transition:

1. persist all state before the old container becomes unavailable;
2. stop launching new compute work in the old container;
3. when `handoffEnabled == true`, allow Portal to switch routing and submit the
   non-empty `pendingRequest.message`; in the new Claude Code runtime, treat
   that automatically submitted prompt as the authoritative continuation request;
4. when `handoffEnabled == false`, expect `pendingRequest.message == ""` and do
   not expect automatic task continuation; the user must start or resume work in
   the new container explicitly;
5. inspect `currentResource` and local effective resources again;
6. inspect persisted progress before repeating any step;
7. repeat a bounded smoke test when appropriate;
8. execute the intended task;
9. capture logs, metrics, outputs, and provenance.

The old Claude Code process must not also continue the same workload. This avoids
duplicate experiments, duplicate writes, and duplicate resource requests.

#### `NO_CHANGE`

When Portal returns `operationType == "NO_CHANGE"` or otherwise confirms that no
container switch will occur:

1. Portal will not resubmit `pendingRequest`;
2. continue the task in the current Claude Code process;
3. do so exactly once;
4. use `currentResource` and local inspection to confirm the environment;
5. do not wait for an automatic prompt that will never arrive.

Resource switching may restart or replace the runtime. Never assume in-memory
state or unpersisted local files survived.

## Resource-related failure recovery

Treat these as signals that the estimate or runtime configuration may be wrong:

- CUDA out of memory;
- host out of memory;
- exit code 137 or SIGKILL with memory pressure evidence;
- allocator failures;
- NCCL failures caused by topology or process-count mismatch;
- CPU thread/process exhaustion;
- disk-backed cache causing unexpected memory growth;
- GPU incompatibility or unsupported kernel architecture.

Do not classify every failure as insufficient hardware. First inspect logs and
distinguish:

- real memory exhaustion;
- memory leak;
- invalid tensor shape;
- software bug;
- unsupported Z1120 or V5000 kernel/environment;
- wrong distributed launch;
- data corruption;
- storage exhaustion;
- transient infrastructure failure.

When the failure is genuinely resource-related:

1. collect the exact error and peak resource usage;
2. update the estimate using observed measurements;
3. determine whether a code/configuration change can solve it without changing
   experiment meaning;
4. apply and smoke-test that change when appropriate;
5. otherwise choose the next supported resource tier;
6. return to Phase 5 and re-inspect the current resources;
7. limit blind retries.

After repeated failures with the same root cause, stop and explain the evidence
instead of indefinitely increasing resources.

## User-facing communication

Before requesting resources, briefly state:

- why CPU or GPU is required;
- the proposed CPU, RAM, GPU model/count, and worker count;
- the main assumptions;
- whether the request is for a smoke test or full experiment.

While queued, report that the current operation remains active. Do not imply
that repeated submissions improve queue priority.

When a conflicting operation exists, present both specifications in a compact,
comparable form.

After execution, report:

- resource environment used;
- command/configuration;
- completion status;
- key outputs;
- deviations from the paper or original plan;
- any remaining reproducibility caveats.

## Portal request validation

Before calling `ensure_resource`, validate:

- `requestId` is non-empty and at most 128 characters;
- `reason` is at most 512 characters;
- `projectId` resolves to a positive integer from `PORTAL_PROJECT_ID`, or from a
  stable workspace basename exactly matching `project{project_id}` when that
  variable is empty; never invent or manually provide it in tool arguments;
- the hook-injected `sessionId` is non-empty and at most 128 characters;
- `clientMessageId` is absent;
- when `handoffEnabled == false`, `pendingRequest.message == ""`;
- when `handoffEnabled == true`, `pendingRequest.message` is non-empty and at
  most 200000 characters;
- when handoff is enabled for V5000, `pendingRequest.message` states that V5000 is a domestic
  accelerator cluster, `nhmegatron` is its domestic-accelerator framework, and
  card enumeration must use `xpu-smi`, not `nvidia-smi`;
- `pendingRequest.parts` contains at most 100 items;
- `pendingRequest.model`, when present, is at most 256 characters;
- `pendingRequest.workingDirectory`, when present, is at most 1024 characters
  and references stable storage;
- `pendingRequest.attachmentRefs` contains at most 100 stable references;
- `pendingRequest` contains no credentials, cookies, tokens, secrets, or
  temporary upload streams;
- `resourceType` is `CPU` or `GPU`;
- `cpu >= 1`;
- `memoryGiB >= 1`;
- CPU requests use `gpuCount == 0` and `gpuType == null`;
- GPU requests use `gpuCount > 0` and `gpuType` equal to `Z1120` or `V5000`;
- GPU count is one of `1`, `2`, `4`, or `8`;
- `workerNum == 1` unless explicit backend support says otherwise;
- no `userId`, provider, runtime, image, or low-level resource UUID is sent;
- the server sends its configured `projectId` only in the documented top-level
  field;
- V5000 requests use exactly one fixed `(gpuCount,cpu,memoryGiB)` tuple;
- Z1120 requests use exactly one fixed `(gpuCount,cpu,memoryGiB)` tuple;
- the available-cluster response includes the selected GPU type;
- `AskUserQuestion` warned, in the language the user is currently using, that
  switching resources will interrupt the user's other active sessions, and
  received an explicit affirmative answer for this exact target specification
  immediately before this new switch request.

## Compact decision algorithm

```text
understand task
→ inspect/generate executable code
→ estimate CPU/RAM/GPU/VRAM/topology
→ inspect current effective resources
→ if sufficient:
     smoke test
     execute
  else:
     GET available clusters and filter GPU candidates
     complete CPU-suitable model/data/repository preparation
     persist and verify reusable artifacts
     GET current provisioning state and read handoffEnabled
     → if no active operation:
          verify persisted preparation
          if handoffEnabled: build self-contained pendingRequest.message
          otherwise: leave pendingRequest.message empty
          AskUserQuestion in the user's language: warn that a successful switch interrupts other active sessions, ask whether to proceed, and explain that insufficient resources queue first and switch automatically once queuing succeeds
          → affirmative: immediately POST smallest sufficient supported specification plus project/message context
          → otherwise: stop without POST
       else if active target equals required target:
          reuse and poll active operation
       else:
          ask keep-versus-replace
          → keep: poll active operation
          → replace:
               before any explicit cancel/replace call, ask the language-matched AskUserQuestion confirmation
               otherwise wait for terminal state, ask the same confirmation, then POST only on affirmation
     → poll until provisioning=false
     → if successful and machine switched:
          if handoffEnabled: Portal submits pendingRequest in the new Runtime
          otherwise: no automatic continuation prompt is expected
          new Runtime re-inspects resources with `xpu-smi` on V5000 or `nvidia-smi` on Z1120
          verifies prepared artifacts
          uses fixed V5000 environment or installs required target dependencies
          smoke test
          executes core task exactly once
       else if NO_CHANGE:
          current Runtime continues exactly once
     → if failed:
          report errorMsg + traceId
          reassess before a new request
→ on confirmed resource-related execution failure:
     measure
     optimize safely or resize
     repeat provisioning flow
```
