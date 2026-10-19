# Compute Orchestrator Claude Code Plugin

This plugin evaluates scientific workloads, prepares reusable data dependencies,
requests CPU or GPU resources through Portal when the interactive container must move,
and submits independently shardable CPU workloads as asynchronous K8s Indexed
Jobs without moving the Claude Code runtime.

## Core lifecycle

1. Understand the task and inspect/generate executable code.
2. Estimate CPU, RAM, GPU, VRAM, and topology requirements.
3. Compare requirements with Portal `currentResource` and local effective limits.
4. When a switch is required and confirmed, finish CPU-suitable preparation first:
   download models/data, clone repositories, and verify persistent artifacts.
   Prepare a continuation message only when `ENABLE_HANDOFF=true`.
5. Call Portal `ensure` with `pendingRequest`.
6. Portal creates/switches the environment and submits `pendingRequest` to the new runtime.
7. The new runtime verifies persisted artifacts, prepares machine-specific dependencies
   when required, smoke-tests, and executes the core computation. GPU-96G first
   classifies work as LLM or non-LLM. It uses
   `/home/scientist/.venv/bin/python` for Python and installs additional compatible
   packages into that shared virtual environment before execution. LLM training uses its fixed preinstalled
   environment and an existing training script under the
   `nhmegatron` repository's `zj_examples/GPU-96G` path. If no local checkout
   exists, the agent first uses the plugin-bundled official example snapshot;
   remote GitLab access is only a fallback. Missing exact variants may be minimally derived from the closest
   same-family official example after a per-GPU VRAM budget; from-scratch GPU-96G
   training logic and launchers remain forbidden. Non-LLM GPU tasks are allowed
   only when their GPU path depends on no GPU library or runtime beyond the
   shared `.venv` environment's torch, which must never be replaced or shadowed.
8. Reassess after confirmed resource-related failures such as OOM.

For distributed CPU work, the lifecycle is separate: validate scripts and
model-declared stable-workspace input/output paths, adapt K8s
`JOB_COMPLETION_INDEX` to the index interface
used by the user's script, discover required environment variables, normalize
directory variables to K8s project paths, estimate advisory capacity, obtain
fixed user confirmation, submit the frozen preflight command/environment, then
monitor status/attempts/logs. K8s CPU jobs do not call `ensure_resource` and do
not switch the current Claude Code container. Install additional Python packages
locally with `/home/scientist/.venv/bin/python -m pip install --ignore-installed`
before preflight; K8s uses the same interpreter and shared virtual environment
path. Each worker repairs `.venv/bin/python3.12` to target
`/opt/conda/bin/python3` before launching the workload.

Portal exposes the 32 GiB resource class as `gpuType: "GPU-32G"`.
GPU-32G requests must use one of the fixed `(GPU, CPU, RAM GiB)` tuples:
`(1,8,64)`, `(2,16,128)`, `(4,32,256)`, or `(8,64,512)`, and must never
exceed 8 cards.
GPU-40G requests use fixed `(GPU, CPU, RAM GiB)` tuples:
`(1,8,100)`, `(2,16,200)`, `(4,32,400)`, or `(8,64,800)`, and must never
exceed 8 cards.
GPU-96G resources are represented as `gpuType: "GPU-96G"`; each GPU has
96 GiB VRAM. Requests must use the single `(GPU, CPU, RAM GiB)` tuple
`(8,128,900)`; 1-, 2-, and 4-card requests are rejected. This tier is selected
only for tasks that genuinely require 8 GPUs, or as a fallback when every other
compatible GPU cluster is confirmed to have no available cards.

## Files

- `skills/compute_orchestrator/SKILL.md`: orchestration policy.
- `skills/compute_orchestrator/references/nhmegatron/zj_examples/GPU-96G`: bundled
  official GPU-96G example scripts for offline template selection.
- `server.py`: MCP tool definitions.
- `portal_client.py`: authenticated Portal API client.
- `models.py`: request validation and secret rejection.
- `resource_inspector.py`: cgroup, RAM, GPU, VRAM, CUDA/PyTorch inspection.
- `handoff_manager.py`: stable handoff persistence and artifact checks.
- `cpu_job_preflight.py`: K8s CPU script/path/index validation and derived launcher.

## MCP tools

- `get_resource_status`
- `get_available_clusters`
- `ensure_resource`
- `inspect_current_resources`
- `persist_handoff`
- `verify_persistent_artifacts`
- `prepare_cpu_job`
- `estimate_cpu_job_capacity`
- `submit_cpu_job`
- `get_cpu_job`
- `list_cpu_jobs`
- `cancel_cpu_job`
- `get_cpu_job_attempts`
- `get_cpu_job_logs`

MCP tools intentionally do not poll indefinitely. The Skill controls polling so
users can see status and the old runtime can stop safely during migration.

Callers keep a compatible, sufficient current GPU rather than switching to a
more suitable cluster. Only when switching is necessary do they query the
available-cluster tool with the required GPU count, rank task-compatible
clusters, prefer one whose derived capacitySufficient value is true, and use the best-suited cluster with queue confirmation when none has enough.
Legacy name-only responses retain unknown-capacity behavior. Clusters absent from
the result are forbidden. Portal creates a new Runtime session after a resource
switch. The ensure request carries the current Claude Code `sessionId`,
injected from trusted `PreToolUse` hook context; callers must not generate it.

The plugin's `UserPromptSubmit` hook adds a resource-classification rule to every
task. Explicit GPU and supported model-training prompts receive a stronger
orchestration directive; otherwise Claude must first infer whether the task or
code may require GPU compute. If so, it must load the Skill before running setup
or workload commands. This also covers GPU needs discovered by Claude rather
than stated by the user.

The plugin also denies Bash commands that directly call Portal with `curl`,
`wget`, Python HTTP clients, or similar tools. Portal traffic must go through the
MCP tools so endpoint construction, request validation, trusted session context,
and audit logging cannot be bypassed. This guard applies even when the Skill has
not loaded.

## Required environment variables

```bash
export PORTAL_BASE_URL="https://portal.example.com"
export PORTAL_ACCESS_TOKEN="..." # raw token or "Bearer ..." are both accepted
export AI_SCIENTIST_STABLE_WORKSPACE_ROOT="/home/scientist/project{project_id}"
# Optional; defaults to /home/scientist/.claude/logs/
# export PLUGIN_LOG_DIR="/path/to/plugin/logs/"
# Optional; defaults to false. Set true to send a continuation message.
# export ENABLE_HANDOFF="true"
# Optional explicit override:
# export PORTAL_PROJECT_ID="123456"
```

When set, `PORTAL_PROJECT_ID` must be a positive integer. When it is absent or
empty, the server extracts the ID from the final directory name of
`AI_SCIENTIST_STABLE_WORKSPACE_ROOT`, which must exactly match
`project{project_id}`. MCP callers do
not supply or generate `project_id`; both resource requests and handoff records
use this server-side value. Do not commit the Portal token. The stable workspace
root must be mounted and accessible from both the old and new containers. Before
each Portal HTTP request, the plugin appends the method, URL, sanitized headers,
and JSON body to `PLUGIN_LOG_DIR/portal_requests.jsonl`. Authorization is
redacted. A logging failure prevents the Portal request from being sent.

K8s CPU submissions use the normal (non-VIP) storage route. They omit
`executionProfile`; `image` is also omitted unless the user explicitly specifies
a pullable custom image.
Every declared input must exist beneath
`AI_SCIENTIST_STABLE_WORKSPACE_ROOT`, and every output must have a writable
parent there. Portal mounts the stable workspace into K8s at the same absolute
path: `/home/scientist/project123/data/input.csv` remains
`/home/scientist/project123/data/input.csv`. VIP jobs are rejected because the
interactive container cannot verify their separate storage volume.

`ENABLE_HANDOFF` accepts `true/false`, `1/0`, `yes/no`, or `on/off`. It
defaults to `false`. When disabled, callers do not need to prepare continuation
text and the server forces `pendingRequest.message` to an empty string. When
enabled, `pendingRequest.message` must be non-empty and self-contained.

## Install dependencies

For production, preinstall dependencies in the Portal container image:

```bash
python3 -m pip install -r requirements.txt
```

## Local development

```bash
export CLAUDE_PLUGIN_ROOT="$PWD"
python3 -m pytest -q
claude --plugin-dir "$PWD"
```

Inside Claude Code, inspect MCP connectivity with `/mcp`.

## Backend contract notes

- Always check Portal business `code`, not only the HTTP status.
- `provisioning` is authoritative for unfinished operations.
- A successful ensure response may mean accepted or reused; it does not mean ready.
- Network retries must preserve `requestId`, `sessionId`, and `pendingRequest`.
- `pendingRequest` must not contain authorization data, cookies, tokens, secrets,
  passwords, or temporary upload streams.
- `NO_CHANGE` means Portal will not resubmit the prompt; the current runtime continues once.
- During a real switch, the old runtime must not execute the same core task after submission.
