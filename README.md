# Compute Orchestrator Claude Code Plugin

This plugin evaluates scientific workloads, prepares reusable data dependencies
in the current CPU container, requests CPU, V100, or V5000 resources through Portal, and
resumes the task in the new Claude Code runtime.

## Core lifecycle

1. Understand the task and inspect/generate executable code.
2. Estimate CPU, RAM, GPU, VRAM, and topology requirements.
3. Compare requirements with Portal `currentResource` and local effective limits.
4. When a switch is required and confirmed, finish CPU-suitable preparation first:
   download models/data, clone repositories, verify artifacts, and persist a handoff.
5. Call Portal `ensure` with `pendingRequest`.
6. Portal creates/switches the environment and submits `pendingRequest` to the new runtime.
7. The new runtime verifies persisted artifacts, prepares machine-specific dependencies
   when required, smoke-tests, and executes the core computation. V5000 training uses
   its fixed preinstalled environment and an existing training script under the
   `nhmegatron` repository's `zj_examples/V5000` path. If no local checkout
   exists, the agent first uses the plugin-bundled official example snapshot;
   remote GitLab access is only a fallback. Missing exact variants may be minimally derived from the closest
   same-family official example after a per-GPU VRAM budget; from-scratch V5000
   training logic and launchers remain forbidden.
8. Reassess after confirmed resource-related failures such as OOM.

Physical NVIDIA V100 hardware is represented by Portal as `gpuType: "Z1120"`.
Z1120 requests must use one of the fixed `(GPU, CPU, RAM GiB)` tuples:
`(1,8,64)`, `(2,16,128)`, `(4,32,256)`, or `(8,64,512)`.
V5000 training hardware is represented as `gpuType: "V5000"`; each GPU has
96 GiB VRAM and requests must use a supported fixed GPU/CPU/RAM tuple.

## Files

- `skills/compute_orchestrator/SKILL.md`: orchestration policy.
- `skills/compute_orchestrator/references/nhmegatron/zj_examples/V5000`: bundled
  official V5000 example scripts for offline template selection.
- `server.py`: MCP tool definitions.
- `portal_client.py`: authenticated Portal API client.
- `models.py`: request validation and secret rejection.
- `resource_inspector.py`: cgroup, RAM, GPU, VRAM, CUDA/PyTorch inspection.
- `handoff_manager.py`: stable handoff persistence and artifact checks.

## MCP tools

- `get_resource_status`
- `get_available_clusters`
- `ensure_resource`
- `inspect_current_resources`
- `persist_handoff`
- `verify_persistent_artifacts`

MCP tools intentionally do not poll indefinitely. The Skill controls polling so
users can see status and the old runtime can stop safely during migration.

Before choosing a GPU target, callers query the available-cluster tool and avoid
clusters absent from its result. Portal creates a new Runtime session after a
resource switch. The ensure request carries the current Claude Code `sessionId`,
injected from trusted `PreToolUse` hook context; callers must not generate it.

The plugin's `UserPromptSubmit` hook adds a resource-classification rule to every
task. Explicit GPU and supported model-training prompts receive a stronger
orchestration directive; otherwise Claude must first infer whether the task or
code may require GPU compute. If so, it must load the Skill before running setup
or workload commands. This also covers GPU needs discovered by Claude rather
than stated by the user.

## Required environment variables

```bash
export PORTAL_BASE_URL="https://portal.example.com"
export PORTAL_ACCESS_TOKEN="..." # raw token or "Bearer ..." are both accepted
export AI_SCIENTIST_STABLE_WORKSPACE_ROOT="/home/scientist/project{project_id}"
# Optional; defaults to /home/scientist/.claude/logs/
# export PLUGIN_LOG_DIR="/path/to/plugin/logs/"
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
