# Compute Orchestrator Claude Code Plugin

This plugin evaluates scientific workloads, prepares reusable data dependencies
in the current CPU container, requests CPU or V100 resources through Portal, and
resumes the task in the new Claude Code runtime.

## Core lifecycle

1. Understand the task and inspect/generate executable code.
2. Estimate CPU, RAM, GPU, VRAM, and topology requirements.
3. Compare requirements with Portal `currentResource` and local effective limits.
4. When a switch is required and confirmed, finish CPU-suitable preparation first:
   download models/data, clone repositories, verify artifacts, and persist a handoff.
5. Call Portal `ensure` with `pendingRequest`.
6. Portal creates/switches the environment and submits `pendingRequest` to the new runtime.
7. The new runtime verifies persisted artifacts, installs machine-specific environment
   dependencies, smoke-tests, and executes the core computation.
8. Reassess after confirmed resource-related failures such as OOM.

Physical NVIDIA V100 hardware is represented by Portal as `gpuType: "Z1120"`.

## Files

- `skills/compute_orchestrator/SKILL.md`: orchestration policy.
- `server.py`: MCP tool definitions.
- `portal_client.py`: authenticated Portal API client.
- `models.py`: request validation and secret rejection.
- `resource_inspector.py`: cgroup, RAM, GPU, VRAM, CUDA/PyTorch inspection.
- `handoff_manager.py`: stable handoff persistence and artifact checks.

## MCP tools

- `get_resource_status`
- `ensure_resource`
- `inspect_current_resources`
- `persist_handoff`
- `verify_persistent_artifacts`

MCP tools intentionally do not poll indefinitely. The Skill controls polling so
users can see status and the old runtime can stop safely during migration.

`ensure_resource.session_id` is injected by the plugin's `PreToolUse` hook from
Claude Code's authoritative hook context. Callers must omit it and must never
generate a replacement value. If Claude Code does not provide a session ID, the
hook blocks the tool call before it reaches Portal.

## Required environment variables

```bash
export PORTAL_BASE_URL="https://portal.example.com"
export PORTAL_ACCESS_TOKEN="..."
export AI_SCIENTIST_STABLE_WORKSPACE_ROOT="/home/scientist/project{project_id}"
# Optional explicit override:
# export PORTAL_PROJECT_ID="123456"
```

When set, `PORTAL_PROJECT_ID` must be a positive integer. When it is absent or
empty, the server extracts the ID from the final directory name of
`AI_SCIENTIST_STABLE_WORKSPACE_ROOT`, which must exactly match
`project{project_id}`. MCP callers do
not supply or generate `project_id`; both resource requests and handoff records
use this server-side value. Do not commit the Portal token. The stable workspace
root must be mounted and accessible from both the old and new containers.

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
- Network retries must preserve `requestId`, `clientMessageId`, and `pendingRequest`.
- `pendingRequest` must not contain authorization data, cookies, tokens, secrets,
  passwords, or temporary upload streams.
- `NO_CHANGE` means Portal will not resubmit the prompt; the current runtime continues once.
- During a real switch, the old runtime must not execute the same core task after submission.
