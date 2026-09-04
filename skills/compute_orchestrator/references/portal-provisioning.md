# Portal provisioning

Read this file only when current resources are insufficient or a Portal
operation is already active.

## Required sequence

1. Call `get_available_clusters` immediately before provisioning. For GPU work,
   select only a cluster included in the latest result. An included cluster may
   queue. An absent cluster must not be selected, submitted, waited for, polled,
   or queued until a later explicit availability call includes it.
2. Finish resource-independent preparation in the current container. Do not
   install dependencies there. Persist reusable artifacts on stable storage.
3. Call `get_resource_status`; its top-level boolean `provisioning` is
   authoritative. Stop if it is missing or not boolean.
4. Use exactly one fixed `AskUserQuestion` contract below, in the user's
   language, immediately before `ensure_resource`.
5. On confirmation, call `ensure_resource` immediately. Do no further work in
   the old container after an accepted switch request.

When `provisioning == false`, ask:

```json
{"questions":[{"header":"切换资源","question":"成功切换资源会中断当前其他活跃的 session，请确认是否执行切换资源操作？如果当前资源不足会先进行排队，排队成功后会自动切换资源。","multiSelect":false,"options":[{"label":"确认切换","description":"确认提交资源切换请求。"},{"label":"取消","description":"不提交资源切换请求。"}]}]}
```

English: header `Switch`; question `A successful resource switch will interrupt
your other active sessions. Please confirm whether to proceed with the resource
switch. If resources are currently insufficient, the request will be queued
first, and resources will switch automatically once queuing succeeds.` Options:
`Confirm switch` / `Submit the resource-switch request.` and `Cancel` / `Do not
submit the resource-switch request.`

When `provisioning == true`, treat the active request as a different target
without comparing specifications, and ask:

```json
{"questions":[{"header":"重新排队","question":"当前存在待排队任务, 若提交新的排队任务, 原排队任务将撤销，新任务重新排队，是否确认提交?","multiSelect":false,"options":[{"label":"确认提交","description":"撤销原排队任务，并提交新的排队任务。"},{"label":"取消","description":"保留原排队任务，不提交新请求。"}]}]}
```

English: header `Requeue`; question `A task is currently waiting in the queue.
Submitting a new queued task will cancel the existing queued task, and the new
task will re-enter the queue from the beginning. Do you confirm submission?`
Options: `Confirm submission` / `Cancel the existing queued task and submit the
new queued task.` and `Cancel` / `Keep the existing queued task and do not
submit a new request.`

Translate the full contract for other languages. Keep the fixed options and
`multiSelect: false`. Treat cancel, Other/free-form, empty, multiple, and unknown
answers as rejection. Earlier consent does not satisfy this gate.

## Request and handoff

Use a stable UUID `requestId`; reuse the exact request after a transport timeout.
The server supplies `projectId`; the PreToolUse hook supplies authoritative
`sessionId`. Never invent either or send `clientMessageId`.

`get_resource_status.handoffEnabled` controls the message:

- false: do not prepare continuation text; the server sends an empty message;
- true: prepare a self-contained continuation with objective, persisted state,
  stable paths, exact next command, dependencies, resource plan, expected
  outputs, and constraints. Never include credentials or temporary references.

For GPU-96G, require `python310_torch29_cuda` in every shell command (or
`conda run -n python310_torch29_cuda`) and verification of `sys.executable` and
the `torch` path/version. For LLM work also name the official template and diff,
VRAM budget, `nhmegatron` constraint, and `xpu-smi` card-count check. For non-LLM
work record dependency/operator eligibility and prohibit replacing that
environment's `torch`.

## Response and polling

Portal business success requires `code == "00000"`; the MCP server unwraps it.
An accepted request is not ready. Poll `get_resource_status` until
`provisioning == false`: 3–5 seconds for active transitions and 10–30 seconds
for `QUEUED`. Treat `CHECKING_SESSION`, `SWITCHING`, `DISPATCHING_PROMPT`, and
`PROMPT_ACCEPTED` as active while provisioning remains true.

On `FAILED`, report `errorMsg` and `traceId`. On a real switch, continue exactly
once in the new runtime after re-inspection. On `NO_CHANGE`, Portal sends no
continuation; continue exactly once in the current runtime.
