# Portal provisioning

Read this file only when current resources are insufficient or a Portal
operation is already active.

## Existing queued request while current resources suffice

Before the first smoke test or workload command for the submitted task, if
`get_resource_status.provisioning == true` and current resources suffice, execute
this fixed `AskUserQuestion` in the language used by the user:

```json
{"questions":[{"header":"运行任务","question":"当前分配资源能够运行新提交的任务。目前存在排队任务，排队成功后会自动切换资源暂停当时正在运行的其他任务，是否继续提交任务。","multiSelect":false,"options":[{"label":"继续","description":"继续提交并运行当前任务。"},{"label":"取消","description":"不提交当前任务。"}]}]}
```

English: header `Run task`; question `The currently allocated resources can run
the newly submitted task. There are queued tasks. Once queued successfully,
resources will be automatically switched, suspending other running tasks. Do you
want to proceed with task submission?` Options: `Continue` / `Submit and run the
current task.` and `Cancel` / `Do not submit the current task.`

`Continue` authorizes this submitted task and directly allows its smoke test and
workload commands. Do not poll, call `ensure_resource`, cancel, or replace the
existing queued request. `Cancel`, Other/free-form, empty, multiple, or unknown
answers forbid execution. Earlier consent does not satisfy this per-task gate.

## Required sequence

1. Call `get_available_clusters(required_gpu_count=estimated GPU count)`
   immediately before provisioning. For GPU work,
   reach this sequence only when the current GPU is incompatible or insufficient,
   then apply the suitability-first, capacity-aware selection in SKILL.md. Select
   only a cluster included in the latest result. An absent cluster must not be
   selected, submitted, waited for, polled, or queued until a later explicit
   availability call includes it.
2. Finish resource-independent preparation in the current container. Do not
   install dependencies there. Persist reusable artifacts on stable storage.
3. Call `get_resource_status`; its top-level boolean `provisioning` is
   authoritative. Stop if it is missing or not boolean.
4. Use exactly one fixed `AskUserQuestion` contract below, in the user's
   language, immediately before `ensure_resource`.
5. On confirmation, call `ensure_resource` immediately. Do no further work in
   the old container after an accepted switch request.

When at least one compatible cluster reports enough cards, ask the following
regardless of `provisioning`. Use the same question when legacy entries make
capacity unknown; unknown capacity is not proof that every resource is
insufficient:

```json
{"questions":[{"header":"切换资源","question":"成功切换资源会中断当前其他活跃的 session 和排队任务，请确认是否执行切换资源操作？如果当前资源不足会先进行排队，排队成功后会自动切换资源。","multiSelect":false,"options":[{"label":"确认切换","description":"确认提交资源切换请求。"},{"label":"取消","description":"不提交资源切换请求。"}]}]}
```

English: header `Switch`; question `A successful resource switch will interrupt
your other active sessions and queued tasks. Please confirm whether to proceed
with the resource switch. If resources are currently insufficient, the request
will be queued first, and resources will switch automatically once queuing
succeeds.` Options: `Confirm switch` / `Submit the resource-switch request.` and
`Cancel` / `Do not submit the resource-switch request.`

When every available compatible cluster has known capacity and reports fewer
cards than the requested GPU count, and `provisioning == false`, ask:

```json
{"questions":[{"header":"进入排队","question":"目前算力资源紧张，您的任务需要进入排队队列。排队成功后任务将立即启动执行，这可能会中断您其他正在运行的对话。请问您是否要进入排队。","multiSelect":false,"options":[{"label":"是的","description":"提交资源请求并进入排队队列。"},{"label":"取消","description":"不提交资源请求。"}]}]}
```

English: header `Join queue`; question `Computing resources are currently
constrained, and your task needs to be placed in the queue. Upon successful
queuing, the task will start immediately, which may interrupt your other ongoing
conversations. Would you like to join the queue?` Options: `Yes` / `Submit the
resource request and join the queue.` and `Cancel` / `Do not submit the resource
request.`

`Yes` permits the immediate `ensure_resource` call. `Cancel`, Other/free-form,
empty, multiple, or unknown answers forbid submission.

Only when every available compatible cluster has known capacity and reports
fewer cards than the requested GPU count, and `provisioning == true`, ask:

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
VRAM budget, `nhmegatron` constraint, and sanitized card-count check. For non-LLM
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
