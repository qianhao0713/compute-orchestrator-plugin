# K8s CPU jobs

Use this flow only for CPU-only work that can be divided into independent
shards. It submits an asynchronous K8s Indexed Job through Portal; it does not
switch, resize, or interrupt the Claude Code container. K8s may place multiple
workers on one physical node, so do not promise a node count.

## Eligibility

- Supported: parameter/grid sweeps, independent files or records, batch
  preprocessing, map-style evaluation, and other shards that communicate only
  through durable outputs.
- Unsupported: MPI, `torch.distributed`, Ray/Dask/Spark clusters, collectives,
  barriers, rendezvous, shared memory across workers, GPUs, or a worker that
  depends on another worker's live process.
- For many short singleton stages, worker 0 may perform the stage while other
  workers wait on an atomic marker in shared storage. Waiting must be bounded,
  idempotent, and fail on timeout. For a long singleton stage, submit it as a
  separate job before or after the parallel job instead of idling workers.
- Prefer a Portal execution profile. A custom image is allowed only when needed
  and must be pullable by K8s. The plugin rejects VIP jobs because current
  stable-workspace visibility cannot prove access to the VIP volume.

## Required sequence

1. Inspect the user's scripts, configuration, command, data, outputs, and
   environment requirements. Estimate shard count, maximum parallelism, CPU,
   memory, ephemeral storage, and deadline per worker. Use the smallest
   sufficient values; `parallelism` must not exceed `shardCount`.
2. Call `prepare_cpu_job` before submission. Declare every script/config/input
   and output. It must be readable or writable from
   `AI_SCIENTIST_STABLE_WORKSPACE_ROOT`; paths outside it fail closed. The tool
   scans declared code for additional hard-coded paths, unsupported collectives,
   and index use, then creates a verified launcher on stable storage. Never edit
   the user's original script merely to submit it.
   In K8s, the project root is
   `K8S_CPU_MOUNT_PATH/<stable-workspace-directory-name>`: for example,
   `/home/scientist/project123/data/input.csv` maps to
   `/mnt/nas/project123/data/input.csv`, not `/mnt/nas/data/input.csv`.
3. K8s provides the zero-based shard number only through
   `JOB_COMPLETION_INDEX`. If user code reads that name directly, no mapping is
   needed. Otherwise tell `prepare_cpu_job` the user's environment variable or
   command-line argument and whether it expects base 0 or 1. The derived
   launcher maps `JOB_COMPLETION_INDEX` to that interface for every worker.
   Never assume a user variable is populated automatically.
4. Prepare a reusable environment once on stable storage when needed. Workers
   may unpack/copy it to `/tmp`; do not let every worker concurrently mutate a
   shared Conda environment or install into shared site-packages. Never put
   secrets in job environment variables.
5. Immediately before asking for consent, call
   `estimate_cpu_job_capacity`. Treat its result as advisory: capacity can
   change before scheduling. Do not disclose node names, exact free resources,
   or raw Portal responses.
6. Execute `AskUserQuestion` with the fixed confirmation and options below.
   Only its affirmative option permits the matching `submit_cpu_job` call.
   Cancellation, free text, an old answer, or a changed request forbids
   submission.
7. Submit with a unique `requestId`. On transport timeout, retry only with the
   same ID and byte-equivalent request; otherwise query/list by the same ID.
8. Poll `get_cpu_job` every 15–30 seconds while actively monitoring. Use
   attempts and logs for diagnosis. A job is fully finished only when its state
   is terminal and `resourcesReleased == true`; do not wait for storage release.
9. Verify declared outputs and per-shard completion records. Report failed
   shards and retries without exposing infrastructure identifiers.

## Fixed submission confirmation

Chinese question:

`将向 K8s 提交 CPU 分布式任务：共 N 个分片，最大并发 M，每个 worker 使用 X 核 CPU 和 Y GiB 内存。该操作不会切换或中断当前 Claude Code 会话。是否确认提交？`

Options, in order:

1. `确认提交` — `提交任务并开始排队或运行。`
2. `取消` — `不提交任务。`

English question:

`A distributed CPU job will be submitted to K8s with N shards, maximum parallelism M, and X CPU cores and Y GiB memory per worker. This will not switch or interrupt the current Claude Code session. Submit the job?`

Options, in order:

1. `Submit` — `Submit the job to queue or run.`
2. `Cancel` — `Do not submit the job.`

Replace only N, M, X, and Y with the verified request values. Do not paraphrase,
add options, preselect an answer, or imply immediate capacity.

## Fixed cancellation confirmation

Chinese question:

`取消任务会停止所有尚未完成的分片，已经写入共享存储的中间结果不会自动删除。是否确认取消任务？`

Options: `确认取消` — `停止未完成的分片。`; `保留任务` — `不取消任务。`

English question:

`Cancelling the job will stop all unfinished shards. Intermediate results already written to shared storage will not be deleted automatically. Cancel the job?`

Options: `Cancel job` — `Stop unfinished shards.`; `Keep job` — `Do not cancel the job.`

## State and failure handling

Nonterminal states include `ACCEPTED`, `SUBMITTING`, `QUEUED`, `RUNNING`,
`RETRYING`, `FAILING`, and `CANCELLING`. Terminal states are `SUCCEEDED`,
`FAILED`, `CANCELLED`, and `UNKNOWN`; inspect attempts/logs before interpreting
`UNKNOWN`. Respect the configured backoff limit and active deadline. Shards
must write outputs atomically or use shard-specific output names so retries are
idempotent. Never infer success solely from submit HTTP status or a running
worker count; Portal business success still requires code `00000`.
