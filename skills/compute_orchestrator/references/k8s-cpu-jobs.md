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
- Do not set `executionProfile`. Omit `image` unless the user explicitly names
  a custom image; when explicitly supplied, it must be pullable by K8s. The
  plugin rejects VIP jobs because current stable-workspace visibility cannot
  prove access to the VIP volume.

## Required sequence

1. Inspect the user's scripts, configuration, command, data, outputs, and
   environment requirements. Estimate shard count, maximum parallelism, CPU,
   memory, ephemeral storage, and deadline per worker. Use the smallest
   sufficient values; `parallelism` must not exceed `shardCount`.
2. Before calling `prepare_cpu_job`, use model reasoning to inspect every entry
   script, imported wrapper, configuration, and command. Declare all inspected
   files in `script_paths`, the complete business input/output set in
   `input_paths`/`output_paths`, and every consumed environment variable in
   exactly one of `required_env_keys` or `optional_env_keys`. Supply required
   non-platform values in `env`. The plugin deliberately does not infer these
   semantics from source strings or filename suffixes; it validates the model's
   declarations, stable paths, and environment completeness. Never edit the
   user's original script merely to submit it.
   Portal mounts the stable workspace into K8s at the same absolute path. For
   example, `/home/scientist/project123/data/input.csv` remains
   `/home/scientist/project123/data/input.csv` inside the worker.
   For Python workloads, use `/home/scientist/.venv/bin/python`. If imports are
   missing, install the smallest compatible dependency set locally first with
   `/home/scientist/.venv/bin/python -m pip install --ignore-installed ...`, then verify imports and
   the executable path with that same interpreter. Launch Python scripts through
   that interpreter rather than relying on a script shebang. Do not create or use
   Conda.
3. Confirm through model reasoning that shards are independent and have no live
   cross-worker coordination, then pass `independent_shards_confirmed=true`.
   MPI, collectives, rendezvous, barriers, distributed runtimes, and a worker
   depending on another live worker remain unsupported; do not call preparation
   when any is used. K8s provides the zero-based shard number through
   `JOB_COMPLETION_INDEX`. Set `uses_native_completion_index=true` only when user
   code reads that exact variable and declare it required. Otherwise pass the
   actual user interface as `worker_index_env` (for example `RANK`) or
   `worker_index_argument`, plus base 0 or 1. The launcher maps the K8s index to
   that interface. Never assume `RANK` or another user variable is populated
   automatically.
4. Confirm the returned environment inventory. Explicitly list every
   path-valued environment variable in `env_path_keys`; do not infer path meaning
   from its name. Required variables must be satisfied by the request, platform,
   or worker-index launcher; optional variables may remain unset. The preflight
   converts declared directory/file environment values from local
   stable-workspace paths to the identical K8s paths and validates input
   readability or output-parent writability. `submit_cpu_job` uses only
   this frozen environment; never add or change variables at submission time.
5. Finish dependency installation before submission. Workers must not run pip or
   mutate the shared `.venv`; they only execute the environment prepared in the
   current container. The generated launcher first replaces
   `/home/scientist/.venv/bin/python3.12` with a symlink to
   `/opt/conda/bin/python3`, because the local `/usr/bin/python3.12` target is not
   available in K8s. Never put secrets in job environment variables.
6. Immediately before asking for consent, call
   `estimate_cpu_job_capacity`. Treat its result as advisory: capacity can
   change before scheduling. Do not disclose node names, exact free resources,
   or raw Portal responses.
7. Execute `AskUserQuestion` with the fixed confirmation and options below.
   Only its affirmative option permits the matching `submit_cpu_job` call.
   Cancellation, free text, an old answer, or a changed request forbids
   submission.
8. Submit with a unique `requestId`. On transport timeout, retry only with the
   same ID and byte-equivalent request; otherwise query/list by the same ID.
9. Poll `get_cpu_job` every 15–30 seconds while actively monitoring. Use
   attempts and logs for diagnosis. A job is fully finished only when its state
   is terminal and `resourcesReleased == true`; do not wait for storage release.
10. Verify declared outputs and per-shard completion records. Report failed
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
