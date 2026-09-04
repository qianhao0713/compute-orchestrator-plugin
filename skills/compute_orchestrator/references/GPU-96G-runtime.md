# GPU-96G runtime

Read this file whenever GPU-96G is a candidate or the current runtime.

## Environment

Unconditionally use the `python310_torch29_cuda` Conda environment. Run every
Python, pip, torchrun, launcher, smoke-test, and workload command by activating
it in the same shell command or with `conda run -n python310_torch29_cuda`.
Never assume activation persists across Claude Code tool calls. Never use base
or another Conda environment. Verify `sys.executable`, the `torch` import path,
and its version before execution.

Use `xpu-smi`, never `nvidia-smi`, to count GPU-96G cards. Compare the observed
count with the requested count.

## LLM branch

Any task involving an LLM follows the fixed Qwen/Llama/DeepSeek `nhmegatron`
workflow. Read [GPU-96G-training.md](GPU-96G-training.md), use an official local
template, and never write the training loop or launcher from scratch.

## Non-LLM branch

Allow an existing non-LLM task only when its complete execution path and direct
and transitive dependencies neither introduce nor require a separate GPU
runtime, backend, library implementation, or binary extension beyond the named
environment's `torch`. Upper-level packages such as a YOLO implementation are
allowed when all accelerator operations flow through that `torch`.

Never install, upgrade, downgrade, replace, or shadow `torch`, including through
dependency resolution. Record and re-check its version and import path around
compatible package installation. Reject GPU-96G if compatibility is uncertain.

## PyTorch operator checklist

Treat as unsupported:

- complex64/complex128 arithmetic, `abs`, `real`, `imag`, `view_as_complex`,
  and `view_as_real`;
- sparse creation/conversion, sparse-dense `mm`, and sparse reductions or
  arithmetic;
- float8 e4m3fn/e5m2 conversion and computation;
- accelerator quantization/dequantization and `torch.quantization` paths;
- `fftn` at 3+ dimensions, `rfft2`, complex-result `fftshift`, `ifftn`,
  `irfft2`, `rfftn`, and `irfftn` (1-D FFT and `fft2`/`ifft2` remain allowed);
- device `torch.nn.functional.ctc_loss` (CPU fallback is allowed);
- `scatter_reduce(..., reduce="add")` (`sum`, `prod`, `mean`, `amax`, and
  `amin` are supported when semantically equivalent).

Treat `torch.linalg.tensorinv` and `torch.linalg.tensorsolve` as restricted by
input shape.

Inspect reachable task/model/dependency paths, dtypes, layouts, shapes,
dimensions, and arguments; do not rely only on text search. Smoke-test the
intended path. A known unsupported path blocks direct execution.

If GPU-96G is the only compatible cluster, the operation is optional, and an
equivalent supported operation or correct CPU fallback exists, first tell the
user the incompatibility, replacement, and correctness/precision/performance/
memory impact. Then implement, re-estimate, and smoke-test. If the operation is
essential or task meaning may change, stop and request a user decision.
