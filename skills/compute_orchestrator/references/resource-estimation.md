# Resource estimation

Estimate from the executable code, configuration, input scale, and a bounded
measurement when possible. Choose the smallest supported tier with headroom.

## CPU and RAM

Include processes, data-loader workers, compilation, preprocessing, BLAS/OpenMP
threads, decompression, model and optimizer state, dataset indexes, prefetch,
process duplication, temporary buffers, and framework overhead. Add at least
20% RAM headroom when uncertainty is material.

## VRAM

For training include parameters, gradients, optimizer/master states,
activations, workspaces, allocator/runtime overhead, communication buffers, and
per-process duplication. For inference include weights, cache, activations,
batch and sequence/image dimensions, workspaces, and framework overhead.

Do not divide total memory by GPU count unless the code actually shards that
state. Data parallelism duplicates model state. Verify multi-GPU support before
requesting more than one GPU.

GPU-32G uses 32 GiB per GPU and architecture `sm70`; verify compiled kernels and
dependencies support it. For GPU-96G keep planned peak comfortably below 96 GiB
per card and run a bounded smoke test before full execution.
