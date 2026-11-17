# Q4_K prefill GEMM / llama.cpp MMQ investigation (gfx1151)

Date: 2026-11-17. Baseline: upstream `main` at `4af0ef0` (merged PR #1).
This is a negative-result research note, not a production performance change.

## Relevant llama.cpp implementation

The locally inspected llama.cpp source snapshot contains:

- `ggml/src/ggml-cuda/mmq.cu`: `ggml_cuda_mul_mat_q`,
  `ggml_cuda_should_use_mmq`, activation quantization, batch- and
  architecture-dependent MMQ selection.
- `ggml/src/ggml-cuda/mmq.cuh`: `mul_mat_q`, specialized warp/CTA tile
  loaders and `stream-k` partitioning. These are not ordinary CUDA GEMMs
  called on unpacked weights.
- `ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`: specialized RDNA 3.5
  configuration for Q4_K/Q5_K/Q6_K and various output-column tile sizes.
- `ggml/src/ggml-cuda/vecdotq.cuh` and `common.cuh`: packed
  integer dot operations, Q4_K/Q8_1 scale/min corrections; on gfx1151,
  HIP `ggml_cuda_dp4a` uses `__builtin_amdgcn_sudot4`.

Key algorithmic difference: llama.cpp quantizes FP32 input activations to
Q8_1-like 32-value blocks and multiplies packed GGUF quantized weights
directly, while the existing Yuuka WMMA prefill path decodes a weight tile
into FP16 in LDS and issues FP16 rocWMMA matrix instructions. At >=128
rows, selected wide Yuuka projections decode into an FP16 scratch and use
rocBLAS instead. Both Yuuka routes are already hand-tuned implementations;
the first is a custom GEMM rather than a plain BLAS wrapper.

## Implemented experiments

`bench/hip/mmq_probe.hip` provides three independently measured approaches:

1. **Scalar SDOT4**: Q4_K + Q8 input, one GPU thread per weight output row,
   four prompt tokens per CTA group. Uses packed 32-bit loads, unsigned
   nibble masks and gfx1151 `sudot4`. Q4_K group scales and mins are
   applied after each 32-value Q8 dot, using its integer sum.
2. **Tiled INT8 rocWMMA**: shared-memory 32x64 output tiles with 32-wide K
   steps, 32-row Q8 activation blocks, packed Q4_K -> int8 LDS staging,
   int32 WMMA accumulation and FP32 subgroup scaling/min correction.
3. **Larger FP16 K tiles**: the existing dequant-on-the-fly FP16 rocWMMA,
   with 32/64-row output tiles and K=64 or K=128 (versus default K=32).
   The 128-wide loader keeps each 32-value Q4_K scale/min subgroup intact.

The code implements these algorithms independently; **no llama.cpp code
was copied**. The extra native benchmark is explicitly opt-in with
`-DYUUKA_BUILD_MMQ_PROBE=ON`. The default model execution plan remains
unchanged; the experiments are not automatically selected.

## Conditions and microbenchmarks

- Hardware: AMD Radeon 8060S, gfx1151, 40 CUs, Ryzen AI MAX+ 395
- Runtime: HIP/ROCm from `/opt/rocm` (test system: ROCm 10.1 layout)
- Same synthetic GGUF Q4_K packed weights and FP32 activations for every
  path; fixed PRNG seed; activation uniform on [-1, 1]
- Weight Q4_K `d=0.015625`, `dmin=0.00390625`, random scale/min
  header bytes and packed nibbles. This reproduces storage and decode
  cost but **does not reflect real trained weight statistics**.
- Shape conventions: `M` prompt rows, `N` output features, `K` input
  features. The corresponding projection shapes occur in the targeted
  Qwen3.5-4B architecture.
- GPU event timings: 2 warmups then 3 measured repetitions, inclusive
  of activation Q8 quantization for the INT8 approaches. No CPU
  transfers occur within timed kernels; GEMM data already resides on GPU.
  Results may vary with temperature and clock behavior.
- Output comparison: Yuuka original FP16 rocWMMA. Q8 approaches have
  ~0.0038 relative RMS output difference (RMS of errors normalized by
  reference RMS), arising mainly from per-32-value activation
  quantization. K tile changes are numerically identical on this fixture.

| Shape (M,N,K) | Existing FP16 WMMA | Scalar SDOT4 | Tiled INT8 WMMA | FP16 64x64 K64 | FP16 32x64 K64 | FP16 32x64 K128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (128,9216,2560) | **0.680 ms** | 17.562 ms | 3.411 ms | 0.914 ms | 0.999 ms | 1.737 ms |
| (512,9216,2560) | **3.054 ms** | 85.233 ms | 14.689 ms | 4.857 ms | 4.215 ms | 9.630 ms |
| (512,2560,9216) | **4.522 ms** | 60.636 ms | 15.550 ms | 6.861 ms | 6.225 ms | 13.620 ms |

The values above are from the corrected Q8 quantization code. An earlier
implementation incorrectly performed warp shuffle only in lanes 0..7,
which read inactive source lanes; that invalid initial result is excluded.
All reported results are post-fix, and the probe fails when Q8-path
relative RMS error exceeds 1%.

**Interpretation:** The naive scalar SDOT4 route is ~13–28x slower; the
simple INT8 WMMA path is ~3.4–5x slower; naive increases to the existing
FP16 WMMA K tile are also slower. These measurements **do not establish
that llama.cpp's fully tuned MMQ algorithm cannot outperform Yuuka's
current paths**. The experimental tiled INT8 kernel has a separate
synchronization/scaling pass for every 32 K values, repeatedly loads
the same quantization metadata, and does not implement llama.cpp's
device-specific memory layouts, fused vector-dot algorithm or
Stream-K scheduling. Thus it is not a faithful optimized port.

## Full-model default prefill baseline

On this same host and `Qwen3.5-4B-Q4_K_M.gguf` with F16 KV,
512-token chunks, single request, 1 warmup / 2 measured repetitions:

| Prompt tokens | Default Yuuka prefill p50 | Throughput |
| ---: | ---: | ---: |
| 512 | 0.504 s | 1,016 tok/s |
| 2,048 | 2.829 s | 724 tok/s |
| 8,192 | 15.992 s | 512 tok/s |

This uses the merged upstream baseline and **does not turn on
`YUUKA_ATTENTION_BLAS`**. Since every new MMQ variant was slower in
isolated GEMM comparisons, none was inserted into the model execution
graph; there is consequently **no end-to-end MMQ speedup claim**.

## Reproduction

```bash
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DYUUKA_BUILD_MMQ_PROBE=ON
cmake --build build --target yuuka_mmq_probe -j8
./build/yuuka_mmq_probe
./build/yuuka_mmq_probe 2560 9216 512
# args: K N M
```

Original prefill baseline:

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:"$PWD"/build
./bin/yuuka-hip bench --model /models/Qwen3.5-4B-Q4_K_M.gguf \
  --backend hip --prompt-tokens 512 --decode-tokens 1 \
  --warmup 1 --reps 2 --chunk-size 512 --kv-type f16
```

## Recommended next investigation

To potentially match llama.cpp, move from these baseline kernels to a
faithfully **tiled and coalesced Q4_K x Q8 MMQ** design: keep per-32 scale
processing local to an INT8 tile, amortize Q8 activation conversion,
reuse packed Q4_K blocks across prompt rows, avoid shared FP32 output
materialization per 32 K, and tune `I/J` occupancy by RDNA3.5's
register/LDS and wave32 constraints. Measure in isolation **first**
and expose per-shape dispatch only where an actual advantage and
real-model logit tolerance are demonstrated. Also profile the
mixed-precision gating/down projections and their weight reuse,
since full-model performance is influenced by more than GEMM.
