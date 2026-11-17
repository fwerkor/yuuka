#include "../common.cuh"
#include "../quant.cuh"
#include "../yuuka_kernels.h"

#include <rocwmma/rocwmma.hpp>

namespace {

constexpr int kBlockN = 64;
// The plain GEMM stages a 64-row tile with a 32-wide K step: 16 KiB of shared
// memory, so four CTAs stay resident per 64 KiB LDS CU. The fused gate/up
// kernel stages three tiles and also uses a 32-wide K step.
constexpr int kFusedBlockK = 32;
constexpr int kThreads = 256;

// write_q4k_word decodes four Q4_K values from one packed 32-bit word. Even
// sub-blocks occupy the low nibble and odd sub-blocks the high nibble of the
// same byte, so four consecutive bytes hold eight values.
__device__ __forceinline__ void write_q4k_word(uint32_t word, bool high_nibble,
                                               float scale, float min,
                                               __half * out) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int shift = 8 * i + (high_nibble ? 4 : 0);
        const int q = static_cast<int>((word >> shift) & 0xF);
        out[i] = __float2half(scale * static_cast<float>(q) - min);
    }
}

// write_q5k_word decodes four Q5_K values from one packed ql word and the four
// matching qh bytes. Bit sub of each qh byte carries the fifth value bit.
__device__ __forceinline__ void write_q5k_word(uint32_t ql, uint32_t qh,
                                               bool high_nibble, int sub,
                                               float scale, float min,
                                               __half * out) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int shift = 8 * i + (high_nibble ? 4 : 0);
        int q = static_cast<int>((ql >> shift) & 0xF);
        if ((qh >> (8 * i)) & static_cast<uint32_t>(1u << sub)) {
            q += 16;
        }
        out[i] = __float2half(scale * static_cast<float>(q) - min);
    }
}

// decode_span decodes Span consecutive weight values at absolute k offset. The
// helper writes exactly out[0..valid-1] and the caller zeroes the rest of the
// span. Span is 8 or 16 and stays inside one 32-value quantization segment, so
// each block header, scale, and min is hoisted once per span. Q4_K and Q5_K use
// vector global loads: their row byte sizes are multiples of 16 and the span
// offsets are 8 or 16-byte aligned by construction. Q4_0, Q5_0, and Q8_0 decode
// with scalar loads.
template <uint32_t DType, int Span>
__device__ __forceinline__ void decode_span(const uint8_t * row, int64_t k,
                                            int valid, __half * out) {
    if (DType == YUUKA_DTYPE_Q4_0) {
        const uint8_t * blk = row + (k / 32) * 18;
        const float d = load_f16_device(blk);
        const int lane0 = static_cast<int>(k & 31);
        for (int j = 0; j < valid; ++j) {
            const int lane = lane0 + j;
            const uint8_t packed = blk[2 + (lane & 15)];
            const int q = (lane < 16) ? (packed & 0xF) : (packed >> 4);
            out[j] = __float2half(d * static_cast<float>(q - 8));
        }
    } else if (DType == YUUKA_DTYPE_Q5_0) {
        const uint8_t * blk = row + (k / 32) * 22;
        const float d = load_f16_device(blk);
        const uint32_t qh = static_cast<uint32_t>(blk[2]) |
                            (static_cast<uint32_t>(blk[3]) << 8) |
                            (static_cast<uint32_t>(blk[4]) << 16) |
                            (static_cast<uint32_t>(blk[5]) << 24);
        const int lane0 = static_cast<int>(k & 31);
        for (int j = 0; j < valid; ++j) {
            const int lane = lane0 + j;
            const uint8_t packed = blk[6 + (lane & 15)];
            const int high =
                (lane < 16) ? static_cast<int>(((qh >> lane) << 4) & 0x10)
                            : static_cast<int>((qh >> ((lane & 15) + 12)) & 0x10);
            const int q = (lane < 16 ? (packed & 0xF) : (packed >> 4)) | high;
            out[j] = __float2half(d * static_cast<float>(q - 16));
        }
    } else if (DType == YUUKA_DTYPE_Q8_0) {
        const uint8_t * blk = row + (k / 32) * 34;
        const float d = load_f16_device(blk);
        const int lane0 = static_cast<int>(k & 31);
        for (int j = 0; j < valid; ++j) {
            const int8_t q = static_cast<int8_t>(blk[2 + lane0 + j]);
            out[j] = __float2half(d * static_cast<float>(q));
        }
    } else if (DType == YUUKA_DTYPE_Q4_K) {
        const int sub = static_cast<int>((k % 256) / 32);
        const int within = static_cast<int>(k & 31);
        const uint8_t * blk = row + (k / 256) * 144;
        const float d = load_f16_device(blk);
        const float dm = load_f16_device(blk + 2);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const float scale = d * static_cast<float>(sc);
        const float min = dm * static_cast<float>(m);
        const uint8_t * qs = blk + 16 + (sub / 2) * 32 + within;
        const bool high_nibble = (sub & 1) != 0;
        if (valid == Span) {
            if constexpr (Span == 16) {
                const uint4 packed = *reinterpret_cast<const uint4 *>(qs);
                write_q4k_word(packed.x, high_nibble, scale, min, out);
                write_q4k_word(packed.y, high_nibble, scale, min, out + 4);
                write_q4k_word(packed.z, high_nibble, scale, min, out + 8);
                write_q4k_word(packed.w, high_nibble, scale, min, out + 12);
            } else {
                const uint2 packed = *reinterpret_cast<const uint2 *>(qs);
                write_q4k_word(packed.x, high_nibble, scale, min, out);
                write_q4k_word(packed.y, high_nibble, scale, min, out + 4);
            }
            return;
        }
        for (int j = 0; j < valid; ++j) {
            const int q = high_nibble ? static_cast<int>(qs[j] >> 4)
                                      : static_cast<int>(qs[j] & 0xF);
            out[j] = __float2half(scale * static_cast<float>(q) - min);
        }
    } else if (DType == YUUKA_DTYPE_Q5_K) {
        const int sub = static_cast<int>((k % 256) / 32);
        const int within = static_cast<int>(k & 31);
        const uint8_t * blk = row + (k / 256) * 176;
        const float d = load_f16_device(blk);
        const float dm = load_f16_device(blk + 2);
        uint8_t sc = 0;
        uint8_t m = 0;
        get_scale_min_k4(sub, blk + 4, sc, m);
        const float scale = d * static_cast<float>(sc);
        const float min = dm * static_cast<float>(m);
        const uint8_t * ql = blk + 48 + (sub / 2) * 32 + within;
        const uint8_t * qh = blk + 16 + within;
        const bool high_nibble = (sub & 1) != 0;
        if (valid == Span) {
            if constexpr (Span == 16) {
                const uint4 packed = *reinterpret_cast<const uint4 *>(ql);
                const uint4 high = *reinterpret_cast<const uint4 *>(qh);
                write_q5k_word(packed.x, high.x, high_nibble, sub, scale, min, out);
                write_q5k_word(packed.y, high.y, high_nibble, sub, scale, min, out + 4);
                write_q5k_word(packed.z, high.z, high_nibble, sub, scale, min, out + 8);
                write_q5k_word(packed.w, high.w, high_nibble, sub, scale, min, out + 12);
            } else {
                const uint2 packed = *reinterpret_cast<const uint2 *>(ql);
                const uint2 high = *reinterpret_cast<const uint2 *>(qh);
                write_q5k_word(packed.x, high.x, high_nibble, sub, scale, min, out);
                write_q5k_word(packed.y, high.y, high_nibble, sub, scale, min, out + 4);
            }
            return;
        }
        for (int j = 0; j < valid; ++j) {
            int q = high_nibble ? static_cast<int>(ql[j] >> 4)
                                : static_cast<int>(ql[j] & 0xF);
            if (qh[j] & static_cast<uint8_t>(1u << sub)) {
                q += 16;
            }
            out[j] = __float2half(scale * static_cast<float>(q) - min);
        }
    }
}

// q6k_span4 decodes four consecutive Q6_K values starting at a multiple of 4
// inside one 256-value block. The group shares one ql word, one qh word, and one
// scale byte, so the four values cost two words instead of eight scalar byte
// loads plus four block-header and scale loads. Q6_K block starts are only
// 2-byte aligned (a block is 210 bytes), so the words are assembled from 16-bit
// halves.
__device__ __forceinline__ void q6k_span4(const uint8_t * blk, int j,
                                          __half * out) {
    const int8_t * sc = reinterpret_cast<const int8_t *>(blk + 192);
    const int n = j / 128;
    const int l = j & 127;
    const uint8_t * ql = blk + n * 64;
    const uint8_t * qh = blk + 128 + n * 32;
    const uint8_t * ql_src = ql;
    const uint8_t * qh_src = qh;
    int ql_shift = 0;
    int qh_shift = 0;
    int sidx = 0;
    if (l < 32) {
        ql_src = ql + l;
        qh_src = qh + l;
        sidx = l / 16;
    } else if (l < 64) {
        ql_src = ql + l;
        qh_src = qh + (l - 32);
        qh_shift = 2;
        sidx = (l - 32) / 16 + 2;
    } else if (l < 96) {
        ql_src = ql + (l - 64);
        ql_shift = 4;
        qh_src = qh + (l - 64);
        qh_shift = 4;
        sidx = (l - 64) / 16 + 4;
    } else {
        ql_src = ql + (l - 64);
        ql_shift = 4;
        qh_src = qh + (l - 96);
        qh_shift = 6;
        sidx = (l - 96) / 16 + 6;
    }
    const uint32_t ql_word = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(ql_src)) |
                             (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(ql_src + 2)) << 16);
    const uint32_t qh_word = static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(qh_src)) |
                             (static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(qh_src + 2)) << 16);
    const float d = load_f16_device(blk + 208) *
                    static_cast<float>(sc[n * 8 + sidx]);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int q = static_cast<int>((ql_word >> (8 * i + ql_shift)) & 0xF) |
                      (static_cast<int>((qh_word >> (8 * i + qh_shift)) & 3) << 4);
        out[i] = __float2half(d * static_cast<float>(q - 32));
    }
}

template <int BlockM, int BlockK>
__device__ __forceinline__ void load_a_tile(__half * tile, const float * x,
                                            int64_t m0, int64_t k0, int64_t rows,
                                            int64_t n_in) {
    for (int idx = threadIdx.x; idx < BlockM * BlockK; idx += kThreads) {
        const int r = idx / BlockK;
        const int c = idx % BlockK;
        const int64_t gr = m0 + r;
        const int64_t gk = k0 + c;
        const float value = (gr < rows && gk < n_in) ? x[gr * n_in + gk] : 0.0f;
        tile[r * BlockK + c] = __float2half(value);
    }
}

// load_b_tile stages a kBlockN x BlockK weight tile. Block-quantized layouts
// decode one contiguous span per thread: the block header is amortized, Q4_K and
// Q5_K use vector global loads, and no decoded span is staged in registers or
// scratch. Q6_K decodes four values per thread from one ql word and one qh word;
// the dense layouts keep the element mapping where consecutive lanes read
// consecutive row bytes, because they have no header worth hoisting above the
// per-value cost.
template <uint32_t DType, int BlockK>
__device__ __forceinline__ void load_b_tile(__half * tile, const uint8_t * w,
                                            int64_t n0, int64_t k0, int64_t n_in,
                                            int64_t n_out, uint64_t row_bytes) {
    constexpr int kThreadsPerRow = kThreads / kBlockN;
    constexpr int kSpan = BlockK / kThreadsPerRow;
    constexpr int kRowsPerPass = kThreads / kThreadsPerRow;
    static_assert(BlockK % kThreadsPerRow == 0, "span must divide the K step");
    static_assert(kSpan == 8 || kSpan == 16 ||
                  (kSpan == 32 && (DType == YUUKA_DTYPE_Q4_K ||
                                   DType == YUUKA_DTYPE_Q5_K)),
                  "unsupported quant weight span");
    if (DType == YUUKA_DTYPE_F32 || DType == YUUKA_DTYPE_F16) {
        for (int idx = threadIdx.x; idx < kBlockN * BlockK; idx += kThreads) {
            const int s = idx / BlockK;
            const int c = idx % BlockK;
            const int64_t gn = n0 + s;
            const int64_t gk = k0 + c;
            float value = 0.0f;
            if (gn < n_out && gk < n_in) {
                const uint8_t * row = w + gn * row_bytes;
                if (DType == YUUKA_DTYPE_F32) {
                    value = reinterpret_cast<const float *>(row)[gk];
                } else {
                    value = load_f16_device(row + gk * 2);
                }
            }
            tile[s * BlockK + c] = __float2half(value);
        }
        return;
    }
    if constexpr (DType == YUUKA_DTYPE_Q6_K) {
        static_assert(BlockK == 32, "the Q6_K span covers 32 K values per row");
        // Eight lanes cover one row and four values per lane, so a pass stages
        // 32 rows.
        const int row_in_group = threadIdx.x / 8;
        const int col0 = (threadIdx.x % 8) * 4;
        for (int r = row_in_group; r < kBlockN; r += kThreads / 8) {
            __half * dst = tile + r * BlockK + col0;
            const int64_t gn = n0 + r;
            const int64_t gk = k0 + col0;
            const bool in_range = gn < n_out && gk < n_in;
            const int64_t remaining = in_range ? n_in - gk : 0;
            const int valid =
                in_range ? static_cast<int>(remaining < 4 ? remaining : 4) : 0;
            const uint8_t * row = w + gn * row_bytes;
            if (valid == 4) {
                q6k_span4(row + (gk / 256) * 210, static_cast<int>(gk % 256), dst);
                continue;
            }
            for (int j = 0; j < valid; ++j) {
                const int64_t col = gk + j;
                dst[j] = __float2half(
                    q6k_value(row + (col / 256) * 210, static_cast<int>(col % 256)));
            }
            for (int j = valid; j < 4; ++j) {
                dst[j] = __float2half(0.0f);
            }
        }
        return;
    }
    const int row_in_group = threadIdx.x / kThreadsPerRow;
    const int col0 = (threadIdx.x % kThreadsPerRow) * kSpan;
    for (int r = row_in_group; r < kBlockN; r += kRowsPerPass) {
        __half * dst = tile + r * BlockK + col0;
        const int64_t gn = n0 + r;
        const int64_t gk = k0 + col0;
        const bool in_range = gn < n_out && gk < n_in;
        const int64_t remaining = in_range ? n_in - gk : 0;
        const int valid =
            in_range ? static_cast<int>(remaining < kSpan ? remaining : kSpan) : 0;
        if constexpr (kSpan == 32) {
            // A Q4_K block has 32-value scale/min subgroups. Reuse the
            // existing vectorized 16-value decoder, staying within one
            // subgroup while benchmarking a wider K tile.
            if (valid > 0) {
                decode_span<DType, 16>(w + gn * row_bytes, gk,
                                       valid < 16 ? valid : 16, dst);
            }
            if (valid > 16) {
                decode_span<DType, 16>(w + gn * row_bytes, gk + 16,
                                       valid - 16, dst + 16);
            }
        } else {
            if (valid > 0) {
                decode_span<DType, kSpan>(w + gn * row_bytes, gk, valid, dst);
            }
        }
        for (int j = valid; j < kSpan; ++j) {
            dst[j] = __float2half(0.0f);
        }
    }
}

// gemm_wmma_kernel computes a BlockM x 64 output tile with eight waves. Each
// wave owns one A fragment and NFragments B fragments; the 32-row variant keeps
// one fragment per wave, and the 64-row variant reuses one decoded B tile
// across two accumulator fragments. A and B staging is double buffered over a
// BlockK-wide K step: the next slice loads while the current slice feeds WMMA.
// The C staging tile aliases the A/B staging memory after the K loop.
template <uint32_t DType, int BlockM, int NFragments, int BlockK>
__global__ void gemm_wmma_kernel(float * out, const float * x, const uint8_t * w,
                                 int64_t n_in, int64_t n_out, int64_t rows,
                                 uint64_t row_bytes) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    constexpr int kWaveCols = kBlockN / 16 / NFragments;
    const int wm = warp / kWaveCols;
    const int wn = warp % kWaveCols;
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * BlockM;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kBlockN;

    __shared__ __align__(16) unsigned char staging[
        2 * (BlockM + kBlockN) * BlockK * sizeof(__half)];
    __half * a_tile = reinterpret_cast<__half *>(staging);
    __half * b_tile = a_tile + 2 * BlockM * BlockK;
    float * c_tile = reinterpret_cast<float *>(staging);

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag0;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag1;
    rocwmma::fill_fragment(c_frag0, 0.0f);
    if constexpr (NFragments == 2) {
        rocwmma::fill_fragment(c_frag1, 0.0f);
    }

    load_a_tile<BlockM, BlockK>(a_tile, x, m0, 0, rows, n_in);
    load_b_tile<DType, BlockK>(b_tile, w, n0, 0, n_in, n_out, row_bytes);
    __syncthreads();

    int current = 0;
    for (int64_t k0 = 0; k0 < n_in; k0 += BlockK) {
        const int next = current ^ 1;
        if (k0 + BlockK < n_in) {
            load_a_tile<BlockM, BlockK>(a_tile + next * BlockM * BlockK, x, m0,
                                        k0 + BlockK, rows, n_in);
            load_b_tile<DType, BlockK>(b_tile + next * kBlockN * BlockK, w, n0,
                                       k0 + BlockK, n_in, n_out, row_bytes);
        }
        const __half * a_cur = a_tile + current * BlockM * BlockK;
        const __half * b_cur = b_tile + current * kBlockN * BlockK;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> b_frag0;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> b_frag1;
        for (int kk = 0; kk < BlockK; kk += 16) {
            rocwmma::load_matrix_sync(a_frag, a_cur + (wm * 16) * BlockK + kk,
                                      BlockK);
            rocwmma::load_matrix_sync(
                b_frag0, b_cur + (wn * NFragments * 16) * BlockK + kk, BlockK);
            rocwmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
            if constexpr (NFragments == 2) {
                rocwmma::load_matrix_sync(
                    b_frag1, b_cur + (wn * NFragments * 16 + 16) * BlockK + kk,
                    BlockK);
                rocwmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
            }
        }
        __syncthreads();
        current = next;
    }

    rocwmma::store_matrix_sync(c_tile + (wm * 16) * kBlockN + (wn * NFragments * 16),
                               c_frag0, kBlockN, rocwmma::mem_row_major);
    if constexpr (NFragments == 2) {
        rocwmma::store_matrix_sync(
            c_tile + (wm * 16) * kBlockN + (wn * NFragments * 16 + 16), c_frag1,
            kBlockN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < BlockM * kBlockN; idx += kThreads) {
        const int r = idx / kBlockN;
        const int c = idx % kBlockN;
        const int64_t gr = m0 + r;
        const int64_t gn = n0 + c;
        if (gr < rows && gn < n_out) {
            out[gr * n_out + gn] = c_tile[r * kBlockN + c];
        }
    }
}

// gemm_wmma_k32_kernel is the 32x64 tile with a 32-wide K step and one
// accumulator fragment per wave. It stays the measured choice for small row
// counts, where the 64-row tile wastes rows on partial M tiles.
template <uint32_t DType>
__global__ void gemm_wmma_k32_kernel(float * out, const float * x,
                                     const uint8_t * w, int64_t n_in,
                                     int64_t n_out, int64_t rows,
                                     uint64_t row_bytes) {
    constexpr int kM32 = 32;
    constexpr int kK32 = 32;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp / (kBlockN / 16);
    const int wn = warp % (kBlockN / 16);
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * kM32;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kBlockN;

    __shared__ __align__(16) __half a_tile[kM32 * kK32];
    __shared__ __align__(16) __half b_tile[kBlockN * kK32];
    __shared__ float c_tile[kM32 * kBlockN];

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag;
    rocwmma::fill_fragment(c_frag, 0.0f);

    for (int64_t k0 = 0; k0 < n_in; k0 += kK32) {
        load_a_tile<kM32, kK32>(a_tile, x, m0, k0, rows, n_in);
        load_b_tile<DType, kK32>(b_tile, w, n0, k0, n_in, n_out, row_bytes);
        __syncthreads();
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> b_frag;
        for (int kk = 0; kk < kK32; kk += 16) {
            if (k0 + kk >= n_in) {
                break;
            }
            rocwmma::load_matrix_sync(a_frag, a_tile + (wm * 16) * kK32 + kk, kK32);
            rocwmma::load_matrix_sync(b_frag, b_tile + (wn * 16) * kK32 + kk, kK32);
            rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    rocwmma::store_matrix_sync(c_tile + (wm * 16) * kBlockN + wn * 16, c_frag,
                               kBlockN, rocwmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < kM32 * kBlockN; idx += kThreads) {
        const int r = idx / kBlockN;
        const int c = idx % kBlockN;
        const int64_t gr = m0 + r;
        const int64_t gn = n0 + c;
        if (gr < rows && gn < n_out) {
            out[gr * n_out + gn] = c_tile[r * kBlockN + c];
        }
    }
}

// gemm_wmma_swiglu_kernel computes two GEMMs that share the same activation
// tile and writes silu(gate) * up in one epilogue. Both weight matrices have
// the same shape and dtype. The C staging tile aliases the A/B staging memory
// after the K loop, and no gate or up intermediate is written to global memory.
template <uint32_t DType, int BlockM, int NFragments>
__global__ void gemm_wmma_swiglu_kernel(float * out, const float * x,
                                        const uint8_t * w_gate,
                                        const uint8_t * w_up, int64_t n_in,
                                        int64_t n_out, int64_t rows,
                                        uint64_t row_bytes) {
    constexpr int kK = kFusedBlockK;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    constexpr int kWaveCols = kBlockN / 16 / NFragments;
    const int wm = warp / kWaveCols;
    const int wn = warp % kWaveCols;
    const int64_t m0 = static_cast<int64_t>(blockIdx.y) * BlockM;
    const int64_t n0 = static_cast<int64_t>(blockIdx.x) * kBlockN;

    __shared__ __align__(16) unsigned char staging[
        2 * (BlockM + 2 * kBlockN) * kK * sizeof(__half)];
    __half * a_tile = reinterpret_cast<__half *>(staging);
    __half * gate_tile = a_tile + 2 * BlockM * kK;
    __half * up_tile = gate_tile + 2 * kBlockN * kK;
    float * c_tile = reinterpret_cast<float *>(staging);

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> gate0;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> gate1;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> up0;
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> up1;
    rocwmma::fill_fragment(gate0, 0.0f);
    rocwmma::fill_fragment(up0, 0.0f);
    if constexpr (NFragments == 2) {
        rocwmma::fill_fragment(gate1, 0.0f);
        rocwmma::fill_fragment(up1, 0.0f);
    }

    load_a_tile<BlockM, kK>(a_tile, x, m0, 0, rows, n_in);
    load_b_tile<DType, kK>(gate_tile, w_gate, n0, 0, n_in, n_out, row_bytes);
    load_b_tile<DType, kK>(up_tile, w_up, n0, 0, n_in, n_out, row_bytes);
    __syncthreads();

    int current = 0;
    for (int64_t k0 = 0; k0 < n_in; k0 += kK) {
        const int next = current ^ 1;
        if (k0 + kK < n_in) {
            load_a_tile<BlockM, kK>(a_tile + next * BlockM * kK, x, m0,
                                    k0 + kK, rows, n_in);
            load_b_tile<DType, kK>(gate_tile + next * kBlockN * kK, w_gate, n0,
                                   k0 + kK, n_in, n_out, row_bytes);
            load_b_tile<DType, kK>(up_tile + next * kBlockN * kK, w_up, n0,
                                   k0 + kK, n_in, n_out, row_bytes);
        }
        const __half * a_cur = a_tile + current * BlockM * kK;
        const __half * g_cur = gate_tile + current * kBlockN * kK;
        const __half * u_cur = up_tile + current * kBlockN * kK;
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> a_frag;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> g_frag0;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> u_frag0;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> g_frag1;
        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> u_frag1;
        for (int kk = 0; kk < kK; kk += 16) {
            rocwmma::load_matrix_sync(a_frag, a_cur + (wm * 16) * kK + kk, kK);
            rocwmma::load_matrix_sync(
                g_frag0, g_cur + (wn * NFragments * 16) * kK + kk, kK);
            rocwmma::load_matrix_sync(
                u_frag0, u_cur + (wn * NFragments * 16) * kK + kk, kK);
            rocwmma::mma_sync(gate0, a_frag, g_frag0, gate0);
            rocwmma::mma_sync(up0, a_frag, u_frag0, up0);
            if constexpr (NFragments == 2) {
                rocwmma::load_matrix_sync(
                    g_frag1, g_cur + (wn * NFragments * 16 + 16) * kK + kk, kK);
                rocwmma::load_matrix_sync(
                    u_frag1, u_cur + (wn * NFragments * 16 + 16) * kK + kk, kK);
                rocwmma::mma_sync(gate1, a_frag, g_frag1, gate1);
                rocwmma::mma_sync(up1, a_frag, u_frag1, up1);
            }
        }
        __syncthreads();
        current = next;
    }

    for (uint32_t i = 0; i < gate0.num_elements; ++i) {
        const float g = gate0.x[i];
        gate0.x[i] = (g / (1.0f + expf(-g))) * up0.x[i];
    }
    if constexpr (NFragments == 2) {
        for (uint32_t i = 0; i < gate1.num_elements; ++i) {
            const float g = gate1.x[i];
            gate1.x[i] = (g / (1.0f + expf(-g))) * up1.x[i];
        }
    }

    rocwmma::store_matrix_sync(c_tile + (wm * 16) * kBlockN + (wn * NFragments * 16),
                               gate0, kBlockN, rocwmma::mem_row_major);
    if constexpr (NFragments == 2) {
        rocwmma::store_matrix_sync(
            c_tile + (wm * 16) * kBlockN + (wn * NFragments * 16 + 16), gate1,
            kBlockN, rocwmma::mem_row_major);
    }
    __syncthreads();
    for (int idx = tid; idx < BlockM * kBlockN; idx += kThreads) {
        const int r = idx / kBlockN;
        const int c = idx % kBlockN;
        const int64_t gr = m0 + r;
        const int64_t gn = n0 + c;
        if (gr < rows && gn < n_out) {
            out[gr * n_out + gn] = c_tile[r * kBlockN + c];
        }
    }
}

uint64_t packed_row_bytes(uint32_t dtype, uint64_t n_in) {
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            return n_in * 4;
        case YUUKA_DTYPE_F16:
            return n_in * 2;
        case YUUKA_DTYPE_Q4_0:
            return (n_in / 32) * 18;
        case YUUKA_DTYPE_Q5_0:
            return (n_in / 32) * 22;
        case YUUKA_DTYPE_Q8_0:
            return (n_in / 32) * 34;
        case YUUKA_DTYPE_Q4_K:
            return (n_in / 256) * 144;
        case YUUKA_DTYPE_Q5_K:
            return (n_in / 256) * 176;
        case YUUKA_DTYPE_Q6_K:
            return (n_in / 256) * 210;
        default:
            return 0;
    }
}

template <uint32_t DType, int BlockM, int NFragments, int BlockK>
void launch_wmma_tile(float * out, const float * x, const uint8_t * w,
                      int64_t n_in, int64_t n_out, int64_t rows,
                      uint64_t row_bytes, hipStream_t stream) {
    const dim3 grid(static_cast<unsigned>((n_out + kBlockN - 1) / kBlockN),
                    static_cast<unsigned>((rows + BlockM - 1) / BlockM));
    gemm_wmma_kernel<DType, BlockM, NFragments, BlockK>
        <<<grid, kThreads, 0, stream>>>(out, x, w, n_in, n_out, rows, row_bytes);
}

// launch_wmma_selected chooses the measured tile. The 64x64 tile with a 32-wide
// K step reuses a decoded B tile across two accumulator fragments, stages
// 16 KiB so four CTAs stay resident, and is used for wide projections at 64 or
// more rows. Tall-K projections keep the 32x64 tile: halving the output columns
// per CTA doubles the CTA count that hides the long K loop, and F16 and Q6_K
// tall-K shapes measured 40-90% slower with the 64-row tile. The 32x64 tile
// also stays for small batches, where the wasted rows of the 64-row tile
// dominate.
template <uint32_t DType>
void launch_wmma_selected(float * out, const float * x, const uint8_t * w,
                          int64_t n_in, int64_t n_out, int64_t rows,
                          uint64_t row_bytes, hipStream_t stream) {
    if (rows >= 64 && n_in <= n_out) {
        launch_wmma_tile<DType, 64, 2, 32>(out, x, w, n_in, n_out, rows,
                                           row_bytes, stream);
        return;
    }
    const dim3 grid(static_cast<unsigned>((n_out + kBlockN - 1) / kBlockN),
                    static_cast<unsigned>((rows + 31) / 32));
    gemm_wmma_k32_kernel<DType><<<grid, kThreads, 0, stream>>>(
        out, x, w, n_in, n_out, rows, row_bytes);
}

template <uint32_t DType, int BlockM, int NFragments>
void launch_wmma_swiglu_tile(float * out, const float * x, const uint8_t * w_gate,
                             const uint8_t * w_up, int64_t n_in, int64_t n_out,
                             int64_t rows, uint64_t row_bytes,
                             hipStream_t stream) {
    const dim3 grid(static_cast<unsigned>((n_out + kBlockN - 1) / kBlockN),
                    static_cast<unsigned>((rows + BlockM - 1) / BlockM));
    gemm_wmma_swiglu_kernel<DType, BlockM, NFragments>
        <<<grid, kThreads, 0, stream>>>(out, x, w_gate, w_up, n_in, n_out, rows,
                                        row_bytes);
}

// launch_wmma_swiglu_selected uses the fused 64-row tile for wide projections at
// 64 or more rows, otherwise the 32-row tile. The fused kernel keeps its
// 32-wide K step because it stages gate and up tiles as well as the activation.
template <uint32_t DType>
void launch_wmma_swiglu_selected(float * out, const float * x,
                                 const uint8_t * w_gate, const uint8_t * w_up,
                                 int64_t n_in, int64_t n_out, int64_t rows,
                                 uint64_t row_bytes, hipStream_t stream) {
    if (rows >= 64 && n_in <= n_out) {
        launch_wmma_swiglu_tile<DType, 64, 2>(out, x, w_gate, w_up, n_in, n_out,
                                              rows, row_bytes, stream);
    } else {
        launch_wmma_swiglu_tile<DType, 32, 1>(out, x, w_gate, w_up, n_in, n_out,
                                              rows, row_bytes, stream);
    }
}

} // namespace

namespace yuuka {

// Built only by the optional MMQ probe; omitted from the production library.
#ifdef YUUKA_MMQ_PROBE
void launch_gemm_wmma_q4k_experiment(float * out, const float * x, const void * w,
                                     uint64_t n_in, uint64_t n_out, uint64_t batch,
                                     int variant, hipStream_t stream) {
    if (!n_in || !n_out || !batch || n_in % 256) {
        return;
    }
    const uint64_t row_bytes = packed_row_bytes(YUUKA_DTYPE_Q4_K, n_in);
    const auto * src = static_cast<const uint8_t *>(w);
    if (variant == 1) {
        launch_wmma_tile<YUUKA_DTYPE_Q4_K, 64, 2, 64>(out, x, src, n_in, n_out,
                                                      batch, row_bytes, stream);
    } else if (variant == 2) {
        launch_wmma_tile<YUUKA_DTYPE_Q4_K, 32, 1, 64>(out, x, src, n_in, n_out,
                                                      batch, row_bytes, stream);
    } else if (variant == 3) {
        launch_wmma_tile<YUUKA_DTYPE_Q4_K, 32, 1, 128>(out, x, src, n_in, n_out,
                                                       batch, row_bytes, stream);
    }
}
#endif // YUUKA_MMQ_PROBE

void launch_gemm_wmma_f32out(float * out, const float * x, const void * w,
                             uint64_t n_in, uint64_t n_out, uint64_t batch,
                             uint32_t dtype, hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || batch == 0) {
        return;
    }
    const uint64_t row_bytes = packed_row_bytes(dtype, n_in);
    if (row_bytes == 0) {
        return;
    }
    const auto * weights = static_cast<const uint8_t *>(w);
    const int64_t m = static_cast<int64_t>(batch);
    const int64_t in = static_cast<int64_t>(n_in);
    const int64_t on = static_cast<int64_t>(n_out);
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            launch_wmma_selected<YUUKA_DTYPE_F32>(out, x, weights, in, on, m,
                                                  row_bytes, stream);
            break;
        case YUUKA_DTYPE_F16:
            launch_wmma_selected<YUUKA_DTYPE_F16>(out, x, weights, in, on, m,
                                                  row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            launch_wmma_selected<YUUKA_DTYPE_Q4_0>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            launch_wmma_selected<YUUKA_DTYPE_Q5_0>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            launch_wmma_selected<YUUKA_DTYPE_Q8_0>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_wmma_selected<YUUKA_DTYPE_Q4_K>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_wmma_selected<YUUKA_DTYPE_Q5_K>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_wmma_selected<YUUKA_DTYPE_Q6_K>(out, x, weights, in, on, m,
                                                   row_bytes, stream);
            break;
        default:
            break;
    }
}

void launch_gemm_wmma_swiglu_f32out(float * out, const float * x,
                                    const void * w_gate, const void * w_up,
                                    uint64_t n_in, uint64_t n_out, uint64_t batch,
                                    uint32_t dtype, hipStream_t stream) {
    if (n_in == 0 || n_out == 0 || batch == 0) {
        return;
    }
    const uint64_t row_bytes = packed_row_bytes(dtype, n_in);
    if (row_bytes == 0) {
        return;
    }
    const auto * gate = static_cast<const uint8_t *>(w_gate);
    const auto * up = static_cast<const uint8_t *>(w_up);
    const int64_t m = static_cast<int64_t>(batch);
    const int64_t in = static_cast<int64_t>(n_in);
    const int64_t on = static_cast<int64_t>(n_out);
    switch (dtype) {
        case YUUKA_DTYPE_F32:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_F32>(out, x, gate, up, in, on,
                                                         m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_F16:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_F16>(out, x, gate, up, in, on,
                                                         m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_0:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q4_0>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_0:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q5_0>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q8_0:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q8_0>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q4_K:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q4_K>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q5_K:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q5_K>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        case YUUKA_DTYPE_Q6_K:
            launch_wmma_swiglu_selected<YUUKA_DTYPE_Q6_K>(out, x, gate, up, in, on,
                                                          m, row_bytes, stream);
            break;
        default:
            break;
    }
}

} // namespace yuuka
