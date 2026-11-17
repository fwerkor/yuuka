#include "../../kernels/hip/quant.cuh"
#include "../../kernels/hip/yuuka_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Minimal Q4_K x Q8_1-like experiment inspired by llama.cpp mul_mat_q.
// This is intentionally a standalone probe, not a production dispatcher.
namespace yuuka {
void launch_gemm_wmma_q4k_experiment(float * out,const float * x,const void * w,
    uint64_t K,uint64_t N,uint64_t M,int variant,hipStream_t stream);
}
namespace {

struct Q8Block {
    float scale;
    int32_t sum;
    uint32_t words[8];
};
static_assert(sizeof(Q8Block) == 40, "unexpected Q8 scratch layout");

__global__ void quantize_q8(const float * x, Q8Block * out, int K) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int idx = blockIdx.x * 8 + warp;
    const int M = gridDim.y;
    const int row = blockIdx.y;
    if (row >= M || idx >= K / 32) return;
    const float v = x[row * K + idx * 32 + lane];
    float amax = fabsf(v);
    for (int offset = 16; offset; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down(amax, offset));
    }
    amax = __shfl(amax, 0);
    const float scale = amax > 0 ? amax / 127.0f : 1.0f;
    const int qi = max(-127, min(127, __float2int_rn(v / scale)));
    int sum = qi;
    for (int offset = 16; offset; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    Q8Block & b = out[row * (K / 32) + idx];
    if (lane == 0) {
        b.scale = scale;
        b.sum = sum;
    }
    // All warp lanes must execute shuffles. Sources 8..31 would be inactive
    // if the shuffle were placed inside the lane < 8 writer branch.
    const int src = 4 * (lane & 7);
    const uint32_t q0 = uint8_t(__shfl(qi, src + 0));
    const uint32_t q1 = uint8_t(__shfl(qi, src + 1));
    const uint32_t q2 = uint8_t(__shfl(qi, src + 2));
    const uint32_t q3 = uint8_t(__shfl(qi, src + 3));
    if (lane < 8) {
        b.words[lane] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
    }
}

template<int MR>
__global__ void gemm_q4k_q8(float * out, const Q8Block * activations,
                            const uint8_t * weights, int K, int N, int M) {
    const int n = int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int m0 = int(blockIdx.y) * MR;
    if (n >= N) return;
    float accum[MR];
    #pragma unroll
    for (int m = 0; m < MR; ++m) accum[m] = 0.0f;
    const uint8_t * row = weights + size_t(n) * (K / 256) * 144;
    constexpr uint32_t mask4 = 0x0f0f0f0fu;
    for (int kb = 0; kb < K / 256; ++kb) {
        const uint8_t * b = row + kb * 144;
        const float d = load_f16_device(b);
        const float dm = load_f16_device(b + 2);
        const uint8_t * qs = b + 16;
        for (int sub = 0; sub < 8; ++sub) {
            uint8_t sc = 0, mn = 0;
            get_scale_min_k4(sub, b + 4, sc, mn);
            const uint32_t * packed = reinterpret_cast<const uint32_t *>(qs + (sub / 2) * 32);
            const float dscale = d * float(sc);
            const float mscale = dm * float(mn);
            int sums[MR];
            #pragma unroll
            for (int m = 0; m < MR; ++m) sums[m] = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t q4 = packed[j];
                const uint32_t w4 = ((sub & 1) ? q4 >> 4 : q4) & mask4;
                #pragma unroll
                for (int m = 0; m < MR; ++m) {
                    if (m0 + m < M) {
                        const Q8Block & a = activations[(m0 + m) * (K / 32) + kb * 8 + sub];
                        sums[m] = __builtin_amdgcn_sudot4(true, int(w4), true,
                                                          int(a.words[j]), sums[m], false);
                    }
                }
            }
            #pragma unroll
            for (int m = 0; m < MR; ++m) {
                if (m0 + m < M) {
                    const Q8Block & a = activations[(m0 + m) * (K / 32) + kb * 8 + sub];
                    accum[m] += a.scale * (dscale * float(sums[m]) - mscale * float(a.sum));
                }
            }
        }
    }
    #pragma unroll
    for (int m = 0; m < MR; ++m) {
        if (m0 + m < M) out[(m0 + m) * N + n] = accum[m];
    }
}


// Packed Q4_K -> int8, Q8 activations -> int8 WMMA. Each block covers
// 32 prompt rows by 64 output channels, accumulating and rescaling each
// 32-value quant sub-block in shared FP32. This avoids redundant per-output
// SDOT4 products and matches the Q4_K per-group scale/min semantics.
__global__ void gemm_q4k_q8_wmma(float * out, const Q8Block * activations,
                                 const uint8_t * weights, int K, int N, int M) {
    constexpr int BM=32, BN=64, BK=32;
    const int tid=threadIdx.x;
    const int wave=tid/32;
    const int wm=wave/4, wn=wave%4;
    const int m0=blockIdx.y*BM, n0=blockIdx.x*BN;
    __shared__ __align__(16) int8_t atile[BM*BK];
    __shared__ __align__(16) int8_t btile[BN*BK];
    __shared__ __align__(16) int32_t itile[BM*BN];
    __shared__ __align__(16) float ctile[BM*BN];

    for (int idx=tid;idx<BM*BN;idx+=blockDim.x) ctile[idx]=0.0f;
    __syncthreads();
    for (int subblock=0;subblock<K/32;++subblock) {
        const int sub=subblock&7;
        const int kb=subblock/8;
        for (int idx=tid;idx<BM*BK;idx+=blockDim.x) {
            const int m=m0+idx/BK;
            const int k=idx%BK;
            atile[idx]=m<M
                ? reinterpret_cast<const int8_t *>(activations[m*(K/32)+subblock].words)[k]
                : 0;
        }
        for (int idx=tid;idx<BN*BK;idx+=blockDim.x) {
            const int n=n0+idx/BK;
            const int k=idx%BK;
            if (n>=N) { btile[idx]=0; continue; }
            const uint8_t * b=weights+size_t(n)*(K/256)*144+kb*144;
            const uint8_t v=b[16+(sub/2)*32+k];
            btile[idx]=int8_t((sub&1)?v>>4:v&0x0f);
        }
        __syncthreads();

        rocwmma::fragment<rocwmma::accumulator,16,16,16,int32_t> acc;
        rocwmma::fill_fragment(acc,0);
        for(int kk=0;kk<32;kk+=16){
            rocwmma::fragment<rocwmma::matrix_a,16,16,16,int8_t,rocwmma::row_major> a;
            rocwmma::fragment<rocwmma::matrix_b,16,16,16,int8_t,rocwmma::col_major> b;
            rocwmma::load_matrix_sync(a,atile+wm*16*BK+kk,BK);
            rocwmma::load_matrix_sync(b,btile+wn*16*BK+kk,BK);
            rocwmma::mma_sync(acc,a,b,acc);
        }
        rocwmma::store_matrix_sync(itile+(wm*16)*BN+wn*16,
                                   acc,BN,rocwmma::mem_row_major);
        __syncthreads();

        for(int idx=tid;idx<BM*BN;idx+=blockDim.x){
            const int m=m0+idx/BN, n=n0+idx%BN;
            if(m>=M||n>=N) continue;
            const Q8Block & a=activations[m*(K/32)+subblock];
            const uint8_t * b=weights+size_t(n)*(K/256)*144+kb*144;
            uint8_t scale=0,mn=0;
            get_scale_min_k4(sub,b+4,scale,mn);
            const float wd=load_f16_device(b)*float(scale);
            const float wm=load_f16_device(b+2)*float(mn);
            ctile[idx]+=a.scale*(wd*float(itile[idx])-wm*float(a.sum));
        }
        __syncthreads();
    }

    for(int idx=tid;idx<BM*BN;idx+=blockDim.x){
        const int m=m0+idx/BN,n=n0+idx%BN;
        if(m<M&&n<N)out[m*N+n]=ctile[idx];
    }
}

void check(hipError_t e, const char * message) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, hipGetErrorString(e));
        std::exit(1);
    }
}
template<class F> float time_ms(F && f, int reps) {
    hipEvent_t start = nullptr, stop = nullptr;
    check(hipEventCreate(&start), "event create");
    check(hipEventCreate(&stop), "event create");
    for (int i = 0; i < 2; ++i) f();
    check(hipDeviceSynchronize(), "warmup");
    check(hipEventRecord(start), "start");
    for (int i = 0; i < reps; ++i) f();
    check(hipEventRecord(stop), "stop");
    check(hipEventSynchronize(stop), "sync");
    float ms = 0;
    check(hipEventElapsedTime(&ms, start, stop), "elapsed");
    check(hipEventDestroy(start), "destroy start event");
    check(hipEventDestroy(stop), "destroy stop event");
    return ms / reps;
}

int run(int K, int N, int M) {
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> byte(0,255);
    std::uniform_real_distribution<float> val(-1.0f,1.0f);
    std::vector<uint8_t> hweights(size_t(N) * (K / 256) * 144);
    for (auto & b : hweights) b = uint8_t(byte(gen));
    for (size_t i = 0; i < hweights.size(); i += 144) {
        hweights[i] = 0x00; hweights[i+1] = 0x24; // F16 0.015625
        hweights[i+2] = 0x00; hweights[i+3] = 0x1c; // F16 0.00390625
    }
    std::vector<float> hx(size_t(M)*K);
    for (auto & v : hx) v = val(gen);
    const size_t output_elements = size_t(M)*N;
    uint8_t * w = nullptr;
    float * x = nullptr, * result = nullptr, * reference = nullptr, * tiled = nullptr, * tuned = nullptr;
    Q8Block * q8 = nullptr;
    check(hipMalloc(reinterpret_cast<void **>(&w), hweights.size()), "weight malloc");
    check(hipMalloc(reinterpret_cast<void **>(&x), hx.size() * sizeof(float)), "activation malloc");
    check(hipMalloc(reinterpret_cast<void **>(&result), output_elements * sizeof(float)), "mmq malloc");
    check(hipMalloc(reinterpret_cast<void **>(&reference), output_elements * sizeof(float)), "wmma malloc");
    check(hipMalloc(reinterpret_cast<void **>(&tiled), output_elements * sizeof(float)), "tiled malloc");
    check(hipMalloc(reinterpret_cast<void **>(&tuned), output_elements * sizeof(float)), "tuned malloc");
    check(hipMalloc(reinterpret_cast<void **>(&q8), size_t(M)*(K/32)*sizeof(Q8Block)), "q8 malloc");
    check(hipMemcpy(w, hweights.data(), hweights.size(), hipMemcpyHostToDevice), "weight copy");
    check(hipMemcpy(x, hx.data(), hx.size()*sizeof(float), hipMemcpyHostToDevice), "activation copy");
    auto mmq = [&]() {
        quantize_q8<<<dim3((K+255)/256, M), 256>>>(x, q8, K);
        gemm_q4k_q8<4><<<dim3((N+127)/128, (M+3)/4),128>>>(result,q8,w,K,N,M);
    };
    auto mmq_tiled = [&]() {
        quantize_q8<<<dim3((K+255)/256, M),256>>>(x,q8,K);
        gemm_q4k_q8_wmma<<<dim3((N+63)/64,(M+31)/32),256>>>(tiled,q8,w,K,N,M);
    };
    auto wmma = [&]() {
        yuuka::launch_gemm_wmma_f32out(reference,x,w,K,N,M,YUUKA_DTYPE_Q4_K,0);
    };
    // Run before timing to report numerical divergence from Q8 activations.
    wmma(); mmq(); mmq_tiled();
    check(hipDeviceSynchronize(), "kernel synchronization");
    std::vector<float> a(output_elements), b(output_elements);
    check(hipMemcpy(a.data(),reference,output_elements*4,hipMemcpyDeviceToHost),"ref copy");
    check(hipMemcpy(b.data(),result,output_elements*4,hipMemcpyDeviceToHost),"mmq copy");
    double sum_error2=0, sum_ref2=0, max_diff=0;
    for (size_t i=0; i<output_elements; ++i) {
        const double diff=double(a[i])-double(b[i]);
        sum_error2+=diff*diff;
        sum_ref2+=double(a[i])*a[i];
        max_diff=std::max(max_diff,std::abs(diff));
    }
    std::vector<float> c(output_elements);
    check(hipMemcpy(c.data(),tiled,output_elements*4,hipMemcpyDeviceToHost),"tiled copy");
    double e2_tiled=0, max_tiled=0;
    for(size_t i=0;i<output_elements;++i) {
        const double d=double(a[i])-double(c[i]);
        e2_tiled+=d*d;max_tiled=std::max(max_tiled,std::abs(d));
    }
    float t_wmma=time_ms(wmma,3), t_mmq=time_ms(mmq,3), t_mmq_tiled=time_ms(mmq_tiled,3);
    std::printf("Q4_K M=%d N=%d K=%d wmma_ms=%.3f mmq_ms=%.3f speedup=%.3fx relative_RMSE=%.6f max_diff=%.5f\n",
        M,N,K,t_wmma,t_mmq,t_wmma/t_mmq,std::sqrt(sum_error2/std::max(sum_ref2,1e-30)),max_diff);
    std::printf("     TILED_MMQ int8_ms=%.3f speedup=%.3fx relative_RMSE=%.6f max_diff=%.5f\n",
                t_mmq_tiled,t_wmma/t_mmq_tiled,std::sqrt(e2_tiled/std::max(sum_ref2,1e-30)),max_tiled);
    if (std::sqrt(sum_error2/std::max(sum_ref2,1e-30)) > 0.01 ||
        std::sqrt(e2_tiled/std::max(sum_ref2,1e-30)) > 0.01) {
        std::fprintf(stderr, "Q8/MMQ numerical error exceeds 1%% relative RMSE\n");
        std::exit(2);
    }
    for (int variant=1;variant<=3;++variant) {
        auto tiled_f16=[&]() { yuuka::launch_gemm_wmma_q4k_experiment(tuned,x,w,K,N,M,variant,0); };
        tiled_f16(); check(hipDeviceSynchronize(), "tuned wmma");
        std::vector<float> d(output_elements);
        check(hipMemcpy(d.data(),tuned,output_elements*4,hipMemcpyDeviceToHost),"tuned copy");
        double err2=0,max_err=0;
        for(size_t i=0;i<output_elements;++i) {
            const double delta=double(a[i])-double(d[i]);
            err2+=delta*delta;max_err=std::max(max_err,std::abs(delta));
        }
        if (err2 > 1e-8 * std::max(sum_ref2,1e-30)) {
            std::fprintf(stderr, "FP16 WMMA K-tile has unexpected numerical divergence\n");
            std::exit(3);
        }
        const float t=time_ms(tiled_f16,3);
        std::printf("     FP16_TILE variant=%d ms=%.3f speedup=%.3fx relRMSE=%.7f max_diff=%.5f\n",
            variant,t,t_wmma/t,std::sqrt(err2/std::max(sum_ref2,1e-30)),max_err);
    }
    check(hipFree(tuned), "free tuned output");
    check(hipFree(tiled), "free int8 output");
    check(hipFree(q8), "free q8");
    check(hipFree(reference), "free reference");
    check(hipFree(result), "free scalar mmq");
    check(hipFree(x), "free activation");
    check(hipFree(w), "free weight");
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    if (argc == 4) return run(std::atoi(argv[1]),std::atoi(argv[2]),std::atoi(argv[3]));
    run(2560,9216,128);
    run(2560,9216,512);
    run(9216,2560,512);
    return 0;
}
