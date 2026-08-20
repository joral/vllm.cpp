// ROCm keep-quant GGUF k-quant GEMM — the kROCM provider for
// `OpId::kMatmulBTQuant` and `OpId::kMatmulBTQuantGrouped`
// (KERNEL-QUANT-CIQ-GEMM-ROCM, issue #1506).
//
// Ported from src/vt/cuda/cuda_quant_dot.cu, which is itself the MMVQ-shaped
// port of llama.cpp `ggml/src/ggml-cuda/mmvq.cu` + `vecdotq.cuh` @ b10451. The
// integer dot cores below are line-for-line the CUDA ones, because they are
// line-for-line the CPU `vec_dot_*` ones: the encoding is the contract and this
// file must not reinterpret it.
//
// WHY THIS FILE EXISTS. Registering these two ops flips
// `GgufQuantComputeAvailable` (gguf_keep_quant.cpp:75) TRUE on `kROCM`, which is
// the whole point: until it is true, `GgufLoadPolicy::FromEnv` defaults
// `keep_quant` OFF and `RouteGgufTensor` expands EVERY quantized weight to bf16
// at load. Measured on gfx1200 before this file existed: 12159 MiB peak RSS and
// 37.1 s against the CPU build's 7038 MiB and 8.2 s on the same
// `Qwen3.5-4B-Q4_K_M.gguf`, identical tokens.
//
// TWO THINGS THIS FILE DOES DIFFERENTLY FROM THE CUDA ONE, both forced:
//
//  1. The CPU fallback STAGES THROUGH HOST MEMORY. The CUDA provider hands its
//     device pointers straight to the CPU op because GB10 is unified memory. A
//     discrete AMD card is not, and since `VT-REFTIER-HOST-ADDRESSABLE` the
//     reference tier says so out loud rather than segfaulting: "this backend
//     does not report its device memory host-addressable, so a host kernel may
//     not dereference what it allocated". So `FallbackToCpu` below copies
//     D2H, runs the CPU kernel on host buffers, and copies H2D.
//
//  2. Wave width is CHECKED, not assumed. The reduction is a wave32 tree. RDNA
//     is wave32 and CDNA is wave64, so a CDNA board would silently reduce over
//     the wrong lane set. `RequireWave32` refuses by name instead.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/tensor.h"
#include "vt/cpu/cpu_quant_blocks.h"

namespace vt::rocm {
namespace {

using vt::cpu::BlockQ3_K;
using vt::cpu::BlockQ4_K;
using vt::cpu::BlockQ5_K;
using vt::cpu::BlockQ6_K;
using vt::cpu::BlockQ8_K;

constexpr int kQK_K = 256;
constexpr int kWave = 32;      // RDNA wavefront; RequireWave32 enforces it
constexpr int kWavesPerBlock = 4;

inline void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("vt rocm: ") + what + ": " +
                             hipGetErrorString(err));
  }
}
inline hipStream_t AsStream(const Queue& q) {
  return static_cast<hipStream_t>(q.handle);
}

// The reduction below is a wave32 shuffle tree. Refuse anything else by name
// rather than computing a wrong partial sum on it (spec §Risks, `## Owed`).
void RequireWave32() {
  static const int warp_size = [] {
    int dev = 0;
    Check(hipGetDevice(&dev), "hipGetDevice");
    hipDeviceProp_t props{};
    Check(hipGetDeviceProperties(&props, dev), "hipGetDeviceProperties");
    return props.warpSize;
  }();
  if (warp_size != kWave) {
    throw std::runtime_error(
        "vt rocm: matmul_bt_quant: this kernel's reduction is wave32 and this "
        "device reports wavefront " + std::to_string(warp_size) +
        " (wave64/CDNA is a named residual of KERNEL-QUANT-CIQ-GEMM-ROCM, "
        "issue #1506); run the model on the CPU device or disable keep-quant "
        "with VT_GGUF_KEEP_QUANT=0");
  }
}

// ─── device scalar helpers (mirrors of the CUDA ones) ────────────────────────

__device__ inline float DF16ToF32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t man = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      int e = -1;
      uint32_t m = man;
      do {
        ++e;
        m <<= 1;
      } while ((m & 0x400u) == 0);
      bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (man << 13);
  }
  float f;
  __builtin_memcpy(&f, &bits, sizeof(f));
  return f;
}

__device__ inline uint16_t DF32ToBF16(float v) {
  uint32_t bits;
  __builtin_memcpy(&bits, &v, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}

__device__ inline int DNearestInt(float v) {
  return static_cast<int>(v < 0.0f ? v - 0.5f : v + 0.5f);
}

// 4-way int8 dot-accumulate.
//
// `__dp4a` is a CUDA spelling and HIP does NOT declare it (verified: ROCm 7.2.3
// / hipClang 22 rejects it with "use of undeclared identifier" for gfx1200), so
// this is the one place the CUDA port cannot be transcribed. The builtin also
// SPLITS BY ARCHITECTURE, and picking the wrong half is a compile error rather
// than a silent slowdown: RDNA3/RDNA4 expose `__builtin_amdgcn_sudot4` while
// CDNA / RDNA2 / gfx906 expose `__builtin_amdgcn_sdot4`. Mirrors llama.cpp
// `ggml/src/ggml-cuda/common.cuh:703-710` (`ggml_cuda_dp4a`), which carries the
// identical split for the identical reason.
//
// The scalar arm is not decoration: it keeps an unlisted target CORRECT rather
// than failing to build, which is what makes adding a gfx a data change.
__device__ inline int DDp4a(int a, int b, int c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) ||  \
    defined(__gfx1200__) || defined(__gfx1201__)
  // RDNA3 / RDNA4 — v_dot4_i32_iu8 via the signed/unsigned-tagged builtin.
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) ||   \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||     \
    defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) ||  \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
  // CDNA / RDNA2 / gfx906 — v_dot4_i32_i8.
  return __builtin_amdgcn_sdot4(a, b, c, false);
#else
  const int8_t* pa = reinterpret_cast<const int8_t*>(&a);
  const int8_t* pb = reinterpret_cast<const int8_t*>(&b);
  return c + pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2] + pa[3] * pb[3];
#endif
}

enum class ActDT : int { kF32 = 0, kF16 = 1, kBF16 = 2 };

__device__ inline float DLoadAct(const void* base, ActDT dt, int64_t idx) {
  switch (dt) {
    case ActDT::kF32: return static_cast<const float*>(base)[idx];
    case ActDT::kF16: return DF16ToF32(static_cast<const uint16_t*>(base)[idx]);
    default: {
      const uint32_t bits =
          static_cast<uint32_t>(static_cast<const uint16_t*>(base)[idx]) << 16;
      float f;
      __builtin_memcpy(&f, &bits, sizeof(f));
      return f;
    }
  }
}

// ─── activation quantizer ────────────────────────────────────────────────────
// One thread per Q8_K super-block. Bit-exact port of QuantizeRowQ8_K
// (cpu_quant_act.cpp / ggml-quants.c:2696), identical to the CUDA kernel.
__global__ void QuantizeQ8KKernel(BlockQ8_K* __restrict__ scratch,
                                  const void* __restrict__ a, ActDT adt,
                                  int64_t a_rs, int64_t m, int64_t nsb) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= m * nsb) return;
  const int64_t i = t / nsb;
  const int64_t sb = t % nsb;
  const int64_t elem0 = i * a_rs + sb * kQK_K;

  float mx = 0.0f;
  float amax = 0.0f;
  for (int j = 0; j < kQK_K; ++j) {
    const float val = DLoadAct(a, adt, elem0 + j);
    const float ax = fabsf(val);
    if (ax > amax) {
      amax = ax;
      mx = val;
    }
  }
  BlockQ8_K& y = scratch[t];
  if (amax == 0.0f) {
    y.d = 0.0f;
    for (int j = 0; j < kQK_K; ++j) y.qs[j] = 0;
    for (int g = 0; g < kQK_K / 16; ++g) y.bsums[g] = 0;
    return;
  }
  const float iscale = -127.0f / mx;
  for (int j = 0; j < kQK_K; ++j) {
    const int v = DNearestInt(iscale * DLoadAct(a, adt, elem0 + j));
    y.qs[j] = static_cast<int8_t>(v < 127 ? v : 127);
  }
  for (int g = 0; g < kQK_K / 16; ++g) {
    int sum = 0;
    for (int ii = 0; ii < 16; ++ii) sum += y.qs[g * 16 + ii];
    y.bsums[g] = static_cast<int16_t>(sum);
  }
  y.d = 1.0f / iscale;
}

// ─── the four super-block dots ───────────────────────────────────────────────
// Line-for-line the CUDA/CPU forms. Only __dp4a -> DDp4a differs.

// VecDotQ3_KQ8_K (quants.c:566).
__device__ inline float DotQ3K(const BlockQ3_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint8_t* hm = xb->hmask;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  const uint8_t* q3 = xb->qs;
  uint8_t m = 1;
  for (int jj = 0; jj < kQK_K; jj += 128) {
    for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
    for (int l = 0; l < 32; ++l) a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) ? 0 : 4));
    a += 32; m = static_cast<uint8_t>(m << 1);
    q3 += 32;
  }
  uint32_t auxs[4];
  __builtin_memcpy(auxs, xb->scales, 12);
  const int8_t* scales = reinterpret_cast<const int8_t*>(auxs);
  uint32_t tmp = auxs[2];
  auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  a = aux8;
  const int8_t* q8p = q8;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 16; ++j) {
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * (q8p[l] * a[l]);
    q8p += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * (q8p[l] * a[l]);
    q8p += 8; a += 8;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum;
}

// VecDotQ4_KQ8_K (quants.c:645).
__device__ inline float DotQ4K(const BlockQ4_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint32_t kmask3 = 0x03030303;
  const uint8_t* q4 = xb->qs;
  const int8_t* q8 = yb->qs;
  uint32_t utmp[4];
  __builtin_memcpy(utmp, xb->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
  int sumi = 0;
  for (int j = 0; j < kQK_K / 16; ++j) sumi += yb->bsums[j] * mins[j / 2];
  int isum = 0;
  for (int sb = 0; sb < kQK_K / 32; ++sb) {
    const int scale = scales[sb];
    const uint8_t* q4b = q4 + (sb / 2) * 32;
    const int8_t* q8b = q8 + sb * 32;
    const int shift = (sb & 1) ? 4 : 0;
    int sub = 0;
    for (int l = 0; l < 32; l += 4) {
      int wv, av;
      __builtin_memcpy(&wv, q4b + l, sizeof(wv));
      __builtin_memcpy(&av, q8b + l, sizeof(av));
      sub = DDp4a((wv >> shift) & 0x0F0F0F0F, av, sub);
    }
    isum += scale * sub;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  return d * isum - dmin * sumi;
}

// VecDotQ5_KQ8_K (quants.c:720).
__device__ inline float DotQ5K(const BlockQ5_K* xb, const BlockQ8_K* yb) {
  const uint32_t kmask1 = 0x3f3f3f3f;
  const uint32_t kmask2 = 0x0f0f0f0f;
  const uint32_t kmask3 = 0x03030303;
  const uint8_t* q4 = xb->qs;
  const uint8_t* hm = xb->qh;
  const int8_t* q8 = yb->qs;
  uint32_t utmp[4];
  __builtin_memcpy(utmp, xb->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);
  int sumi = 0;
  for (int j = 0; j < kQK_K / 16; ++j) sumi += yb->bsums[j] * mins[j / 2];
  int isum = 0;
  for (int sb = 0; sb < kQK_K / 32; ++sb) {
    const int scale = scales[sb];
    const uint8_t* q4b = q4 + (sb / 2) * 32;
    const int8_t* q8b = q8 + sb * 32;
    const int shift = (sb & 1) ? 4 : 0;
    int sub = 0;
    for (int l = 0; l < 32; l += 4) {
      int wv, hv, av;
      __builtin_memcpy(&wv, q4b + l, sizeof(wv));
      __builtin_memcpy(&hv, hm + l, sizeof(hv));
      __builtin_memcpy(&av, q8b + l, sizeof(av));
      const int lo = (wv >> shift) & 0x0F0F0F0F;
      const int hi = ((hv >> sb) & 0x01010101) << 4;
      sub = DDp4a(lo | hi, av, sub);
    }
    isum += scale * sub;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  const float dmin = DF16ToF32(xb->dmin) * yb->d;
  return d * isum - dmin * sumi;
}

// VecDotQ6_KQ8_K (quants.c:800).
__device__ inline float DotQ6K(const BlockQ6_K* xb, const BlockQ8_K* yb) {
  const uint8_t* q4 = xb->ql;
  const uint8_t* qh = xb->qh;
  const int8_t* q8 = yb->qs;
  int8_t aux8[kQK_K];
  int8_t* a = aux8;
  for (int j = 0; j < kQK_K; j += 128) {
    for (int l = 0; l < 32; ++l) {
      a[l + 0] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
      a[l + 32] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
      a[l + 64] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
      a[l + 96] = static_cast<int8_t>(
          static_cast<int8_t>((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
    }
    a += 128; q4 += 64; qh += 32;
  }
  a = aux8;
  const int8_t* q8p = q8;
  int is = 0;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int j = 0; j < kQK_K / 16; ++j) {
    const int scale = xb->scales[is++];
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
    q8p += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8p[l] * a[l]);
    q8p += 8; a += 8;
  }
  const float d = DF16ToF32(xb->d) * yb->d;
  int isum = 0;
  for (int l = 0; l < 8; ++l) isum += aux32[l];
  return d * isum;
}

// The encodings this provider serves NATIVELY. Kept in step with
// `IsRocmKeepQuantSupported` — the enum and the predicate are one fact.
enum class WType : int { kQ3_K = 0, kQ4_K = 1, kQ5_K = 2, kQ6_K = 3 };

template <WType W>
__device__ inline float DotSuperblock(const void* w, const BlockQ8_K* a);
template <>
__device__ inline float DotSuperblock<WType::kQ3_K>(const void* w, const BlockQ8_K* a) {
  return DotQ3K(static_cast<const BlockQ3_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ4_K>(const void* w, const BlockQ8_K* a) {
  return DotQ4K(static_cast<const BlockQ4_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ5_K>(const void* w, const BlockQ8_K* a) {
  return DotQ5K(static_cast<const BlockQ5_K*>(w), a);
}
template <>
__device__ inline float DotSuperblock<WType::kQ6_K>(const void* w, const BlockQ8_K* a) {
  return DotQ6K(static_cast<const BlockQ6_K*>(w), a);
}

// ─── the GEMMs ───────────────────────────────────────────────────────────────
// One WAVE per output element (i,j); the 32 lanes split the K super-blocks and a
// shuffle tree sums the partials. The integer core is order-independent (exact);
// only the float scale sum is reassociated against the CPU's sequential add,
// which is what the NMSE band in the gate measures.

template <WType W, typename OutT>
__global__ void QuantDotGemmKernel(OutT* __restrict__ out,
                                   const uint8_t* __restrict__ weight,
                                   const BlockQ8_K* __restrict__ act, int64_t m,
                                   int64_t n, int64_t nsb, size_t w_row_bytes,
                                   size_t w_block_bytes) {
  const int64_t wave = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (wave >= m * n) return;
  const int64_t i = wave / n;
  const int64_t j = wave % n;
  const int lane = threadIdx.x;

  const uint8_t* w_row = weight + static_cast<size_t>(j) * w_row_bytes;
  const BlockQ8_K* a_row = act + i * nsb;

  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += kWave) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    partial += DotSuperblock<W>(w_sb, a_row + sb);
  }
#pragma unroll
  for (int off = kWave / 2; off > 0; off >>= 1)
    partial += __shfl_down(partial, off, kWave);

  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) {
      out[i * n + j] = partial;
    } else {
      out[i * n + j] = DF32ToBF16(partial);
    }
  }
}

// GROUPED variant: wave per (p, n); the weight row is selected by the per-group
// expert index — row (expert_ids[p]*N + n) of the stacked [E*N,K] weight. Same
// core, so it is numerically identical to the per-expert dense call.
template <WType W, typename OutT>
__global__ void QuantDotGemmGroupedKernel(OutT* __restrict__ out,
                                          const uint8_t* __restrict__ weight,
                                          const BlockQ8_K* __restrict__ act,
                                          const int32_t* __restrict__ expert_ids,
                                          int64_t P, int64_t n, int64_t nsb,
                                          size_t w_row_bytes, size_t w_block_bytes,
                                          bool bcast) {
  const int64_t wave = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (wave >= P * n) return;
  const int64_t p = wave / n;
  const int64_t j = wave % n;
  const int lane = threadIdx.x;

  const int64_t e = expert_ids[p];
  const uint8_t* w_row = weight + static_cast<size_t>(e * n + j) * w_row_bytes;
  const BlockQ8_K* a_row = act + (bcast ? 0 : p) * nsb;

  float partial = 0.0f;
  for (int64_t sb = lane; sb < nsb; sb += kWave) {
    const void* w_sb = w_row + static_cast<size_t>(sb) * w_block_bytes;
    partial += DotSuperblock<W>(w_sb, a_row + sb);
  }
#pragma unroll
  for (int off = kWave / 2; off > 0; off >>= 1)
    partial += __shfl_down(partial, off, kWave);

  if (lane == 0) {
    if constexpr (sizeof(OutT) == 4) {
      out[p * n + j] = partial;
    } else {
      out[p * n + j] = DF32ToBF16(partial);
    }
  }
}

// ─── host side ───────────────────────────────────────────────────────────────

bool IsRocmKeepQuantSupported(DType dt, WType* out) {
  switch (dt) {
    case DType::kQ3_K: *out = WType::kQ3_K; return true;
    case DType::kQ4_K: *out = WType::kQ4_K; return true;
    case DType::kQ5_K: *out = WType::kQ5_K; return true;
    case DType::kQ6_K: *out = WType::kQ6_K; return true;
    // Everything else — Q2_K, the IQ codebooks, MXFP4, Q4_0, Q8_0 — DELEGATES
    // to the CPU op below. It stays CORRECT and runs at CPU speed, which is the
    // disposition cuda_quant_dot.cu records for its own unserved types. Widening
    // this set is `## Owed` in specs/rocm-keepquant-gemm.md and wants a reaching
    // checkpoint first.
    default: return false;
  }
}

ActDT ActDTypeOf(DType dt) {
  switch (dt) {
    case DType::kF32: return ActDT::kF32;
    case DType::kF16: return ActDT::kF16;
    case DType::kBF16: return ActDT::kBF16;
    default:
      throw std::runtime_error(
          "vt rocm: matmul_bt_quant: activation must be f32/f16/bf16");
  }
}

// A device tensor copied to host, and the host Tensor that views it. The
// discrete-GPU reason this exists is in the file header.
struct HostMirror {
  std::vector<uint8_t> bytes;
  Tensor view;
};

HostMirror MirrorToHost(hipStream_t s, const Tensor& t, size_t nbytes) {
  HostMirror h;
  h.bytes.resize(nbytes);
  Check(hipMemcpyAsync(h.bytes.data(), t.data, nbytes, hipMemcpyDeviceToHost, s),
        "keepquant CPU-fallback D2H");
  h.view = t;
  h.view.data = h.bytes.data();
  h.view.device = Device{DeviceType::kCPU, 0};
  return h;
}

size_t TensorBytes(const Tensor& t) {
  // Rows are packed on the innermost dim; stride[0] may exceed shape[1] for a
  // column slice, so size from the stride to copy exactly what the kernel reads.
  const int64_t rows = t.shape[0];
  const int64_t row_elems = t.rank == 2 ? t.stride[0] : t.shape[0];
  if (IsBlockQuant(t.dtype))
    return static_cast<size_t>(rows) * RowSizeBytes(t.dtype, t.shape[1]);
  return static_cast<size_t>(rows) * static_cast<size_t>(row_elems) *
         SizeOf(t.dtype);
}

// Run the CPU keep-quant kernel over HOST mirrors of the device tensors and copy
// the result back. See the file header for why the CUDA shortcut is unavailable.
void FallbackToCpu(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  hipStream_t s = AsStream(q);
  HostMirror ha = MirrorToHost(s, a, TensorBytes(a));
  HostMirror hb = MirrorToHost(s, b, TensorBytes(b));
  const size_t out_bytes = TensorBytes(out);
  std::vector<uint8_t> host_out(out_bytes);
  Check(hipStreamSynchronize(s), "keepquant CPU-fallback drain");

  Tensor ho = out;
  ho.data = host_out.data();
  ho.device = Device{DeviceType::kCPU, 0};
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  reinterpret_cast<MatmulFn>(GetOp(OpId::kMatmulBTQuant, DeviceType::kCPU))(
      cq, ho, ha.view, hb.view);

  Check(hipMemcpyAsync(out.data, host_out.data(), out_bytes,
                       hipMemcpyHostToDevice, s),
        "keepquant CPU-fallback H2D");
  Check(hipStreamSynchronize(s), "keepquant CPU-fallback H2D drain");
}

void FallbackToCpuGrouped(Queue& q, Tensor& out, const Tensor& act,
                          const Tensor& weight, const Tensor& expert_ids) {
  hipStream_t s = AsStream(q);
  HostMirror ha = MirrorToHost(s, act, TensorBytes(act));
  HostMirror hw = MirrorToHost(s, weight, TensorBytes(weight));
  HostMirror he = MirrorToHost(
      s, expert_ids,
      static_cast<size_t>(expert_ids.shape[0]) * SizeOf(expert_ids.dtype));
  const size_t out_bytes = TensorBytes(out);
  std::vector<uint8_t> host_out(out_bytes);
  Check(hipStreamSynchronize(s), "keepquant-grouped CPU-fallback drain");

  Tensor ho = out;
  ho.data = host_out.data();
  ho.device = Device{DeviceType::kCPU, 0};
  Queue cq{Device{DeviceType::kCPU, 0}, nullptr};
  reinterpret_cast<MatmulBTQuantGroupedFn>(
      GetOp(OpId::kMatmulBTQuantGrouped, DeviceType::kCPU))(cq, ho, ha.view,
                                                           hw.view, he.view);

  Check(hipMemcpyAsync(out.data, host_out.data(), out_bytes,
                       hipMemcpyHostToDevice, s),
        "keepquant-grouped CPU-fallback H2D");
  Check(hipStreamSynchronize(s), "keepquant-grouped CPU-fallback H2D drain");
}

// Q8_K activation scratch, grown in place. One BlockQ8_K per (row, super-block).
struct ActScratch {
  void* p = nullptr;
  size_t bytes = 0;
  BlockQ8_K* Get(size_t need) {
    if (need > bytes) {
      if (p != nullptr) (void)hipFree(p);
      Check(hipMalloc(&p, need), "hipMalloc keepquant activation scratch");
      bytes = need;
    }
    return static_cast<BlockQ8_K*>(p);
  }
};
thread_local ActScratch g_act;

template <WType W>
void LaunchDense(void* out, DType out_dt, const uint8_t* w, const BlockQ8_K* act,
                 int64_t m, int64_t n, int64_t nsb, size_t w_row_bytes,
                 size_t w_block_bytes, hipStream_t s) {
  const dim3 block(kWave, kWavesPerBlock);
  const int64_t waves = m * n;
  const dim3 grid(static_cast<unsigned>((waves + kWavesPerBlock - 1) / kWavesPerBlock));
  if (out_dt == DType::kF32) {
    QuantDotGemmKernel<W, float><<<grid, block, 0, s>>>(
        static_cast<float*>(out), w, act, m, n, nsb, w_row_bytes, w_block_bytes);
  } else {
    QuantDotGemmKernel<W, uint16_t><<<grid, block, 0, s>>>(
        static_cast<uint16_t*>(out), w, act, m, n, nsb, w_row_bytes, w_block_bytes);
  }
}

template <WType W>
void LaunchGrouped(void* out, DType out_dt, const uint8_t* w, const BlockQ8_K* act,
                   const int32_t* ids, int64_t P, int64_t n, int64_t nsb,
                   size_t w_row_bytes, size_t w_block_bytes, bool bcast,
                   hipStream_t s) {
  const dim3 block(kWave, kWavesPerBlock);
  const int64_t waves = P * n;
  const dim3 grid(static_cast<unsigned>((waves + kWavesPerBlock - 1) / kWavesPerBlock));
  if (out_dt == DType::kF32) {
    QuantDotGemmGroupedKernel<W, float><<<grid, block, 0, s>>>(
        static_cast<float*>(out), w, act, ids, P, n, nsb, w_row_bytes,
        w_block_bytes, bcast);
  } else {
    QuantDotGemmGroupedKernel<W, uint16_t><<<grid, block, 0, s>>>(
        static_cast<uint16_t*>(out), w, act, ids, P, n, nsb, w_row_bytes,
        w_block_bytes, bcast);
  }
}

void MatmulBTQuantRocm(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  WType w{};
  if (!IsRocmKeepQuantSupported(b.dtype, &w)) {
    FallbackToCpu(q, out, a, b);
    return;
  }
  RequireWave32();

  const int64_t m = a.shape[0], k = a.shape[1], n = b.shape[0];
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt rocm: matmul_bt_quant: K must be a whole number of 256-element "
        "Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  hipStream_t s = AsStream(q);

  BlockQ8_K* act = g_act.Get(static_cast<size_t>(m * nsb) * sizeof(BlockQ8_K));
  const int64_t qthreads = m * nsb;
  constexpr int kQBlock = 256;
  QuantizeQ8KKernel<<<static_cast<unsigned>((qthreads + kQBlock - 1) / kQBlock),
                      kQBlock, 0, s>>>(act, a.data, ActDTypeOf(a.dtype),
                                       a.stride[0], m, nsb);

  const size_t w_block_bytes = static_cast<size_t>(BlockBytes(b.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;
  switch (w) {
    case WType::kQ3_K:
      LaunchDense<WType::kQ3_K>(out.data, out.dtype,
                                static_cast<const uint8_t*>(b.data), act, m, n,
                                nsb, w_row_bytes, w_block_bytes, s);
      break;
    case WType::kQ4_K:
      LaunchDense<WType::kQ4_K>(out.data, out.dtype,
                                static_cast<const uint8_t*>(b.data), act, m, n,
                                nsb, w_row_bytes, w_block_bytes, s);
      break;
    case WType::kQ5_K:
      LaunchDense<WType::kQ5_K>(out.data, out.dtype,
                                static_cast<const uint8_t*>(b.data), act, m, n,
                                nsb, w_row_bytes, w_block_bytes, s);
      break;
    case WType::kQ6_K:
      LaunchDense<WType::kQ6_K>(out.data, out.dtype,
                                static_cast<const uint8_t*>(b.data), act, m, n,
                                nsb, w_row_bytes, w_block_bytes, s);
      break;
    // No silent arm. Past IsRocmKeepQuantSupported there is no fallback left,
    // and #1029 is what a missing case costs: the launch simply does not happen,
    // the error check reports success, and the output buffer is never written.
    default:
      throw std::runtime_error(
          "vt rocm: matmul_bt_quant: no kernel for weight dtype " +
          std::string(Name(b.dtype)) +
          " past the supported predicate (KERNEL-QUANT-CIQ-GEMM-ROCM)");
  }
  Check(hipGetLastError(), "matmul_bt_quant launch");
}

void MatmulBTQuantGroupedRocm(Queue& q, Tensor& out, const Tensor& act,
                              const Tensor& weight, const Tensor& expert_ids) {
  WType w{};
  if (!IsRocmKeepQuantSupported(weight.dtype, &w)) {
    FallbackToCpuGrouped(q, out, act, weight, expert_ids);
    return;
  }
  RequireWave32();

  const int64_t P = out.shape[0], n = out.shape[1], k = act.shape[1];
  if (k % kQK_K != 0) {
    throw std::runtime_error(
        "vt rocm: matmul_bt_quant_grouped: K must be a whole number of "
        "256-element Q8_K super-blocks");
  }
  const int64_t nsb = k / kQK_K;
  const bool bcast = act.shape[0] == 1;
  const int64_t Pa = bcast ? 1 : P;
  hipStream_t s = AsStream(q);

  BlockQ8_K* qact = g_act.Get(static_cast<size_t>(Pa * nsb) * sizeof(BlockQ8_K));
  const int64_t qthreads = Pa * nsb;
  constexpr int kQBlock = 256;
  QuantizeQ8KKernel<<<static_cast<unsigned>((qthreads + kQBlock - 1) / kQBlock),
                      kQBlock, 0, s>>>(qact, act.data, ActDTypeOf(act.dtype),
                                       act.stride[0], Pa, nsb);

  const size_t w_block_bytes = static_cast<size_t>(BlockBytes(weight.dtype));
  const size_t w_row_bytes = static_cast<size_t>(nsb) * w_block_bytes;
  const int32_t* ids = static_cast<const int32_t*>(expert_ids.data);
  switch (w) {
    case WType::kQ3_K:
      LaunchGrouped<WType::kQ3_K>(out.data, out.dtype,
                                  static_cast<const uint8_t*>(weight.data), qact,
                                  ids, P, n, nsb, w_row_bytes, w_block_bytes,
                                  bcast, s);
      break;
    case WType::kQ4_K:
      LaunchGrouped<WType::kQ4_K>(out.data, out.dtype,
                                  static_cast<const uint8_t*>(weight.data), qact,
                                  ids, P, n, nsb, w_row_bytes, w_block_bytes,
                                  bcast, s);
      break;
    case WType::kQ5_K:
      LaunchGrouped<WType::kQ5_K>(out.data, out.dtype,
                                  static_cast<const uint8_t*>(weight.data), qact,
                                  ids, P, n, nsb, w_row_bytes, w_block_bytes,
                                  bcast, s);
      break;
    case WType::kQ6_K:
      LaunchGrouped<WType::kQ6_K>(out.data, out.dtype,
                                  static_cast<const uint8_t*>(weight.data), qact,
                                  ids, P, n, nsb, w_row_bytes, w_block_bytes,
                                  bcast, s);
      break;
    // Same reason as the dense switch above. #1029.
    default:
      throw std::runtime_error(
          "vt rocm: matmul_bt_quant_grouped: no kernel for weight dtype " +
          std::string(Name(weight.dtype)) +
          " past the supported predicate (KERNEL-QUANT-CIQ-GEMM-ROCM)");
  }
  Check(hipGetLastError(), "matmul_bt_quant_grouped launch");
}

// Table fill only, no HIP calls — same rationale as the other ROCm registrars.
// This is what flips GgufQuantComputeAvailable TRUE on kROCM, so the GGUF loader
// stops expanding every quantized weight to bf16 at load.
struct Registrar {
  Registrar() {
    RegisterOp(OpId::kMatmulBTQuant, DeviceType::kROCM,
               reinterpret_cast<void*>(static_cast<MatmulFn>(&MatmulBTQuantRocm)));
    RegisterOp(
        OpId::kMatmulBTQuantGrouped, DeviceType::kROCM,
        reinterpret_cast<void*>(
            static_cast<MatmulBTQuantGroupedFn>(&MatmulBTQuantGroupedRocm)));
  }
} registrar;

}  // namespace
}  // namespace vt::rocm
