// ROCm keep-quant GEMM gate (KERNEL-QUANT-CIQ-GEMM-ROCM, issue #1506). The
// kROCM provider for `OpId::kMatmulBTQuant` / `kMatmulBTQuantGrouped`
// (src/vt/rocm/rocm_quant_dot.hip) is measured against the LANDED CPU
// keep-quant reference (src/vt/cpu/cpu_quant_gemm.cpp — the oracle, itself
// gated by test_ops_quant_dot.cpp) and against an INDEPENDENT f64
// dequantize-then-dot.
//
// Structure mirrors tests/vt/test_cuda_quant_dot.cpp so the two read side by
// side. Three differences, each deliberate:
//
//  1. FOUR encodings, not eleven — Q4_K/Q6_K/Q3_K/Q5_K. A census of 43 GGUFs on
//     the gate box reaches exactly these four plus types the CPU already serves,
//     and `AGENTS.md` §"Nothing lands dead" refuses a kernel arm no checkpoint
//     exercises. Everything else DELEGATES to the CPU op, which case (b) gates.
//
//  2. Every output buffer is POISONED before the call and every element is
//     asserted overwritten. This is not decoration. #1029 landed green because a
//     grouped dispatch with a missing case quantized the activation, launched
//     NOTHING, and returned — and `cudaGetLastError()` reported success because
//     there had been no launch to fail. An all-zeros buffer hides that; a
//     sentinel cannot.
//
//  3. Case (c) gates the GROUPED op in the same file as the dense one, because
//     registering the dense op ALONE is the hazard this row exists to avoid:
//     `GgufQuantComputeAvailable` (gguf_keep_quant.cpp:75) probes only the dense
//     op, but `qwen3_5.cpp:6193` KqGrouped calls MatmulBTQuantGrouped default-ON
//     and un-device-gated. A dense-only registration flips `keep_quant` true on
//     ROCm, keeps MoE blocks compressed, and then kills the first expert GEMM on
//     models that load today.
//
// Skips cleanly (returns) when no ROCm backend is present, so the CPU CI leg is
// green; it only asserts on a real AMD device. Compiled everywhere as a bit-rot
// guard, exactly like tests/vt/test_rocm_backend.cpp.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/quant.h"  // vt::cpu::BlockToFloat
#include "vt/tensor.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

namespace {

// CPU-vs-ROCm: the Q8_K activation quant and the whole INTEGER dot are
// bit-identical to the CPU reference by construction, so only the per-super-block
// float scale sum is reassociated (wave reduction vs the CPU sequential add).
// Same tight band the CUDA gate holds.
constexpr double kMaxNmseVsCpu = 1e-6;
// vs the f64 dequant reference: the band test_ops_quant_dot.cpp uses. This
// measures ACTIVATION quantization error, not the kernel.
constexpr double kMaxNmseErr = 5e-4;
// The poison value. Chosen finite and large so a partial write is visible in the
// NMSE too, not only in the sentinel scan.
constexpr float kPoison = -12345.0F;

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kROCM, 0}; }

bool HasRocm() { return vt::TryGetBackend(DeviceType::kROCM) != nullptr; }

struct WeightCase {
  DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;
  int dmin_off;
  const char* name;
};

// Offsets restated independently from ggml-common.h, matching the rows in
// tests/vt/test_cuda_quant_dot.cpp for the four types this row serves.
const WeightCase kCases[] = {
    {DType::kQ3_K, 256, 110, 108, -1, "q3_K"},
    {DType::kQ4_K, 256, 144, 0, 2, "q4_K"},
    {DType::kQ5_K, 256, 176, 0, 2, "q5_K"},
    {DType::kQ6_K, 256, 210, 208, -1, "q6_K"},
};

// The dtype case (b) uses to prove the CPU delegation: Q2_K is a real encoding
// the CPU serves and this row deliberately does NOT implement natively.
constexpr DType kDelegatedDType = DType::kQ2_K;
constexpr int64_t kDelegatedBlockBytes = 84;
constexpr int kDelegatedDOff = 80;
constexpr int kDelegatedDminOff = 82;

void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
}

std::vector<uint8_t> RandomBlocks(int64_t nblocks, int64_t block_bytes, int d_off,
                                  int dmin_off, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    if (d_off >= 0) put_f16(d_off, 0.0125F * jitter);
    if (dmin_off >= 0) put_f16(dmin_off, 0.0075F * jitter);
  }
  return bytes;
}

Tensor DevTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = p;
  t.dtype = dt;
  t.device = Gpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

Tensor HostTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t = DevTensor(p, dt, shape);
  t.device = Cpu();
  return t;
}

// How many elements the callee left at the sentinel. Zero is the contract.
int64_t PoisonSurvivors(const std::vector<float>& out) {
  int64_t n = 0;
  for (float v : out)
    if (v == kPoison) ++n;
  return n;
}

}  // namespace

// ── (a) the dense per-type gate ───────────────────────────────────────────────
TEST_CASE("ROCm keep-quant GEMM == CPU reference and f64 dequant, and it WRITES") {
  if (!HasRocm()) {
    MESSAGE("no ROCm backend on this host; ROCm keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  for (const WeightCase& c : kCases) {
    // Decode (M=1) through prefill (M=512), plus an odd N that catches a
    // wave/chunking assumption; K = 8 super-blocks (model-ish).
    const int64_t k = 8 * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{4}, int64_t{32}, int64_t{512}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(std::string(c.name));
        CAPTURE(m);
        CAPTURE(k);
        CAPTURE(n);

        std::vector<uint8_t> wq = RandomBlocks(n * (k / c.block_elems),
                                               c.block_bytes, c.d_off,
                                               c.dmin_off, 0x5EEDU);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());

        // --- CPU oracle (the landed keep-quant kernel over host tensors) ------
        std::vector<float> cpu_out(static_cast<size_t>(m * n), kPoison);
        {
          Tensor at = HostTensor(a.data(), DType::kF32, {m, k});
          Tensor bt = HostTensor(wq.data(), c.dtype, {n, k});
          Tensor ot = HostTensor(cpu_out.data(), DType::kF32, {m, n});
          vt::MatmulBTQuant(cq, ot, at, bt);
        }
        REQUIRE(PoisonSurvivors(cpu_out) == 0);  // the oracle writes too

        // --- ROCm path (device tensors) ---------------------------------------
        std::vector<float> rocm_out(static_cast<size_t>(m * n), kPoison);
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        void* d_w = gpu.Alloc(wq.size());
        void* d_o = gpu.Alloc(rocm_out.size() * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        // Poison the DEVICE buffer, not just the host copy: a kernel that never
        // launches leaves exactly these bytes behind.
        gpu.Copy(gq, d_o, rocm_out.data(), rocm_out.size() * sizeof(float));
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        Tensor bt = DevTensor(d_w, c.dtype, {n, k});
        Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
        vt::MatmulBTQuant(gq, ot, at, bt);
        gpu.Copy(gq, rocm_out.data(), d_o, rocm_out.size() * sizeof(float));
        gpu.Synchronize(gq);
        gpu.Free(d_a);
        gpu.Free(d_w);
        gpu.Free(d_o);

        // THE #1029 ASSERTION: every element was written. A dispatch that
        // launches nothing and reports success dies here and nowhere else.
        CAPTURE(PoisonSurvivors(rocm_out));
        REQUIRE(PoisonSurvivors(rocm_out) == 0);

        // --- f64 independent reference: decode weight + f32 activation dot ----
        std::vector<float> w(static_cast<size_t>(n * k));
        vt::cpu::BlockToFloat(c.dtype)(wq.data(), w.data(), n * k);

        double num_ref = 0, den_ref = 0, num_cpu = 0, den_cpu = 0;
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t jj = 0; jj < n; ++jj) {
            double ref = 0;
            for (int64_t p = 0; p < k; ++p)
              ref += static_cast<double>(a[static_cast<size_t>(i * k + p)]) *
                     static_cast<double>(w[static_cast<size_t>(jj * k + p)]);
            const double got = rocm_out[static_cast<size_t>(i * n + jj)];
            const double cpu = cpu_out[static_cast<size_t>(i * n + jj)];
            num_ref += (got - ref) * (got - ref);
            den_ref += ref * ref;
            num_cpu += (got - cpu) * (got - cpu);
            den_cpu += cpu * cpu;
            REQUIRE(std::isfinite(got));
          }
        }
        const double nmse_ref = den_ref > 0 ? num_ref / den_ref : num_ref;
        const double nmse_cpu = den_cpu > 0 ? num_cpu / den_cpu : num_cpu;
        CAPTURE(nmse_ref);
        CAPTURE(nmse_cpu);
        CHECK(nmse_ref <= kMaxNmseErr);    // quantization error vs f64 dequant
        CHECK(nmse_cpu <= kMaxNmseVsCpu);  // matches the CPU oracle
      }
    }
  }
  gpu.DestroyQueue(gq);
}

// ── (b) the delegation arm ────────────────────────────────────────────────────
// With keep_quant true on ROCm, a dtype this row does not implement natively
// still arrives at the ROCm provider. It must DELEGATE to the CPU op and produce
// CPU-equal output — the disposition cuda_quant_dot.cu:1835 records ("falls to
// the CPU arm below and still emits CORRECT tokens, just at CPU speed"). A
// throw here would convert a slow path into a broken one.
TEST_CASE("ROCm keep-quant delegates an unimplemented dtype to the CPU op") {
  if (!HasRocm()) {
    MESSAGE("no ROCm backend on this host; ROCm delegation gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  const int64_t k = 8 * 256, m = 4, n = 7;
  std::vector<uint8_t> wq =
      RandomBlocks(n * (k / 256), kDelegatedBlockBytes, kDelegatedDOff,
                   kDelegatedDminOff, 0x5EEDU);
  std::vector<float> a(static_cast<size_t>(m * k));
  GenerateData(1.0F, a.size(), a.data());

  std::vector<float> cpu_out(static_cast<size_t>(m * n), kPoison);
  {
    Tensor at = HostTensor(a.data(), DType::kF32, {m, k});
    Tensor bt = HostTensor(wq.data(), kDelegatedDType, {n, k});
    Tensor ot = HostTensor(cpu_out.data(), DType::kF32, {m, n});
    vt::MatmulBTQuant(cq, ot, at, bt);
  }

  std::vector<float> rocm_out(static_cast<size_t>(m * n), kPoison);
  void* d_a = gpu.Alloc(a.size() * sizeof(float));
  void* d_w = gpu.Alloc(wq.size());
  void* d_o = gpu.Alloc(rocm_out.size() * sizeof(float));
  gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
  gpu.Copy(gq, d_w, wq.data(), wq.size());
  gpu.Copy(gq, d_o, rocm_out.data(), rocm_out.size() * sizeof(float));
  Tensor at = DevTensor(d_a, DType::kF32, {m, k});
  Tensor bt = DevTensor(d_w, kDelegatedDType, {n, k});
  Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
  vt::MatmulBTQuant(gq, ot, at, bt);
  gpu.Copy(gq, rocm_out.data(), d_o, rocm_out.size() * sizeof(float));
  gpu.Synchronize(gq);
  gpu.Free(d_a);
  gpu.Free(d_w);
  gpu.Free(d_o);

  REQUIRE(PoisonSurvivors(rocm_out) == 0);
  for (size_t i = 0; i < rocm_out.size(); ++i) {
    CAPTURE(i);
    REQUIRE(std::isfinite(rocm_out[i]));
    CHECK(rocm_out[i] == doctest::Approx(cpu_out[i]).epsilon(1e-6));
  }
  gpu.DestroyQueue(gq);
}

// ── (c) the grouped arm ───────────────────────────────────────────────────────
// The arm that must not be left unreached. See the file header.
TEST_CASE("ROCm grouped keep-quant GEMM == CPU grouped golden and it WRITES") {
  if (!HasRocm()) {
    MESSAGE("no ROCm backend on this host; ROCm grouped keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  for (const WeightCase& c : kCases) {
    const int64_t k = 4 * c.block_elems;
    const int64_t n = 8, e = 4, p = 6;
    CAPTURE(std::string(c.name));

    std::vector<uint8_t> wq = RandomBlocks(e * n * (k / c.block_elems),
                                           c.block_bytes, c.d_off, c.dmin_off,
                                           0xA11CEU);
    std::vector<float> a(static_cast<size_t>(p * k));
    GenerateData(2.0F, a.size(), a.data());
    std::vector<int32_t> ids(static_cast<size_t>(p));
    for (int64_t i = 0; i < p; ++i) ids[static_cast<size_t>(i)] = int32_t(i % e);

    std::vector<float> cpu_out(static_cast<size_t>(p * n), kPoison);
    {
      Tensor at = HostTensor(a.data(), DType::kF32, {p, k});
      Tensor wt = HostTensor(wq.data(), c.dtype, {e * n, k});
      Tensor et = HostTensor(ids.data(), DType::kI32, {p});
      Tensor ot = HostTensor(cpu_out.data(), DType::kF32, {p, n});
      vt::MatmulBTQuantGrouped(cq, ot, at, wt, et);
    }
    REQUIRE(PoisonSurvivors(cpu_out) == 0);

    std::vector<float> rocm_out(static_cast<size_t>(p * n), kPoison);
    void* d_a = gpu.Alloc(a.size() * sizeof(float));
    void* d_w = gpu.Alloc(wq.size());
    void* d_e = gpu.Alloc(ids.size() * sizeof(int32_t));
    void* d_o = gpu.Alloc(rocm_out.size() * sizeof(float));
    gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
    gpu.Copy(gq, d_w, wq.data(), wq.size());
    gpu.Copy(gq, d_e, ids.data(), ids.size() * sizeof(int32_t));
    gpu.Copy(gq, d_o, rocm_out.data(), rocm_out.size() * sizeof(float));
    Tensor at = DevTensor(d_a, DType::kF32, {p, k});
    Tensor wt = DevTensor(d_w, c.dtype, {e * n, k});
    Tensor et = DevTensor(d_e, DType::kI32, {p});
    Tensor ot = DevTensor(d_o, DType::kF32, {p, n});
    vt::MatmulBTQuantGrouped(gq, ot, at, wt, et);
    gpu.Copy(gq, rocm_out.data(), d_o, rocm_out.size() * sizeof(float));
    gpu.Synchronize(gq);
    gpu.Free(d_a);
    gpu.Free(d_w);
    gpu.Free(d_e);
    gpu.Free(d_o);

    CAPTURE(PoisonSurvivors(rocm_out));
    REQUIRE(PoisonSurvivors(rocm_out) == 0);

    double num = 0, den = 0;
    for (size_t i = 0; i < rocm_out.size(); ++i) {
      REQUIRE(std::isfinite(rocm_out[i]));
      const double d = rocm_out[i] - cpu_out[i];
      num += d * d;
      den += static_cast<double>(cpu_out[i]) * cpu_out[i];
    }
    const double nmse = den > 0 ? num / den : num;
    CAPTURE(nmse);
    CHECK(nmse <= kMaxNmseVsCpu);
  }
  gpu.DestroyQueue(gq);
}
