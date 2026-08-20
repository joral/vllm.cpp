# ROCm keep-quant GGUF k-quant GEMM — `KERNEL-QUANT-CIQ-GEMM-ROCM`

Issue: [#1506](https://github.com/mudler/vllm.cpp/issues/1506).
Row: `KERNEL-QUANT-CIQ-GEMM-ROCM` ([kernel matrix](../kernel-matrix.md)).
Base SHA: `cffe59b02e86d347dbf22d033d3e15e552cc3aba` (`upstream/main`, 2026-08-20).
Pull request shape: **one PR** for spec and implementation, recorded at row
claim in [developer preferences](../developer-preferences.md).

The ROCm sibling of [`KERNEL-QUANT-CIQ-GEMM-CUDA`](cuda-keepquant-gemm.md). That
row's landing contract — "registering it flips `GgufQuantComputeAvailable` TRUE
on `kCUDA`" — is the same contract here, and §Design records why that flip is
the dangerous part rather than the goal.

## Now

`SPIKE`. Spec committed; no kernel written. Next action: the red-first per-type
unit gate over a poisoned output buffer (§Tests, step 1), on gfx1200.

## The gap, measured

`vt::OpId::kMatmulBTQuant` is registered on `kCPU`
(`src/vt/cpu/cpu_quant_gemm.cpp:303`) and `kCUDA`
(`src/vt/cuda/cuda_quant_dot.cu:1991`), and on nothing else. `src/vt/rocm/`
holds 18 `.hip` files and none mentions `MatmulBTQuant`.

The visible effect is not a refusal. The chain:

1. `GgufQuantComputeAvailable()` (`gguf_keep_quant.cpp:75`) probes
   `OpRegistered(kMatmulBTQuant, CurrentPlatform().device_type())`.
2. `GgufLoadPolicy::FromEnv` (`:169`) takes that as the `keep_quant` default.
3. `RouteGgufTensor` (`:141`) returns `kExpandBf16` for every quantized weight
   when `keep_quant` is false. `p.expand_nk` (`:172`) rides the same flag.

So a HIP build dequantizes the whole checkpoint at load, silently.

Measured 2026-08-20, gfx1200 (RDNA4), ROCm 7.2.3, `Qwen3.5-4B-Q4_K_M.gguf`
(2.74 GB, arch `qwen35`), same prompt, `--max-tokens 4`, through
`examples/vllm-cli`. Peak RSS from `/proc/<pid>/status` `VmHWM`.

| build | device | `keep_quant` | peak RSS | wall |
|---|---|---|---|---|
| CPU | `cpu` | on (auto) | 7038 MiB | 8.2 s |
| HIP | `auto` | off (auto) | 12159 MiB | 37.1 s |
| HIP | `auto` + `VT_GGUF_KEEP_QUANT=1` | forced on | — | `no kernel for op MatmulBTQuant (id 75) on device rocm (type 5)` (`op_provider.cpp:518`) |

Identical tokens in the first two rows. **1.73x peak RSS, 4.5x wall.** The cost
is a load-time cliff at scale, not a slowdown: a ~16 GB k-quant expands to
roughly 55-60 GB and does not fit on a 64 GB host that runs the same file on
CPU. Today the HIP build is strictly worse than the CPU build for GGUF.

## Scope

Register `kMatmulBTQuant` **and** `kMatmulBTQuantGrouped` on `kROCM`, with a
native wave32 kernel for **Q4_K, Q6_K, Q3_K, Q5_K** and a CPU delegation for
every other block dtype.

Four types, not the CUDA row's eight, because those four are what the
checkpoints on the gate box actually contain: a census of 43 local GGUFs found
Q4_K (2327 tensors), Q6_K (889), Q5_K (256) and Q3_K (180) alongside already-CPU
types, and no reaching checkpoint for IQ2_S/IQ1_S/MXFP4 on this host. Landing a
kernel arm nothing exercises is the shape `AGENTS.md` §"Nothing lands dead"
refuses.

Out of scope, listed under `## Owed`: CUDA-parity type coverage, wave64/CDNA,
a native grouped kernel, and the Q8_0-activation legacy types.

## Upstream anchors

vLLM has no GGUF k-quant GEMM, so per `AGENTS.md` §"When vLLM has no
implementation" the secondary oracle is **llama.cpp** (registry id `llama-cpp`,
pinned `b10451`, [oracle file](../oracles/llama-cpp.md)).

- `ggml/src/ggml-cuda/mmvq.cu` — the MMVQ shape this kernel mirrors.
- `ggml/src/ggml-cuda/vecdotq.cuh` — `vec_dot_q4_K_q8_1` and siblings.
- `ggml/src/ggml-hip/CMakeLists.txt:63` — the HIP build globs
  `../ggml-cuda/*.cu`; there is no separate ROCm kernel tree upstream.
- `ggml/src/ggml-cuda/common.cuh:733` — `__dp4a` wrapped with a HIP/GCN-asm
  branch and a scalar fallback.

In-tree port source: `src/vt/cuda/cuda_quant_dot.cu` (`DotQ4K` `:578`,
`DotSuperblock` `:719`, `IsCudaKeepQuantSupported` `:1588`, registration
`:1991`). 52 `__dp4a`, 3 `__vsub4`, one `__shfl_sync`; no tensor cores, no PTX,
no async copy.

**Oracle caveat, stated rather than assumed.** The `llama-cpp` oracle records
`gateable = no` and names [#857](https://github.com/mudler/vllm.cpp/issues/857)
as owing the build-and-run measurement. This row does not need that measurement
to gate correctness, because its primary reference is the in-tree CPU
`kMatmulBTQuant` on the identical bytes — an oracle that demonstrably runs here.
llama.cpp is used for the ENCODING semantics only.

## Design

### The flip is the risk, not the feature

Registering `kMatmulBTQuant` on `kROCM` changes the load path for **every** GGUF
on **every** ROCm device, because the probe is global and the policy is
load-time. Two consequences drive the design.

**(a) The grouped op must land in the same change.** `GgufQuantComputeAvailable`
probes only `kMatmulBTQuant`, but `qwen3_5.cpp:6193` `KqGrouped` calls
`vt::MatmulBTQuantGrouped` and is **default-ON** (`VT_QWEN35_GROUPED_MOE`,
`:6167`) and not device-gated; `deepseek_v4.cpp:571,1428` do the same. Register
the dense op alone and the probe flips true, MoE GGUFs keep their blocks
compressed, and the first expert GEMM dies with `no kernel for op
MatmulBTQuantGrouped on device rocm` — a NEW failure on models that load today.
This is [#1029](https://github.com/mudler/vllm.cpp/issues/1029)'s shape exactly:
"past that predicate there is no fallback left."

**(b) Unsupported dtypes must delegate, not throw.** With `keep_quant` true, a
Q2_K or IQ2_XXS tensor also stays compressed and reaches the ROCm provider. The
CUDA provider's answer is a CPU delegation
(`cuda_quant_dot.cu:1835` dense, `:1928` grouped) guarded by
`IsCudaKeepQuantSupported`, and its comment records the reason: an unsupported
type "falls to the CPU arm below and still emits CORRECT tokens, just at CPU
speed." Mirror it: `IsRocmKeepQuantSupported(DType, WType*)` returning false
routes to `GetOp(..., DeviceType::kCPU)`.

Both switches take a `default:` that **throws and names the dtype**. #1029
landed green precisely because a missing case silently launched nothing and
`cudaGetLastError()` reported success.

### Kernel

One `src/vt/rocm/rocm_quant_dot.hip`, structured as the CUDA TU is: a
`DotSuperblock<WType>` template specialised per block type, over a Q8_K
activation, warp-per-output-row.

- **Wave width.** gfx1200 is wave32, so the CUDA lane algebra maps directly.
  `__shfl_sync`'s mask argument has no HIP analogue; use `__shfl_xor` with an
  explicit width. Guard the TU on wave32 and refuse wave64 by name rather than
  computing a wrong reduction on CDNA — see `## Owed`.
- **`__dp4a`.** HIP exposes it; gfx1200 has `v_dot4_i32_iu8`. Verify it lowers
  to the instruction rather than a scalar expansion (`llvm-objdump` the cubin)
  and record which, because it decides whether the speed axis is even
  addressable. llama.cpp's `common.cuh:733` fallback is the model if it does
  not.
- **Activation.** Reuse the existing ROCm Q8_K quantizer rather than porting
  one — `6251de146` (`perf(rocm)`) already tuned it, and #1294 measures it at
  35% of ROCm decode GPU time. This kernel consumes its output.

### Not a byte-identity claim

The ROCm reduction order will differ from the CPU one, so this is a near-tie
gate against the CPU keep-quant path, not a bitwise one. What IS claimed
byte-exact is the loader: with `VT_GGUF_KEEP_QUANT=0` the ROCm path must remain
byte-identical to today's expanded-bf16 behaviour, which is the same-binary A/B
that makes the flip reviewable.

## Tests

Red-first, in this order. Each step states the mutation that must turn it red.

1. **Per-type dot, poisoned output.** For each of Q4_K/Q6_K/Q3_K/Q5_K: fill the
   output with a sentinel, run `vt::MatmulBTQuant` on a kROCM queue, compare
   against `vt::cpu::BlockToFloat` + f32 dot. RED first by running before the
   kernel exists. The sentinel is not decoration — it is the only thing that
   catches #1029's launch-nothing-and-return-success failure, which an
   all-zeros buffer hides.
   *Mutation:* delete one `DotSuperblock` specialisation; the case must fail,
   not fall through to a neighbour.
2. **Unsupported-dtype delegation.** A Q2_K tensor on a kROCM queue must
   produce CPU-equal output via the fallback, and the `default:` arm must throw
   with the dtype named for a dtype in neither set.
   *Mutation:* remove the `default:`; the throw test must fail.
3. **Grouped composition.** `vt::MatmulBTQuantGrouped` over a stacked
   `[E*N,K]` tower with real expert ids, poisoned output, CPU-compared. This is
   the arm #1029 shows is easy to leave unreached.
4. **Loader flip, same binary.** `GgufQuantComputeAvailable()` true on kROCM;
   and `VT_GGUF_KEEP_QUANT=0` reproduces today's peak RSS and tokens exactly.
5. **Model level, gfx1200.** `Qwen3.5-4B-Q4_K_M.gguf` through `vllm-cli`:
   tokens equal to the CPU build's, peak RSS at or below the CPU build's 7038
   MiB (from 12159).
6. **MoE model, gfx1200.** One MoE GGUF, to exercise step 3 in production and
   prove the (a) hazard is closed rather than argued.

## Gates

Correctness before speed, per `AGENTS.md` §Gates.

| # | Axis | Condition |
|---|---|---|
| G1 | Correctness, unit | Steps 1-3 green; every mutation red |
| G2 | Loader byte-identity | `VT_GGUF_KEEP_QUANT=0` tokens + peak RSS unchanged vs base SHA |
| G3 | Correctness, model | Step 5 tokens == CPU build, both gate files |
| G4 | Memory | HIP peak RSS <= CPU peak RSS on step 5 (12159 -> <=7038 MiB) |
| G5 | No regression | `ctest -R 'rocm\|cross_device'` green on gfx1200 |
| G6 | Speed | REPORTED, not gated, this row |

G6 is deliberately report-only. The decode-path speed question belongs to
[#487](https://github.com/mudler/vllm.cpp/issues/487) (M=1 matmuls landing on a
hipBLASLt 128x128 tile GEMM, 77% of decode GPU time) and
[#1294](https://github.com/mudler/vllm.cpp/issues/1294); this row must not
quietly annex them. Record the numbers, claim nothing.

## Risks

- **The flip is global.** Every ROCm GGUF load changes at once. G2 is the
  control that makes it reviewable; without it there is no same-binary A/B.
- **`__dp4a` may not lower to the instruction on gfx1200.** Then the kernel is
  scalar and the memory win lands while the speed win does not. Measured, not
  assumed — see §Design. Does not block G1-G5.
- **Wave64.** The reduction is wave32; a CDNA board would silently compute a
  wrong result if the guard is missing. Refuse by name.
- **Q8_K activation contention.** #1294 already puts the quantizer at 35% of
  decode GPU time; this kernel raises its call count. May make #1294 worse
  before it makes it better, which is the honest framing for that row's owner.
- **No CI.** This lands on hardware only one contributor has. G5 is the local
  ratchet; there is no upstream ROCm CI to catch a regression.

## Evidence

- Three-run measurement above, gfx1200 / ROCm 7.2.3, 2026-08-20, this worktree.
- Build report: `-DVLLM_CPP_HIP=ON -DVLLM_CPP_HIP_ARCHITECTURES=gfx1200`
  configures and builds `vllm-cli` clean (494 targets) at base SHA, in
  `nix develop .#rocm-shell`.
- Gap re-verified against `upstream/main` `cffe59b02` after finding local `main`
  86 commits behind: no `kMatmulBTQuant` kROCM registration, no quant file in
  `src/vt/rocm/`.

## Stop conditions

Stop and report rather than widening scope if any of these holds.

- G2 cannot be made to pass, i.e. the flip is not separable from a behaviour
  change on the expanded path. That is a loader-design finding and needs its own
  row.
- `__dp4a` lowers to a scalar expansion AND the measured memory win is under
  1.2x on step 5. The premise of the row would be wrong and it should be
  re-scoped before more kernel work.
- The grouped arm turns out to need a native kernel to be correct rather than
  merely slow. That is a second row, not a scope extension here.
- Any gate needs a value from `.env`, which does not exist. Ask for the single
  value; never infer a host or a path.

## Owed

Named here so they are visible debt rather than silence, per `AGENTS.md`
§"Nothing lands dead".

- **CUDA-parity type coverage** — Q2_K, IQ2_XXS, IQ3_XXS, IQ2_S on ROCm. They
  delegate to CPU after this row: correct, slow. Owed to this row; needs a
  reaching checkpoint on the gate box first.
- **Wave64 / CDNA.** This row is wave32-only and refuses wave64 by name.
- **Native grouped kernel.** The grouped arm delegates per-expert to CPU
  initially where the dense kernel does not apply.
- **Q8_0-activation legacy types** (Q4_0, Q8_0, MXFP4) — out of the Q8_K family
  on CUDA too, same disposition.
- **Async scheduling on ROCm.** The HIP build reports
  `max_concurrent_batches=1` where CPU reports `2`. Noticed while measuring,
  uninvestigated, recorded in #1506. Not this row's work.
