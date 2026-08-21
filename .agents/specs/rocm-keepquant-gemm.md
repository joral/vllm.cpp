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

`ACTIVE`. The kernel is written and gated on gfx1200: `test_rocm_quant_dot`
**3/3 - 53157/53157**, red-first captured, mutation-proven, and the model-level
arm measured. Next action: the MoE-model arm (§Tests step 6) and a decode-only
speed measurement, because §Outcome retracts this row's own wall-time figure.

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
| HIP | `auto` | off (auto) | 12159 MiB | 37.1 s (COLD, see below) |
| HIP | `auto` + `VT_GGUF_KEEP_QUANT=1` | forced on | — | `no kernel for op MatmulBTQuant (id 75) on device rocm (type 5)` (`op_provider.cpp:518`) |

Identical tokens in the first two rows. The memory figure is **1.73x peak RSS**
and it holds. **The 37.1 s does NOT**: it was the first read of that file on this
host, so it measured page-cache misses rather than dequantization, and the warm
A/B in `## Outcome` reverses its direction. Retracted here rather than deleted,
because the number reached issue #1506 before it was checked.

The load-time cliff is what makes this row worth doing and it is untouched by
that retraction: a ~16 GB k-quant expands to roughly 55-60 GB and does not fit on
a 64 GB host that runs the same file on CPU.

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

## Our baseline

What this tree does on ROCm before the kernel, measured rather than read
(gfx1200 / ROCm 7.2.3, `Qwen3.5-4B-Q4_K_M.gguf`, `examples/vllm-cli`):

- `kMatmulBTQuant` registered on `kCPU` (`cpu_quant_gemm.cpp:303`) and `kCUDA`
  (`cuda_quant_dot.cu:1991`) only; `src/vt/rocm/` has 18 `.hip` files and no
  keep-quant GEMM.
- `keep_quant` therefore defaults OFF on `kROCM`, and every quantized weight is
  expanded to bf16 at load: **12159 MiB peak RSS**, against the CPU build's
  7038 MiB with keep-quant on. Tokens identical.
- Forcing `VT_GGUF_KEEP_QUANT=1` past the probe: `no kernel for op
  MatmulBTQuant (id 78) on device rocm (type 5)` (`op_provider.cpp:518`).
- The Q8_K ACTIVATION quantizer already exists on ROCm and has been tuned
  (`6251de146`), so only the weight side is missing.

## Upstream chain

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
MatmulBTQuantGrouped on device rocm`. This is
[#1029](https://github.com/mudler/vllm.cpp/issues/1029)'s shape exactly: "past
that predicate there is no fallback left."

**CORRECTION (measured at W7).** An earlier draft of this section, and the
commit that landed the kernel, called that "a NEW failure on models that load
today". On ROCm that is **overstated**. A Qwen3.5/3.6 MoE GGUF does not run on a
discrete AMD card at all: it dies earlier, at `kSharedExpertGate`, which is
registered on `kCPU` and nowhere else, and which the reference tier may not
cover because this device's memory is not host-addressable
([#1590](https://github.com/mudler/vllm.cpp/issues/1590)). So the grouped op
protects a path that is currently UNREACHABLE on this backend rather than one
that currently works. It still ships here, because it becomes required the
moment #1590 is fixed and because leaving a predicate with no arm behind it is
the #1029 defect regardless of who reaches it first. The reasoning was right;
the claim about today's behaviour was not.

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

## Port map

| Source | Destination | Note |
|---|---|---|
| `cuda_quant_dot.cu:578` `DotQ4K` | `rocm_quant_dot.hip` `DotQ4K` | line-for-line; `__dp4a` -> `DDp4a` |
| `:620` `DotQ5K`, `:530` `DotQ3K`, `:662` `DotQ6K` | same names | line-for-line |
| `:176` `QuantizeQ8KKernel` | same name | unchanged; one thread per super-block |
| `:774` `QuantDotGemmKernel` | same name | `__shfl_down_sync` -> `__shfl_down`, wave32 |
| `:814` `QuantDotGemmGroupedKernel` | same name | same substitution |
| `:1588` `IsCudaKeepQuantSupported` | `IsRocmKeepQuantSupported` | four types, not eight |
| `:1835`/`:1928` CPU fallback | `FallbackToCpu`/`FallbackToCpuGrouped` | REWRITTEN: stages through host memory (see §Design) |
| `:1991` `Registrar` | `Registrar` | `kCUDA` -> `kROCM`, both ops |
| `cpu_quant_blocks.h` block structs | included as-is | the encoding is shared, not re-declared |

No new codebook tables: the four k-quants carry their scales in-block, so unlike
the IQ family there is nothing to vendor.

## Dependencies

- `vt::cpu::BlockQ{3,4,5,6}_K` / `BlockQ8_K` (`src/vt/cpu/cpu_quant_blocks.h`) —
  shared block layouts, included rather than re-declared.
- The CPU `kMatmulBTQuant` / `kMatmulBTQuantGrouped` providers — both the
  correctness oracle and the delegation target for unserved dtypes.
- `RegisterOp` / `GetOp` (`vt::OpProvider`) — the seam the flip rides.
- ROCm >= 6.1 with `__builtin_amdgcn_sudot4` (RDNA3/4) or `sdot4`
  (CDNA/RDNA2/gfx906). Verified on ROCm 7.2.3 / hipClang 22.
- NOT a dependency: `hipBLASLt`. This kernel does its own integer dot.

## Work breakdown

| # | Slice | State |
|---|---|---|
| W1 | Red-first gate: three cases, poisoned buffers, over the CPU oracle | DONE — RED captured, then 3/3 - 53157 |
| W2 | `rocm_quant_dot.hip`: four dots, Q8_K quant, dense + grouped GEMM, wave32 guard | DONE |
| W3 | `IsRocmKeepQuantSupported` + host-staged CPU delegation + throwing `default:` | DONE |
| W4 | CMake registration; the flip of `GgufQuantComputeAvailable` on `kROCM` | DONE |
| W5 | Model-level arm: tokens + peak RSS on gfx1200, and the `VT_GGUF_KEEP_QUANT=0` byte-identity control | DONE |
| W6 | `ctest -R 'rocm\|cross_device'` no-regression | DONE — 5/6; the 1 failure is pre-existing #1513, control-proven |
| W7 | MoE-model arm (the grouped op in a real GGUF) | BLOCKED by #1590, not owed to this row |
| W8 | Decode-only speed measurement, to attribute §Outcome's 1.8x | OWED |

## Tests to port

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
- **~~`__dp4a` may not lower to the instruction on gfx1200.~~ RESOLVED, and the
  premise was wrong twice.** HIP does not declare `__dp4a` at all (ROCm 7.2.3 /
  hipClang 22: "use of undeclared identifier"), and the AMD builtin SPLITS by
  architecture — RDNA3/RDNA4 take `__builtin_amdgcn_sudot4`, CDNA/RDNA2/gfx906
  take `__builtin_amdgcn_sdot4`. With the right one the device assembly carries
  **576 `v_dot4_i32_iu8`**, so it is the real instruction and the speed axis is
  addressable.
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

## Outcome

Recorded at implementation, per `AGENTS.md` §"Spec before code" (an `## Outcome`
records what was measured, what was rejected, and why each default has its
value).

### What was measured

`test_rocm_quant_dot` on gfx1200 / ROCm 7.2.3: **3/3 cases, 53157/53157
assertions**. RED first — before `rocm_quant_dot.hip` existed all three cases
threw `no kernel for op MatmulBTQuant (id 78) on device rocm (type 5)`.

Mutation-proven rather than merely green: dropping the per-sub-block scale in
`DotQ4K` (`isum += scale * sub` -> `isum += sub`) fails **25 assertions** across
the dense and grouped Q4_K arms at NMSE 1.09 against the 1e-6 band, and the tree
was restored byte-for-byte (sha256 `4ffc3a34d034ef75...`) with 53157/53157
green after.

Model level, `Qwen3.5-4B-Q4_K_M.gguf`, warm, three runs per arm, same binary:

| arm | peak RSS | wall |
|---|---|---|
| `keep_quant` ON (this kernel) | **6066-6074 MiB** | 3.169 / 3.182 / 3.279 s |
| `keep_quant` OFF (expand bf16) | 12152 MiB | 1.769 / 1.770 / 1.830 s |

Tokens identical in both arms and identical to the CPU build (" Paris.\nA").

- **G1** unit correctness: PASS, mutation-proven.
- **G2** loader byte-identity: PASS — `VT_GGUF_KEEP_QUANT=0` reproduces 12152
  MiB and the same tokens, so the flip is the only difference.
- **G3** model tokens: PASS.
- **G4** memory: **PASS with margin** — 12159 -> 6066 MiB is 2.0x, and below the
  CPU build's 7038 MiB, which the gate only required matching.
- **G5** no regression: **PASS.** `ctest -R 'rocm|cross_device'` on gfx1200 is
  5/6. The one failure is `test_backend_cross_device:2063`, the bf16 arm of
  `kMoeSiluMul`, and it is PRE-EXISTING with an empirical control: removing
  `rocm_quant_dot.hip` from the build entirely reproduces the identical
  failure, same line, same 359/360. Filed as
  [#1513](https://github.com/mudler/vllm.cpp/issues/1513) against `BACKEND-ROCM`
  rather than carried silently.
- **G6** speed: report-only, and the report is UNFAVOURABLE — see next.

### The row retracts its own headline number

The issue and the first draft of this spec claimed the expanded path costs
`4.5x wall`. **That was cold page-cache**, measured on the first-ever read of
that file on this host. The warm A/B above puts keep-quant at about **1.8x MORE
wall time**, not less. Corrected in issue #1506 (comment), and left visible here
rather than quietly edited.

The direction is not yet attributed and this row does not attribute it. The
wall figure is dominated by load plus a 5-token prompt and isolates no decode;
the expanded arm hands its GEMMs to a tuned hipBLASLt while this kernel is a
wave-per-output MMVQ shaped for M=1. `AGENTS.md` §Gates forbids trading
correctness for throughput and this row's G6 was made report-only precisely so a
speed result could not be annexed from #487/#1294 — that clause is what caught
this.

**The case for the row is unchanged, because it was never speed.** Halving
resident weights is what makes a ~16 GB k-quant loadable on a 64 GB host at all,
and G4 delivers that with margin.

### Why the defaults have their values

- **Four types, not eight.** A 43-GGUF census on the gate box reaches Q4_K,
  Q6_K, Q5_K and Q3_K and no others. Everything else delegates and stays
  correct.
- **`__builtin_amdgcn_sudot4`, not `__dp4a`.** HIP does not declare `__dp4a`;
  the AMD builtin splits RDNA3/4 (`sudot4`) from CDNA/RDNA2/gfx906 (`sdot4`).
  576 `v_dot4_i32_iu8` in the gfx1200 device assembly confirms the instruction.
- **The CPU fallback stages through host memory.** The CUDA provider passes
  device pointers straight to the CPU op because GB10 is unified. A discrete AMD
  card is not, and since `VT-REFTIER-HOST-ADDRESSABLE` the reference tier says
  so by name instead of segfaulting. Found by running the red-first test, not by
  reading.
- **Wave32 is checked, not assumed.** `RequireWave32` refuses a wave64 device by
  name rather than reducing over the wrong lane set.

## Owed (added at implementation)

- **§Tests step 6, the MoE-model arm: BLOCKED, not owed.** Attempted at W7 on
  `Qwen3.6-14B-A3B-VibeForged-v2-Q4_K_M.gguf` (arch `qwen35moe`, 8.5 GB, fits
  the 15.9 GiB card). It cannot run: `no kernel for op SharedExpertGate (id 67)
  on device rocm`, identically with `VT_GGUF_KEEP_QUANT=0`, so it is independent
  of this row. Filed as [#1590](https://github.com/mudler/vllm.cpp/issues/1590)
  against `BACKEND-ROCM`. The grouped op therefore stands on its unit gate
  alone, and this row says so rather than implying production coverage it does
  not have. No other vehicle was available: the corpus MoE alternatives are the
  same `qwen35moe` family, a `deepseek2` arch the GGUF dispatch does not handle,
  or too large for 15.9 GiB VRAM.
- **A decode-only speed measurement**, one tool, isolating decode from load, to
  attribute the 1.8x above. Until it exists this row asserts nothing about
  throughput.
