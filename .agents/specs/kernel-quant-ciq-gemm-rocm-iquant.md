# Spec: KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT

- Issue: [#1940](https://github.com/mudler/vllm.cpp/issues/1940)
- Row: `KERNEL-QUANT-CIQ-GEMM-ROCM-IQUANT` — the I-quant coverage half of the
  ROCm keep-quant GEMM. Sibling, not overlapping, of
  `KERNEL-QUANT-CIQ-GEMM-ROCM` (the RDNA4 WMMA tensor-core tile for the
  four formats already ported: Q8_0/Q4_K/Q5_K/Q6_K). That row's own scope
  section names this gap and explicitly declines it: "Q4_0/Q2_K/Q3_K/IQ2_*/
  IQ3_*/MXFP4 keep-quant formats. Unrelated gap (#1940), untouched by this
  row." Owning feature row: `BACKEND-ROCM` (#41).
- Base: `bcade48d6f7e6666f88ffaaf3d2b78af24c35d7d` (`upstream/main`,
  2026-09-05).
- Pull request shape: separate spec and implementation pull requests
  (developer decision 2026-09-05, recorded in
  `.agents/developer-preferences.md`). This pull request lands the spec only.

## Scope

`src/vt/rocm/rocm_grouped_gemm.hip` ports exactly four keep-quant formats:
Q8_0, Q4_K, Q5_K, Q6_K (`DotQ8_0`/`DotQ4K`/`DotQ5K`/`DotQ6K`, lines
209-323). Every other block dtype the GGUF reader recognizes still expands to
bf16 before reaching ROCm compute, because `DeviceKeepQuantSupported`'s
`kROCM` arm (`gguf_keep_quant.cpp:136-145`) only admits those four. This is
the crash mode a real checkpoint hits today: an IQ4_XS or IQ3_XXS tensor on a
ROCm run expands to full bf16 in host RAM at load time rather than staying
compressed, which is large enough to exhaust host RAM on a consumer box
(measured: a 35B-A3B MoE checkpoint's IQ4_XS tensors alone would expand
19.4 GB on disk to ~70 GB resident — see the "Motivating case" below).

This row ports the **scalar dot kernel and device admission** for two
formats, in this order:

1. **IQ4_XS** (ggml id 23, `BlockIQ4_XS`, `cpu_quant_blocks.h:228-234`,
   136-byte block) — first, because it is the format actually blocking the
   motivating checkpoint's expert-down projections.
2. **IQ3_XXS** (ggml id 18, `BlockIQ3_XXS`, `cpu_quant_blocks.h:163-167`,
   98-byte block) — second, because its dot body is a clean single
   integer accumulator (see Design) and de-risks the harness before IQ4_XS's
   trickier float-accumulation body.

Both share the `kQK_K` = 256-element super-block and the `BlockQ8_K`
activation encoding already resident on this ROCm path (same activation
quantizer the four ported formats use — `QuantizeQ8KKernel`, no new
activation kernel needed).

Out of scope, and not attempted in this row:

- **The other five owed ROCm formats** named in the same refusal message —
  Q4_0, Q2_K, Q3_K, IQ2_XXS, IQ2_S, MXFP4. Real gaps, tracked by the same
  issue #1940, left for a follow-on row once this row's harness (test
  structure, oracle wiring, nwarps re-measurement recipe) exists to extend
  rather than invent twice.
- **IQ1_S, IQ1_M, IQ1_XXXS, IQ2_XS, IQ3_S, IQ4_NL.** Not requested by the
  motivating case; #1940 already separates these into "needs a fresh port"
  bucket distinct from the five above.
- **Any WMMA/tensor-core tile for these two formats.** `KERNEL-QUANT-CIQ-GEMM-ROCM`
  established the tensor-core lever for the four existing k-quants as a
  follow-on to their scalar tier, not a precondition of it. This row lands
  the scalar `dp4a`-tier equivalent for IQ4_XS/IQ3_XXS first, matching that
  same order; a tensor-core tile for codebook formats is a separate,
  harder problem (grid lookups do not vectorize into a WMMA int8 tile the
  way a linear-scale/min k-quant does) and is explicitly owed, not attempted,
  here.
- **ROCm device-fit / residency sizing** (`--fit`, `VT_DEVICE_WEIGHT_BUDGET_BYTES`,
  `GgufExpertTowersReachSlotLane`). That is `BACKEND-ROCM-IQ-EXPERT-RESIDENCY`,
  already merged for the formats it covers. This row is the compute
  prerequisite that lets keep-quant residency apply to IQ4_XS/IQ3_XXS
  tensors on ROCm at all; it does not touch the fit/budget logic itself.
- **The nwarps=8 decode-scaling table** (`ROCM-KQUANT-NWARPS-DECODE`,
  merged). Reused as-is for the initial port (both formats' `nsb = K/256`
  decomposition matches the existing four); see Risks for why it may not
  transfer and is re-measured rather than assumed.

### Motivating case

`joral`'s ROCm box (RX 9060 XT, gfx1200, 16 GiB VRAM, 62 GiB host RAM)
SIGSEGVs loading a 35B-A3B MoE checkpoint at IQ4_XS: `DeviceKeepQuantSupported`
routes all IQ4_XS/IQ3_XXS/IQ1_S/Q2_K/Q3_K towers to `kExpandBf16`, and the
19.4 GB on-disk IQ4_XS tensors alone expand past what a streamed-expert lane
can hold in 62 GiB host RAM — a silent host-memory SIGSEGV, not a clean
refusal. `Q5_K_M` is the only currently-safe quant family on that box. This
row's IQ4_XS port is the fix for that specific crash; IQ3_XXS is included
because the same checkpoint family typically pairs IQ4_XS gate/up projections
with IQ3_XXS or IQ2_XXS down projections (the pattern already documented for
`unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_XXS` in
`KERNEL-QUANT-CIQ-IQUANT`/`quantization-matrix.md`).

## What already exists to port from

Both formats are **already ported on CPU and CUDA** — issue #1940's own
table under-reports this for IQ4_XS (it lists IQ4_XS among formats "CPU and
CUDA do not port," which is stale; `cuda_quant_dot.cu:650` (`DotIQ4XS`) and
`cpu_quant_dot.cpp` (`VecDotIQ4_XSQ8_K`) both exist and are wired into
`IsCudaKeepQuantSupported`/`HasQuantDotKernel` today). This row is a **third
backend adaptation of an existing algorithm**, not a fresh port from the
llama.cpp oracle:

| Format | CUDA dot (adapt from) | CPU dot (cross-check oracle) | llama.cpp source |
|---|---|---|---|
| `IQ4_XS` | `DotIQ4XS`, `cuda_quant_dot.cu:650` | `VecDotIQ4_XSQ8_K`, `cpu_quant_dot.cpp:844` | `quants.c:1283` `ggml_vec_dot_iq4_xs_q8_K_generic` |
| `IQ3_XXS` | `DotIQ3XXS`, `cuda_quant_dot.cu:429` | `VecDotIQ3_XXSQ8_K`, `cpu_quant_dot.cpp` | `quants.c:999` `ggml_vec_dot_iq3_xxs_q8_K_generic` |

The CUDA bodies are the ones to adapt line-for-line into
`rocm_grouped_gemm.hip`'s existing `__device__ inline float Dot*(const
Block*, const BlockQ8_K*)` shape (matching `DotQ4K`/`DotQ5K`/`DotQ6K`'s
signature), because CUDA already carries the oracle-verified accumulation
order (see Risks) that a fresh transcription from the CPU generic body could
silently reassociate.

`IQ3_XXS` additionally needs the `d_iq3xxs_grid`/`d_ksigns_iq2xs`/
`d_kmask_iq2xs` device tables. CUDA generates these into
`cuda_quant_iq_tables.cuh`; this row generates the equivalent HIP device
tables the same way (same source grids from `cpu_quant_iq_tables.h`, ported
generator, not hand-copied — a wrong grid still decodes plausibly, per the
existing FNV-1a table-seal precedent in `test_ops_quant_dot.cpp:717`).
`IQ4_XS` needs only `d_kvalues_iq4nl` (16 entries, already small enough to
inline as a `__constant__`/`__device__` array directly, matching CUDA's
approach).

## Design

Add `DotIQ4XS`/`DotIQ3XXS` to `rocm_grouped_gemm.hip` alongside
`DotQ4K`/`DotQ5K`/`DotQ6K` (lines ~253-323), same `__device__ inline float
Dot*(const Block*, const BlockQ8_K*)` signature, dispatched through the same
`nsb = K / 256` per-warp superblock loop the four existing formats already
use (`KQuantGemmK`/`GroupedKQ8K`). No new activation quantizer, no new
launch-site shape — this is a dispatch-table extension, not a new kernel
family.

Wire the two new dtypes into:

- `DeviceKeepQuantSupported`'s `kROCM` arm (`gguf_keep_quant.cpp:136-145`) —
  add `dt == vt::DType::kIQ4_XS || dt == vt::DType::kIQ3_XXS`.
- The two refusal-message switches in `rocm_grouped_gemm.hip` (`:889`,
  `:959`) — remove both from the "owed" list in the message text, add
  dispatch cases.
- Whatever grouped/plain GEMM dtype switch selects `DotQ4K` etc. today
  (mirror the CUDA `WType` enum shape if ROCm has an equivalent, otherwise
  the existing block-dtype `switch`).

## Upstream anchor

llama.cpp, pin `b10451` per `.agents/upstream-sync.md`.

- `ggml/src/ggml-cpu/quants.c:1283` `ggml_vec_dot_iq4_xs_q8_K_generic`.
- `ggml/src/ggml-cpu/quants.c:999` `ggml_vec_dot_iq3_xxs_q8_K_generic`.
- `ggml/src/ggml-common.h:454-460` `block_iq4_xs`, `:385-400` `block_iq3_xxs`.
- `ggml/src/ggml-cuda/mmvq.cu:387-388` — the comment this issue's title
  refers to: llama.cpp's own `nwarps`-scaling table excludes `Q3_K` and the
  `IQ2`/`IQ3` families by name because their vec_dot does a grid lookup,
  citing register pressure and lookup-table contention at higher thread
  counts. Whether that split holds for our own dot bodies is unmeasured
  (see Risks); IQ4_XS's dot has no grid lookup (a 16-entry codebook fits a
  register array) and is not implicated by that comment.

## Risks

- **FMA contraction on IQ4_XS's float-accumulation body.** IQ4_XS's dot is
  the one format in this row (and in the whole quant-dot family) whose core
  is not a single integer accumulator: it forms `d1`/`d2` as f32 and folds
  in per-sub-block `sumf +=` steps, eight per super-block
  (`cuda_quant_dot.cu:606-680`, extensively commented on exactly this
  point). On CUDA that required `__fmul_rn`/`__fadd_rn` in place of ordinary
  `*`/`+`, because nvcc's default `-fmad=true` silently contracts the
  textual two-rounding sequence into a single-rounding FMA and two of eight
  real super-blocks then disagreed with the oracle by 1-4 ULP. **This may
  not reproduce on ROCm**: `CMakeLists.txt:414` already applies
  `-ffp-contract=off` to `$<COMPILE_LANGUAGE:HIP>` project-wide, unlike CUDA
  where the project's `-ffp-contract=off` is CXX-only and never reaches
  `.cu`/`.cuh` translation units. Verify this empirically before assuming it
  (a W0-style probe: compile the naive `sumf += d1 * x` form, diff against
  the CPU oracle on the same real super-blocks CUDA's golden vectors use,
  and inspect the generated ISA for `v_fma_f32` if any block disagrees) —
  do not carry the CUDA workaround over unexamined, and do not assume the
  flag alone is sufficient without a measured check, matching how the CUDA
  side only added the intrinsics after measuring a real disagreement rather
  than as a precaution.
- **The nwarps=8 decode table (`ROCM-KQUANT-NWARPS-DECODE`) may not transfer.**
  Both new formats share the existing `nsb = K/256` decomposition, so they
  compile against the same launch shape as Q4_K/Q5_K/Q6_K with no code
  change required to use it. Whether `nwarps=8` is still the right choice
  for IQ4_XS/IQ3_XXS's different per-lane work (a 16-entry codebook lookup
  per element for IQ4_XS; a grid-table lookup plus sign unpacking for
  IQ3_XXS) is exactly the open question issue #1940 was filed to test, per
  llama.cpp's own upstream exclusion of these families from its analogous
  table (`mmvq.cu:387-388` above). Re-measure rather than inherit the
  existing table's choice; a regression here is a decode-throughput
  question, not a correctness one, and does not block landing if the
  existing nwarps value is merely suboptimal rather than wrong.
- **IQ3_XXS's grid tables are shared-shape with sibling grids.** `kIq3xxsGrid`
  (256 u32) and other IQ-family grids in `cpu_quant_iq_tables.h` are the
  same element count as unrelated grids; the CPU/CUDA precedent guards this
  with an FNV-1a seal specifically because a decoder pointed at the wrong
  same-shaped grid still decodes plausibly. Generate the HIP device table
  the same way CUDA's is generated (from the single source-of-truth grid),
  and reuse or extend the seal test rather than hand-transcribing constants
  into the `.hip` file.
- **Interaction with `BACKEND-ROCM-IQ-EXPERT-RESIDENCY`'s already-merged
  fit logic.** That row's `GgufExpertTowersReachSlotLane` and the ROCm
  device-fit budget were built and measured against a world where IQ4_XS/
  IQ3_XXS towers still expand to bf16 on ROCm. Landing keep-quant compute
  for them changes their resident size (compressed, not bf16-expanded) and
  therefore changes what `--fit` decides — re-run that row's device-fit
  gate after this lands rather than assuming its prior sizing still holds;
  flag a regression there as `NEEDS_DECISION` rather than silently
  reconciling it inside this row.

## Tests

- Extend `test_ops_quant_dot.cpp`'s existing IQ4_XS/IQ3_XXS `vec_dot`
  golden-vector gates (`iq2xs_iq4xs_dot_golden.h`, already committed and
  sourced from real `unsloth/GLM-5.3-Flash-GGUF` checkpoint bytes) to a new
  `test_rocm_quant_dot.cpp`, same shape as the CUDA gate
  (`test_cuda_quant_dot.cpp`): NMSE ≤ 5e-4 vs the independent f64
  dequant-then-dot reference for IQ3_XXS; bit-exact (not NMSE) for IQ4_XS
  against the same real-checkpoint golden values CUDA's gate uses, since
  bit-exactness is the property the FMA-contraction risk above is actually
  about.
- `test_backend_cross_device.cpp`: add both formats to the CPU-vs-ROCM
  cross-check, NMSE ≤ 5e-4 (matching the existing four formats' gate shape
  there).
- Rerun `ROCM-KQUANT-NWARPS-DECODE`'s own measurement recipe
  (`rocprofv3 --kernel-trace` on a real quant-matched trace workload) for
  IQ4_XS/IQ3_XXS specifically, to answer the nwarps question this issue was
  filed to test — record the result (transfers / does not transfer) rather
  than assuming either.
- `ctest -R 'rocm|cross_device'`, zero regression on the four existing
  formats' numerics.
- End-to-end: reload the motivating checkpoint (or a same-format synthetic
  fixture if the real 35B-A3B artifact is not staged on the gate host) on
  `isravale` (RX 9060 XT, gfx1200) or an `rc`-leased ROCm fleet device, and
  confirm keep-quant residency replaces the prior bf16 SIGSEGV — this is
  the row's actual acceptance criterion, not merely the unit-level dot
  gates.

## Owed

- The other five formats named in the same ROCm refusal message (Q4_0,
  Q2_K, Q3_K, IQ2_XXS, IQ2_S, MXFP4): tracked by #1940, left for a
  follow-on row.
- A WMMA/tensor-core tile for IQ4_XS/IQ3_XXS, if the scalar tier's measured
  throughput warrants one (mirroring how `KERNEL-QUANT-CIQ-GEMM-ROCM`
  followed the existing four formats' scalar tier): not attempted here.
- The nwarps re-measurement itself, if it is not completed within this
  row's implementation wave for lack of GPU time: record as `PENDING` on a
  named lease/box, never silently dropped.

## Stop conditions

- `NEEDS_DECISION`: any request to extend this row's scope to the other
  five owed ROCm formats, to MXFP4, or to a tensor-core tile for either
  format.
- `NEEDS_DECISION`: if the FMA-contraction probe (Risks) shows
  `-ffp-contract=off` does *not* fully protect IQ4_XS's accumulation on
  this toolchain, before deciding whether to port CUDA's `__fmul_rn`-style
  workaround (HIP's equivalent, if one exists) or to gate the format at
  NMSE instead of bit-exactness the way Q4_K/Q5_K/Q6_K already are.
- 20 failed attempts within the implementation wave: stop, report the
  measured position and the next traceable hypothesis.

## Now

`SPIKE`. This pull request lands the spec only; no product code changes in
this change. Next: W0 probes the FMA-contraction question on target
hardware (gfx1200), then W1 ports `DotIQ4XS` (the harder, float-accumulation
body) and W2 ports `DotIQ3XXS`, each with its own focused gate before the
combined `ctest` sweep.
