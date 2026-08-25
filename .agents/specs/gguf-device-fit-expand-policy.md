# GGUF-DEVICE-FIT-EXPAND-POLICY — the device-fit bound stops assuming the loader picks the cheaper residency

Issue: [#1870](https://github.com/mudler/vllm.cpp/issues/1870).
Owed, filed separately: [#1928](https://github.com/mudler/vllm.cpp/issues/1928)
(ROCm has no `kMoeGroupedGemmBf16` provider).
Base: `3574065e7eb6a968fc57928a28cf5fb59b748778` (`origin/main` at the claim).

## Scope

`CheckDeviceWeightFit` (`gguf_device_fit.h`, ENG-EXPERT-STREAM issue #1123) is
the load-time refusal that stops a GGUF whose weights cannot be staged onto the
target device from reaching the first forward and dying there instead. Its
per-tensor term, `StagedBytes`, is `min(gguf_bytes, elems * model_dtype_bytes)`
— the smaller of "kept quantized" and "expanded to the model dtype". The header
defends this as a lower bound: whichever the loader actually does, this is not
larger than it.

That defence has an unstated precondition: it assumes the loader is FREE to pick
the cheaper of the two for a given tensor. `RouteGgufTensor`
(`gguf_keep_quant.h`) says otherwise — it is a total decision over
`{keep_quant, keep_f16, nvfp4_fp4, cpu_ref}` and a tensor's role, and states
its own totality plainly: "Anything else is kExpandBf16 — the decision is
total and never throws." When an operator sets `VT_GGUF_KEEP_QUANT=0` on a
device where `keep_f16` and `nvfp4_fp4` are also off — every ROCm device
today, since `nvfp4_fp4` requires `kMatmulNvfp4`, CUDA-only, and `keep_f16`
rides `expand_nk`, which rides `keep_quant` — `RouteGgufTensor` returns
`kExpandBf16` for EVERY tensor, unconditionally. The bound still takes
`min(gguf_bytes, elems * 2)`, which for a compressed GGUF picks the on-disk
term nearly every time, so it reports a footprint roughly a quarter of what the
load will actually stage. On a 16 GiB ROCm card the refusal that exists
specifically to replace an allocator crash with a named message never fires,
and the load reaches `hipMalloc: out of memory` instead — reproduced on
`RX 9060 XT` (gfx1200, 16304 MiB) in #1870 on two checkpoints, one dense, one
MoE.

IN SCOPE:

- Make the footprint exact, not merely a safe lower bound, in the one case
  where `RouteGgufTensor`'s totality guarantee already makes the per-tensor
  answer known rather than assumed: every residency-shrinking flag off.
- Wire the one production call site (`model_loader.cpp`) to tell the bound
  when that condition holds, from the SAME resolved `GgufLoadPolicy` the
  loader already computes for the expert-stream lane check beside it.
- Document the precondition `docs/ENVIRONMENT.md:94` omits: the toggle needs
  roughly 4x the file size in device memory, and what refuses when it does not
  fit.

OUT OF SCOPE:

- Any per-tensor role classification inside `gguf_device_fit.cpp` for the
  MIXED case (some flags on, some tensors keep, some expand). That still needs
  role information this file does not have, is still a genuine approximation
  problem, and is unchanged by this row — see Risks.
- The `kMoeGroupedGemmBf16` ROCm provider gap #1870 names as a "related gap,
  same area". Filed as its own issue, #1928, and recorded under `## Owed`
  below rather than attempted here: the CUDA implementation is a single TU
  (`cuda_matmul_nvfp4.cu:1478-2732`) with a WMMA prefill path, a split-K
  decode path, persistent CUDA-graph-safe scratch, and a fused
  reduce+SwiGLU epilogue — a from-scratch HIP kernel port needing its own
  spec and hardware-gated evidence, not a slice of this one.
- Moving any residency DEFAULT. `keep_quant`, `keep_f16`, `nvfp4_fp4` availability
  rules are untouched; this row only makes the bound agree with what they
  already decide.

## Upstream chain

No upstream vLLM mirror. Pinned vLLM (`.agents/upstream-sync.md`, `555967922`)
has no GGUF load format, so there is no `DeviceConfig`/`memory_profiling`
counterpart that answers "will this fit before I pay for it" — the same header
comment `gguf_device_fit.h` opens with. This row edits a vllm.cpp-original
predicate against its own prior design, not an upstream port.

## Design

`RouteGgufTensor`'s switch is total: `kNvfp4Fp4` needs `nvfp4_fp4`, `kKeepQuant`
needs `keep_quant`, `kKeepF16` needs `keep_f16`, and every tensor that clears
none of those three gates is `kExpandBf16` — regardless of role, dtype or
shape. So the conjunction `!(keep_quant || keep_f16 || nvfp4_fp4)` is not a
heuristic; it is the exact condition under which the footprint can stop
guessing and charge the expanded size unconditionally, because that is the
ONLY residency `RouteGgufTensor` can produce.

`StagedBytes` and `GgufStagedWeightFootprint` gain one new parameter,
`policy_forces_full_expand` (default `false`), and `CheckDeviceWeightFit`
forwards it through. Default `false` is a byte-for-byte no-op for every
existing caller and test: unset, the function takes the same
`min(gguf_bytes, elems * model_dtype_bytes)` it always has. Set `true`,
`StagedBytes` returns the expanded term outright — no `min`, because there is
nothing left to be uncertain about.

The one production call site, `model_loader.cpp`'s GGUF branch, already
resolves `GgufLoadPolicy::FromEnv()` once for the expert-stream lane check
beside this one (`GgufExpertTowersReachSlotLane`). That resolution is hoisted
into a local, reused by both calls (previously two separate `FromEnv()` calls
computed the identical policy twice), and
`!(policy.keep_quant || policy.keep_f16 || policy.nvfp4_fp4)` is passed as the
new argument. `cpu_ref` needs no term of its own: it forces `kExpandBf16`
unconditionally too, but every `cpu_ref` load is a CPU load, and
`CheckDeviceWeightFit` already returns before computing anything when
`needs_weight_staging` is false — the oracle switch never reaches a device
that stages weights in the first place (`RouteGgufTensor`'s comment: "the
oracle switch wins over everything").

**Why not thread a `GgufLoadPolicy` all the way into the footprint and drop
`min` argument entirely?** Considered and rejected. The MIXED case — some
residency flag on, but a given tensor's role/dtype/shape makes it ineligible
(a ragged K, an unsupported encoding, `DeviceKeepQuantSupported` false) — still
needs the per-tensor role this file deliberately does not carry (see the
header's own note on `GgufExpertTowersReachSlotLane` needing the loader to ask
the SAME routing question the model's own loader asks, tensor by tensor,
because only the loader knows roles). `min()` stays the defensible bound for
that case, unchanged. The new parameter narrows the exact claim to precisely
the sub-case where role does not matter, and says so.

## Risks and decisions

- **The MIXED case is still approximate, and stays so.** A load with
  `keep_quant=1` on a checkpoint carrying one dtype ROCm's keep-quant list
  does not cover (`DeviceKeepQuantSupported`) still under-counts that one
  tensor's contribution exactly as it did before this row. Not a regression:
  `policy_forces_full_expand` is false whenever any flag is on, so that path is
  byte-for-byte unchanged. Recording it here rather than silently accepting is
  what the header already does for the file/load-scope over-count (#1136); this
  is the same shape, named rather than fixed, because fixing it needs role
  information this predicate does not have.
- **`cpu_ref` gets no explicit term.** Argued above: every `cpu_ref` load is a
  CPU load in this tree (the flag exists to reproduce the historical dequant
  path for the parity oracle, and nothing stages weights on the CPU platform),
  so `CheckDeviceWeightFit`'s existing `!needs_weight_staging` early return
  already keeps it out of scope. If a future device both stages weights and
  wants `cpu_ref` this term would need revisiting; nothing in this tree does
  that today, and `test_gguf_device_fit.cpp` already has fixtures using
  `PolicyWith(..., /*cpu_ref=*/true)` this row's tests build on.
- **A caller that has NOT resolved a `GgufLoadPolicy` still gets the safe
  (approximate, never over-tight) answer.** Default `false` is the
  conservative choice — under-counting is what #1870 is about, so the new
  parameter's default must never accidentally introduce it for a caller this
  row does not touch. Tests pin the default.

## Tests

**`tests/vllm/model_executor/test_gguf_device_fit.cpp`**, red first:

- A fixture combining the existing two-tensor file's Q8_0 and F32 tensors
  (`BuildTwoTensorGguf`, already in the file) with
  `policy_forces_full_expand=true`: the expected footprint is the SUM OF
  EXPANDED SIZES (`64 + 16 = 80`), not the existing `min`-based `50`. Asserted
  against `GgufStagedWeightFootprint` directly, and against
  `CheckDeviceWeightFit` at a budget between 50 and 80 (refuses only with the
  new parameter, passes without it) — the case that is RED before the fix,
  because today's `StagedBytes` has no such parameter and cannot distinguish
  the two loads.
- A default-parameter case pinning `policy_forces_full_expand=false` reproduces
  the EXISTING `kExpectedLowerBound=50` result byte for byte, so the new
  parameter's default is proven to be a no-op rather than merely documented as
  one.
- A production-shaped case: a policy built with
  `PolicyWith(/*keep_quant=*/false, /*keep_f16=*/false, /*nvfp4_fp4=*/false,
  /*cpu_ref=*/false)` derives `policy_forces_full_expand` the same way
  `model_loader.cpp` does (`!(keep_quant || keep_f16 || nvfp4_fp4)`), so the
  test exercises the exact boolean expression the call site uses rather than a
  hand-picked `true`/`false` literal that could drift from it.

**`tests/vllm/entrypoints/test_gguf_device_fit_reach.cpp`** — reachability half,
already registers a fake staging platform; extended with one case setting
`VT_GGUF_KEEP_QUANT=0` (and no other override) against a fixture sized to fit
the MIN-based bound but not the full-expand bound, proving the production call
site now refuses where it previously loaded past the point of no return.

## Gates

- `./build-hip/tests/test_gguf_device_fit` (ROCm build, this box, gfx1200)
- `./build-hip/tests/test_gguf_device_fit_reach`
- `ctest --test-dir build-hip -R 'gguf_device_fit'`
- `python3 scripts/check-env-doc.py`
- `scripts/agent-preflight.sh --staged`
- A real hardware repro: the two checkpoint arms #1870 reports (or an
  equivalent locally available dense + MoE GGUF pair sized to expand past 15.92
  GiB), `--device rocm`, `VT_GGUF_KEEP_QUANT=0`, before and after, showing the
  raw `hipMalloc: out of memory` crash replaced by
  `CheckDeviceWeightFit`'s named refusal.

## Evidence

Recorded in the pull request body: the red output of the new test cases before
the fix, green after, the fresh review's mutation results, and the before/after
transcript of the real hardware repro (crash vs. named refusal, exact bytes
reported vs. budget).

## Owed

- **#1928** — the ROCm `kMoeGroupedGemmBf16` provider gap #1870's "related gap,
  same area" section names. Not fixed in flow; scoped out above.
- **The MIXED-case approximation** stays exactly as conservative (and exactly
  as capable of under-counting on a role this predicate cannot see) as it was
  before this row. No issue filed for it beyond the standing `#1136` note the
  header already carries, because this row changes nothing about that case.

## Stop conditions

Stop and report rather than widening scope if the reachability fixture cannot
be sized to separate the two bounds without also tripping the existing
`nextn`-block over-count case (`kExpectedLowerBound + kNextnStaged`) — that
would mean the two rows' fixtures interact and need reconciling together
rather than in this row alone.

Stop rather than touching any residency DEFAULT (`keep_quant`,
`GgufQuantComputeAvailable`, `DeviceKeepQuantSupported`, ...). This row moves no
default and no availability rule.

## Now

`ACTIVE`. Spec committed; implementation, tests and the real-hardware repro
follow in this same pull request per the recorded PR-shape preference.
