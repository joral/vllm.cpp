# ROCm decode-attention WMMA at head_dim=128 — a spike proving the lever is real, with a real crossover

**Row:** `BACKEND-ROCM` (backend-matrix, `ACTIVE`).
**Claim:** `CLAIM-ROCM-DECODE-ATTN-D128-WMMA`.
**Issue:** `TBD` — this spike owes its **own** issue, to be filed and its number
substituted here before this spec's PR merges. It is
deliberately *not* filed against
[#488](https://github.com/mudler/vllm.cpp/issues/488): #488 is a measurement
issue that explicitly asserts no cause, this spike is one candidate answer to
it and does not close it, and #382 already owns the scalar arm — three PRs
pointing at one issue makes the required issue↔spec↔PR agreement meaningless.
#488 is the **parent observation**; this issue references it.
**Depends on:** the scalar `d=128` arm in
[rocm-decode-attn-d128.md](rocm-decode-attn-d128.md) (#382), same branch. That
spec's `bf16_decode_opt` gate carries a `decode_wmma` disjunct so `d==128`
reaches this dispatch under `VT_ATTN_DECODE_WMMA=1` alone; without it this arm
would silently require both flags. Measured against the scalar kernel as
baseline, not the original `PagedAttnOnline`.
**Base:** `row/ROCM-DECODE-ATTN-D128`, same branch (no stacking needed —
additive to the scalar fix, not competing with it).
**Board:** AMD Radeon RX 9060 XT (`gfx1200`, Navi 44, RDNA4, discrete, 32 CUs),
ROCm 7.2.3, hipClang/Clang 22.0.0.
**Nature of this record:** a **spike**. The kernel and its runtime-selected
default are landed and gated, but the spike's job was to prove the lever is
real and find its shape — not to ship an auto-tuned, always-on production
path. §6 names what a follow-on production claim needs.

---

## 1. What this proves

**A real, substantial, context-length-dependent win**, with a clean
mechanistic explanation and a measured crossover:

| Context | Qwen3-0.6B scalar → WMMA (TPOT) | Qwen3-1.7B scalar → WMMA (TPOT) |
|---|---|---|
| 128 | 10.04 → 10.82 ms (**+8% slower**) | 20.38 → 21.24 ms (+4% slower) |
| 256 | 10.57 → 10.86 ms (+3% slower) | 20.80 → 21.32 ms (+3% slower) |
| 512 | 11.36 → 10.90 ms (**4% faster**) | 21.62 → 21.42 ms (~1% faster) |
| 1024 | 12.87 → 10.70 ms (**17% faster**) | 23.40 → 21.67 ms (**7% faster**) |
| 2048 | 16.47 → 12.00 ms (**27% faster**) | 26.82 → 23.12 ms (**14% faster**) |

Every cell is 2 reps, tightly agreeing (see §5 for the full log). **Crossover
sits around 400-500 tokens** on both model sizes, and the WMMA advantage grows
with context past that point.

Getting here required finding and fixing two real bugs along the way (§3, §4)
— this section states the destination; §2-§5 are the honest route, including
two dead ends that turned out to be measurement artifacts, not results.

## 2. Design: three iterations

Scope stays what the scalar-fix spec named: **`QG=2`, `d=128` only** (Qwen3-0.6B/
1.7B's GQA ratio — the models this was measured on). `QG=4`/`8` (Qwen3-4B) is
still a named, unbuilt follow-on, same as before.

**v1 — single-wave, whole-sequence.** One CTA (one wave, 32 lanes) per
`(query token, kv head)` pair, WMMA for QK^T (`rocwmma::` 16x16x16 tiles,
matching this file's existing `PagedAttnPrefillWmmaWave` convention), scalar
online-softmax for PV. Single-wave by construction: `PagedAttnPrefillWmmaWave`
carries an inherited warning — never spread one `mma_sync` across multiple
waves in one block, it hangs on gfx1201 (§3.5 has the full provenance and what
is/isn't independently verified about that claim).

**v2 — partition + reduce.** Splits the KV sequence into `kDecWmmaTPar=64`-
token partitions, each an independent single-wave CTA
(`PagedAttnDecodeGqaWmmaPartial`), merged by a second kernel
(`PagedAttnDecodeGqaWmmaReduce`) doing the same online-softmax combine
`PagedAttnDecodeOptBf16T` already uses to merge its 8 warps — just reading
partial `(m, l, o)` state from global scratch instead of shared memory. Restores
occupancy by launching more independent CTAs instead of packing more waves
into one (which §3.5's constraint forbids).

**v3 — vectorized LDS staging.** v2's K/V/Q shared-memory staging loaded one
`bf16` element per lane per iteration. Switched to `uint4` (8 elements/lane),
matching `LoadRowEplBf16`'s convention elsewhere in this file. This is the
version in §1's numbers.

## 3. Bug found: `__gfx1200__`/`__gfx1201__` are device-pass-only

**v1 and v2's first "measurements" were invalid — no WMMA kernel had ever
actually launched.** `VT_ROCWMMA_OK` (this file's existing macro, defined
`#if defined(__gfx1200__) || defined(__gfx1201__)`) is checked at the top of
the host dispatch function to guard the WMMA kernel launch. A standalone
probe settles it directly:

```
$ hipcc --offload-arch=gfx1200 -o /tmp/macrotest macrotest.hip && /tmp/macrotest
HOST:   __gfx1200__ is NOT defined
DEVICE: __gfx1200__ check result = 1
```

HIP's split host/device compilation defines the architecture macro only for
the device pass. A host-side `#if defined(VT_ROCWMMA_OK)` is **always false**
on this toolchain (hipClang/Clang 22.0.0) — it silently deletes the guarded
launch statement, not just the kernel body. `VT_ATTN_DECODE_WMMA=1` was a
no-op the entire time v1 and v2 were being A/B'd; every throughput delta
"measured" in that window was noise (confirmed: 4-rep A/B swung sign both
ways at the same shape, e.g. Qwen3-0.6B +1.8%/-9.2%/+6.6%/-4.5% across
identically-configured reps).

**Fix:** `IsGfx1200Or1201()`, a real runtime `hipGetDeviceProperties` query
(cached once), mirroring vLLM's own `is_navi_gpu()`
(`csrc/rocm/attention.cu`: `arch.find("gfx11")==0 || arch.find("gfx12")==0`)
and this project's `DeviceCuCount` caching pattern. Replaces the host-side
`#if defined(VT_ROCWMMA_OK)` guard around the launch. The kernel body itself
keeps its own `#if !defined(VT_ROCWMMA_OK) return; #else ... #endif` — that
one *is* correctly evaluated per-device-target and stays.

### 3.6 The same mechanism reaches already-shipped prefill kernels — CONFIRMED

The host-launch guard around `PagedAttnPrefillSharedKWmma`
(`rocm_paged_attn.hip:2071`) uses the identical `#if defined(VT_ROCWMMA_OK)`
pattern. That was flagged here as a concrete concern; it has since been
**confirmed statically**, by preprocessing the host compilation pass with the
build's own command (`clang++ … -E --cuda-host-only`, flags taken verbatim from
`build-hip/compile_commands.json`):

| symbol in the host pass | count |
|---|---|
| `prefill_sharedk_wmma` (the string literal inside **both** launch sites) | **0** |
| `PagedAttnPrefillSharedKWmma` | 1 — the definition only, no launch |
| `rocwmma` (the header) | **0** |
| `PagedAttnDecodeGqaWmmaPartial` (this spike's runtime-gated arm) | 2 — definition **and** launch |

Both launch sites are deleted from the host pass. **The shipped
`PagedAttnPrefillSharedKWmma` path at `d=256` and `d=512` has never executed**;
prefill at those head dims has always fallen through to the scalar
`PagedAttnPrefillSharedK` below it. Whatever performance was attributed to the
WMMA prefill kernel when it landed was the scalar kernel's.

**Deliberately not fixed here.** It is a defect in already-merged, default-on
production behavior, in a different code path, and needs its own red-first test
(a host-pass assertion or a kernel-trace gate — the bug's whole nature is that
it is invisible to every existing green test). Folding it into this spike would
under-test both. Owed its own issue; standing item 4 in §6.

## 4. Bug found: the decode-shaped kernel was also firing on prefill

After §3's fix, a second bug surfaced immediately via a runtime diagnostic:
the WMMA branch sits at the `flash_fallback:` label, which is *also* where a
**prefill** call (`total_q` = the whole prompt) lands when `d=128` doesn't
match the dedicated prefill-WMMA kernels' `{256,512}` gate — a fallback that
was safe when the destination was the generically-correct scalar kernel, but
not once the destination was a kernel built and tested only for decode.
Observed directly: a 2048-token prompt's prefill call launched a
`(total_q=2048) x (num_kv_heads=8) x (num_partitions=32)` grid —
**524,288 single-wave blocks** for one prefill step.

This also explains why an *earlier* attempt to reconcile a profiled-vs-wall-
clock discrepancy failed to reconcile: the scalar kernel's own rocprofv3 trace
was contaminated by the same 56 giant prefill-shaped dispatches (`n=3584`
included them, `avg=566us`), while the WMMA trace had — by then — already
excluded its own prefill contribution. Comparing the two traces without
separating prefill-shaped from decode-shaped dispatches (`Grid_Size_X`, since
this file reports total thread count not block count: 256 = decode `total_q=1`,
524288 = this prefill case) was comparing different things and looked like a
contradiction. Filtered to decode-shaped dispatches only, both numbers agree
with each other and with wall-clock (§5).

**Fix:** `total_q <= 4` added to the WMMA dispatch condition — matches this
project's existing "decode-skinny" `M<=4` convention (the GEMM lever, #487).

## 5. Evidence

**Correctness**, gfx1200: full `ctest` green throughout every iteration (v1,
v2, v3, both bug fixes) — `rocm|cross_device` 4/4 each time, including the
`d=128` GQA case from the scalar-fix spec, run with `VT_ATTN_DECODE_WMMA=1`
after every change.

**Kernel-level, same-tool, decode-shaped only** (`rocprofv3 --kernel-trace`,
native Nix `rocprofiler-sdk` — no container needed for *this* comparison since
it traces only our own binary; see §6 for why the oracle side still needs one),
Qwen3-0.6B, 2048-token context, filtered to `Grid_Size_X<=300` (decode-shaped):

| Kernel | avg/call |
|---|---|
| `PagedAttnDecodeGqaBf16` (scalar) | 259.352 us |
| `PagedAttnDecodeGqaWmmaPartial` + `Reduce` | 86.559 + 23.327 = 109.886 us |

**2.36x faster** at kernel level — bigger than the 27% wall-clock TPOT win,
consistent with attention being only part of each decode layer's cost (GEMMs,
RMSNorm etc. are unaffected and dilute the wall-clock delta).

**Context-length sweep**, wall-clock TPOT, `vllm-bench`, 2 prompts,
concurrency 1, seed 0, 2 reps/point (full log, illustrative subset in §1):

```
RESULT model=Qwen3-0.6B ctx=128  wmma=0 tpot_ms=10.04   wmma=1 tpot_ms=10.81
RESULT model=Qwen3-0.6B ctx=512  wmma=0 tpot_ms=11.36   wmma=1 tpot_ms=10.90
RESULT model=Qwen3-0.6B ctx=2048 wmma=0 tpot_ms=16.46   wmma=1 tpot_ms=11.99
RESULT model=Qwen3-1.7B ctx=128  wmma=0 tpot_ms=20.36   wmma=1 tpot_ms=21.24
RESULT model=Qwen3-1.7B ctx=512  wmma=0 tpot_ms=21.63   wmma=1 tpot_ms=21.42
RESULT model=Qwen3-1.7B ctx=2048 wmma=0 tpot_ms=26.82   wmma=1 tpot_ms=23.13
```

**Mechanistic explanation for the crossover**, not just a curve fit:
`kDecWmmaTPar=64` means `num_partitions` at `ctx=512` is 8 —
`8 partitions x 8 kv_heads = 64` independent WMMA CTAs, landing almost exactly
on the scalar kernel's fixed `8 CTAs x 8 warps = 64` total resident waves.
Below that context WMMA has fewer partitions than the scalar kernel has warps
(occupancy-starved, loses); above it, more (wins, and the gap widens because
the scalar kernel's per-call cost grows with context while each WMMA
partition's cost stays roughly flat at `kDecWmmaTPar` tokens).

**Architecture-scope confirmation, Qwen3.5-0.8B** (`head_dim=256`, GQA
ratio 4 — outside this kernel's `d=128`/`qg=2` gate by construction):
flag-invariant at both ends, as expected — `ctx=128`: 19.11 vs 19.21ms;
`ctx=2048`: 30.27 vs 30.20ms. Confirms the gate is doing what it says, not a
real cross-architecture comparison (that needs the follow-on in §6).

## 6. Decision and follow-ons

**Land default-OFF** (`VT_ATTN_DECODE_WMMA` unset = scalar path, matching this
file's `git.getenv(...) == "1"` convention). A spike proves the lever real; it
does not make an unconditional case for flipping default behavior, and
default-ON here would *regress* every request under ~400-500 tokens of
context, which is a real, common case, not a corner one.

**What a follow-on production claim needs, named and not built here:**

1. **Automatic dispatch by context length.** The crossover is real,
   measured, and mechanistically explained — a runtime `if (seqlen > crossover)`
   switch is the obvious next step, but the exact crossover threshold should be
   swept more finely (this spike used 5 points a factor of 2 apart) and
   revalidated once `kDecWmmaTPar` itself is tuned (below), since the
   crossover depends on it directly.
2. **Tune `kDecWmmaTPar` (currently 64, unswept).** The mechanistic story in
   §5 predicts the crossover point moves with this constant; a smaller value
   would cross over earlier (more partitions sooner) at the cost of more
   per-partition fixed overhead (LDS staging repeated more often, more reduce
   work). Not measured.
3. **`QG=4`/`8` (Qwen3-4B and larger GQA ratios).** Same scope gap the
   scalar-fix spec named; still open. Qwen3.5-0.8B's `d=256` full-attention
   layers are a second, distinct extension (different `EPL`, different WMMA
   tile shape) — "porting more pieces over for Qwen3.5" is real, unbuilt work,
   not a small delta.
4. **Fix the shipped prefill-WMMA dead-launch bug (§3.6).** No longer a
   suspicion — **confirmed** by host-pass preprocessing:
   `PagedAttnPrefillSharedKWmma` has never launched at `d=256` or `d=512`. Owes
   its own issue (body drafted), its own red-first test that can actually see a
   dead launch, and a re-measure of whatever was attributed to that kernel.
5. **Independently verify the gfx1201 multi-wave `mma_sync` hang claim**
   (§2, inherited from `PagedAttnPrefillWmmaWave`'s comment, itself sourced to
   PR #317 on 2x R9700/gfx1201 hardware, not independently reproduced on this
   gfx1200 board or this ROCm version). If it turns out not to reproduce here,
   a multi-wave WMMA design becomes viable and could close much of the
   occupancy gap directly rather than via the partition workaround.
6. **Same-tool oracle comparison** (`rocprofv3` inside the pinned
   `vllm-rocm-oracle:555967922-gfx1200` container, matching #487/#488's
   original methodology) to see where this leaves the gap to vLLM's own
   kernel. Attempted this session; blocked on a real Nix-glibc-vs-Ubuntu-
   container ABI mismatch (the container's glibc 2.35 vs this Nix toolchain's
   2.42). The native-Nix `rocprofiler-sdk` package
   (`/nix/store/.../rocprofiler-sdk-7.2.3/bin/rocprofv3`) sidesteps this for
   tracing *our own* binary (used throughout §5) but does not reach the
   oracle, which still needs the container.

## 7. Scope

**In scope.** The three kernel iterations in §2 (only v3 ships), the two bug
fixes in §3-§4, `IsGfx1200Or1201`, `EnsureDecodeWmmaScratch`, and this spec.

**Out of scope, named above.** §6 items 1-6 in full.

## 8. Reproduction

```sh
nix develop .#rocm-shell --command bash -c '
  cmake -S . -B build-hip -G Ninja -DVLLM_CPP_HIP=ON \
    -DVLLM_CPP_HIP_ARCHITECTURES=gfx1200 -DROCM_PATH=$ROCM_PATH \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build build-hip -j"${JOBS:-8}"
'
flock "$HOME/gpu.lock" -c '
  nix develop .#rocm-shell --command ctest --test-dir build-hip --output-on-failure
'
# Context sweep (both flags, several --input-len points, fixed --output-len):
VT_ATTN_DECODE_WMMA=0|1 build-hip/examples/vllm-bench --model <path> \
  --num-prompts 2 --input-len <N> --output-len 32 --concurrency 1 --seed 0
# Kernel-level trace (native Nix rocprofv3, no container needed for own-binary trace):
VT_ATTN_DECODE_WMMA=0|1 /nix/store/*-rocprofiler-sdk-*/bin/rocprofv3 \
  --kernel-trace --output-format csv -d <dir> -- build-hip/examples/vllm-bench ...
```

## Outcome (2026-08-12)

**Spike complete — the lever is real, and its shape is known.** Two design
dead-ends turned out to be bugs, not negative results: a HIP host/device
macro-scoping bug (`__gfx1200__` is device-pass-only) meant the first two
kernel iterations never actually ran, and a dispatch-placement bug let the
decode-only kernel fire on full prefill calls, contaminating every
measurement taken before both were found and fixed. Once clean, the result is
unambiguous: WMMA loses by 3-8% under ~400-500 tokens of context and wins by
4-27% above it, on both model sizes tested, with a mechanistic (not just
curve-fit) explanation for exactly why. Landed default-OFF, gated correctly
(4/4 `ctest` green throughout), because a spike proving a lever is real is not
the same claim as an unconditional default flip — six follow-ons named in §6,
none built here, the automatic-dispatch-by-context-length one being the
obvious next step once `kDecWmmaTPar` itself is tuned.
