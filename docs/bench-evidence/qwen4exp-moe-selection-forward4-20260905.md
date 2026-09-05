# The `VT_MOE_SEL_FP` budget counts MoE calls, so it already reaches the disagreeing forwards, 5 September 2026

Wave MOESEL of [`MODEL-MM-QWEN4-EXP`](../../.agents/specs/qwen4-exp-flash-next.md),
[#2969](https://github.com/mudler/vllm.cpp/issues/2969), on the open clause of
[#2547](https://github.com/mudler/vllm.cpp/issues/2547) and the next step named by
[#2877](https://github.com/mudler/vllm.cpp/issues/2877).

**The one-line result.** Reaching a model forward on which the two arms disagree
needs **no code change, no larger budget, and no lease**. `VT_MOE_SEL_FP` counts
MoE block invocations and not model forwards, `qwen4_exp` runs 48 MoE blocks in
every forward, and wave MOEDIV already spent 384 calls. That is eight forwards.
Forward 4 is calls 192 to 239 and forward 7 is calls 336 to 383, so **the tap's
window already reached all three disagreeing forwards on 2 September 2026**. This
document proves that mapping from data committed in this repository.

**Read that claim exactly.** It says the window reaches those forwards. It does
**not** say that an instrument observed a disagreement there, and it does not say
the arms were measured to diverge at forward 4. Those are different statements,
and the second one is still unmeasured.

**What is still owed is one arm, not one instrument.** MOEDIV's decode columns are
VOID because its CUDA arm answered the degenerate pre-#2550 sequence
`11751 271 271 271 271 271 0 0`, so the comparison ran against a different input.
The reading that closes the clause is the same tap at the same budget against a
**non-degenerate** CUDA arm. That run needs the 68 GB artifact and a GPU. It was
submitted and its state is in [section 5](#5-the-measurement-on-the-matched-pair).

## 1. The claim this corrects

`.agents/specs/qwen4-exp-flash-next.md` carried the heading "NO INSTRUMENT HAS YET
OBSERVED A DISAGREEING STEP" over this reasoning:

> `LayerFp` returns early on `s.step >= s.budget`
> (`src/vllm/model_executor/models/qwen4_exp_forward.cpp:118`), so every
> fingerprint this row has taken covers model forwards **0, 1 and 2**.

That statement is correct for `VT_Q4EXP_LAYER_FP` and wrong for `VT_MOE_SEL_FP`.
The two instruments count different things.

| Instrument | Counter | Increment site | `=N` covers |
|---|---|---|---|
| `VT_Q4EXP_LAYER_FP` | `LayerFpState::step` | `LayerFpEndStep`, once per model forward, `qwen4_exp_forward.cpp:170` | forwards 0 to `N-1` |
| `VT_MOE_SEL_FP` | `MoeSelFpCall()` | end of `MoeSelFp`, once per `MoeBlock` invocation, `qwen3_5.cpp:7167` | MoE calls 0 to `N-1` |

`MoeBlock` runs once for every decoder layer. The layer loop at
`qwen4_exp_forward.cpp:491` reaches the call at `:691` under no condition, and
`qwen4_exp_weights.cpp:760` loads MoE weights for every layer inside the loop at
`:746`. `qwen4_exp` has 48 layers and no dense layer, so one forward is 48 MoE
calls.

### #2877 states its own refutation two sentences before its conclusion

This is worth pointing at, because the arithmetic that overturns the claim was
already on the page when the claim was written. #2877's headline paragraph, quoted
verbatim in
[the ARMTOKENS evidence file](qwen4exp-gdn-chunked-token-ids-20260904.md), reads:

> MOEDIV's committed digests count **48** MoE calls at `T=5` and **336** at
> `T=1` — 8 forwards for 8 tokens — so this window is tokens `11751 13 15767`,
> **which AGREE on both arms**. The three ids that disagree are at indices **4, 6
> and 7**. **No instrument on this row has yet observed a single disagreeing
> step.**

The first sentence counts eight forwards in the `VT_MOE_SEL_FP` window. The last
sentence treats that window as three forwards. **The count and the conclusion are
about different instruments, and the paragraph does not say so.** The three-forward
window belongs to `VT_Q4EXP_LAYER_FP`, whose budget is counted in forwards. The
eight-forward count belongs to `VT_MOE_SEL_FP`, whose budget is counted in MoE
block invocations, and it is stated correctly.

The quoted paragraph is left byte-for-byte where it appears, because it is a
quotation of an issue body. An editorial note beside it carries this reading.

## 2. The mapping, derived from committed data

`docs/bench-evidence/qwen4exp-moe-selection-20260902/digests-CPU-CTRL.txt` is
committed in this repository and holds all 384 digest lines of MOEDIV's CPU
control arm. It answers the structural question with no share and no device:

| Property | Value |
|---|---|
| Digest lines | 384 |
| Digests with `T=5`, the prefill | 48, and they are exactly calls 0 to 47 |
| Digests with `T=1`, the decode | 336, and they are exactly calls 48 to 383 |
| Digests with any other `T` | 0 |
| Last digest `lines=` | 576, which is 48 x 5 plus 336 x 1 |

Because the prefill block is contiguous, has one call for each layer, and every
later call carries `T=1`, forward `f` occupies calls `48f` to `48f+47`:

| Forward | Calls | Phase | Sampled id | Arms agree |
|---|---|---|---|---|
| 0 | 0 to 47 | prefill, `T=5` | `11751` | yes |
| 1 | 48 to 95 | decode | `13` | yes |
| 2 | 96 to 143 | decode | `15767` | yes |
| 3 | 144 to 191 | decode | `411` | yes |
| **4** | **192 to 239** | decode | `2029` against `1928` | **no** |
| 5 | 240 to 287 | decode | `11` | yes |
| **6** | **288 to 335** | decode | `1092` against `628` | **no** |
| **7** | **336 to 383** | decode | `369` against `567` | **no** |

With `max_tokens = 8` the `k`-th sampled id comes from forward `k-1`, which is why
the three disagreeing ids at indices 4, 6 and 7 are produced at forwards 4, 6
and 7.

## 3. The comparator, and the positive control that qualifies it

`selfwd.py` groups both arms' `moesel` output by forward. It **derives** the
blocks-per-forward from the data instead of assuming 48, then asserts that the
prefill calls are contiguous from 0, that the decode count is an exact multiple of
that number, and that both arms report the same `T` for every call. A run that
fails any of those checks is reported as UNSTRUCTURED and no per-forward verdict is
quoted, because an off-by-one in this mapping would attribute a flip to the wrong
forward.

**The comparator was qualified against committed data before it was pointed at a
new run.** Run against MOEDIV's own `CPU-CTRL` and `CUDA-PROD` output, it
reproduces the upper endpoint of #2552's layer-0 bracket exactly:

| Quantity | `selfwd.py` on MOEDIV output | Recorded by #2552 and PREFILLDIV |
|---|---|---|
| Layer 0 MoE input, CPU arm | `x=3613.82031` | `L00 mhc.mix` 3613.82031 |
| Layer 0 MoE input, CUDA arm | `x=3615.62777` | `L00 mhc.mix` 3615.62777 |
| Layer 0 selection outcome | FLIP at token 2 | "layer 0 already flips at token 2" |

Reproducing both the tensor pair and the flip location, from a different script
than the one that first reported them, is what licenses a later reading from this
comparator.

**The value axes are reported as ordering only.** `rel(sumabs)` is a difference of
norms and cannot rank magnitudes, and every ratio taken from it on this row is
withdrawn. `selfwd.py` therefore prints, for each of `x`, `logit`, `exp` and `shr`,
the first call at which the axis stops being bit-identical, and prints no ratio.

## 4. Why `VT_MOE_SEL_FP` and not a wider `VT_Q4EXP_LAYER_FP`

#2552 brackets the layer-0 expert-flip threshold on the `L00 mhc.mix` tap:

| Pair | `L00 mhc.mix` | Layer-0 selection |
|---|---|---|
| CPU-CTRL against CUDA `VT_GDN_CHUNKED=0` | `2.139e-05` | no flip, all five tokens agree |
| CPU-CTRL against CUDA-PROD | `4.999e-04` | flip at token 2 |
| **CPU-chunked against CUDA-chunked**, the algorithm-matched pair | **`4.324e-05`** | **unknown** |

`4.324e-05` sits inside the bracket, so the norm cannot say which side of the
threshold the matched pair is on. A selection is a discrete property with bimodal
error, and `VT_MOE_SEL_FP` reports it directly as set equality. That is the axis
the bracket cannot resolve and this tap can.

The matched pair is ARMTOKENS run 2's `CPU-CHUNKED` against `CUDA-CTRL`, both at
`VT_CPU_QUANT_REPACK=0`. Its identity is fixed by the rendered diff, which reads
`L00 mhc.mix` 3615.47142 against 3615.62777 for `rel` 4.324e-05. Because this tap's
`x` axis is that same tensor, a run of the matched pair must report those two
numbers at call 0, and the job asserts them.

## 5. The measurement on the matched pair

**Host.** `thor:gpu0` under an `rc` lease, which is the box that recorded both
sequences. `rc` job `9e0864da-9b37-4309-b863-04810de0e068`, submitted
2026-09-05T21:10:41Z.

**Same image, not a rebuild.** The job runs wave ARMTOKENS' server, sha256
`1d129fa0ab96663bea8f50f715117596241a7f2f8ae77e877ba5853bb198792f`, built from
`b767ebda4e55122b1a5473b9aa4027da67f77b75`. That is the binary that produced both
recorded id sequences, and it already carries the tap. The job changes exactly one
thing about that measurement, which is an environment variable that turns on a
readback. A rebuild would have put a second difference between this reading and
the divergence it is about.

**Artifact.** The released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S, shard 1
sha256 `88a1420825a9304063e882ada29d438263617f51ac8923d438d927496693bafd`, staged
worker-local and verified inside the lease before any arm runs.

**Arms.** Three, one binary, `VT_MOE_SEL_FP=512` on each.

| Arm | Device | Environment | Expected ids |
|---|---|---|---|
| A-CPU | `cpu` | `VT_CPU_QUANT_REPACK=0` | `11751 13 15767 411 2029 11 1092 369` |
| B-CUDA | `cuda` | `VT_CPU_QUANT_REPACK=0` | `11751 13 15767 411 1928 11 628 567` |
| C-CPU2 | `cpu` | `VT_CPU_QUANT_REPACK=0` | same as A-CPU |

Arm C repeats arm A. **A against C is the negative control.** A null result is a
live outcome for this measurement, and zero flips is also what a broken comparison
prints, so a zero from A against C is what makes a zero from A against B readable.

**The instrument must not change the answer.** Each arm asserts its ids against the
sequence the same binary produced with the tap off. A mismatch exits 50, which is
distinct from every instrument-failure exit, because an instrument whose failure
looks like a result is this row's defining trap.

**State.** PENDING. Read by id with `rc jobs -o json` and not from `rc ps`,
because `rc ps` lists only pending and running work, so absence there is not
cancellation:

| Field | Value at 2026-09-05T21:36:32Z |
|---|---|
| `id` | `9e0864da-9b37-4309-b863-04810de0e068` |
| `state` | `queued` |
| `exit_code` | `None` |
| `device_id` | `thor:gpu0` |
| `queued_at` | `2026-09-05T21:10:41Z` |
| `started_at` | `None` |
| `finished_at` | `None` |

The job had not dequeued 26 minutes after submission. `thor:gpu0` was held for
that whole period by an unrelated job. **A queued job survives the death of the
client that submitted it**, so this id stays valid and the next reader queries it
by id rather than resubmitting. See section 6.

## 6. What this document does not establish

- **Whether the matched pair's layer-0 selection flips.** The job that answers it
  was queued behind another job on `thor:gpu0` and had not started when this
  document was written. Nothing here places `4.324e-05` on either side of the
  bracket.
- **Anything about forwards 4, 6 and 7 on a non-degenerate pair.** MOEDIV's
  reading at those forwards exists and is VOID, because its CUDA arm ran a
  different input. This document corrects the reason the reading is unavailable.
  It does not supply one.
- **Any magnitude.** No ratio is quoted from `rel(sumabs)`, in either direction.
- **A cause for the three disagreeing ids.** #2552 named two floor terms and found
  both to be faithful mirrors of vLLM. That remains the position.

## 7. The ARMTOKENS raw fingerprints, and what they permit without a lease

The spec recorded that the ARMTOKENS diffs cannot be re-rendered without a
device, because no arm's raw `fp.txt` was committed. The premise is right and the
conclusion is withdrawn. The job output was never lost.

| Item | Value |
|---|---|
| Location | `/workspace/armtokens-2612/out2/<ARM>/fp.txt` on the fleet share, which is `/mnt/nas_share/rc/armtokens-2612/out2/<ARM>/fp.txt` from a host that mounts it |
| Arms | `CPU-CHUNKED`, `CPU-SEQ`, `CUDA-CTRL` |
| Size | 1314 lines each, 3 `taps=` end markers each |
| Unique `(step, L, tag)` taps | 1311, against the 42 the committed diff reported |

**The 42 is a parser defect and not missing data.** `run2-job.sh`'s differ keys on
`f.get('L')`, and it builds its field map by splitting tokens on `=`. The tap
prints the layer as `L%+03lld`, for example `L+00`, which carries no `=`, so `L`
never parses and every one of the 48 layers folds onto one key for each
`(step, tag)` pair.

**What this permits.** A reader can re-render the ARMTOKENS per-layer comparison
at full coverage with the repaired differ and no lease. **What it does not
permit** is a magnitude. `rel(sumabs)` is a difference of norms, so a re-render
may state which taps are bit-identical and in what order they stop being so, and
it may not publish a ratio.

**The share is fleet scratch and not a record surface.** A wave that holds a lease
on this row must still capture `fp.txt` for each arm.

## 8. Provenance

The job script and comparator are `job.sh` and `selfwd.py` in
[`qwen4exp-moe-selection-forward4-20260905/`](qwen4exp-moe-selection-forward4-20260905/).
The structural claim in section 2 is reproducible from this repository alone:

```sh
python3 - docs/bench-evidence/qwen4exp-moe-selection-20260902/digests-CPU-CTRL.txt <<'EOF'
import sys
rows = []
for line in open(sys.argv[1]):
    f = dict(t.split('=', 1) for t in line.split() if '=' in t)
    rows.append((int(f['call']), int(f['T'])))
rows.sort()
pre = [c for c, T in rows if T > 1]
dec = [c for c, T in rows if T == 1]
assert pre == list(range(48)), pre[:5]
assert dec == list(range(48, 384)), dec[:5]
print('forwards =', len(rows) // 48, 'forward 4 =', 4 * 48, '..', 5 * 48 - 1)
EOF
```
