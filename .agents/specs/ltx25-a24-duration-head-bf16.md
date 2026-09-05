# LTX-2.5 — the DURATION HEAD's bfloat16 arm (A24, wave 6, the last)

Row: `LTX25-A24-DURATION-HEAD-BF16`
Issue: [#2955](https://github.com/mudler/vllm.cpp/issues/2955)
Parent scope: `.agents/specs/ltx25-completion-scope.md` §A.7 (A24), operator-owned
Wave 1: `.agents/specs/ltx25-a24-text-tower-bf16.md` (#2676, merge `8e582a5f9`)
Wave 2: `.agents/specs/ltx25-a24-connector-bf16.md` (#2720, merge `77704c8d0`)
Wave 3: `.agents/specs/ltx25-a24-video-vae-bf16.md` (#2786, merge `c20fb2ba2`)
Wave 4: `.agents/specs/ltx25-a24-leaves-bf16.md` (#2850, merge `d2b1bda2b`)
Wave 5: `.agents/specs/ltx25-a24-upsampler-bf16.md` (#2857) — this row is its `## Owed`
The wiring this row stands on: `.agents/specs/ltx25-duration-head-wire.md` (#2900, `5386a9eb8`)
Oracle: `.agents/oracles/ltx-2.md`, `fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`
Base: `5386a9eb8`

---

## Now

`ACTIVE`. This is the **eighth and final** component of gap A24. What A24 still
owes after it is in `## Owed`, and none of it is a ninth component.

---

## 0. GATEABILITY, SETTLED BEFORE A LINE WAS WRITTEN

Waves 1-5 each gated with no checkpoint, no lease and no GPU, by **executing**
the pinned upstream module on synthetic fixtures. The dispatch for this row
required that question answered first and a `NEEDS_DECISION` returned if the
answer was no. **The answer is yes**, and these are the commands that say so, run
at this row's base on this box:

```text
git -C ~/_git/LTX-2 rev-parse HEAD      fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca  (the pin)
git -C ~/_git/LTX-2 status --porcelain  empty
python3 -c 'import torch,numpy,einops'  torch 2.11.0+cu130, numpy 2.3.5, einops 0.8.2
DurationHead(...).to(bfloat16)(v, a)    RAN on CPU, returned tensor([1.0312], dtype=torch.bfloat16)
narrowing check, 15 of 15 parameters    every parameter moved under `.to(bfloat16)`
```

The narrowing check is quoted because it is the probe trap that has caught four
sessions in this family: reading a parameter AFTER `.to(bfloat16)` narrows it in
place and yields a false 0/0. Every probe in this row captures the f32
parameters **before** any cast, and the counts are 16/16, 512/512, 384/384,
768/768, 47/48, 256/256, 384/384, 12/12, 1/1 and so on — a real population, not
an artefact. `attention_pooler.cross_attn.in_proj_bias` moving 47 of 48 rather
than 48 of 48 is the control that the count is measuring something.

**No lease was taken, no GPU work was run, nothing was downloaded, and
`CHECKPOINT_ROOT` was not resolved.** None was needed.

---

## 1. Scope

**IN.**

* `src/vllm/model_executor/models/ltx2_duration_head.cpp` — `Ltx2DurationPredict`
  and `Ltx2DurationAttentionPool` gain the bf16 arm. They read every tensor
  through `Ltx2VaeWeights::Get` today, which **refuses a bf16 bag by name**, so a
  head loaded at `kBF16` cannot be run at all at this row's base.
* `include/vllm/model_executor/models/ltx2_duration_head.h` — the DTYPE block,
  whose "f32, the parity dtype of this gate" sentence this row falsifies.
* `src/vllm/multimodal/ltx2_video.cpp` — the two `Ltx2LoadDurationHeadWeights`
  call sites (`:1932`, `:1941`) ask for `kBF16`, and the render path gains the
  runtime-width and storage counters.
* `include/vllm/multimodal/ltx2_video.h` — those trace fields.
* `scripts/gen-ltx2-duration-wire-goldens.py` — a new bf16 section, a pure
  addition, with its own refusal.
* `tests/vllm/models/test_ltx2_pipeline.cpp`, `tests/vllm/multimodal/test_ltx2_video.cpp`.

**OUT, and owed by name in `## Owed`.** The **FP8 and NVFP4** arms, which are
A22 and which `ReadTensorBytes` already refuses by name. A **CUDA arm**: the
head's only device seams are `vt::MatmulBT` and `vt::AttentionCross` and it runs
on a default-constructed `vt::Queue{vt::Device{}, nullptr}`, so a device arm is a
residency row and not a dtype row. A **real-weight render**: no LTX-2.5
checkpoint carrying a `duration_head.` block is on the NAS at this base, and
`CHECKPOINT_ROOT` does not resolve on this box. That is a pre-existing hole in
the *f32* arm too — #2900 landed the wiring against synthesized safetensors —
and this row neither widens nor closes it.

---

## 2. Upstream anchors, each read at `fd4ded7f`

| what | where |
|---|---|
| the ONE pipeline dtype, `torch.bfloat16` | `ltx-pipelines/.../distilled.py:109` |
| the head is constructed with it | `distilled.py:163-165`, `DurationPredictor.from_checkpoint(checkpoint_path, dtype, device)` |
| `DurationHead.forward` | `ltx-core/.../duration_head/duration_head.py:89-118` |
| `AttentionPooler.forward` | `duration_head.py:45-49` |
| the pooler is `torch.nn.MultiheadAttention` | `duration_head.py:38-43` |
| the modality embeddings are bare `nn.Parameter`s | `duration_head.py:74`, `:77` |
| `gelu(..., approximate="tanh")` | `duration_head.py:116` |
| the output is `exp` of the regression | `duration_head.py:117-118` |

`.to(dtype)` narrows a `Parameter` and a registered buffer alike, so all fifteen
of the head's tensors are bf16 upstream. Nothing in this module is annotated
f32, and nothing here argues for one — unlike the audio VAE, whose f32 §A.7
records as *argued* rather than owed.

---

## 3. The rules, MEASURED — five that hold and three honest negatives

Every row was produced by executing the module at the pin on CPU, torch
2.11.0+cu130, single-threaded, with the f32 parameters captured before any cast.
A rule that separates nothing is reported as separating nothing.

### 3.1 The five that reproduce, with their rejected hypothesis

| # | site | upstream's rule | rejected hypothesis | separating |
|---|---|---|---|---|
| R1 | every `nn.Linear` (6 sites) | f32 accumulate, bias added **in f32**, **ONE** rounding on store | round the matmul, then add the bias in bf16 (two roundings) | 25/80, 584/2048, **1024/4096**, 206/768, 83/256, 63/256 |
| R2 | `+ self.video_modality_emb` | the embedding is a **narrowed Parameter** and the **add rounds** | the sum kept in f32 | 38/80 |
| R3 | `gelu(approximate="tanh")` | f32 opmath, round **once** on store | the tanh argument narrowed first | 74/200 |
| R4 | `log_duration.exp()` | f32 opmath, round **once** on store | the seconds returned unrounded | **flips the FRAME COUNT** — §3.3 |
| R5 | the pooled attention output | f32 accumulate, round the pooled result | softmax and scores taken at bf16 | 69 of 93 pooler fixtures |

**R1 is the one whose obvious C++ spelling is wrong in a way a reader will not
see.** `torch.nn.functional.linear` at bf16 is bit-exactly `bf16(f32(a) @ f32(w)^T
+ f32(b))` at every shape probed, including the shipped 4096 and 2048 widths. The
plausible alternative — round the GEMM, then add the bias — disagrees in a
**quarter of all outputs** and type-checks identically.

**The f64 accumulator this file already uses is SAFE on this arm, and that is the
opposite of the polarity that applies to its f32 arm.** An f64 accumulation is
bit-equal to torch's bf16 Linear at every probed shape (0 of 4096 at the widest),
because at a bf16 store the f32-vs-f64 reduction difference sits far below one
ulp. The same holds for `GeluTanh`, whose f64 pointwise expression — which
`ltx2_duration_head.cpp:69-78` records as *wider than upstream* and therefore
visible debt on the f32 arm — is bit-equal to upstream's f32 `gelu` at bf16 in
0 of 200. **This row does not narrow that expression and does not need to**, and
the reason is written here so the next reader does not re-derive it. The debt
stays owed on the f32 arm.

### 3.2 THE THREE NEGATIVES, and the one that decides the whole gate design

**N1. The forward's output is a FUNNEL, and a whole-chain value gate on it
cannot see a single intermediate rule.** `DurationHead.forward` returns ONE bf16
scalar per batch row, through `exp`. Swept over 504 configurations of width,
head count, query count, token count and parameter amplitude, the correct C++
chain reproduces upstream **bit-exactly**, and so does every wrong one:

```text
case        correct  two_round   add_f32   emb_f32 soft_bf16  attn_f32  gelu_narrow  exp_f32
Shipped           0          0         0         0         0         0            0  0.0037866
Small             0          0         0         0         0         0            0  0.0032780
VideoOnly         0          0         0         0         0         0            0  0.0006394
AudioOnly         0          0         0         0         0         0            0  0.0007653
```

Eight mantissa bits, applied once to a single number, absorb every intermediate
difference. **A gate built on the shipped fixture alone would be green under six
of the seven mutations this row claims to detect**, and it would look exactly
like a passing test. This is the mute switch §A.7's family has now hit in six
consecutive waves, found here before a line of C++ was written rather than in
review.

**What it drives.** Coverage is measured **per RULE across fixtures**, never per
fixture, which is wave 5's own correction applied at the outset. Over a 66-fixture
sweep, 59 give a bit-exact correct chain, and among those 59 each rule is
separated by at least one:

```text
rule separated in N of the 59 bit-exact fixtures:
  two_round 7 | add_f32 5 | emb_f32 5 | soft_bf16 3 | attn_f32 7 | gelu_narrow 3 | exp_f32 59
```

The generator emits the fixtures that separate, records the coverage as a golden,
and **refuses to write when any rule reaches zero**. A rule whose coverage falls
to zero reds at generation time instead of quietly becoming decoration.

**N2. `torch`'s bf16 attention core does not reproduce from an f32-accumulate
reference, and the control proves the reference is right.** A hand-written mirror
of `F.multi_head_attention_forward` reproduces torch's **f32** `MultiheadAttention`
in **0 of 32** elements — bit-exact, so the reference is correct. The same mirror
at bf16 differs in **18 of 32**, and `scaled_dot_product_attention` at bf16
differs from `bf16(f32 core)` in **2 of 32**. This is wave 5's `Conv3d` finding
in a different operator: oneDNN selects a different route for the bf16 kernel.
The consequence is bounded and stated rather than hidden — the port keeps the
attention core at f32 on narrowed inputs and rounds the pooled output, which is
the closest reachable mirror, and the **pooler is gated only at fixtures where
that chain is bit-exact**, which the sweep found in 93 of 504.

**N3. The pooler's `query_tokens` narrowing is UNMEASURABLE by any value gate
here.** Among the 93 bit-exact pooler fixtures, holding `query_tokens` at f32
instead of narrowing it separates in **0**. This is wave 5's `BlurDownsample`
kernel again: the parameter genuinely narrows — 32 of 32 entries move under
`.to(bfloat16)`, which is the control that the probe is not blind — and the
difference is then absorbed by the rounding of its own projection. It is gated
by a **count** rather than a value, and recorded in `## Owed`.

### 3.3 R4 is not a rounding curiosity; it changes how many frames get rendered

The head's output feeds `Ltx2DurationPredictFrames`, so the last rounding is the
one a user sees. Sweeping the synthetic head's `mlp_out.bias` across `[-1, 1.5)`
at 25 fps on the shipped widths, **10 of 1000 samples flip the frame count**:

```text
bias=+0.2775  f32 1.300527 s -> 33 frames | bf16 1.296875 s -> 25 frames
bias=+0.4975  f32 1.620556 s -> 41 frames | bf16 1.617188 s -> 33 frames
bias=+0.9650  f32 2.586406 s -> 65 frames | bf16 2.578125 s -> 57 frames
bias=+1.3675  f32 3.868124 s -> 97 frames | bf16 3.859375 s -> 89 frames
```

Eight frames, a third of a second, on the `8k + 1` grid. That is the production
consequence of this row, it is an **integer**, and an integer is the right shape
for it: a discrete selection has bimodal error, not a tolerance, so the gate
asserts frame-count **equality** and never a band.

---

## 4. Design

**The weights are STORAGE; the activations are ARITHMETIC. The row claims both,
and says which observable measures which.** Wave 5 failed review for gating an
arithmetic width while claiming a storage one, so the two claims are separated
here by construction:

* **Storage — the weight bag.** `Ltx2LoadDurationHeadWeights` already fills
  `Ltx2VaeWeights::bf16` for `kBF16` and moves a BF16 source **word for word**
  rather than round-tripping it. The claim is that the resident bag is *half the
  bytes*, and the observable is `Ltx2VaeWeights::Bytes()` — the tree's own
  measurement, which had no caller on this component. The gate compares **two
  bags built from the same tensor set** and asserts an exact 2:1 ratio, so no
  number is quoted.
* **Arithmetic — the activations.** They stay in `std::vector<float>` holding
  narrowed values, exactly as wave 2's connector does, and this row **does not
  claim** the activation buffers are half. Saying so is the point: the claim that
  is true is gated, and the claim that is not true is not made.

**Reading the bag.** `Ltx2DurationPredict` and `Ltx2DurationAttentionPool` take
every tensor through `Ltx2VaeWeights::Get`, which refuses a bf16 bag by name. A
`WeightRef` helper resolves a name against whichever arm the bag carries —
`Get` or `GetBf16` widened through `vt::BF16ToF32` — mirroring the connector's
`RegisterAt` (`ltx2_connector.cpp:179-188`), which solved the same problem
against the same bag.

**Narrowing at upstream's store points, and nowhere else.** A `NarrowBuffer`
equivalent to the connector's (`ltx2_connector.cpp:47-55`, through
`vt::F32ToBF16`) is applied at exactly the sites §3.1 measured: after each of the
six `Linear`s including its bias add (R1), after the modality-embedding add (R2),
after the GELU (R3), after the pooled attention output (R5), and after the `exp`
(R4). It is applied nowhere else, because a narrowing upstream does not do is the
same class of defect as one it does.

**A third width refuses by name**, mirroring `Ltx2LoadDurationHeadWeights`'s own
`kF32 || kBF16` check, with a message token no other site in this tree emits —
wave 4 established that asserting a shared refusal string gates a different site.

**The call sites.** `ltx2_video.cpp:1932` and `:1941` both ask for `kBF16`. They
are reverted **independently** in mutation, because wave 5's M9 reverted two
loader sites together and could not see either one alone.

---

## 5. Risks

1. **The funnel (§3.2 N1).** The single largest risk in this row and the reason
   its gate is shaped the way it is. Mitigated by per-rule coverage across
   fixtures plus a generator refusal; if any rule's coverage reaches zero the
   generator reds rather than emitting.
2. **The probe trap.** Reading a parameter after `.to(bfloat16)` yields a false
   0/0. Every probe captures parameters before any cast, and §0 quotes the
   resulting counts including the 47-of-48 control.
3. **The attention core (§3.2 N2).** Bounded by gating the pooler only at
   fixtures where the correct chain is bit-exact. If that population ever empties,
   the pooler gate is dead and must be replaced, not relaxed.
4. **The reader-anchor rot.** `test_ltx2_video`'s reader-anchor case reads
   `src/vllm/multimodal/ltx2_video.cpp` at RUN TIME. The list has rotted eight
   times. It is re-derived with the case's own rule **after** this row's edits,
   and no ltx2 suite is run while a mutation is applied — every mutation is
   restored **and rebuilt** before the next run.
5. **Regenerating a committed golden.** #2855 moved 4970 lines by changing the
   thread count. The bf16 section is a pure addition and `git diff --numstat` is
   checked for zero deletions.

---

## 6. Tests

1. **Red first, through the production entry point #2900 built.** `Load` with a
   `duration_head_path`, then `Generate` with an auto duration. At this base the
   bf16 arm refuses inside `Ltx2VaeWeights::Get`; the case asserts the bf16
   frame count and reds on that refusal.
2. **Per-rule value cases** against the new goldens, each asserting upstream's
   answer is equal AND the rejected hypothesis is **different** (`separating > 0`).
3. **The rule-coverage golden**, asserted non-zero for every rule, so a fixture
   set that stops separating reds rather than passing.
4. **The pooler**, at the bit-exact fixtures only.
5. **The STORAGE ratio**: two bags, same tensor set, `Bytes()` exactly 2:1.
6. **The frame count on the production path** — the §3.3 consequence, an integer
   equality, reached through `Load` + `Generate`.
7. **The `query_tokens` narrowing by COUNT** (§3.2 N3), with the control that
   the count can be non-zero.
8. **The third-width refusal**, on its own token.
9. **Mutations**, each confirmed *applied* and *compiled* before its result is
   read and each restored byte-for-byte: delete the production call site; revert
   each loader call site **alone**; flip each of R1-R5; size the bag by
   `sizeof(float)`.

---

## 7. Gates

Run from the row worktree. Every command here was executed before this section
was claimed.

```sh
# G0 — oracle identity. Nothing in §3 holds if this is another revision.
git -C ~/_git/LTX-2 rev-parse HEAD              # fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca
git -C ~/_git/LTX-2 status --porcelain          # empty

# G1 — the TWO production loader call sites, counted by string.
grep -c 'Ltx2LoadDurationHeadWeights(' src/vllm/multimodal/ltx2_video.cpp   # 2

# G2 — the goldens are upstream's, regenerated at the pin, and a pure ADDITION.
python3 scripts/gen-ltx2-duration-wire-goldens.py --ltx2 ~/_git/LTX-2 \
    --out tests/vllm/models/ltx2_duration_wire_goldens.inc
git diff --numstat -- tests/vllm/models/ltx2_duration_wire_goldens.inc   # deletions 0

# G2b — determinism: thread-count independent (#2855).
OMP_NUM_THREADS=1 python3 scripts/gen-ltx2-duration-wire-goldens.py \
    --ltx2 ~/_git/LTX-2 --out /tmp/dh1.inc
OMP_NUM_THREADS=8 python3 scripts/gen-ltx2-duration-wire-goldens.py \
    --ltx2 ~/_git/LTX-2 --out /tmp/dh8.inc
cmp /tmp/dh1.inc /tmp/dh8.inc

# G3 — the focused suites.
ctest --test-dir build -R 'ltx2_video|ltx2_pipeline' --output-on-failure

# G4 — the full gate, and the two checks preflight SKIPS.
scripts/agent-preflight.sh
python3 scripts/check-pr-size.py --base origin/main --head HEAD
python3 scripts/agent-issue-index.py --refresh && python3 scripts/check-agent-record.py
python3 scripts/agent-pr-body.py --pr <N>
```

---

## 8. Evidence

Recorded in `## Outcome` at `DONE`: the §0 gateability commands literally; the
§3 probes with their separating counts including the three negatives and the
control that each negative is not blind; the literal red of §6.1 and the literal
green after in doctest's own output; every mutation with its literal assertion
count including the production-call-site deletion; and the gate log lines rather
than an exit code.

---

## 9. Stop conditions

* **A GPU lease, a mount or a download turns out to be needed.** Stop and report;
  §0 settled that none is, and if that is wrong the row's premise is wrong.
* **A rule's coverage reaches zero** and no fixture separates it. Record it as
  unmeasurable by name, as §3.2 N3 does — never widen a band to reach it.
* **The band would have to reach a rejected rule's distance.** The gate is then
  dead. Replace it; do not relax it.
* Return `NEEDS_DECISION` rather than widening a golden tolerance to pass.

---

## Owed

* **The FP8 and NVFP4 arms** of the duration head — A22, upstream's four
  quantization policies (`quantization_factory.py:22-26`). `ReadTensorBytes`
  already refuses them by name.
* **A CUDA arm.** The head runs on a default-constructed `vt::Queue`; a device
  arm is a residency row, not a dtype row.
* **A real-weight render.** No checkpoint carrying a `duration_head.` block is
  on the NAS at this base and `CHECKPOINT_ROOT` does not resolve on this box.
  Pre-existing on the f32 arm; neither widened nor closed here.
* **`query_tokens`' narrowing is unmeasurable by any value gate at this
  component's shipped widths** (§3.2 N3): 0 of 93 bit-exact pooler fixtures
  separate it. It is gated by a count, and the value gate for it is owed.
* **`GeluTanh`'s f64 pointwise expression stays WIDER than upstream on the f32
  arm.** It is bit-equal at bf16 (§3.1), so this row neither needs nor makes the
  change; `ltx2_duration_head.cpp:69-78` keeps recording it as visible debt.
* **A24's population beyond its eight components.** §A.7 returns the **audio
  VAE** under its own `## Owed`: upstream constructs `AudioDecoder` in
  `self.dtype` too (`distilled.py:156`) and this tree runs it f32, *argued*
  rather than owed, because upstream's own BWE path pins the vocoder chain to
  float32 (`vocoder.py:575-580`) and this port extends that contract up to the
  spectrogram decoder above it. **Whether that extension is sound is a question
  no marker sweep can answer**, and it is not this row's to settle.
* **§A.7's anchor for this component**, `ltx2_duration_head.h:55-58`, is the
  DTYPE block this row rewrites, so the line numbers move. §A.7 is
  operator-owned: recorded here, not edited there — the same handling waves 4
  and 5 gave their own findings.
