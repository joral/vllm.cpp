# Review: `619d193946542d71c46d86efde7366caed113d22` — W2 and W3 of `BACKEND-ROCM`

**Reviewer role:** read-only (fresh reviewer, REV-STATIC | REV-MUTATION | REV-FULL-GATE)
**Worktree:** `/tmp/rev-rocm-w2` at `619d193946542d71c46d86efde7366caed113d22`
**Scope:** `46d2f4c6..619d1939` — two commits, three files (+204/−8)
**Sha256 proof of restore:** `src/vllm/platforms/rocm.cpp` = `14bd0ba82868b8bcf15e10a6450c5f68d8a1946d9299dd302fcd4c85c5badfee` (unchanged before, during, and after all scratch mutations)

---

## Static review

**Product code** — one line in `src/vllm/platforms/rocm.cpp:81`:

```cpp
bool support_static_graph_mode() const override { return true; }
```

Verified against the pinned oracle container source
(`vllm-rocm-oracle:555967922-gfx1200`):

- `rocm.py:1001-1002` — `return True`, unconditional ✓
- `cuda.py:662` — `return True` ✓
- `interface.py:1191-1195` — default `return False` ✓

The mirror is exact. Upstream does not gate this predicate on arch (`on_gfx9`/
`on_gfx1x` gates elsewhere are for attention kernel selection, not graph capture
support — independent concerns). The W0 placeholder comment about "stays false"
is removed, and the new comment correctly documents both D1 (hipBLASLt lazy init)
and D6 (DestroyGraph Check asymmetry) as the capture-contract hazards the
decode-graph class must honour on HIP.

**No model-level edit** — confirmed by inspection. Every decode-graph class gates
on `support_static_graph_mode() && SupportsGraphCapture()` (both true for kROCM
after W1+W2) with no `is_cuda()` anywhere in the model code. The seam claim in §2
holds.

**Test change** — `CHECK(rocm.support_static_graph_mode())` added to
`test_rocm_backend.cpp:292`. Mutation test confirmed it catches the defect.

**Spec** — I reviewed every claim against my own independent measurements.

---

## Mutation evidence

Flipped `return true` to `return false` in `rocm.cpp:81`. Rebuilt. Ran
`test_rocm_backend`:

```
ERROR: CHECK( rocm.support_static_graph_mode() ) is NOT correct!
values: CHECK( false )
test cases: 9 | 8 passed | 1 failed
```

Restored via `git checkout`; sha256 verified byte-for-byte:
`14bd0ba8...5badfee`. Rebuilt clean; 9/9 passed again.

---

## Full gate (run on own worktree at `619d1939`)

| Gate | Result |
|---|---|
| **1** Clean `-Werror` build, gfx1200, 0 warnings | **PASS** — full rebuild from scratch, 0 warnings in build log |
| **2** `ctest -R 'rocm\|cross_device'` | **PASS** — 4/4; `test_rocm_backend` 9 cases / **1059** assertions |
| **7** `agent-preflight.sh` | All relevant checks OK; 6 failures all known-unrelated |
| `check-device-leakage` | DSR **32 == 32**, ratchet holds |
| `check-agent-record` | OK |

**Known-unrelated preflight failures** (all reproduce on unmodified base):
- `audit-live-rows` / `test_audit_live_rows` — 1 abandoned ACTIVE row
  (`ENG-FORGE-COAUTHOR`), present on `origin/main`
- `test_release_archive` / `test_release_metadata` — NixOS `/nix/store` RPATHs
- `test_agent_onboard` — host `init.defaultBranch=main` vs test's `master`
- `role-undeclared` — this session's own read-only claim

`origin/main..HEAD` = 8 commits (W1 4 commits + W2 + W3 + 2 others), so
`doc-checkpoint range` and `commit-trailers` widen the range. Neither fires.

---

## Independent re-measurement of every load-bearing claim

### Claim 1: "Capture buys ~nothing" — REPRODUCED

My A/B on Qwen3-0.6B, 128in/128out batch 8, 3 reps:

| Rep | Capture ON (tok/s) | Capture OFF (tok/s) | Delta |
|---|---|---|---|
| 1 | 192.83 | 191.15 | +0.9% |
| 2 | 187.96 | 188.52 | −0.3% |
| 3 | 191.93 | 189.08 | +1.5% |

Spec reported +3.2%. My average delta is +0.7% (within noise — difference < 1
SE). Both agree: capture does not meaningfully move throughput. The 126 replays
prove capture genuinely engaged at this config.

Qwen3-1.7B: ON 150.03/147.96, OFF 148.68/138.16 — flat within run variance.
Qwen3-4B: ON 100.57/101.67, OFF 100.51/100.67 — flat.

### Claim 2: "Not a batch-size artifact" — REPRODUCED

Single-stream (concurrency 1, 4 prompts):

| | tok/s | TPOT |
|---|---|---|
| Capture ON | 50.30 | 13.79 ms |
| Capture OFF | 49.72 | 13.98 ms |

No gain in capture's most favourable configuration. The reasoning (batch 8
amortises per-step launch cost over 8 tokens) is correct.

### Claim 3: "Capture removes no host CPU work" — REPRODUCED

My `bash time` on the same 4-prompt concurrency-1 workload:

| | wall | user | sys |
|---|---|---|---|
| Capture ON | 11.746s | 14.048s | **14.186s** |
| Capture OFF | 11.801s | 13.836s | **13.792s** |

Sys is marginally HIGHER with capture ON (Δ = +0.39s), matching the spec's
pattern (Δ = +0.55s). The crux claim reproduces. `bash time` is coarse — it
doesn't isolate decode-phase CPU — but it's adequate for the claim being made.
The spec appropriately names `rocprof` as the next step.

### Claim 4: Gate 4 (126 replays at batch 8) — REPRODUCED

```
[Qwen3DenseDecodeGraph] captured dense decode graph for padded size S=8 (real B=8)
[Qwen3DenseDecodeGraph] dense decode graph: 126 total replays across 1 captured size(s)
```

### Claim 5: Byte-identical output, 4 models — REPRODUCED

| Model | Capture ON | Capture OFF | Match |
|---|---|---|---|
| Qwen3-0.6B | ` 1000000` | ` 1000000` | ✓ |
| Qwen3-1.7B | ` Paris. 1. What is the capital of France? 2. What` | identical | ✓ |
| Qwen3-4B | ` in which country? The capital of France is in France. But wait, that` | identical | ✓ |
| Qwen3.5-0.8B | `:\nA. Paris\nB. London\nC. London\nD.` | identical | ✓ |

### Claim 6: 2×2 oracle anti-symmetry — REPRODUCED

Ran both containers on the identical prompt:

| Build | `enforce_eager=True` | `enforce_eager=False` |
|---|---|---|
| vLLM 0.19.1 | ` 1000000` | ` Paris, and the capital of the United` |
| vLLM 555967922 (pin) | ` Paris. The capital of the United States` | ` 1000000` |

Exactly matches the spec. Perfectly anti-symmetric. Both builds flip on capture
config in opposite directions.

---

## Findings (severity-descending)

### INFO-1 — Gate 3 attribution vs W2 work breakdown

**Path:** `.agents/specs/rocm-decode-graph.md`, §7 gate 3 vs §9 W2.

**Evidence:** Gate 3 section says "MET in W2, on four models rather than the one
required," but the W2 work breakdown says "Gates 3 and 4 met above, on
Qwen3-0.6B AND Qwen3.5-0.8B." The 1.7B and 4B correctness checks were performed
during W3 (as part of the A/B), not W2.

**Violated rule:** Record accuracy (AGENTS.md: "each fact lives in exactly one
of them" / git cannot disagree with the tree).

**Required remediation:** Change "MET in W2, on four models" to "MET in W2/W3,
on four models," or split the table to show which models were checked in W2 vs
W3. Minor — the four-model table is correct and authoritative; only the
"in W2" label overstates which week produced which rows.

### INFO-2 — Spec's A/B delta is on the high side

**Path:** `.agents/specs/rocm-decode-graph.md`, §7 gate 5 table.

**Evidence:** The spec reports +3.2% A/B delta for Qwen3-0.6B. My independent
3-rep measurement averaged +0.7% (within noise). The conclusion is unchanged
(capture buys nothing) and the §10 stop condition triggers identically
regardless. But a reader might attribute more meaning to +3.2% than the data
supports.

**Violated rule:** No rule violated — the direction and conclusion are correct.
This is an observation about measurement noise, not a factual error.

**Required remediation:** None required. A note that the +3.2% is near the top
of the observed noise band would improve precision, but the negative result is
honest either way.

### INFO-3 — D6 teardown risk on the recapture path

**Path:** `.agents/specs/rocm-decode-graph.md` §8 D6; `src/vt/rocm/rocm_backend.hip:257,301-304`;
`src/vllm/model_executor/models/qwen3.cpp:601`.

**Evidence:** The `DestroyGraph`/`EndCapture` `Check()` asymmetry (HIP throws,
CUDA ignores) is a latent risk. With W2 engaging the model path, the recapture
path in `qwen3.cpp:601` calls `b.DestroyGraph(s.graph)` without explicit
synchronization when block-table columns change. Currently safe because graph
capture runs only on the depth-1 sync path (`input.device_token_ids != nullptr`
guard at `qwen3.cpp:703`), which has implicit synchronization between steps. But
this is an unguarded path for future async work.

**Violated rule:** None currently — properly recorded in the spec as a
consequence for W2.

**Required remediation:** None for this review. The risk is documented and the
guard preventing async access (`device_token_ids != nullptr`) is in place.

### INFO-4 — Enabling capture for untested models

**Path:** `src/vllm/platforms/rocm.cpp:81` (the one-line change).

**Evidence:** Enabling capture unconditionally for all decode-graph models
(Qwen3 dense/MoE, DeepSeek-V2/V4, Voxtral, Laguna, etc.) when only Qwen3 dense
and Qwen3.5 were tested. However, capture failures are LOUD
(`hipStreamEndCapture` throws on any illegal op mid-capture, including the D1
lazy-init hazards). No silent corruption risk. An untested model hitting a shape
not covered by pre-warm would abort visibly, not produce wrong output.

**Violated rule:** None — upstream is unconditional too, and the failure mode is
loud.

**Required remediation:** None. The behavior mirrors upstream's unconditional
`True`, and any failure surfaces as a thrown error, not silent corruption.

---

## Remaining concern

The §1 "SUPERSEDED BY MEASUREMENT" marker is methodologically the right call —
a pre-registered prediction shown wrong is more valuable than one silently
rewritten — but the two superseded paragraphs still present the fit's
conclusions in present tense ("the expected outcome is all three sizes
converging on ~1.36x"). A reader who skims the bold marker might still absorb
the prediction as current. This is a readability concern, not a correctness
defect.

---

## Verdict: **PASS**

The one-line product change is an exact mirror of upstream (`rocm.py:1001-1002`,
unconditional `True`). The test assertion catches its mutation. All gates pass on
the unchanged head. Most critically, I independently reproduced every
load-bearing claim in the spec — the negative A/B result, the single-stream
null, the CPU-usage pattern, gate 4's 126 replays, gate 3's byte-identical
outputs across four models, and the 2×2 oracle anti-symmetry. The negative
result is honest, the refutation is well-argued, and the methodology
(pre-registered prediction kept and superseded, not rewritten; stop condition
triggered as written) is sound.
