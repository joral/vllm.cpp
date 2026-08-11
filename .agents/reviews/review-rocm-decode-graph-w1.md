## Review: `150b8f40` — W1 of `BACKEND-ROCM` (hipGraph capture seam)

**Reviewer contract:** `.agents/prompts/reviewer.md` (`prompt-contract-version: 1`)
**Commit:** `150b8f40bc81507b0a2f6f37c30dc1e27474417e`
**Diff range:** `a706350d..150b8f40` (3 files, +260/-4)
**Board:** AMD Radeon RX 9060 XT, `gfx1200`, Navi 44, RDNA4, discrete, ROCm 7.2.3

### `REV-STATIC` — Complete

**Faithfulness to the CUDA mirror (`cuda_backend.cu`).** I diffed all six virtuals
line by line against the `--- CUDA-graph capture/replay` block. The port is
call-for-call. Three deviations exist, all identified and assessed:

| Deviation | Assessment |
|---|---|
| `hipGraphInstantiate` 5-arg `(&exec_, graph, nullptr, nullptr, 0)` vs CUDA 3-arg `(&exec_, graph, 0)` | **Correct.** Verified the ROCm 7.2.3 header signature: `hipGraphInstantiate(hipGraphExec_t*, hipGraph_t, hipGraphNode_t*, char*, size_t)`. Passing `nullptr, nullptr, 0` is the correct no-error-node, no-log-buffer instantiation. |
| `Check()` on `hipGraphExecDestroy`/`hipGraphDestroy` where CUDA ignores returns | **Design choice**, consistent with `Free`/`FreePinned` in the same file. However — see finding LOW-1: the stated justification is factually wrong. |
| `VT_BENCH_PROFILE_CONTROL` omitted from `ReplayGraph` | **Deliberate**, spec §2, documented in code comment. |

**Claim verification:**
- **Claim 1** (`SupportsGraphCapture()` true): **Verified.** Base impls in
  `backend.cpp` throw via `VT_CHECK(false, ...)`; `rocm_backend.hip` overrides
  to `true`.
- **Claim 2** (replay re-executes): **Verified** via mutation — the load-bearing
  step 5 is caught by every relevant mutation.
- **Claim 3** (pre-warmed GEMM): **Verified.** The test pins the mitigation,
  which is the correct design choice — forbidding the cold-throw path would
  ratchet against a future capture-safe allocator.
- **Claim 4** (no model-level changes): **Verified.** `rocm.cpp:67` has only a
  comment; no `support_static_graph_mode()` override exists. Inherits `false`
  from `interface.h:189`.
- **Claim 5** (`VT_BENCH_PROFILE_CONTROL` not ported): **Verified** in code and
  spec.

### `REV-MUTATION` — Complete

All mutations were run in a scratch worktree (`git worktree add <path> 150b8f40`)
on a copy of `rocm_backend.hip`, rebuilt, tested, then restored byte-for-byte
(sha256 `4a6abb18...` before and after, proven each time). Scratch worktree
removed after review.

| # | Mutation | Result | Catches |
|---|---|---|---|
| 1 | `SupportsGraphCapture() → false` | RED: 9/10 pass, 1 fail | Step 1 assertion |
| 2 | `Replay() → no-op` | RED: 6/10 pass, 4 fail | Steps 4–5 (stored path) |
| 3 | `EndCapture` executes once before returning | RED: 9/10 pass, 1 fail | Step 3 (recorded-not-executed) |
| 4 | `ReplayGraph() → no-op` | RED: 7/10 pass, 3 fail | Steps 4–5 (handle path) |
| 5 | `ReplayGraph` fires once then freezes | RED: 8/10 pass, 2 fail | Step 5 only — **the load-bearing step** |
| 6 | `Replay() → no-op` on GEMM test | RED: 0/2 pass, 2 fail | Pre-warm replay correctness |
| 7 | `BeginCapture() → no-op` | RED: exception at `hipStreamEndCapture` | Capture mode never entered |

Mutations **not caught** (coverage gaps, not implementation bugs — the code is
correct):

| # | Mutation | Result | Why not caught |
|---|---|---|---|
| 8 | `EndCaptureGraph` returns stale `exec_` | GREEN: 10/10 | Both graphs capture the identical `Copy(dst, src)` — test cannot distinguish them |
| 9 | `EndCaptureGraph` leaks `graph_t` | GREEN: 10/10 | No leak detection in unit test |
| 10 | `EndCapture` skips prior `exec_` destroy | GREEN: 10/10 | Same — leak, not correctness |

The author's two reported mutations (ReplayGraph no-op → 3 failures; freeze →
step 4 passes, step 5 fails) are independently confirmed by mutations 4 and 5
above.

### `REV-FULL-GATE` — Complete, run once on unchanged `150b8f40`

| Gate | Result |
|---|---|
| 1: HIP build, gfx1200, `-Werror` | **0 warnings**, clean build. |
| 1: Non-HIP CPU syntax check (`vllm_rocm_platform_syntax_check`) | **Compiles clean** — `rocm.cpp` + `test_rocm_backend.cpp` object-compiled. |
| 2: `ctest -R 'rocm\|cross_device'` | **3/3** (`test_backend_cross_device`, `test_rocm_arch`, `test_rocm_backend`). |
| 2: New capture case | **10/10 assertions**. |
| 2: New GEMM case | **2/2 assertions**. |
| 7: `check-agent-record` | **OK** |
| 7: `check-public-doc-tables` | **OK** |
| 7: `check-test-registration` | **OK** |
| 7: `check-device-leakage` | **OK**, DSR **32 == 32**, ratchet holds |
| 7: `test_release_archive` + `test_release_metadata` | **FAIL** — confirmed on base `a706350d` (NixOS `/nix/store` RPATHs vs bundle-relative). Known-unrelated. |
| 7: `test_agent_onboard` | **FAIL** — confirmed on base (host `init.defaultBranch=main` vs test's `master`). Known-unrelated. |

**Note:** The review envelope's cmake command omits `-DVLLM_CPP_HIP=ON`.
Without it, `VLLM_CPP_HIP` defaults to `AUTO` which resolves to `OFF`
(`CMakeLists.txt:297-298`), producing a CPU-only build with no HIP code
compiled. The dev shell greeting's command (`-DVLLM_CPP_HIP=ON
-DROCM_PATH=$ROCM_PATH`) is the correct one. The author evidently used the
correct flags; the gate results are valid.

### Findings (severity-descending)

---

**LOW-1 — False `[[nodiscard]]` claim in code comment and commit message**

`src/vt/rocm/rocm_backend.hip`, comment above `DestroyGraph`:

> `hipGraphExecDestroy is [[nodiscard]] where cudaGraphExecDestroy is not, so the CUDA leg's bare call would not compile here.`

Same claim repeated in the commit message:

> `hipGraphExecDestroy and hipGraphDestroy are [[nodiscard]] where their cudaGraph* counterparts are not, so the mirrored bare calls do not compile.`

**Evidence:** I compiled a bare
`hipGraphExecDestroy(reinterpret_cast<hipGraphExec_t>(graph))` call (no `Check`,
no `(void)` cast) under the same `-Werror` HIP build. It compiled with **0
warnings**. The ROCm 7.2.3 header declares `hipError_t hipGraphExecDestroy(hipGraphExec_t graphExec)`
with no `[[nodiscard]]` attribute. Same for `hipGraphDestroy`.

**Violated rule:** Code comments and commit messages must state accurate
technical facts. The `Check()` is a valid design choice (consistency with
`Free`/`FreePinned`), but the stated justification is false — a future
maintainer would be misled into thinking the check is mandatory.

**Required remediation:** Rewrite the comment to justify `Check()` on its
actual merits — "Checked, matching Free/FreePinned in the same file: a failing
destroy is a leak this backend would rather report than swallow" — without the
false `[[nodiscard]]` claim.

---

**INFO-1 — Behavioral asymmetry: `DestroyGraph` throws where CUDA silently ignores**

`src/vt/rocm/rocm_backend.hip`, `DestroyGraph`: CUDA's leg ignores
`cudaGraphExecDestroy`'s return; HIP's leg `Check()`s it, throwing on failure. A
decode-graph teardown (W2+) that destroys an exec still in flight would throw on
HIP where CUDA succeeds silently. Not a W1 correctness issue — the model-level
path is not engaged (`support_static_graph_mode()` = false), and the tests
always `Synchronize` before `DestroyGraph`. Flag for W2: the decode-graph class
must synchronize before destroying on HIP.

---

**INFO-2 — Test coverage gap: handle API independence not distinguishable**

`tests/vt/test_rocm_backend.cpp`, handle variant section: returning stale
`exec_` instead of the local `exec` from `EndCaptureGraph` passes the test
(mutation 8, GREEN) because both graphs capture the identical `Copy(dst, src)`
operation. Not an implementation bug — the code correctly returns the local
`exec`. A stronger test would use distinct operations per path, but this is not
required for W1's scope.

---

### `remaining_concern`: NONE

The `Check()` on `hipGraphExecDestroy` in `EndCapture` (destroying a prior
`exec_` during recapture) could throw if the prior exec is still in flight,
where CUDA would silently ignore. The same asymmetry as INFO-1 applies. For W1,
the test always synchronizes before re-capturing. This is a W2 concern, not a
W1 gap.

---

### Verdict: **PASS**

The implementation is a faithful call-for-call port of the CUDA capture block to
hipGraph, with correct API adaptations (`hipGraphInstantiate` 5-arg form). Every
guarantee the tests claim to pin — re-execution over persistent buffers (the
load-bearing one), recorded-not-executed, handle API path — is caught by
mutation testing. The declared gates (1, 2, 7) pass cleanly with only
confirmed-known-unrelated failures. The one substantive finding (LOW-1: false
`[[nodiscard]]` comment) is a documentation inaccuracy that does not affect
correctness.
