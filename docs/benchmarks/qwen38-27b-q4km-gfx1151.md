# `qwen38-27b-q4km-gfx1151` — Qwen3.8-27B Q4_K_M on Strix Halo, three engines

llama.cpp, the pinned vLLM and vllm.cpp were run on one board, one artifact, one
prompt and one token count, interleaved in a single lease. This file has what
each engine did, including the engine that did not run.

## Disposition

Mixed, and the mixture is the point.

**Two engines produced figures. Ours produced none.** vllm.cpp faulted the board
on every leg. That is reported below as a fault rate, not as a missing row and
not as a number with an asterisk, because for that arm the fault rate is the
result.

## Correctness comes first, and it is not passing

**This is a survey. No ratio on this page is a gated result.** It is published
under a recorded developer decision of 2026-09-04, and the correctness state is
part of the table rather than a footnote:

| comparison | divergent prompts |
|---|---|
| `TOKEN_GATE` | **FAIL** |
| vllm.cpp vs llama.cpp `b10451` | 3 of 6 |
| vllm.cpp vs vLLM (`torch.compile`) | 5 of 6 |
| vLLM (`torch.compile`) vs llama.cpp `b10451` | 3 of 6 |
| vLLM (eager) vs llama.cpp `b10451` | 4 of 6 |

Every divergence is a near-tie at about 0.125 nats, one bf16 ULP.

**No deterministic denominator exists on this path.** llama.cpp's greedy decode
is not stable across its own kernel paths, and vLLM disagrees with itself across
`enforce_eager` on both 27B models tried. Limb 3 of the ratified methodology is
unsatisfied and may be unsatisfiable here.

These four counts were **carried into this run from earlier evidence**, not
re-measured by it. The fold script writes them as constants. They describe the
same board and the same artifact, and no leg of this run re-derived them.

## Subject

| | |
|---|---|
| artifact | `Qwen3.8-27B-Q4_K_M.gguf`, 17,106,775,008 B, sha256 `7e78da5d…fe169` |
| repo | `unsloth/Qwen3.8-27B-GGUF` @ `fe1e2a23d973adb629709749dc4f6756df66ef10` |
| device | AMD Radeon 8060S, `gfx1151`, ROCm 7.2.4, 64 GiB, `strix:gpu0` |
| lease | `rc` job `b51afb48-ddb6-438a-b30c-acf46980e918`, boot `a5bc8128…f88032e1` |
| prompt | `The capital of France is`, 5 tokens, greedy, `--temperature 0` |
| generation | 64 tokens |

The artifact's sha256 was recomputed on the worker before any timing ran.

## Method

Four rounds. Every arm ran in every round, and the arm order rotates each round,
so drift along the session cannot be read as a difference between engines. Run 1
of each in-process arm is cold and is discarded. A clock sampler ran at 4 Hz
beside every leg, writing to worker-local disk.

```sh
# llama.cpp, pure decode
llama-bench -m <gguf> -p 0 -n 64 -ngl 99 -r 3 -o json

# vLLM, production configuration (NOT --enforce-eager)
LLM(model=<gguf>, quantization="gguf", gpu_memory_utilization=0.75,
    max_model_len=4096, enforce_eager=False)

# vllm.cpp
vllm-cli --model <gguf> --prompt 'The capital of France is' --max-tokens 64 \
  --temperature 0 --repeat 4 --max-num-seqs 1
```

## Results

| engine | legs | fault rate | figure | spread |
|---|---|---|---|---|
| llama.cpp `b10451` | 4 of 4 | 0 of 4 | **12.219 tok/s** decode | 0.133% |
| vLLM `0.26.0.dev0+g5559679229` | 4 of 4 | 0 of 4 | **6.734 tok/s** whole completion | 0.192% |
| vLLM, decode only (derived) | 4 of 4 | 0 of 4 | **11.056 tok/s** | 0.835% |
| **vllm.cpp `11fed3ba5`** | **0 of 4** | **4 of 4, `rc=139`** | **none — FAULTED** | — |
| llama.cpp `llama-cli` control | 0 of 4 | 0 of 4 | none — harness defect | — |

### The ratio depends on which definition you take

`llama-bench -p 0` excludes the prompt and every per-request cost. The vLLM leg
times a whole `generate()` call. Those are two different quantities, and the
difference between them is not small here:

| basis | llama.cpp / vLLM |
|---|---|
| whole completion vs pure decode, **mismatched** | 1.814x |
| **decode against decode** | **1.105x** |

**The 1.814x figure compares two different things and should not be quoted.**
The survey's own fold emits it because the leg that would have supplied a
matched llama.cpp row timed out (see below). Read the 1.105x.

### How the decode-only vLLM figure was derived

It is a derivation from two token counts, not a direct reading, and is labelled
so wherever it appears. Each leg timed 64-token and 128-token completions of the
same prompt. The slope between them removes whatever fixed cost each call pays:

| leg | t(64) s | t(128) s | slope tok/s | 1-token call s |
|---|---|---|---|---|
| r1 | 9.5094 | 15.3057 | 11.041 | 3.860 |
| r2 | 9.4913 | 15.2916 | 11.034 | 3.848 |
| r3 | 9.5018 | 15.2541 | 11.126 | 3.853 |
| r4 | 9.5052 | 15.2861 | 11.071 | 3.860 |

Median slope **11.056 tok/s**, spread 0.835%.

The slope implies a fixed cost of about **3.72 s per `generate()` call**. The
independently measured 1-token call reads **3.85 s**, which is that same fixed
cost plus one token. Two separate readings of the overhead agree to about 4%, so
the linear model the slope assumes is at least consistent with the data. It rests
on two token counts and is not a substitute for measuring decode directly.

That overhead is what makes vLLM's 64-token whole-completion figure 6.734 rather
than 11.056. **This page does not attribute it.** Whether it is prefill of a
5-token prompt, scheduling, detokenization or synchronization was not measured.

### Per-leg readings

**llama.cpp**, `tok/s` and the clock window over each leg:

| leg | tok/s | 3 reps | sclk MHz mean | busy % mean |
|---|---|---|---|---|
| r1 | 12.2296 | 12.2996 / 12.1756 / 12.2136 | 1924.9 | 45.9 |
| r2 | 12.2133 | 12.2632 / 12.1841 / 12.1927 | 2062.8 | 66.4 |
| r3 | 12.2221 | 12.2796 / 12.2139 / 12.1729 | 2079.2 | 65.7 |
| r4 | 12.2163 | 12.2068 / 12.2336 / 12.2084 | 2097.4 | 67.3 |

Median 12.2192, spread 0.133% of the median.

**vLLM**, warm runs only:

| leg | whole-completion tok/s | sclk MHz mean | busy % mean | load s |
|---|---|---|---|---|
| r1 | 6.7302 | 1736.2 | 51.3 | 527.4 |
| r2 | 6.7431 | 1855.3 | 56.8 | 491.1 |
| r3 | 6.7355 | 1905.0 | 58.8 | 476.5 |
| r4 | 6.7332 | 2030.4 | 64.8 | 429.0 |

Median 6.7344, spread 0.192%. Across all 12 warm repetitions: median 6.7362,
range 6.7235 to 6.7656.

The `sclk` means climb across rounds on both arms, from 1925 to 2097 MHz on
llama.cpp and 1736 to 2030 MHz on vLLM, while the throughput of both moves by
less than 0.2%. The board warms over the session and neither figure follows it.

## The vllm.cpp arm faulted on every leg

All four legs died the same way:

```text
[vt op-provider] op=... device=5 selected=vt-native   (17 to 21 ops)
HW Exception by GPU node-1 (Agent handle: 0x...) reason :GPU Hang
```

Exit 139, about 80 s into each leg, on the first forward pass. Board clocks
during those legs read 749 to 923 MHz mean and 6.1 to 12.2 percent busy, so the
board never reached a working state. `VT_OP_PROVIDER_STATS=1` reported
`reference_tier_notices=0` on every leg and every op resolved `selected=vt-native`,
so no host fallback ran.

**This row is void as an engine verdict, and it is published rather than
omitted.** The binary measured was built from `11fed3ba56b8f823c07032416982a44a8c0967b5`,
which is an **ancestor of `6b97a6800`**, the commit that fixed exactly this hang
by refusing `hipMallocManaged` on a part that reports `PageableMemoryAccess = 0`.
That fix reached `main` as `27da7787e` at 2026-09-02 15:11 UTC. This survey
started 2026-09-04 22:29 UTC, two days and seven hours later, and did not carry
it, because the
harness records our revision and never asserts it, unlike the llama.cpp source
manifest which it does assert and would fail on. That gap is
[#2933](https://github.com/mudler/vllm.cpp/issues/2933); the fault itself and its
fix are [#2511](https://github.com/mudler/vllm.cpp/issues/2511).

**Whether the current head completes this workload on `gfx1151` is not
established by this run, and is not claimed here.** It needs a fresh lease with
the revision asserted. The commit that fixed #2511 measured 0 failures in 21 legs
against 17 in 21 for the managed allocator, but that was a different workload in
a different lease and it is not this survey's result.

## The end-to-end control arm was lost

`llama-cli` was in the design so that llama.cpp would have a row on the same
whole-completion definition as vLLM, rather than the survey comparing pure
decode against a whole request. It ignored `-no-cnv`, opened its interactive
chat UI, answered the prompt, and then printed a `> ` prompt against
`/dev/null` until the 25-minute timeout, producing about 24 GB per leg. Mean
`sclk` was 606 MHz at 0.2 percent busy, so the board was idle throughout. That is
[#2935](https://github.com/mudler/vllm.cpp/issues/2935).

**It does not affect the llama.cpp figure**, which comes from `llama-bench`, a
different binary that completed 4 of 4. What it costs is the matched-definition
row, which is why the 1.105x above had to be derived from vLLM's slope instead of
read off a llama.cpp leg.

Each leg did print its own rate before it hung: generation 12.2, 12.2, 12.2 and
12.1 t/s. That is llama.cpp's self-reported figure at one decimal place, and it
agrees with `llama-bench`. It corroborates and is not counted as a leg.

## Agreement with the landed denominator

The landed llama.cpp denominator is **12.233 tok/s**, median of 6 legs, spread
0.303%, range 12.218 to 12.255, taken in a different lease on 2026-09-02.

This run reads **12.219**, median of 4, spread 0.133%, range 12.213 to 12.230.

The two differ by 0.11% and their ranges overlap between 12.218 and 12.230. They
agree. The landed figure stands; nothing here supersedes it.

## Limitations

- **No vllm.cpp figure exists on this page.** There is no throughput number for
  our own engine on this board, at this artifact, at any revision.
- The four correctness counts are transcribed from earlier evidence and were not
  re-measured by this run.
- The decode-only vLLM figure is derived from two token counts and assumes the
  cost is linear in tokens. It is not a direct decode measurement.
- The matched whole-completion comparison has no llama.cpp side at all, because
  that arm timed out.
- vLLM ran on the host out of its own venv while the other arms ran in a
  container. Both open `/dev/kfd` directly. The asymmetry is recorded, not
  corrected.
- llama.cpp's custody chain has one open link, carried forward: the source is
  pinned by content manifest and the binaries are pinned as bytes, but no
  compiler ran in this lease.
- Clock sampling on AMD is ad-hoc. No in-tree harness samples AMD clock state.

## Evidence

`docs/bench-evidence/qwen38-27b-q4km-gfx1151-survey-20260904/` holds the job
log, the three harness scripts, the clock sampler, the per-leg captures and the
fold's `RESULT`. JSON captures carry a `.json.txt` suffix so that they classify
as evidence rather than as public documents.

Raw per-leg artefacts, including the 24 GB `llama-cli` captures which are not
committed, remain at `/mnt/nas_share/rc/strix-survey-2497/out/survey-20260904/`.

Issues: [#2921](https://github.com/mudler/vllm.cpp/issues/2921) the survey,
[#2497](https://github.com/mudler/vllm.cpp/issues/2497) the missing quant-matched
number, [#2933](https://github.com/mudler/vllm.cpp/issues/2933) the unasserted
revision, [#2935](https://github.com/mudler/vllm.cpp/issues/2935) the control arm.
