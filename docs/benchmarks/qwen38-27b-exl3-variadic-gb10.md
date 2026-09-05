# `qwen38-27b-exl3-variadic-gb10` — Qwen3.8-27B EXL3 under a mixed-length serving load

Their engine and ours, on one GB10 board, behind one client, over a prompt-length
distribution that spans short questions to long code reviews, swept over
concurrency, reported as percentiles.

Its method is [`variadic-load-methodology`](variadic-load-methodology.md). Read
that before you quote anything here. Its predecessor is
[`qwen38-27b-exl3-gb10`](qwen38-27b-exl3-gb10.md), which measured one prompt band
at concurrency 1 and published means; nothing on this page supersedes its
ratios, because the two pages measure different loads.

## Disposition

**PENDING.** The harness, its controls and its method are committed. The run is
queued on `dgx:gpu0`. No number on this page is measured yet, and the tables
below carry the shape the report writes and no values.

## Two things that will be true of every number here

**No correctness gate covers this run, and none can.** Both engines sample at
`T = 0.6`, so the two token streams are not expected to match and a token-exact
comparison cannot run on them. Nothing scores the text either engine writes.
This is the predecessor's position and it is unchanged.

**The comparator serializes generation.** `tools/serve_openai.py:45` creates
`gen_lock` and line 315 holds it around the entire `generator.iterate()` loop.
Their own docstring at line 18 says "requests are serialized (batch-1 draft);
concurrent callers queue." At any concurrency above 1 their figures therefore
measure queueing and ours measure batching. That is a property of the software
as shipped and it is the reason a ratio at `c = 8` is not an engine-versus-engine
kernel result. It is stated here, above the tables, rather than under them.

## Subject

| | |
|---|---|
| target | `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` @ `19441ac874c4018295da848e250f23511361cda4` |
| draft | `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` @ `4f0436269bca761b071f05319e8e04a87cc633f9` |
| device | NVIDIA GB10, compute capability 12.1, `sm_121a` |
| tree | pinned by the job; the exact commit and binary md5 are in `results.txt` |
| comparator engine | `MiaAI-Lab/exllamav3` @ `63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed`, staged as a tarball with sha256 `18b49d64e6a171bcbfd06bd02f139fc53189e04b5ff8f510a3afbd622dd372d4` |
| host | `dgx:gpu0`, inside an `rc` lease |

Both checkpoint shard sha256 values are recomputed **on the device** before any
leg runs, and a mismatch aborts the job. The expected values are
`7b77214fe58ff15fed0b4af55e3cd92f38842b8711886d68954e8071ff8270c6`,
`411c83bb1070b27f3d670fc93e38dca0f17eb66429f64b5706901b12613188b2` and
`6b2e3afc694a343b7f3f0edfe5925e460762fc9ede4699165b577ca0733c8e56`.

## The load

Four length bands from three corpora, each pinned by sha256:

| Band | Share | Source | sha256 |
|---|---|---|---|
| `S` short question | 35% | GSM8K `test.jsonl` | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| `M` code completion | 40% | HumanEval `HumanEval.jsonl` | `1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2` |
| `L` prose summary | 15% | vLLM `benchmarks/sonnet.txt` | `d58663195ba6780da5f029b920c7ac00cad1e435ee1df7e03bf9ec2470f8dea4` |
| `XL` long code review | 10% | HumanEval `HumanEval.jsonl` | as above |

`M` is the predecessor's whole workload, kept so the two pages share one band.

The corpus is built by `benchmarks/variadic/build_corpus.py` from those bytes,
that seed and those weights, and nothing else. The realised token-length
histogram is read back from each server's own `usage.prompt_tokens`.

Matched on both sides: the box, the client, the corpus, `max_tokens: 192`,
`temperature: 0.6`, `top_p: 0.95`, `top_k: 20`, `seed: 0`, the chat endpoint,
each model's own chat template, thinking enabled, and no `ignore_eos`.

Concurrency rungs 1, 4 and 8. Two rounds. Round 2 reverses both the arm order
and the rung order, so a monotone drift over the session appears as a rung-to-rung
disagreement between the rounds rather than as a shape in the sweep.

Warmup is `max(C, 4)` requests, run to completion before the measured window
opens, and recorded rather than discarded, so both views come from one set of
records.

## Realised prompt-length histogram

Pending.

## Throughput and the round-to-round spread

Pending. Every cell will carry both rounds' values and their spread, because
this repository has one unchanged binary reading 36.82 and 78.86 tok/s in the
same session at `c = 8`.

## Percentiles

Pending. p50, p90, p95, p99 and the maximum for time to first token, time per
output token, inter-token latency and end-to-end latency, warm only and with
warmup, pooled across both rounds.

## Streaming granularity

Pending. The predecessor measured 4.46 completion tokens per streamed chunk on
our side and 1.08 on theirs, which inflates our time to first token at the
median for a reason that is accounting rather than engine. Both the raw and the
chunk-corrected figures will appear here for both engines, with the correction's
method and its estimate flagged as an estimate.

## Cold start

Pending. In the predecessor, request `i = 0` cost us 27.5 s and 25.2 s time to
first token against their 7.9 s and 1.9 s, over four legs, and that one request
per leg is what made our mean read worse than theirs.

## Acceptance

Pending. Ours comes from `GET /metrics`, which serves the four vLLM
speculative-decoding families since `58a7162ba` closed
[#2770](https://github.com/mudler/vllm.cpp/issues/2770). Theirs comes from the
`Job` counter its server wrapper forwards. The predecessor had one side only.

## What is not matched

Each engine runs at its own published configuration, which is the shape this
comparison was built for and is not one configuration.

- **Context and KV cache.** Theirs is `-cs 262144` with an NVFP4 KV cache. This
  engine refuses an NVFP4 KV cache by name
  ([#2620](https://github.com/mudler/vllm.cpp/issues/2620)), so their cache
  configuration is not one our arm can take.
- **Prefix caching.** Disabled explicitly on our side. Their paged cache reuse
  is not configurable from their server wrapper, so leaving ours on would be an
  unmatched cache rather than a matched one.
- **Their engine revision is our choice.** Their card pins none.
- **Their stop rule.** Their generator ends at
  `max_new_tokens - 1 - num_draft_tokens`, so a request for 192 returns 184 from
  them and 192 from us. Each side is divided by the tokens it produced.

## Evidence

[`docs/bench-evidence/qwen38-27b-exl3-variadic-20260905/`](../bench-evidence/qwen38-27b-exl3-variadic-20260905/README.md).
Per-request records stay on the share at `/mnt/nas_share/rc/exl3-variadic/out/`.

## Reproduce this run

See [the methodology page](variadic-load-methodology.md#running-it). The
checkpoint download and verification steps are the predecessor's
([`qwen38-27b-exl3-gb10`](qwen38-27b-exl3-gb10.md#1-get-the-weights)) and are
not repeated here.
