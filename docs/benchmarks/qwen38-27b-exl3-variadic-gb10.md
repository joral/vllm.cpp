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

**PARTIAL.** The variadic sweep is queued on `dgx:gpu0` and its tables below are
empty.

One section is not empty. The predecessor's four legs recorded per-request
timings under the same definitions this harness uses, and its client then
reduced them to means and threw the distribution away. The records did not go
anywhere, so [the percentile method applied to them](#the-percentile-method-applied-to-the-predecessors-own-records)
carries measured numbers today. It reads the same run differently; it is not a
new measurement and it adds no samples.

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


## The percentile method, applied to the predecessor's own records

`benchmarks/variadic/adapt_headtohead.py` converts the four legs of
[`qwen38-27b-exl3-gb10`](qwen38-27b-exl3-gb10.md) into the record shape
`report.py` reads. The field definitions already agreed: `ttft` is the wall time
to the first content-carrying chunk, `itl` is a gap between streamed chunks,
`latency` is client-side, and the token counts come from each server's `usage`.
Request `i = 0` is tagged as warmup, which is that page's own stated rule and the
one its head-to-head legs did not apply.

**The percentile tables that came out of it are published on that page**, under
[time to first token](qwen38-27b-exl3-gb10.md#time-to-first-token), and they are
not repeated here. In short: its mean-TTFT conclusion held at the median and
reversed above p90, and one request out of 164 was most of the difference.

Two readings from the same conversion are not on that page.

### Inter-token latency measures the same thing on both sides once the chunking is removed

| arm | tokens per chunk | itl p50 ms | itl mean ms | itl mean / tokens per chunk | tpot mean ms |
|---|---|---|---|---|---|
| ours | 4.47 | 82.2 | 82.3 | 18.4 | 18.6 |
| theirs | 1.08 | 0.0 | 24.0 | 22.3 | 22.3 |

Their p50 inter-token latency is 0.0 ms. That is not a fast engine; it is a
server wrapper that holds back 16 characters
(`tools/serve_openai.py:74`) and then releases several chunks with no gap
between them. Ours is 82.2 ms because one chunk carries 4.47 tokens.

Divide each engine's mean chunk gap by its own tokens per chunk and both land on
that engine's mean time per output token, 18.4 against 18.6 and 22.3 against
22.3. That is why this harness reports time per output token as the primary
inter-token axis and prints the chunk statistics beside the raw gaps.

### A bound on what the chunking costs our time to first token

The harness's own correction needs the first chunk's character count, which the
predecessor's client did not record, so `report.py` prints that column as absent
for these legs. A weaker correction is available from what the files hold, and
it is a **bound, not the harness's estimate**: subtract
`(mean tokens per chunk - 1) x mean time per output token`, which is 64.5 ms from
ours and 1.7 ms from theirs. It assumes the first chunk carries that engine's
average number of tokens, which the harness does not assume.

| percentile | ours raw ms | ours bounded ms | theirs raw ms | theirs bounded ms |
|---|---|---|---|---|
| p50 | 863.1 | 798.5 | 762.7 | 761.0 |
| p90 | 1219.0 | 1154.4 | 1739.2 | 1737.5 |
| p95 | 1371.6 | 1307.0 | 1959.9 | 1958.2 |
| p99 | 1535.8 | 1471.2 | 2850.8 | 2849.0 |

Chunking accounts for about two thirds of their median lead and none of our tail
lead. Their median advantage survives the correction, at 5% rather than the 12%
the raw figures show.

**The bound does not neutralise their hold-back, and should not be read as
doing so.** Their wrapper emits `pending[:len(pending) - HOLD_BACK]`, so its
first chunk contains the text minus the 16 characters that caused the delay. The
correction removes a chunk SIZE from both engines. It does not restore the time
either engine spent before its first chunk left the server.

### What this section does not do

It adds no samples, runs no engine and changes no configuration. Every unmatched
axis and every limitation carries over from the predecessor page unchanged.

## Realised prompt-length histogram

**The published histogram is the one the servers count**, from
`usage.prompt_tokens`, and it is pending with the run. Each engine renders its
own chat template, which adds tokens neither the corpus nor a local tokenizer
sees.

The corpus's own shape is not pending. Measured with the target checkpoint's
`tokenizer.json` by `benchmarks/variadic/corpus_tokens.py`, over the raw prompt
text of the 144-prompt corpus at seed 0, sha256
`6d6420fc4c0a55019dc24da3dd5b410e2743a6b268f2ee2fb2c89ae21d3cfc5e`:

| band | n | min | p50 | p90 | max | mean |
|---|---|---|---|---|---|---|
| `S` | 50 | 26 | 56 | 79 | 108 | 58 |
| `M` | 58 | 43 | 118 | 224 | 407 | 134 |
| `L` | 22 | 687 | 878 | 1070 | 1087 | 882 |
| `XL` | 14 | 2237 | 2928 | 3094 | 3233 | 2804 |
| **all** | 144 | 26 | 108 | 1083 | 3233 | 481 |

| prompt tokens | prompts |
|---|---|
| 0-127 | 84 |
| 128-255 | 21 |
| 256-511 | 3 |
| 512-1023 | 18 |
| 1024-2047 | 4 |
| 2048 and up | 14 |

The shortest prompt is 26 tokens and the longest is 3,233, a span of 124x, with
a median of 108 and a mean of 481. The predecessor's whole workload is the `M`
band alone: 43 to 407 tokens.

That gap between the median and the mean is the point of the mix. A quarter of
the corpus is above 512 tokens and carries most of the prefill, while more than
half of it is under 128 tokens, so the scheduler sees short and long requests in
the same queue rather than one length repeated.

The longest prompt plus 192 output tokens plus a chat template fits inside the
8,192-token `--max-model-len` the run pins, which is what this measurement was
taken to check before a lease was spent on finding out.

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
