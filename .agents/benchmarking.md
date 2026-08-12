# Task guide — measuring performance

How to produce a number worth believing. The rules are in
[`AGENTS.md`](../AGENTS.md); this is the method.

## The denominator

vLLM is the bar, quant-matched, in its **production** configuration. Never
benchmark against `--enforce-eager` and call it parity. llama.cpp may appear
only as an explicitly labelled secondary comparison.

Both sides run the pinned oracle on identical model artifacts, prompts, token
counts, batching, concurrency, and sampling. If the two sides differ in any of
those, the ratio means nothing.

Prove the oracle actually *runs* the model before trusting it as a
denominator — constructing a config proves nothing.

## Getting a clean measurement

One GPU job at a time. Take the box lock before any measurement, stop competing
services, and never run two large models at once — unified-memory boxes reboot
rather than swap.

Calibrate the noise band from repeated identical legs *before* interpreting a
delta. Discard cold legs for a named cause, never because they are
inconvenient. Use paired, order-alternated A/B legs and a majority rule; a
single pair is an anecdote.

Prefer an instrument that is immune to page-cache effects (GPU-active time per
step) over wall clock when the host is doing heavy I/O.

Budget the disk before the run. A production RelWithDebInfo CUDA build tree is
about **169 GiB** — the build contract claimed ~3 GiB until 2026-08-10, a 56x
underestimate on the one number that decides whether a grid fits. A full disk
does not fail loudly: it voids the binding through memory-return tolerance while
still emitting plausible ratios. Leave real headroom, and delete the tree once
the evidence directory is captured (evidence is tens of MiB).

Two ratio sets that disagree may be two different HARNESSES rather than a
regression. Compare their absolute numbers before believing either; ratios are
scale-invariant and hide an order-of-magnitude mismatch completely. If the
change between the readings is provably inert (a byte-identical refactor),
suspect the measurement, not the code.

## Reading a profile

**A whole-run kernel ranking is a trap.** It sums prefill and decode, so the
top-percentage kernel is frequently one-time prefill work that no decode step
touches. Use a decode-only window or diff two sequence lengths. A `Max` far
above the `Median` means you are looking at a mixture, not a hot loop.

Profile the entire step, not only the kernels. Several of the largest wins here
were host-side waste, not slow math.

Before accepting a gap as "GPU-bound", trace both implementations with the same
tool on the same workload and compare what actually ran.

### Same-tool tracing when the oracle only runs in a container (ROCm)

The rule above is hard to obey on ROCm: vLLM runs in a container, our binary is
built outside it, and a profiler installed on each side is two tools, not one.
The answer is to run OUR binary INSIDE the oracle's container, so one
`rocprofv3` and one ROCm runtime observe both. Both AMD images already ship
`rocprof`, `rocprofv2` and `rocprofv3` at `/opt/rocm/bin` — nothing to install.

On NixOS this needs four things, and each was found by hitting it:

1. **Bind-mount the store**: `-v /nix:/nix:ro`. Our ELF interpreter and RPATH
   are `/nix/store` paths; with the store visible they resolve inside the
   container unchanged.
2. **Scope `LD_LIBRARY_PATH` to the CHILD, not the container.** Setting it via
   `docker run -e` puts our glibc in front of the container's own tools and
   breaks them — `rocprofv3` is a script, and `/usr/bin/env` dies with
   `undefined symbol: __tunable_is_initialized`. Wrap the target instead:
   `rocprofv3 … -- bash -c 'export LD_LIBRARY_PATH=…; exec <our-binary> …'`, so
   the profiler starts clean and only the traced process is affected.
3. **Give the child BOTH library sets.** Ours (build dir, the hipBLAS overlay,
   and the `ldd`-derived nix dirs) AND the container's
   (`/opt/rocm/lib:/usr/lib/x86_64-linux-gnu`) — `rocprofv3` injects a library
   into the target that needs container deps such as `libsqlite3.so.0`. Derive
   the nix set from the binary itself:

   ```sh
   ldd <binary> | grep -oE '=> /nix/store/[^ ]+' \
     | sed 's|^=> ||' | xargs -n1 dirname | sort -u
   ```

   Take `dirname` of the resolved path. A regex ending in `/lib` silently
   mismatches — greedy `[^ ]+` matches inside `…/libfoo.so` — producing
   `/lib/lib` paths that resolve to nothing and fail confusingly later.
4. **Mount the model and an output dir**, and `chmod 777` the output dir — the
   container writes the `.db` as root.

Sanity-check the plumbing by running the binary WITHOUT the profiler first. It
should produce the same tokens it produces natively; if it does not, the
environment substitution changed behaviour and nothing measured afterwards is
comparable.

Note this runs our binary against the CONTAINER's ROCm, not the host's. Same
version and soname is what makes it sound (verify both), and it is arguably
more correct for a comparison — but if a profile shows something surprising,
that substitution is the first thing to suspect.

Results land in a rocpd SQLite `*_results.db`. The `top_kernels` view gives
`name, total_calls, total_duration, average, percentage`; the `kernels` view has
per-dispatch `start`/`end`, which is what the decode-only windowing above needs.
The oracle's trace in particular is dominated by model load and graph-capture
warmup — bucket dispatch counts over time to find the phase boundary, then take
the final burst. Skipping that step compares our decode against their startup.

## Recording it

Record the exact build and run recipe, revisions, model hashes, environment,
clock and contention state, raw output, and the same-binary A/B. Reproduce on an
idle box before acceptance.

Record every required axis — throughput, latency, memory — as both values and
ratios. An axis below floor is an open gap, not a rounding error.

Never record a ceiling. An apparent same-architecture limit is an unresolved
implementation difference; name the next traceable hypothesis instead.

Accepted and pending results go in [`benchmark-record.md`](benchmark-record.md)
and `docs/BENCHMARKS.md`. Method specific to one lever stays in
[`parity-lever-protocol.md`](parity-lever-protocol.md).
