# GLM-4.7-Flash routes through DeepSeek-V2's config-driven router dtype, and upstream's is a class property

**Row:** `MODEL-TEXT-GLM4-MOE-LITE-ROUTER-F32` — new row, `ACTIVE`.
**Issue:** [#2928](https://github.com/mudler/vllm.cpp/issues/2928).
**Date:** 2026-09-04. **Base:** `c796fea41`.
**Predecessor rows:** `MODEL-TEXT-GLM4-MOE-LITE-GATE-2839` (the gate that
measures the gap, [#2839](https://github.com/mudler/vllm.cpp/issues/2839) /
[#2906](https://github.com/mudler/vllm.cpp/pull/2906)),
`MODEL-TEXT-glm4-moe-lite-glm4-moe-lite-for-causal-lm` (the port).

## Scope

One mirror repair on the `Glm4MoeLiteForCausalLM` forward: the MoE router logit
dtype. In scope are the GLM registry's parse, the load path that feeds the
forward, and a checkpoint-free test that measures the difference through the
model forward.

Out of scope: re-capturing any golden (needs a GPU and a 58.2 GiB snapshot,
neither of which is on this host), the near-tie gap artifacts ([#2929](https://github.com/mudler/vllm.cpp/issues/2929)),
the `routed_scaling_factor` application point ([#2930](https://github.com/mudler/vllm.cpp/issues/2930)),
and this model's speed axis.

## What is true at `c796fea41`

**Which revision this was compared against.** Every upstream anchor below was
read in the local checkout at `5559679229`, which is the revision the committed
`tests/parity/goldens/glm4_moe_lite_greedy/` artifacts were captured on and the
PRIOR parity pin. The ACTIVE pin is `e126687a9a`, at which no gate has run in
this tree (`.agents/upstream-sync.md`, [#2794](https://github.com/mudler/vllm.cpp/issues/2794)).
Re-reading `glm4_moe.py` at the active pin is owed as O5.

**Upstream, read at `5559679229` — the pin the goldens were captured on.**
`Glm4MoeLiteForCausalLM`'s MoE block is `Glm4MoeLite`, a bare subclass of
`Glm4MoE` (`glm4_moe_lite.py:86-87`, instantiated at `:161-165`). `Glm4MoE`'s
gate is `nn.Linear(hidden_size, n_routed_experts, bias=False,
dtype=torch.float32)` (`glm4_moe.py:141-146`), it is fed
`hidden_states.to(dtype=torch.float32)` (`:218`), and the layer declares
`router_logits_dtype=torch.float32` (`:205`). No config key participates. The
fp32 router is a property of the class.

`DeepseekV2MoE` resolves the same dtype from the config
(`deepseek_v2.py:308-314` through `_get_moe_router_dtype`, `:123-133`), which
returns fp32 only for `model_type == "glm_moe_dsa"` or an explicit
`moe_router_dtype: "float32"`.

**Ours.** `glm4_moe_lite_registry.cpp` composes the DeepSeek-V2 forward and
loader, so GLM's router dtype comes from `ParseDeepseekV2Params`, which mirrors
`_get_moe_router_dtype` at `deepseek_v2_weights.cpp:332`, and `deepseek_v2.cpp:363`
sizes the logit buffer from the resolved flag. The published
`zai-org/GLM-4.7-Flash` `config.json` declares no `moe_router_dtype` and its
`model_type` is `glm4_moe_lite`, so the flag resolves **false** and our router
logits are rounded to bf16 before the top-k.

**Why it bites this model and not the vehicle that gated the block.**
GLM-4.7-Flash routes top-4 of 64 experts, `topk_method: noaux_tc` (sigmoid
scores plus `e_score_correction_bias` for the selection), `norm_topk_prob: true`,
`routed_scaling_factor: 1.8`. bf16 carries an 8-bit mantissa, ~4e-3 relative, and
the rounding lands in front of a **discrete** rank-4 boundary, so the error is
bimodal rather than a tolerance. DeepSeek-V2-Lite — the vehicle the block was
gated on — is top-2 of 4, softmax, greedy, no bias.

`deepseek_v2.h:164-168` states that "a token gate cannot see this either way".
That holds for a store that is merely too wide. It does not hold for one that is
too narrow in front of a top-k, and this row's test measures the difference.

## Design

**A — the GLM registration answers for GLM.** `ParseDeepseekV2Params` keeps
mirroring `_get_moe_router_dtype` and keeps refusing to read `model_type`
(`deepseek_v2_weights.cpp:321-330` argues that, and it is right: that parser
serves `DeepseekV2ForCausalLM`). The GLM registry TU gains one exported
`ParseGlm4MoeLiteParams(config)` that composes it and then sets
`router_dtype_is_f32 = true`, which is what `Glm4MoE` being a different class
from `DeepseekV2MoE` means in a port that composes them. `glm_moe_dsa.cpp:353`
already does exactly this for GLM-5.3.

**B — every GLM entry point resolves through that one function.** The registry's
`parse_config`, `load_weights` and `make_kv_cache` hooks all call it, so the
params the forward reads and the params the config hook validates cannot drift.

**C — the difference is measured through the forward, not asserted.** A new
checkpoint-free case builds a GLM-4.7-Flash-shaped synthetic model (`noaux_tc`
sigmoid router with the correction bias, 64 experts, top-4, `norm_topk_prob`,
`routed_scaling_factor` 1.8) and runs the CPU forward under both arms.

## Tests and expected verdicts

- **T1** `ParseGlm4MoeLiteParams` on a config shaped like the published
  `zai-org/GLM-4.7-Flash` one — no `moe_router_dtype` key — resolves
  `router_dtype_is_f32 == true`, and `ParseDeepseekV2Params` on the same bytes
  resolves `false`. RED before the change (both false).
- **T2** at GLM's router shape the two arms produce **different** logits through
  the model forward, so the dtype is observable rather than merely wider, and the
  params the GLM production path resolves select the f32 arm. RED before the
  change (the production arm is the bf16 one).
- **T3** at GLM's real router configuration, the bf16 store changes the selected
  top-4 SET. This is the MECHANISM measured directly through
  `vt::MoeRouterTopK`; it holds independently of the repair and is not a
  regression gate.
- **T4** the PRODUCTION path — `ModelRegistry::Load` over a synthetic
  GLM-4.7-Flash checkpoint on disk, then `ModelRegistry::Forward` on its default
  configuration — produces the f32 arm and not the bf16 one. This is the
  reachability case: it is what the loader call site owes.
- **T5** the SACRED engine gate. **NOT RUNNABLE HERE and not claimed.** No
  `zai-org/GLM-4.7-Flash` snapshot is on this host or on the NAS.

## Measured

Built `-O0`, CPU only (no CUDA toolkit on this host), at `c796fea41`.

RED, before the one-line repair, with the whole file in place (M1 below):

```
[doctest] test cases:  5 |  2 passed | 3 failed | 0 skipped
[doctest] assertions: 23 | 20 passed |  3 failed
```

GREEN, after:

```
[doctest] test cases:     5 |     5 passed | 0 failed | 0 skipped
[doctest] assertions: 16085 | 16085 passed | 0 failed
```

The three numbers the cases print:

| | measurement |
|---|---|
| T2 | the router dtype is observable in the GLM forward at **6 of 16** fixed seeds; worst \|f32 - bf16\| logit `0.00100262` |
| T3 | the bf16 store changes the top-4 **SET** for **34 of 4096** tokens (0.83%) at 64 experts / top-4 / sigmoid+bias / renormalize / scale 1.8 |
| T4 | checkpoint seed 61 separates the arms; `ModelRegistry::Forward` matches the f32 arm byte for byte |

T3's 0.83% is per MoE layer per token on `N(0,1)` synthetic logits, so it is the
magnitude of the mechanism and not a prediction for the checkpoint. Taken at face
value over GLM-4.7-Flash's 46 MoE layers it would put at least one changed
selection on roughly a third of token positions, which is the order of the
SACRED gate's 59 divergent positions — but the real logit distribution is not
this one, and nothing here measures the checkpoint.

### Mutations

Each rebuilt (`BUILD_RC=0` every time, so no mutation was caught by the
compiler) and each restored, with the restore verified by `sha256sum -c` against
a baseline taken before the first mutation.

| # | Mutation | Result |
|---|---|---|
| M1 | remove `p.router_dtype_is_f32 = true;` from `ParseGlm4MoeLiteParams` | RED `5 \| 2 passed \| 3 failed`, `assertions: 23 \| 20 passed \| 3 failed` — T1, T2 and T4 all fire |
| M2 | delete the production loader call site `weights.params = ParseGlm4MoeLiteParams(config);` | RED `5 \| 4 passed \| 1 failed`, 2 assertions — **only T4**, and it reports the registry's output as the bf16 arm. This is the reachability proof: T1, T2 and T3 stay green without the call site |
| M3 | remove the bf16 round trip T3's claim rests on (`logits_bf16 = logits_f32`) | RED — the SET-difference count falls to `0/4096`, so the 34 is caused by the store and by nothing else |
| M4 | make T2's second arm f32 as well | RED — `0/16` seeds differ, `worst 0`, so T2's comparison is not vacuous |

### Regression

Run on the repaired tree; `DeepseekV2ForCausalLM` is byte-identical because
`ParseDeepseekV2Params` is untouched.

| suite | result |
|---|---|
| `test_glm4_moe_lite_router_dtype` | 5/5, 16085 assertions |
| `test_glm4_moe_lite_load` | 3/3, 15 assertions |
| `test_deepseek_v2_forward` | 12/12, 2061 assertions |
| `test_deepseek_v2_load` | 4/4, 14 assertions |
| `test_ops_moe_router_grouped` | 14/14, 941 assertions |

## Gates

```sh
cmake --build build -j 3 --target test_glm4_moe_lite_router_dtype \
      test_glm4_moe_lite_load test_deepseek_v2_forward test_deepseek_v2_load \
      test_ops_moe_router_grouped
./build/tests/test_glm4_moe_lite_router_dtype
./build/tests/test_glm4_moe_lite_load
./build/tests/test_deepseek_v2_forward
./build/tests/test_deepseek_v2_load
./build/tests/test_ops_moe_router_grouped
scripts/agent-preflight.sh
```

## Stop conditions

Stop and report `NEEDS_DECISION` before changing `ParseDeepseekV2Params` to read
`model_type`, and before touching any assertion in
`tests/vllm/models/test_glm4_moe_lite_paged_engine.cpp`. That gate is red because
it measures something; a forward repair cannot and must not make it green from
this host, because what it compares is a frozen `our_ids.npy`.

## Owed

- O1. **This repair is not verified against the oracle, and the row does not
  claim it is.** The gap it addresses is 59 positions
  ([#2839](https://github.com/mudler/vllm.cpp/issues/2839)); what is established
  here is that the router dtype is a mirror divergence on the block that feeds
  those tokens, and that it is observable in a forward. Whether it is the whole
  gap is owed to whoever next holds a GPU and the snapshot, who must re-capture
  `our_ids.npy` and re-run the SACRED gate.
- O2. `docs/USAGE.md` still carries no checkpoint row for
  `Glm4MoeLiteForCausalLM` — no revision, no sha256 — which `CLAUDE.md`
  §"Say which weights, and from where" requires. Inherited from
  `MODEL-TEXT-GLM4-MOE-LITE-GATE-2839` O3 and not repaired here, because the
  values have to be read off an artifact this host does not have.
- O3. The near-tie gap artifacts are identically zero in 12 of 18 golden
  directories while `our_ids` diverges at 13 to 83 positions in each
  ([#2929](https://github.com/mudler/vllm.cpp/issues/2929)). Established from the
  committed artifacts alone; the re-capture that would settle the cause needs the
  checkpoints.
- O5. The upstream anchors are read at `5559679229`, not at the active pin
  `e126687a9a`. `Glm4MoE`'s fp32 gate is a class property rather than a config
  key, so it is unlikely to have moved, but "unlikely" is not "read". Owed to the
  pin-reconciliation queue ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
- O4. `routed_scaling_factor` is applied to the routing weights rather than the
  routed output ([#2930](https://github.com/mudler/vllm.cpp/issues/2930)). It is
  a recorded deviation whose only gate vehicle had `routed_scaling_factor: 1.0`,
  which made it bit-identical there; GLM's 1.8 makes it live. Moving it changes
  `vt::MoeCombine`'s contract for every model on this block, so it needs its own
  spec rather than an in-flow fix.
