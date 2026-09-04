// SPEC-DFLASH2 A2-3 (#2911) — the async placeholder fill's PER-REQUEST rule.
//
// WHAT THE FILL IS. Under async scheduling the scheduler ships `-1` placeholders
// for a verify step's draft positions (async_scheduler.py:43-45) because it
// schedules step N+1 before step N's drafts exist. The real values are the ones
// this runner proposed at the previous step's sampling, and `execute_model`
// patches a LOCAL copy of `scheduled_spec_decode_tokens` with them before
// `update_req_spec_token_ids` reads it.
//
// WHAT A2-3 CHANGED. The fill used to read `pending_drafts_`, a HOST, ragged,
// req_id-keyed structure. It now reads the same per-req_state buffer the
// combine's draft scatter reads (`InputBatch::draft_tokens`, mirroring
// `RequestStates.draft_tokens`, vllm/v1/worker/gpu/states.py:71-77 @ pin
// 5559679229bc961848b121ccdeaa8fa5d79bec98). One producer, one residence, two
// consumers — which is the shape upstream has, and it is what stops the fill and
// the scatter from disagreeing about what was drafted.
//
// WHY THE RULE IS A NAMED FUNCTION. `combine_sampled_and_draft_tokens` reads the
// same buffer with the same row arithmetic, and the CUDA kernel writes that
// arithmetic a third time. One rule with three expressions is how this row's
// predicate split shipped (#2710), so the fill's reading is extracted here and
// the call site applies it rather than agreeing with it by inspection.
//
// WHY EVERY CASE BELOW THAT MATTERS HAS `num_reqs > 1`. A per-request rule feeding
// a per-STEP route is the exact shape that survived 27 mutations in
// `test_combine_row_predicate.cpp` because every test used `num_reqs == 1`: at
// one request a per-request and a per-step reading agree on every input. The
// MIXED cases here are the inputs that separate them — some rows drafting, some
// not, and different k per drafting row.
//
// WHAT THIS FILE DOES NOT CLAIM. Nothing here asserts anything about a queue, an
// event, or a kernel that has not been waited on. `vt::cpu::CpuBackend::
// CreateQueue()` returns the same null handle as the main queue, so on the CPU
// tier a "second queue" is the first queue and every event call is a no-op; a
// test claiming otherwise would be green for a reason unrelated to its subject.
// The buffer's LIFETIME claim is made where the buffer is declared and rests on
// it being runner-lifetime rather than on a test.
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/v1/worker/gpu/prepare_inputs.h"

using vllm::v1::FillDraftsForRow;

namespace {

// A [num_req_states, stride] row-major buffer whose row r reads
// 100*r + 0, 100*r + 1, ... so a row read from the wrong slot is unmistakable.
std::vector<int32_t> Buffer(int num_req_states, int stride) {
  std::vector<int32_t> b(static_cast<size_t>(num_req_states) *
                         static_cast<size_t>(stride));
  for (int r = 0; r < num_req_states; ++r) {
    for (int c = 0; c < stride; ++c) {
      b[static_cast<size_t>(r) * static_cast<size_t>(stride) +
        static_cast<size_t>(c)] = 100 * r + c;
    }
  }
  return b;
}

}  // namespace

TEST_CASE("A2-3 fill: a row's placeholders are filled from ITS req_state row") {
  const std::vector<int32_t> buf = Buffer(/*num_req_states=*/4, /*stride=*/3);

  const std::vector<int32_t> got =
      FillDraftsForRow(buf, /*draft_tokens_stride=*/3, /*req_state_idx=*/2,
                       /*num_valid=*/3, /*num_placeholders=*/3, "r2");
  REQUIRE(got.size() == 3u);
  CHECK(got[0] == 200);
  CHECK(got[1] == 201);
  CHECK(got[2] == 202);
}

// The stride is the speculator's MAX draft length and pads every shorter row.
// Filling from the stride rather than from the placeholder count writes the pad
// over a position the scheduler never reserved — the combine has the identical
// distinction (`num_draft_tokens` and NOT `draft_tokens_stride`) and this is the
// fill's half of it.
TEST_CASE("A2-3 fill: the count comes from the placeholders, not from the stride") {
  const std::vector<int32_t> buf = Buffer(/*num_req_states=*/4, /*stride=*/4);

  // The scheduler fit only 2 of the 4 drafted positions into this step's budget.
  const std::vector<int32_t> got =
      FillDraftsForRow(buf, /*draft_tokens_stride=*/4, /*req_state_idx=*/1,
                       /*num_valid=*/4, /*num_placeholders=*/2, "r1");
  REQUIRE(got.size() == 2u);
  CHECK(got[0] == 100);
  CHECK(got[1] == 101);
}

// ── THE MIXED STEP, and it is why this file is not a single-request test ─────
//
// Three requests in one step. Row 0 drafts 3, row 1 drafts NOTHING (a plain
// decode row, or a prefill chunk the proposer skipped), row 2 drafts 1. A
// per-request rule that quietly used the step's total draft count, or the first
// row's count, or the batch row instead of the req_state slot, agrees with the
// correct rule on every single-request input and disagrees here.
TEST_CASE("A2-3 fill: a MIXED step fills each row from its own slot and count") {
  const std::vector<int32_t> buf = Buffer(/*num_req_states=*/8, /*stride=*/3);
  const int stride = 3;

  // Row 0: req_state slot 5, three drafts, three placeholders.
  const std::vector<int32_t> r0 =
      FillDraftsForRow(buf, stride, /*req_state_idx=*/5, /*num_valid=*/3,
                       /*num_placeholders=*/3, "a");
  REQUIRE(r0.size() == 3u);
  CHECK(r0[0] == 500);
  CHECK(r0[1] == 501);
  CHECK(r0[2] == 502);

  // Row 1: req_state slot 6, NO drafts and NO placeholders. The fill must return
  // nothing rather than reading slot 6's stale row, which a rule keyed off the
  // step's total would happily do.
  const std::vector<int32_t> r1 =
      FillDraftsForRow(buf, stride, /*req_state_idx=*/6, /*num_valid=*/0,
                       /*num_placeholders=*/0, "b");
  CHECK(r1.empty());

  // Row 2: req_state slot 7, ONE draft. A rule that took its width from row 0
  // would splice three ids over one placeholder.
  const std::vector<int32_t> r2 =
      FillDraftsForRow(buf, stride, /*req_state_idx=*/7, /*num_valid=*/1,
                       /*num_placeholders=*/1, "c");
  REQUIRE(r2.size() == 1u);
  CHECK(r2[0] == 700);
}

// The two refusals the pre-A2-3 fill carried, restated on the new source. They
// are refusals and not clamps because a placeholder the worker cannot fill would
// otherwise embed a literal -1 in the model's input ids.
TEST_CASE("A2-3 fill: fewer drafts than placeholders is refused, not padded") {
  const std::vector<int32_t> buf = Buffer(/*num_req_states=*/4, /*stride=*/3);

  // Proposed 1, scheduler placed 3.
  CHECK_THROWS(FillDraftsForRow(buf, /*draft_tokens_stride=*/3,
                                /*req_state_idx=*/0, /*num_valid=*/1,
                                /*num_placeholders=*/3, "short"));

  // Proposed nothing at all while placeholders were scheduled — the state the
  // pre-A2-3 fill named "placeholders scheduled without a matching propose".
  CHECK_THROWS(FillDraftsForRow(buf, /*draft_tokens_stride=*/3,
                                /*req_state_idx=*/0, /*num_valid=*/0,
                                /*num_placeholders=*/2, "none"));
}

// THE BOUND THE CUDA SCATTER DOES NOT HAVE. `LaunchCombineSampledAndDraftTokens`
// reads `draft_tokens[req_state_idx * stride + b]` with nothing checking it, so
// a row past the allocation is an unchecked device read whose quiet outcome is
// garbage drafts and a silent acceptance loss. The buffer is sized by the
// req_state pool so this cannot arise, and the refusal is what says so out loud
// if a future caller sizes it by `num_reqs` instead.
TEST_CASE("A2-3 fill: a req_state row past the buffer is refused") {
  const std::vector<int32_t> buf = Buffer(/*num_req_states=*/4, /*stride=*/3);

  // Slot 4 is one past the last row this buffer holds.
  CHECK_THROWS(FillDraftsForRow(buf, /*draft_tokens_stride=*/3,
                                /*req_state_idx=*/4, /*num_valid=*/3,
                                /*num_placeholders=*/3, "past"));

  // Slot 3 is the last row and must NOT be refused — a bound that is off by one
  // in the safe direction removes the top slot from service, and a batch that
  // fills its pool would refuse a request that is perfectly addressable.
  CHECK_NOTHROW(FillDraftsForRow(buf, /*draft_tokens_stride=*/3,
                                 /*req_state_idx=*/3, /*num_valid=*/3,
                                 /*num_placeholders=*/3, "last"));

  CHECK_THROWS(FillDraftsForRow(buf, /*draft_tokens_stride=*/3,
                                /*req_state_idx=*/-1, /*num_valid=*/1,
                                /*num_placeholders=*/1, "negative"));
}

// A non-speculative runner carries a zero-width buffer. Reaching the fill with
// placeholders to fill and no buffer to fill them from is a bookkeeping defect,
// not a step to serve.
TEST_CASE("A2-3 fill: placeholders against a zero-width buffer are refused") {
  const std::vector<int32_t> empty;

  CHECK_THROWS(FillDraftsForRow(empty, /*draft_tokens_stride=*/0,
                                /*req_state_idx=*/0, /*num_valid=*/0,
                                /*num_placeholders=*/1, "nospec"));

  // No placeholders and no buffer is the non-speculative path itself, and it
  // must stay inert rather than refuse.
  CHECK_NOTHROW(FillDraftsForRow(empty, /*draft_tokens_stride=*/0,
                                 /*req_state_idx=*/0, /*num_valid=*/0,
                                 /*num_placeholders=*/0, "nospec"));
}
