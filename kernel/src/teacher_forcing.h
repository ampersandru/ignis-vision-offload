#pragma once
// GitHub #267 (test-only): a teacher-forced verify round's verdict for one
// lane (`ignis_decode_options::forced_tokens`, kernel/include/ignis_step.h)
// -- OURS. Host-only, no CUDA type: the CTest unit
// kernel/tests/test_teacher_forcing.cpp builds it with no CUDA library, so it
// runs on a card another process holds.
//
// The lane accepts the leading drafts its forced text agrees with, at most
// `extent` of them, and takes the forced token after them as its correction:
// `licensed[0..n)` are those drafts and `licensed[n]` is `forced[n]`, what
// the accept kernel writes for a target that sampled the text. The run is
// then `{anchor, drafts[0..n)}`, cut as any run is (ignis_cut_and_redirect),
// and `forced[n]` is its successor: the sequence's pending token.
//
// So `forced[n]` must be a vocabulary id. The one exception is the pad (-1)
// after a recording that ends at a stop id: the stop cuts the run, nothing
// recorded follows it, and the sequence is left with no pending token --
// refused by every round until a prefill gives it one, as the replay of
// recorded turns prefills the next turn's context. Any other id outside the
// vocabulary is refused: a pad past a recording that ends without a stop
// would leave a sequence that goes on with nothing to feed, and an id past
// the vocabulary would be the next round's embedding row. `forced[0..n)`
// need no check: they equal drafts, which are vocabulary ids.

#include "ignis_reasoning_redirect.h"

#include <cstdint>

struct IgnisForcedVerdict {
  int32_t accepted; /* n: the leading drafts the forced text agrees with, 0..extent */
  bool refused;     /* forced[n] cannot be the run's successor; licensed is unwritten */
};

inline IgnisForcedVerdict ignis_teacher_forced_verdict(int32_t anchor, const int32_t *drafts,
                                                       const int32_t *forced, int32_t extent,
                                                       int32_t vocab, const int32_t *stop_ids,
                                                       uint32_t stop_count, int32_t *licensed) {
  int32_t n = 0;
  while (n < extent && drafts[n] == forced[n]) {
    ++n;
  }
  bool stopped = ignis_is_stop_id(anchor, stop_ids, stop_count);
  for (int32_t j = 0; j < n && !stopped; ++j) {
    stopped = ignis_is_stop_id(drafts[j], stop_ids, stop_count);
  }
  const int32_t successor = forced[n];
  const bool in_vocabulary = successor >= 0 && successor < vocab;
  if (!in_vocabulary && !(successor == -1 && stopped)) {
    return {n, true};
  }
  for (int32_t j = 0; j < n; ++j) {
    licensed[j] = drafts[j];
  }
  licensed[n] = successor;
  return {n, false};
}
