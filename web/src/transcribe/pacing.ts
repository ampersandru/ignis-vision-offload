// A file fed over Realtime faster than real time (spec speech/07, "Sources",
// Q99). The page paces it: it sends while the audio not yet transcribed stays
// under N seconds, a viewer setting (`filePace.ts`); the server's own limit,
// `speech.max_queued_audio_seconds`, is spec 08's.
//
// The audio not yet transcribed runs from the start of the oldest utterance
// whose text has not come yet to the audio sent. Two cases the spec's words
// ("past the end of the last utterance whose completed came") would stall on:
// - Silence: with no utterance under way or waiting, there is nothing to
//   transcribe, so a long silence in a file is not a backlog.
// - An utterance under way past N: only more audio can end it (a silence, the
//   server's length guard, a speaker change), so past N the page still sends
//   it at real time, as a microphone would, and stops only while utterances
//   wait for their text.

/** The audio not yet transcribed, in ms of the session's audio. */
export function backlogMs(sentMs: number, pending: { startMs: number }[]): number {
  if (pending.length === 0) return 0;
  return sentMs - Math.min(...pending.map((p) => p.startMs));
}

/** The ms of audio to send now: a burst under N, real time past it while an utterance is under way, else none. */
export function allowance({
  backlogMs,
  limitMs,
  open,
  sinceLastMs,
  burstMs,
}: {
  backlogMs: number;
  limitMs: number;
  /** An utterance is under way: started, not stopped. */
  open: boolean;
  /** Since the page last looked: the real-time share. */
  sinceLastMs: number;
  /** The most one look sends. */
  burstMs: number;
}): number {
  if (backlogMs < limitMs) return Math.min(burstMs, limitMs - backlogMs);
  return open ? sinceLastMs : 0;
}

/** The share of the file transcribed, and the time left at the pace so far; none before any has been. */
export function estimate({ doneMs, totalMs, elapsedMs }: { doneMs: number; totalMs: number; elapsedMs: number }): { fraction: number; leftMs: number | null } {
  const fraction = totalMs > 0 ? Math.min(1, doneMs / totalMs) : 1;
  if (fraction >= 1) return { fraction: 1, leftMs: 0 };
  if (doneMs <= 0) return { fraction, leftMs: null };
  return { fraction, leftMs: Math.round(((totalMs - doneMs) * elapsedMs) / doneMs) };
}
