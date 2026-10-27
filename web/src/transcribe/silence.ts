import { viewerChoice } from "./viewerChoice.ts";

// The silence that ends an utterance in the Transcribe view (Q122), sent as
// `server_vad`'s `silence_duration_ms`. The server's default, 500 ms, cuts a
// speaker's phrase at every breath: utterances of a second or two, and the
// model's language guessed afresh on each. With speakers, a change of voice
// ends an utterance anyway, so the silence only has to end a real pause. A
// per-viewer setting, this browser's (viewerChoice.ts); a change holds at once.

export const UTTERANCE_SILENCE_KEY = "ignis.transcribeSilence";

/** The choices the view offers, in ms. */
export const UTTERANCE_SILENCE_CHOICES = [500, 1_000, 1_500, 2_000];

/** 1.5 s until the viewer says otherwise (owner, 2026-10-27). */
export const DEFAULT_UTTERANCE_SILENCE_MS = 1_500;

const silence = viewerChoice({ key: UTTERANCE_SILENCE_KEY, choices: UTTERANCE_SILENCE_CHOICES, fallback: DEFAULT_UTTERANCE_SILENCE_MS });

/** The silence the next listening asks for. */
export const utteranceSilence = silence.get;
export const chooseUtteranceSilence = silence.choose;
export const reloadUtteranceSilence = silence.reload;
export const useUtteranceSilence = silence.use;
