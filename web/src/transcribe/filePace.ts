import { viewerChoice } from "./viewerChoice.ts";

// N: how far ahead of its transcript a file is sent, in seconds of audio not
// yet transcribed (spec speech/07, Q99; pacing.ts). A per-viewer setting, this
// browser's (viewerChoice.ts). The server's own limit,
// `speech.max_queued_audio_seconds` (120 by default), refuses an append past
// it, so every choice stays under it. With speakers a file's utterance waits
// up to 30.4 s of audio for the diarizer (spec speech/08), so the lowest
// choice still sends ahead past that.

export const FILE_PACE_KEY = "ignis.transcribeFilePace";

/** The choices the view offers, in seconds. */
export const FILE_PACE_CHOICES = [45, 60, 90];

export const DEFAULT_FILE_PACE_SECONDS = 60;

const pace = viewerChoice({ key: FILE_PACE_KEY, choices: FILE_PACE_CHOICES, fallback: DEFAULT_FILE_PACE_SECONDS });

/** N for the next file. */
export const filePace = pace.get;
export const chooseFilePace = pace.choose;
export const reloadFilePace = pace.reload;
export const useFilePace = pace.use;
