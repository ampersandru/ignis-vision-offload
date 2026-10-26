// A Transcript (spec speech/07): the record of one conversation heard in the
// Transcribe view, its utterances in time order, each with its times, its
// speaker and its text, and never its audio (GLOSSARY.md).
//
// - **Speakers** are the page's labels, not the diarizer's channels. A
//   session's channel (`stream:index`) maps to a label the first time it
//   speaks, numbered Speaker 1, 2… in that order; a new session (a Resume, a
//   Continue) numbers its channels afresh, so a voice that comes back is a new
//   speaker until the user merges it. On a call the microphone is Me and the
//   shared audio, without its speakers, Others: one label each, whatever the
//   session.
// - **Corrections** are mappings in the page and nothing goes back to the
//   server (Q109): a rename names a label, past and future; a merge maps the
//   channels of both to one label, so the future utterances of either follow;
//   a move and a split are one-off. A split has no audio to read, so its time
//   is proportional to the characters and marked approximate (Q98).
//
// Every function returns a new Transcript and leaves the one it was given alone.

export type SourceKind = "microphone" | "call" | "file";

/** A label: a voice the diarizer told apart, or a call's Me and Others. */
export type Speaker = { id: number; name: string; role: "voice" | "me" | "others" };

export type Utterance = {
  /** Unique within its Transcript. */
  id: number;
  /** From the start of the Transcript's clock. */
  startMs: number;
  endMs: number;
  /** A split guessed the time (Q98): written with `~`. */
  approxStart?: true;
  approxEnd?: true;
  /** Its label; `null` with no speakers. */
  speaker: number | null;
  text: string;
};

export type Transcript = {
  id: number;
  title: string;
  source: SourceKind;
  /** The wall-clock start of its first live listening, in epoch ms; `null` for a file's. */
  startedAt: number | null;
  /** The listened time so far: where the next listening's clock starts (a Continue, Q97). */
  clockMs: number;
  utterances: Utterance[];
  speakers: Speaker[];
  /** A channel, `stream:index`, or `me` and `others` on a call, to its label. */
  channels: Record<string, number>;
  /** The voices numbered so far: the next is Speaker `voices + 1`, whatever was merged or renamed since. */
  voices: number;
};

/** Who an utterance's audio is, as the listening knows it. */
export type Voice = { kind: "speaker"; index: number } | { kind: "me" } | { kind: "others" };

/** An utterance as the listening hands it over: times on the Transcript's clock. */
export type Heard = { stream: string; startMs: number; endMs: number; text: string; voice: Voice | null };

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

const two = (n: number) => String(n).padStart(2, "0");

/** "25 Oct" and "15:42", in English whatever the browser's locale: a title and an export read the same everywhere. */
export const dayAndTime = (at: Date) => ({ day: `${at.getDate()} ${MONTHS[at.getMonth()]}`, time: `${two(at.getHours())}:${two(at.getMinutes())}` });

/** "25 Oct 15:42 · Call" (Q101); a file is named by its name. */
export function titleFor(source: SourceKind, at: Date, fileName?: string): string {
  const what = source === "file" ? (fileName ?? "File") : source === "call" ? "Call" : "Microphone";
  const { day, time } = dayAndTime(at);
  return `${day} ${time} · ${what}`;
}

export function createTranscript(id: number, { source, at, fileName }: { source: SourceKind; at: Date; fileName?: string }): Transcript {
  return {
    id,
    title: titleFor(source, at, fileName),
    source,
    startedAt: source === "file" ? null : at.getTime(),
    clockMs: 0,
    utterances: [],
    speakers: [],
    channels: {},
    voices: 0,
  };
}

const nextId = (items: { id: number }[]) => Math.max(0, ...items.map((i) => i.id)) + 1;

/** A new label: a numbered voice, or Me or Others. */
function newSpeaker(t: Transcript, role: Speaker["role"]): { transcript: Transcript; id: number } {
  const id = nextId(t.speakers);
  const voices = role === "voice" ? t.voices + 1 : t.voices;
  const name = role === "me" ? "Me" : role === "others" ? "Others" : `Speaker ${voices}`;
  return { transcript: { ...t, speakers: [...t.speakers, { id, name, role }], voices }, id };
}

const channelOf = (stream: string, voice: Voice) => (voice.kind === "speaker" ? `${stream}:${voice.index}` : voice.kind);

/** The label a channel speaks under, made the first time it speaks. */
function labelFor(t: Transcript, stream: string, voice: Voice): { transcript: Transcript; id: number } {
  const channel = channelOf(stream, voice);
  const known = t.channels[channel];
  if (known !== undefined) return { transcript: t, id: known };
  const made = newSpeaker(t, voice.kind === "speaker" ? "voice" : voice.kind);
  return { transcript: { ...made.transcript, channels: { ...made.transcript.channels, [channel]: made.id } }, id: made.id };
}

/** In time order: after every utterance that starts no later. */
function placed(utterances: Utterance[], utterance: Utterance): Utterance[] {
  const at = utterances.findIndex((u) => u.startMs > utterance.startMs);
  return at === -1 ? [...utterances, utterance] : [...utterances.slice(0, at), utterance, ...utterances.slice(at)];
}

export function addUtterance(t: Transcript, heard: Heard): Transcript {
  let next = t;
  let speaker: number | null = null;
  if (heard.voice) {
    const label = labelFor(t, heard.stream, heard.voice);
    next = label.transcript;
    speaker = label.id;
  }
  const utterance: Utterance = { id: nextId(next.utterances), startMs: heard.startMs, endMs: heard.endMs, speaker, text: heard.text };
  return { ...next, utterances: placed(next.utterances, utterance) };
}

const withUtterance = (t: Transcript, id: number, change: (u: Utterance) => Utterance): Transcript => ({
  ...t,
  utterances: t.utterances.map((u) => (u.id === id ? change(u) : u)),
});

/** The label a correction names: one of the Transcript's, or a new voice. */
function target(t: Transcript, to: number | "new"): { transcript: Transcript; id: number } {
  return to === "new" ? newSpeaker(t, "voice") : { transcript: t, id: to };
}

/** Every utterance of the speaker, past and future; a blank name keeps the one it had. */
export function renameSpeaker(t: Transcript, id: number, name: string): Transcript {
  const trimmed = name.trim();
  if (trimmed === "") return t;
  return { ...t, speakers: t.speakers.map((s) => (s.id === id ? { ...s, name: trimmed } : s)) };
}

/** One utterance to another speaker, or a new one: one-off, its channel still speaks as before. */
export function moveUtterance(t: Transcript, utterance: number, to: number | "new"): Transcript {
  const label = target(t, to);
  return withUtterance(label.transcript, utterance, (u) => ({ ...u, speaker: label.id }));
}

/** `from` becomes `into`: its utterances, and the channels that spoke as it, so its future utterances follow. */
export function mergeSpeakers(t: Transcript, from: number, into: number): Transcript {
  if (from === into) return t;
  return {
    ...t,
    speakers: t.speakers.filter((s) => s.id !== from),
    utterances: t.utterances.map((u) => (u.speaker === from ? { ...u, speaker: into } : u)),
    channels: Object.fromEntries(Object.entries(t.channels).map(([channel, label]) => [channel, label === from ? into : label])),
  };
}

/**
 * Two utterances where one was: the text before `at` and the text after it, the second to `to`. With no audio
 * kept, the time between is proportional to the characters before it, and approximate (Q98).
 */
export function splitUtterance(t: Transcript, utterance: number, at: number, to: number | "new"): Transcript {
  const whole = t.utterances.find((u) => u.id === utterance);
  if (!whole) return t;
  const first = whole.text.slice(0, at).trimEnd();
  const second = whole.text.slice(at).trimStart();
  if (first === "" || second === "") return t;
  const label = target(t, to);
  const cut = Math.round(whole.startMs + ((whole.endMs - whole.startMs) * at) / whole.text.length);
  const { approxEnd, ...start } = whole;
  const before: Utterance = { ...start, text: first, endMs: cut, approxEnd: true };
  const after: Utterance = {
    id: nextId(label.transcript.utterances),
    startMs: cut,
    endMs: whole.endMs,
    approxStart: true,
    ...(approxEnd ? { approxEnd } : {}),
    speaker: label.id,
    text: second,
  };
  const index = label.transcript.utterances.findIndex((u) => u.id === utterance);
  const utterances = [...label.transcript.utterances];
  utterances.splice(index, 1, before, after);
  return { ...label.transcript, utterances };
}

export function editText(t: Transcript, utterance: number, text: string): Transcript {
  return withUtterance(t, utterance, (u) => ({ ...u, text }));
}
