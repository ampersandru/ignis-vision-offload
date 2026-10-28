import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import type { SpeechStatus } from "../api/model.ts";
import { unavailable } from "../speech/availability.ts";
import { HintText } from "../speech/Dictation.tsx";
import { listening, subscribeListening } from "../speech/listening.ts";
import { caption } from "../ui/classes.ts";
import { IconMic } from "../ui/icons.tsx";
import type { Version, WorkspaceFile } from "./files.ts";
import { BUTTON, Source } from "./looks.tsx";
import { clock, type Recorder, type RecorderState } from "./recorder.ts";

// The workspace panel's audio (#395): an audio file plays beside the
// transcript the model reads of it, which the panel can ask for before the
// model does; and Record makes a new audio file from the microphone, under
// the rules every microphone of the page follows (speech/availability.ts).

/** Why a transcript cannot be had now, from the models listing; null when it can, or when the server has not said (the press finds out). */
export function transcriptUnavailable(speech: SpeechStatus | undefined): string | null {
  if (!speech || speech.available) return null;
  return unavailable({ secure: true, speech, blocked: false })?.sentence ?? null;
}

/** An audio file as the panel shows it: the player, then its transcript or the way to it. */
export function AudioView({
  name,
  src,
  transcript,
  unavailable: why,
  transcribing,
  error,
  onTranscribe,
}: {
  name: string;
  /** The recording's URL; null while it is read, or when the page holds no bytes of it. */
  src: string | null;
  /** The transcript kept with the version, once computed. */
  transcript: string | undefined;
  /** Why no transcript can be had now. */
  unavailable: string | null;
  transcribing: boolean;
  error: string | null;
  onTranscribe: () => void;
}) {
  return (
    <>
      {src && <audio controls preload="metadata" src={src} aria-label={`Play ${name}`} className="w-full" />}
      <p className={caption}>Its transcript, as the model reads it</p>
      {transcript !== undefined ? (
        <Source text={transcript} />
      ) : (
        <div className="flex flex-col items-start gap-1.5">
          <p className="text-xs leading-snug text-ash">Not transcribed yet: the model's first read of it transcribes it.</p>
          <button type="button" disabled={why !== null || transcribing} className={`${BUTTON} disabled:cursor-default disabled:opacity-45 disabled:hover:bg-surface`} onClick={onTranscribe}>
            {transcribing ? "Transcribing…" : "Transcribe"}
          </button>
          {why && (
            <p className="text-xs leading-snug text-ash">
              <HintText text={why} />
            </p>
          )}
          {error && <p className="text-xs leading-snug text-fault">{error}</p>}
        </div>
      )}
    </>
  );
}

/** An audio file of the workspace in the panel: its bytes loaded once per file's bytes, and Transcribe through the workspace module. */
export function AudioViewer({
  file,
  version,
  speech,
  load,
  transcribe,
}: {
  file: WorkspaceFile;
  version: Version;
  speech?: SpeechStatus;
  load: (version: Version) => Promise<Blob | null>;
  /** Computes the version's transcript and keeps it; why it could not, or null. */
  transcribe: (file: WorkspaceFile) => Promise<string | null>;
}) {
  const [src, setSrc] = useState<{ key: string; url: string } | null>(null);
  const [transcribing, setTranscribing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loading = useRef({ load, version });
  loading.current = { load, version };
  // The panel reads the files again after every change: the player keeps playing while the bytes are the same.
  const key = version.bytes?.id ?? "";
  useEffect(() => {
    if (!key) return;
    let url: string | null = null;
    let live = true;
    void loading.current.load(loading.current.version).then(
      (blob) => {
        if (!live || !blob) return;
        url = URL.createObjectURL(blob);
        setSrc({ key, url });
      },
      () => {},
    );
    return () => {
      live = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [key]);
  return (
    <AudioView
      name={file.name}
      src={src?.key === key ? src.url : null}
      transcript={version.derived}
      unavailable={transcriptUnavailable(speech)}
      transcribing={transcribing}
      error={error}
      onTranscribe={() => {
        setTranscribing(true);
        setError(null);
        void transcribe(file).then((why) => {
          setTranscribing(false);
          setError(why);
        });
      }}
    />
  );
}

/** Record, or the recording under way: its time, its level and the limit it stops at, with Stop and save and Cancel. */
export function RecorderView({
  state,
  unavailable: why,
  onStart,
  onStop,
  onCancel,
}: {
  state: RecorderState;
  /** Why the microphone cannot start now (availability.ts); a blocked one stays pressable, which asks again. */
  unavailable: { sentence: string; blocked: boolean } | null;
  onStart: () => void;
  onStop: () => void;
  onCancel: () => void;
}) {
  if (state.phase !== "idle") {
    const recording = state.phase === "recording";
    return (
      <div role="status" className="flex basis-full flex-wrap items-center gap-2 bg-surface px-2.5 py-2 text-xs">
        <span className="dictation-dot inline-block size-[7px] bg-[#ff5a1f]" aria-hidden />
        <span className="font-display font-semibold tabular-nums text-ink">
          {recording
            ? `Recording ${clock(state.seconds)}${state.limit !== null ? ` of ${clock(state.limit)}` : ""}`
            : state.phase === "starting"
              ? "Starting the microphone…"
              : "Saving the recording…"}
        </span>
        {recording && (
          <span className="h-1 w-14 bg-line" aria-hidden>
            <span className="block h-full bg-[#ff5a1f]" style={{ width: `${Math.round(state.level * 100)}%` }} />
          </span>
        )}
        <span className="ml-auto flex items-center gap-2">
          {recording && (
            <button type="button" className="font-display font-semibold text-ember hover:text-ink" onClick={onStop}>
              Stop and save
            </button>
          )}
          {state.phase !== "saving" && (
            <button type="button" className="font-display text-ash hover:text-ink" onClick={onCancel}>
              Cancel
            </button>
          )}
        </span>
      </div>
    );
  }
  const struck = why !== null && !why.blocked;
  return (
    <>
      <button
        type="button"
        title="Record from the microphone into a new audio file"
        disabled={struck}
        className={`${BUTTON} disabled:cursor-default disabled:opacity-45 disabled:hover:bg-surface`}
        onClick={onStart}
      >
        <IconMic struck={why !== null} />
        Record
      </button>
      {(why || state.note) && (
        <p className="basis-full text-xs leading-snug">
          {why && (
            <span className="block text-ash">
              <HintText text={why.sentence} />
            </span>
          )}
          {state.note && (
            <span className={`block ${state.note.startsWith("Saved") ? "text-ash" : "text-fault"}`}>
              <HintText text={state.note} />
            </span>
          )}
        </p>
      )}
    </>
  );
}

/** The panel's recorder: the microphone's rules, and `save`, asked at the press, taking the recording once it ends. */
export function Recording({ recorder, speech, save }: { recorder: Recorder; speech?: SpeechStatus; save: () => (file: File) => Promise<void> }) {
  const state = useSyncExternalStore(recorder.subscribe, recorder.getState, recorder.getState);
  // A listening in the Transcribe view holds the page's speech (spec speech/07, Q85).
  const transcribing = useSyncExternalStore(subscribeListening, listening, listening);
  // Outside https and localhost the browser gives no page a microphone; a test has no such flag.
  const secure = typeof globalThis.isSecureContext === "boolean" ? globalThis.isSecureContext : true;
  const why = state.phase === "idle" ? unavailable({ secure, speech, blocked: state.blocked, listening: transcribing }) : null;
  return (
    <RecorderView
      state={state}
      unavailable={why}
      onStart={() => {
        if (why && !why.blocked) return;
        void recorder.start(save());
      }}
      onStop={() => void recorder.stop()}
      onCancel={() => recorder.cancel()}
    />
  );
}
