# 07 - the Transcribe view: a conversation transcribed in the Playground

GitHub: #362 (master #350). Track: Transcribe (spec 08 is the diarizer). Feature: `spec.md`. Needs 05
(`server_vad`); the speakers and every server change it relies on are spec 08's. Web only.

ADRs: 0067 (the speech model), 0069 (the speakers).

Decided 2026-10-25 in the Transcribe grill (Q81-Q118, the log in `.scratch/speech-grill-2026-10-23/`), with the
spike Q91 asked for: the finding [Nemotron on the CPU](../../findings/2026-10-25-nemotron-on-the-cpu.md).

## Problem Statement

- The Playground takes speech only as **dictation** (spec 04): one session per dictation, ended by Done or by its
  first server commit (Q80), its transcript landing in a composer. Nothing transcribes a whole conversation: a
  meeting in a room, a call on the PC, a recording.
- The owner wants a section that does: it hears a conversation, writes it down utterance by utterance with who spoke,
  lets the labels be renamed and corrected live, and keeps the result (the grill's ask).

## Solution

**The implementer designs the view with the `frontend-design` skill** (Q103). No mockup was made; the owner trusts
the skill. The view follows the Playground's existing style and components.

### The view (Q81, Q84)

- **A new header tab, Transcribe**, beside chat, Decide, Image, Monitor and Server (`web/src/app/Header.tsx`'s
  `View`). It is mounted from its first visit on, as Decide is, so a listening goes on while the user looks at
  another view. The header shows that a listening is under way, from any view.
- **What it holds:** a **Transcript**, the record of one conversation heard there. Its **utterances** come in order,
  each with its times, its **speaker** and its text, and never its audio (GLOSSARY.md).

### Sources (the grill's first round, Q93, Q94)

- **Microphone:** a conversation in the room. One Realtime session.
- **Call:** the microphone plus one `getDisplayMedia` pick: a tab (its audio), the whole screen with system audio
  (on Windows every application's sound, mixed), or a window where the browser offers its audio (Chrome 141+
  `windowAudio`). **Two sessions, never mixed**: the microphone's voice is "Me", the shared audio's are the others
  (Q93, Q112).
  - Before the call starts the view says to use headphones: with speakers, the far voices come back into the
    microphone and are transcribed twice (Q93).
  - The two sessions' times are aligned on their wall-clock starts, and their utterances are merged in time order.
- **File:** a recording the user picks. It is decoded in the browser (`decodeAudioData` into an
  `OfflineAudioContext` at 16 kHz mono: wav, mp3, ogg, m4a, webm, an mp4's audio) and fed over Realtime faster than
  real time (Q94):
  - **The client paces it** (Q99): it sends while the audio not yet transcribed stays under N seconds. That audio is
    the audio past the end of the last utterance whose `completed` came. N is a view setting, per viewer, 60 s by
    default. The server's own limit is spec 08's.
  - A progress bar and an estimate of the time left.
  - The session asks for the file geometry (spec 08).

### The session (Q83; not a dictation's)

A listening opens its session(s) at Start and closes them at Stop, which commits the last item. It is **not** the
dictation's session. `web/src/speech/realtime.ts`'s `openTranscription` today configures a dictation, and the view
must not inherit three of its behaviours. It gets a sibling, or a parameter, that:
- asks for `server_vad` at the server's defaults (`turn_detection: {type: "server_vad"}` with no lengths, so the
  `speech.vad_*` fields hold), never the viewer's auto-stop silence (Q80);
- treats **every** `input_audio_buffer.committed` as one utterance, and never ends at the first;
- opts in to `session.ignis.provisional_tail: true` (Q82) and, when Speakers is on, `session.ignis.diarization:
  "live"`, or `"file"` for a file (spec 08). It leaves `session.ignis.max_utterance_seconds` unset, so the server's
  length guard holds (spec 08).

The shared pieces stay shared: the capture (`capture.ts`, `pcm.ts`), the refusal and availability handling
(`availability.ts`), the busy wait. One listening takes one of `speech.max_sessions`, or two on a call.

### Live text (Q82)

- The utterance under way is drawn as the dictation strip draws it: its fixed text, then the provisional tail drawn
  light. It settles on `completed`.
- With Speakers on, the server fixes no text before an item ends (spec 08): the whole utterance under way is the
  provisional tail until `completed`.
- **Following** (Q102): the view follows new utterances. Scrolling up stops following and shows "Jump to live".

### Context and language (Q87, Q88)

- A free **context** field (topic, names, jargon) is sent as the session's `prompt` at open. Changed mid-listening,
  it goes out in a `session.update` and holds from the next utterance. Earlier utterances are never sent as context.
- No `language` is ever sent: the model detects it per utterance.

### Utterances and speakers (Q92, Q98, Q100, Q109, Q112)

- **Times** are relative to the listening's start, `mm:ss` (`h:mm:ss` past an hour): an utterance starts at its
  `speech_started` and ends at its `speech_stopped` (spec 08 gives both times to every item). A live listening also
  keeps its wall-clock start, shown in the Transcript's header. A paused and resumed listening continues its clock.
- **Speakers.** "Speaker 1, 2…" are numbered in the order they first speak in the Transcript, from spec 08's
  `ignis.input_audio_transcription.speaker`. On a call the microphone's utterances are "Me", and the shared audio's
  are diarized into Speaker n, or "Others" with Speakers off.
- **A per-listening Speakers toggle**, on by default (Q112). Off, no label is shown, or only Me/Others on a call. With
  the diarizer unavailable (`/v1/models`' `speech.diarizer`, spec 08) the toggle is struck, with its reason in the
  way spec 04 states an unavailable mic.
- **Corrections** are mappings in the page, and nothing goes back to the server (Q109):
  - **rename** a speaker: every one of their utterances, past and future;
  - **move** an utterance to another speaker or a new one;
  - **merge** two speakers: one label, and the future utterances of either take it;
  - **split** an utterance at the cursor: two utterances, the second to a chosen or new speaker. No audio is kept, so
    the split acts on the text alone. The split time is proportional to the characters and marked approximate (`~`)
    (Q98).
- **The text is editable inline**, even while listening, on every completed utterance. The edited text is what export
  and Open in chat carry (Q96).

### The Transcript (Q86, Q97, Q101)

- **Saved** in the page's store (`web/src/store/`, IndexedDB, #171) as a new kind of session, held to the store's
  contract tests. Each utterance is saved as it completes. Text, times, speakers and the corrections are saved; the
  audio never is.
- **Title:** automatic, "date time · source" (for example "25 Oct 15:42 · Call"), renamable. It is listed in the
  view's sessions panel, as the chat's and Decide's are.
- **Pause** stops the capture. **Resume** goes on, opening a new session when the server closed the old one
  (`server.socket_idle_close`), with the times continuing from the offset.
- **Continue** on a saved Transcript opens a new listening that appends to it.
- **A reload** keeps every completed utterance and ends the listening.

### While listening (Q85)

The mics of the other views are struck, saying "Listening in Transcribe", and open no second session. A dictation
under way when a listening starts ends as a dictation does when another mic starts (spec 04's "one dictation per
page").

### Out of the view (Q89, Q95)

- **Export**, computed in the page from the utterances:
  - the clipboard and `.txt`: one line per utterance, `[mm:ss] Speaker: text`;
  - `.md`: the title and start in a header, then `**Speaker** (mm:ss): text` per utterance;
  - `.srt` and `.vtt`: one cue per utterance from its start to its end, the speaker in front of its text (in `.vtt`
    as a `<v Speaker>` voice tag). An approximate split time is written as it is.
- **Open in chat:** a new chat, its composer seeded with the Transcript as Markdown, never sent by itself. No summary
  is built into the view.

## Acceptance

1. The Transcribe tab is in the header, mounted from its first visit; a listening survives a look at another view,
   and the header shows it from every view (Q84).
2. A listening's session asks for `server_vad` with no lengths, `session.ignis.provisional_tail: true`, and
   `session.ignis.diarization` as the Speakers toggle and the source say. It treats every commit as an utterance and
   sends no `session.ignis.max_utterance_seconds`. A test against a fake socket speaking spec 08's events pins it
   (Q83, Q82, spec 08).
3. The three sources work: the microphone (one session), a call (two sessions, "Me" and the shared audio, merged in
   time order, the headphones sentence before it starts), a file (decoded at 16 kHz mono, paced under N seconds not
   yet transcribed, N a viewer setting defaulting to 60, with progress and an estimate) (Q93, Q94, Q99).
4. Live text: fixed text and the light provisional tail, settling on `completed`; with Speakers on, the whole
   utterance stays provisional until `completed` (Q82, spec 08).
5. The view follows new utterances; scrolling up stops it and shows "Jump to live" (Q102).
6. The context field goes out as `prompt` at open and by `session.update` when changed; no `language` is ever sent
   (Q87, Q88).
7. Times are relative to the start, from each item's `speech_started` and `speech_stopped`; the wall-clock start of a
   live listening is in the header; a resumed listening continues its clock (Q100, Q97).
8. Speakers are numbered by first appearance from the speaker event; a call's microphone is "Me"; the Speakers
   toggle is on by default, and struck with its reason when the diarizer is unavailable (Q112, spec 08).
9. Rename, move, merge and split behave as specified. A merge holds for future utterances, and a split's time is
   proportional and marked `~` (Q92, Q98, Q109).
10. Completed utterances' text is editable while listening; export and Open in chat carry the edit (Q96).
11. The Transcript is saved in the store as each utterance completes, never with audio, under an automatic, renamable
    title, and listed in the sessions panel; a reload keeps the completed utterances (Q86, Q101, Q97).
12. Pause, Resume (with a new session after an idle close) and Continue behave as specified (Q97).
13. While listening, the other views' mics are struck with "Listening in Transcribe" and open no session (Q85).
14. The five exports produce the formats above from the edited Transcript; Open in chat seeds a new chat's composer
    and sends nothing (Q95, Q89).
15. A refused upgrade, a busy server and an unavailable speech model show spec 04's states (the shared
    `availability.ts`); a lost connection keeps every completed utterance.
16. `docs/user/playground/transcribe.md` describes the view and the Playground section of the site lists it.
    `docs/user/playground/sessions.md` names the new kind of session. The page's screenshot comes from
    `make docs-screenshots`, as the other Playground pages' do (AGENTS.md, `docs/agents/user-docs.md`).
17. The tests use synthetic tones and public audio only (FLEURS, VoxPopuli); no real conversation and no name from
    one reaches a commit or an issue (the privacy rule, Q90).
18. The web tests pass, and `cargo test` passes workspace-wide (AGENTS.md).

## Out of Scope

- The diarizer, the speaker event, the cuts, the length guard, the server's queue limit: spec 08.
- A summary inside the view (Q89).
- Keeping the audio, re-transcribing an utterance (Q86).
- A language picker (Q88).
- More than one `getDisplayMedia` pick on a call (Q93).
- `diarized_json` for files on the batch route: a parked idea (Q94).
- Corrections that teach the diarizer (Q109).

## Notes from #361 (2026-10-26, ideas, not ACs)

The owner's third scenario, two people working together for a whole day, raised three things for this view's build:
- **Speakers across sessions.** A quiet socket closes after `server.socket_idle_close` (300 s); Resume opens a new
  session, and a new diarizer stream numbers its speakers afresh (the speaker event carries the stream's channel). The
  view maps channels to its own Speaker n per stream, so after a Resume the same person can come back as a new
  Speaker until the user merges them: worth a sentence in the view, or a merge suggestion.
- **The cost of a microphone left open.** With silence still flowing the socket never idles, and the diarizer steps on
  the silence all day: about 0.33 s of nine threads every 2 s on the i9 (some 20 %), more on four cores. Pause is how
  a user stops it.
- **Hours are not measured.** Nothing has run for hours; the owner's live check of a long session is the test of its
  memory and stability.
