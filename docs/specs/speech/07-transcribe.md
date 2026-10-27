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

## As built

Built 2026-10-26 on branch `speech-transcribe` (#362), web only, in `web/src/transcribe/` (the view, its model and
its listening) and `web/src/speech/` (the session, the capture, the strike of the other mics). The guide's page is
`docs/user/playground/transcribe.md`; its picture is the scene `transcribe` of `docs/site/scripts/scenes.mjs`.

### The pieces

- **The session** is `openListening` in `speech/realtime.ts`, the dictation's sibling (AC 2): `server_vad` with no
  lengths, the provisional tail, `diarization` `live` or `file` when Speakers is on (omitted otherwise, and always on
  a call's microphone), the context as `prompt`, no `language`, no `max_utterance_seconds`. Its tests drive a fake
  socket with spec 08's events.
- **The listening** is `ListeningController` (`transcribe/listening.ts`), one lane per source, each lane a session; a
  lane's session is a stream of its own for the speakers. The clock starts at the Transcript's (a Continue) and
  stands still while paused; a lane's origin on it is its first audio. Its tests drive fake sessions, captures, a
  fake decoder and fake timers.
- **The Transcript** (`transcribe/transcript.ts`) maps a stream's channel to the page's label the first time it
  speaks; rename, move, merge and split are functions of it; it is stored as `transcribe-sessions`
  (`transcribe/stored.ts`), round-tripped on the memory and IndexedDB adapters.
- **The state** is the page's (`transcribe/useTranscribe.ts`), as the Image view's is: the header's badge and the view
  each subscribe to the listening themselves, so its level meters redraw them and not the page.

### Decisions taken while building

- **The file's pace** (Q99). The audio not yet transcribed runs from the start of the oldest utterance whose text has
  not come to the audio sent, and is none when no utterance is under way or waiting. The spec's words ("past the end
  of the last utterance whose completed came") stall on a silence in a file longer than N (nothing completes, nothing
  more is sent) and on an utterance under way longer than N (only more audio ends it). Past N the page still sends an
  utterance under way at real time, and stops only while utterances wait for their text. With the file geometry
  `completed` lags an item's end by up to 30.4 s of audio, so items over about 30 s go stop-and-go at N = 60. An
  append refused for the queue limit (`ignis_audio_queue_full`, its `event_id` echoed) is sent again from there after
  1 s, after the file's end too. One look sends up to 5 s of appends, so if the server refuses one and takes a later
  one of the same look, that later audio goes twice: the pace keeps the backlog under N, and N under the server's
  default limit, so a refusal needs a server limit set lower. N's choices are 45, 60 and 90 s: 120 would sit at the
  server's default limit, and below about 45 s a file with speakers runs near real time.
- **A file's Pause** stops the sending and commits nothing: a commit would cut the utterance under way in two. A live
  source's Pause commits it, as Stop commits the last item (the spec says only that Pause "stops the capture"), and
  only when one is under way: a commit with no speech since the last item would make an item of the silence.
- **An end flushes first.** Stop and a file's end send each session an empty `session.update` and commit the
  utterance under way only once it is answered: the server reads events in order and runs its VAD as it reads each
  append, so by then the VAD's events for the last audio sent have come (the empty update changes no context, so it
  prefills nothing), and an utterance whose start was still on its way is not lost. The session closes
  once every utterance's text has come; a session that closes while one still waits ends the listening as a lost
  connection.
- **A Continue's clock** starts at the larger of the clock the last listening left and the last utterance's end: a
  reload in the middle of a listening keeps the utterances but never hands the clock back.
- **A call's Pause** lets the microphone go and keeps the shared tab, window or screen picked (not sent), so Resume
  asks for no new pick; the browser's own Stop sharing stops the listening as Stop does.
- **A file's title** is its name ("25 Oct 15:42 · standup.m4a"), not "· File": the list tells files apart.
- **The page-wide flag** that strikes the other mics (AC 13) is up while the listening captures, not while it is
  paused: a paused listening holds no microphone, and a dictation then works; Resume ends it as Start does.
- **An unavailable diarizer refuses the whole `session.update`** that asks for speakers (spec 08's server), so the
  session's audio would wait for a configuration that never comes: `openListening` sends the configuration again
  without `diarization`, and the view turns Speakers off with the server's reason. A server whose `/v1/models` says
  nothing of a diarizer is taken as one without: it would refuse the field.
- **Corrections use native menus** (the utterance's speaker, Split at the cursor, Merge into), which open over a
  clipped `.cut` box and on a phone.

### The owner's live check (2026-10-27)

The first try on a call (grill Q121-Q123):
- The server's log showed items of 1 to 2.5 s, each in a language of its own (Thai, Chinese, Portuguese, Hindi;
  the 3.7 and 5.5 s ones Italian), and the far side cut often with one speaker heard: the server's VAD silence,
  500 ms, not speaker changes.
- **Q121: a call holds the mic while the far side speaks.** The page reads the shared audio's level (`HOLD_LEVEL`,
  about -34 dBFS) and sends the microphone's session silence of the same length while it is loud and for
  `HOLD_AFTER_MS` (500 ms) after, so its times stay the clock's and the voices that come back through the speakers
  are not transcribed as Me. On by default, with a switch in the deck (headphones need none); speech over the far
  side is lost while it holds.
- **Q122: the view asks for its own silence** (`silence_duration_ms`), a per-viewer setting of 0.5, 1, 1.5 or 2 s,
  1.5 s by default; it amends "server_vad at the server's defaults" above. A change holds at once.
- **Q123, the language, is open.** Q88 stands: the owner first tries the Context field (an Italian context biases
  the model's guess) before choosing among a language field, a language held after the first long utterance (a
  server change), or nothing more. `transcription.languages` with several entries is no help: the server detects
  among all thirty then.

### Notes (2026-10-27, Q124-Q127)

The owner asked, after the first live check, for the view to sum a conversation up as it goes: a kind of meeting
chosen, the model estimating what that kind wants (a work call's topics, people, actions; a game's games and wins),
in a tab of the right panel as the chat's General and Tools are, kept up during a call, each feature behind a
switch. Built as a draft for the owner to try, beyond Q81-Q123 (transcribe/notes.ts, notesWriter.ts, Notes.tsx):
- **Q124: a Notes tab** beside Tools, open by default (the header's right drawer is "Notes"). Four kinds, each its
  sections: General (Summary, Key points, Next steps), Work call (Summary, Topics, People, Decisions, Action items,
  Open questions), Game session (Games, Players, Results, Highlights), Lesson (Topics, Key concepts, Questions,
  To review). Write notes / Update notes / Stop, Copy notes. The notes and their kind live on the Transcript
  (`notes`, under version 1: a field the page cannot read is dropped on restore, not the Transcript); a new one
  starts at the kind picked last (`ignis.transcribeNoteKind`). Nothing goes to the model before a press or the
  switch below.
- **Q125: the request keeps the prefix.** The system prompt is the kind's alone and constant; the user turn is the
  title and the transcript's `.txt`, which a listening only appends to. The notes written before are not sent back:
  each writing starts from the transcript, so a correction holds and an error does not compound. The headings stay
  as the prompt writes them and the text is in the transcript's language: probed on the 27B (2026-10-27), asking for
  the headings in the conversation's language turned the whole notes English, while "keep each heading as given,
  the text in the transcript's language, a transcript in Italian gets notes in Italian" wrote Italian notes under
  the English headings. A kind that does not fit writes a dash in every section (a Game session on a work call).
- **Q126: Keep updating while listening**, a per-viewer switch, off by default (`ignis.transcribeNotesLive`). A
  look every 5 s: the first notes once 3 utterances are in, then at least 60 s from the last writing's start while
  something new was said, a writing at once when another kind is picked, and one more when the listening ends,
  after the writing under way. A writing that failed waits the 60 s before the next, whatever the cadence says.
- **Q127: the model.** The model loaded, never named, so no request switches (the image model loaded, or none
  serving, strikes the tab with its reason); `class: "agent"`, `reasoning_effort: "none"`, temperature 0.3, top_p
  0.9, at most 1,500 tokens; a full engine is waited out a second at a time, up to 60 times (api/busy.ts, now
  shared with the agents and the safety check). A transcript past the model's context is sent from where its tail
  fits (3 characters a token, 2,500 tokens kept for the prompt and the notes), and the line says from when. The
  probe on the 27B: 6 utterances, 517 prompt tokens, 355 written, 2 s.
- Open for the owner: kinds or sections of their own; the notes in the exports and in Open in chat; whether this
  stays in #362 or becomes a ticket of its own.

### The call's context, its languages, suggestions and corrections (2026-10-27, Q128-Q135)

The owner's second round, approved whole (grill Q128-Q135), built in six slices, one commit each:
- **S2, the panels (Q129).** The right panel's tabs are Call (the context and the speakers), Notes, and Settings
  (Utterance ends after, File pace); it opens on Call, and the header's drawer button says Call. The title, the
  start and the exports leave the scroller: a heading above the sheet with Copy, a Download menu (.txt, .md, .srt,
  .vtt) and Open in chat, so they stay in sight in a long transcript. Hold my mic stays in the deck: it is the
  listening's.
- **S1, the context (Q128).** A Transcript keeps what its transcription is told, `context` under version 1
  (topic, terms, anything else, the languages, `appliedAt`), read back like the notes. The Call tab shows it
  applied, as a record with its time and its share of `speech.context_max_tokens` (512, about three characters a
  token, the server keeping the last), or a draft framed in the ember after Edit: nothing goes out as it is typed,
  where every keystroke was a `session.update` and a prefill before. Apply sends one update (prompt, keywords,
  language; `null` clears what is no longer told), which holds from the next utterance; a Start applies a draft
  left open. The topic and anything else are the `prompt`, the terms and the speakers' names the user gave (never
  "Speaker n", Me or Others) the `keywords`, which the server refuses with `<`, `>` or a line break, so the field
  does too. A rename during a listening is sent as it is made. The deck's first line says what is told, with Edit.
- **S4, each utterance's language (Q131).** The server sends it on request: `session.ignis.item_language: true`,
  then `ignis.input_audio_transcription.language {item_id, content_index, language}` (the ISO code the model labelled
  the item with) after the speaker and before `completed`; none for an item labelled `None`. Every listening asks
  for it; a server older than it refuses the key with `param: session.ignis.item_language`, and the page configures
  the session again without it, saying nothing. An utterance keeps its `language`; the language suggestion (S5)
  reads them.
- **S5, the suggestions (Q132, Q134).** A Transcript keeps its `suggestions` (each open, accepted or dismissed, and
  the kinds accepted always), read back like the notes. The languages come from the utterances' `language` once six
  are in, no model: Force the top language when it holds 70 % and the others came only on utterances under 2.5 s
  (the owner's first call), else Detect those with 15 % and two utterances. The topic, a title (only over the
  listening's own) and the terms come from a look by the text model: first at six utterances, then 90 s from the
  last look's start with three new, once more at the end (assist.ts's Follower, shared with the notes), a failed look
  waiting the gap; Suggest now on demand. An offer made before, whatever became of it, is not made again; a term
  taken out is dismissed alone. Always applies every later one of its kind in that Transcript at once. Accepting
  tells the context (applied, so sent) or renames. The bell beside the tabs counts what waits. Suggest while
  listening, a per-viewer switch in Settings, on by default (they only propose).
  - **Q134 measured, and amended.** The engine reuses an earlier prompt only whole (the checkpoint match is
    all-or-nothing at its length, `crates/core/src/checkpoint.rs`): on the 27B a request whose prompt began with
    the last one's reused 559 of 563 tokens, two tasks after one transcript 64 (the system prompt's first page),
    the transcript in its own message 64 too. So a task after the transcript, or a task shared, is never reused;
    the task in the system prompt with the transcript after it, which a listening only appends to, reused 448 of
    565 on the next request. Each task keeps its own chain: assist.ts's preamble and the task as the system
    prompt, the title and the `.txt` as the user turn; the notes and the suggestions share the shape, not a
    prefix. Measured over HTTP; the socket goes to the same engine.
  - **The look's task asks for the misheard names first** (`FIX: [mm:ss] heard => meant`), then TERM, TOPIC,
    TITLE. Asked for the terms alone, the 27B with thinking off copied a misheard name as heard ("Enidesk"); asked
    to check them first, it fixed it and spelled the term AnyDesk, twice out of two at temperature 0.2 (greedy said
    "Eni Desk"). The FIX lines are S6's; S5 reads the rest leniently (bullets, bold, NONE).
- **S6, the fixes (Q133).** A FIX line (`[mm:ss] heard => meant`, `→` and `->` too) becomes a fix suggestion when
  both sides are one to four words the server could take as a keyword, differ beyond case, and `heard` stands as
  words of its own (no letter or digit around it, any case) in at least one utterance: the fix covers every
  utterance that holds it then. Fixes stand side by side, one card each. Accepted, each of those utterances gets the
  replacement, keeps `heard` (its text before its first fix) and `fixed` (the words put in, marked dotted in the
  ember), and the right spelling joins the terms, so the speech model spells it right from the next utterance.
  Undo fix puts the text back as heard, an edit since with it; a split leaves neither half a way back. The exports,
  the notes and Open in chat read the fixed text. Always applies the later fixes as they come; only names and terms,
  never a rephrasing (the task says so, and a side over four words is dropped).

### Not done here

- No live check: the view was built against fakes and the dev mock, which has no Realtime socket. The owner's live
  check (a room, a call, a file; a long session) is the test of the whole.
- The picture of the guide shows a made-up call the scene puts in the page's store; no recording reaches the guide.
