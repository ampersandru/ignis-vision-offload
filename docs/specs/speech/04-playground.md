# 04 - dictation in the Playground

GitHub: #354 (master #350). Phase 1. Feature: `spec.md`. Needs 03 (the Realtime
route); its components can be built against a fake socket first.

ADRs: 0066 (the speech model), 0028 (exposure always keyed). Specs:
playground/06 (the image studio), decide (the Decide view).

Decided 2026-10-23/24 in the speech grill. The behaviour below is the
mockup the owner validated (Q17, "perfetto"), kept untracked with the
decision log (`dictation-canvas/`: Main, States, Decide, Studio). Its copy
strings are the UI copy. Details the grill left open were decided by the
owner on 2026-10-24, cited as `(owner, 2026-10-24)`.

## Problem Statement

- The Playground has no microphone: no `getUserMedia`, no `MediaRecorder`,
  no audio anywhere in `web/src` (`ignis-asr-seams` §1).
- Three composers take prose:
  - chat (`web/src/conversation/Composer.tsx`, textarea `:159`, add menu
    `:116-131`, Send `:183-185`, hint line `:190-196`);
  - the image studio (`web/src/image/StudioComposer.tsx:73`);
  - Decide (`web/src/decide/QuestionCard.tsx:108`, the prose evidence at
    `EvidenceEditor.tsx:110`).
- Two Decide textareas take JSON, not prose: `EvidenceEditor.tsx:123` and
  `Bench.tsx:224`.
- `web/src/api/model.test.ts:197` uses `kind: "audio"` as the kind the
  client does not know.

## Solution

### One mic, everywhere (Q3)

One component serves chat, Decide and the image studio. Chat comes first,
and all three are in this spec. The transcript lands at the cursor, is
editable, and is never sent by itself (Q4). The two JSON textareas get no
mic: speech is for prose (Q17).

### The dictation (Q5, Q33, Q21, Q43)

- **Always over Realtime** (Q33), push-to-talk too: one session per
  dictation, opened at the press. The press is also the signal that loads the
  speech model (Q53), and in phase 3 the loan's (Q32, Q56). The session is
  configured with:
  - `turn_detection: null`;
  - PCM16 at 16 kHz, the declared extension (Q21);
  - the context (below);
  - the provisional-tail opt-in, `session.ignis.provisional_tail: true` (Q43
    b; owner, 2026-10-24).
- **Capture:** an AudioWorklet downsamples the mic to 16 kHz mono and sends
  `input_audio_buffer.append`, one every 100 ms (owner, 2026-10-24).
- **Done** commits. `completed` inserts the transcript and the session
  closes. **Esc** clears the buffer, closes the session, and discards the
  dictation (Q17).
- **Language:** none is sent, so the model detects it. The Playground never
  forces one (Q6, Q51).
- **Context** (Q15, Q17):
  - in chat and in the studio, the last reply and the composer's text;
  - in Decide, the question and the evidence text, never raw JSON.
  - The order is the last reply first, the composer text last (owner,
    2026-10-24).
    The Playground sends them whole, and the server's cap
    (`speech.context_max_tokens`, keeping the last 512 tokens, spec 02)
    makes "the tail of the last reply" by construction, with no second
    threshold.

### What the user sees (Q17)

**The mic.** An icon button beside Send, in the ash of the prompt box's
other icons: a new `IconMic`, 16 viewBox, stroke 1.5. The add menu stays on
the left. Its tooltip: "Dictate. Or hold Ctrl+Shift+Space while you speak."

**The shortcut** (owner, 2026-10-24): Ctrl+Shift+Space, scoped to the
composer (the Playground has no global hotkeys). Held longer than 400 ms it
is push-to-talk, and the release commits. Tapped, it toggles. A click on the
mic toggles too (Q5, Q17).

**While listening:**
- the mic becomes an ember **Done** chip with a square stop glyph, in Send's
  colour;
- Send is disabled (line background) until the dictation ends, so a
  dictation is never sent by itself (Q4);
- the composer's underline becomes the running heat bar.

**The strip** sits above the prompt box, as wide as it (the attachment
strip's `max-w-3xl`), on kiln dark `#1c2026`. It has `role="status"` and
`aria-live="polite"`, and holds:
- a level meter of 28 bars (`#ff5a1f` when loud, `#4a525c` when quiet);
- the settled words in light text (`#eae8e4`), the provisional tail in grey
  (`#939ba4`), clamped to two lines. "Listening…" shows before the first
  word;
- a timer with a pulsing dot;
- "Discard Esc";
- a 2 px running heat bar at its foot, the one signature element.

Under `prefers-reduced-motion` the heat bar, the dot and the meter stop
animating.

**Finishing.** Done becomes a disabled "Finishing" chip. The strip stays
until the text lands. The inserted text is selected, in the ember selection
tint, at the cursor, with a space added on a side that needs one.

**The hint line** (under the prompt box, where Send's reasons appear today;
no tooltip):

| state | text | colour |
|---|---|---|
| idle | Enter sends, Shift+Enter adds a line. Dictate with the mic, or hold Ctrl+Shift+Space while you speak. | ash |
| listening | Listening. Done puts the text at the cursor; Esc discards it. | ash |
| finishing | Finishing the last words. | ash |
| inserted (6 s) | Inserted N words at the cursor. Nothing is sent until you press Send. | ash |
| discarded (4 s) | Dictation discarded. | ash |
| connection lost | Dictation stopped: the connection to the server closed. Insert what was heard, or discard it. | fault |
| busy (`503`, `Retry-After`, Q44) | Speech is busy with another dictation. Trying again in N s. (counting down, then retrying) | warn |

**Connection lost.** The strip keeps what was heard, with "Insert what was
heard" and "Discard". What was heard is the stable text and the last
provisional tail (owner, 2026-10-24).

**Unavailable.** The mic is struck through and dimmed (opacity 0.45),
labelled "Dictate (unavailable)". The reason is one sentence in the hint
line:

| cause | text | from |
|---|---|---|
| plain http, not localhost | The microphone needs https or localhost. Open the Playground through an https proxy, or start the server with `--server-expose cloudflare-quick`. | Q18 |
| no speech model | No speech model on this server. Download one with `ignis model download <id>`. | Q39 |
| speech turned off | Speech is turned off on this server (`speech.model = off`). | Q39 |
| browser blocked the mic | Microphone access is blocked for this site. Allow it in the browser's site settings, then press the mic again. | Q17 |
| no CPU path, no room | Speech is unavailable on this machine: the CPU lacks AVX2 and the card has no free room to lend. | Q35 |

The model-dependent reasons come from spec 02's `GET /v1/models`
(its top-level `speech` object, Q55). The secure-context reason
comes from `window.isSecureContext`. ignis serves no TLS itself, but
`--server-expose cloudflare-quick` gives an https URL that is always keyed
(ADR 0028), and the mic works there (Q18).

### Decide (Q17, Q3)

- One mic per prose field, on its label row: Question, and the text
  Evidence. Labels: "Dictate into Question", "Dictate into Evidence".
- The field being dictated into keeps its ember border. The strip sits under
  that field, not above the page, and the text lands at that field's cursor.
- Starting the mic on another field ends the current dictation first and
  inserts what was heard.
- The JSON evidence and the request editor have no mic.

### The image studio (Q17, Q3)

As in chat: the Done chip sits beside "Switch & Generate", which is disabled
while listening. The listening hint adds "Generate waits until you press
it."

## Acceptance

1. One mic component serves chat, Decide's prose fields and the image
   studio. The JSON textareas have none (Q3, Q17).
2. The transcript is inserted at the cursor, selected and editable. Send
   (Generate, in the studio) is disabled while listening and finishing, and
   nothing is ever sent by a dictation (Q4, Q17).
3. A click on the mic toggles. Ctrl+Shift+Space works inside the composer:
   push-to-talk when held past 400 ms, a toggle on a tap (Q5, Q17; owner,
   2026-10-24).
4. Partials appear live in the strip, settled words light and the
   provisional tail grey, with the level meter and the timer (Q5, Q17, Q43).
5. Esc clears the buffer, closes the session, keeps the composer as it was,
   and shows "Dictation discarded." (Q17).
6. Every dictation is one Realtime session with `turn_detection: null`, PCM16
   at 16 kHz and `session.ignis.provisional_tail: true`.
   Done commits (Q33, Q21, Q43).
7. The context sent is the last reply and the composer text, or in Decide the
   question and the evidence text, never raw JSON (Q15, Q17).
8. No `language` is ever sent (Q51).
9. Outside https and localhost the mic is disabled, with the secure-context
   sentence (Q18).
10. Each unavailable cause shows the struck-through mic and its sentence
    from the table (Q35, Q39, Q17).
11. A `503` with `Retry-After` shows the busy sentence, counts down, and
    retries once (Q44, Q35, Q17).
12. A closed connection keeps what was heard in the strip, with "Insert what
    was heard" and "Discard", and the fault sentence (Q17).
13. In Decide, each prose field has its own mic and strip, the field keeps
    its ember border, and starting another field's mic ends and inserts the
    current dictation first (Q17).
14. In the image studio, "Switch & Generate" waits while listening (Q17).
15. Every string in the tables above is the shipped copy, apart from the
    `<id>` filled in (Q17).
16. The strip is `role="status"`, `aria-live="polite"`, and its animations
    stop under `prefers-reduced-motion` (Q17).
17. `model.test.ts` knows `kind: "speech"`, and keeps `"audio"` as the
    unknown example (Q11).
18. The web tests pass, and `cargo test` passes workspace-wide (AGENTS.md,
    not a grill decision).

## Out of Scope

- A personal glossary for the context: later (Q15).
- A language picker: the Playground never forces a language (Q51).
- Global hotkeys outside the composer.
