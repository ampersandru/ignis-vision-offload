# Playground: keep the chat history

GitHub: #171

## Why

Sessions live in React state only: a reload, a closed tab or a new build
of the page loses every conversation, with its tool runs, figures and
attachments. Memory notes already survive (localStorage); the chats do
not. The Decide tab's decisions are lost the same way. The Image view's
sessions already survive (spec playground/06, the image store).

## Decisions (2026-10-19, the owner)

- **Where it lives: the browser only.** The page's image store
  (`web/src/store/`, IndexedDB; spec playground/06) holds the chat's sessions
  as documents of `chat-sessions` and the decisions as documents of
  `decide-sessions`, beside `image-sessions`. Private to the browser and the
  origin, so `--expose` mixes nobody's chats, and the static `/ui/` page works
  as is. A store on ignis stays open.
- **What is kept: everything, whole.** Messages with reasoning, tool calls and
  results, agent, web, local and picture runs, questions, figures, the
  session's log, attached files (a PDF's text included), and a decision's
  draft, set-aside evidence, last answer and refusal — a loaded log of up to
  32 MB included. No cap: a store that refuses is said, not worked around.
- **Chat pictures: in the image store by SHA-256.** A prompt image keeps its
  redraw (the data URI the wire carries) inline; the picture beside it — the
  file at its own size, and a marked picture's source and marked area — goes
  into the image store when the prompt is sent, named on the image by id
  (`PromptImage.kept`). One copy however many sessions, chat or image, name it;
  deleting a session deletes only what no remaining session of either view
  names.
- **Export and import: later**, not in this work. So are "delete all" and
  settings per session: the settings stay the page's, as before.

## Solution

- **A stamp per session**, given when the store first hears of it and kept
  for good (`store/docs.ts`): a UUID key the document is written under, and an
  order — the moment it was stamped, after every stamp before it — the list
  is sorted by. The page's numeric ids stay its own and are handed out again
  at each load, past every id the page has given: two tabs counting from the
  same place would otherwise write each other's sessions, and a session
  written since a load would sort by its new id.
- **Written as it changes.** Every change makes a new session object; each
  changed one is written 400 ms after its last change, and no later than 1 s
  from the first one waiting while changes never pause (a reply streaming).
  What is waiting is written when the page goes away (`pagehide`). A session
  with nothing in it (chat: no message and no file; Decide: untouched) is never
  written. A session the list no longer holds is deleted from the store; its
  pending write is dropped first. The session each view has open is noted under
  `active-session` (`chat`, `decide`).
- **Read once, when the view mounts.** Newest first, in the order the list
  showed them; the session that was open opens again. What the page did before
  the store answered stays on top, and open. A turn does not wait for the
  store: the ids the stored sessions come back with are past the page's.
- **Settled on reading.** A page that went away mid-turn left nothing running:
  a reply streaming comes back stopped with its text; agent, web, local and
  picture runs under way come back stopped; a waiting question skipped; a
  switch under way stopped; every call left without a result is answered as
  stopped, right after the results it has, so the next turn sends a
  well-formed history. A decision being asked comes back not running, its
  earlier answers kept, with a refusal saying the page closed before the
  answers came back. A decision's question uids are reserved, so the next
  question added is not numbered as one of them.
- **A schema version per document.** `version: 1`, the first: there is no
  older one to migrate from. A document of a later version was written by a
  newer page and is left out, unread and untouched. Reads are tolerant of
  fields an older writer did not write; a version 2 adds its migration to the
  read (`readStored`).
- **Pictures across the two views.** Deleting a chat session deletes the
  pictures no remaining chat session names and no Image view session names;
  deleting an Image view session the same the other way. While the other view
  has not read its sessions yet, a delete removes no picture: one only those
  sessions name would go.
- **A store that refuses.** The write fails, the session list says what the
  browser answered, and the page goes on in memory.

## Seams

- `web/src/store/` — `saver.ts` (when a session is written), `docs.ts` (keys,
  order, what changed, merge), `useStoredSessions.ts` (the glue between a
  view's list and the store).
- `web/src/sessions/stored.ts` — the chat's document: to and from, settle,
  pictures named, kept, read back and freed.
- `web/src/decide/stored.ts` — the decision's document; `decide/model.ts`
  `reserveUids`.
- `web/src/app/useConversation.ts`, `web/src/decide/DecideView.tsx` — the
  views; `web/src/app/App.tsx` — each view keeps the other's pictures when it
  deletes a session (`useImageSessions` `live`, `useConversation` `elsewhere`).

## Acceptance

1. A chat session is stored as plain data — no Blob, no Date — and reads back
   equal, the pictures beside the redraw left to the store by id.
   (sessions/stored.test)
2. Stored sessions read back newest first by their stamp, whatever ids they
   had, with new ids above every stored message id and the page's own; the
   order holds through a second load; a stamp given after a load sorts after
   every stamp read; a later schema and what is not a session are left out; a
   document without a log or files reads. (sessions/stored.test,
   store/docs.test)
3. Reading settles a session left mid-turn: the streaming reply stopped with
   its text, runs stopped, a waiting question skipped, a switch stopped, every
   unanswered call answered right after its results — but not a failed
   reply's, which the history drops. (sessions/stored.test)
4. A sent prompt's full-size picture and markup pictures go into the store and
   are named on the image; a store that refuses leaves the redraw alone; a
   session names its stored pictures; they read back beside the redraw, and a
   missing one is left out. Deleting a session frees the pictures no chat or
   Image view session names, and none while the Image view has not read its
   sessions. (sessions/stored.test)
5. A decision is stored as plain data and reads back equal; one being asked
   reads back not running with its answers and a refusal; the next question
   is numbered past the stored ones; a later schema is left out.
   (decide/stored.test)
6. Writes wait 400 ms after the last change, at most 1 s while changes never
   pause, per session; a forgotten session is never written; the page going
   away writes what waits. (store/saver.test)
7. The stored sessions join the page: what the page did first stays on top and
   open, else the session last open opens, else the newest; an empty store
   leaves the page as it is. A look at the list names the sessions changed, by
   identity, and the ones gone. (store/docs.test)
8. Reload the page mid-conversation: the sessions list, the active session and
   every finished turn come back; a streaming reply comes back stopped with its
   text; the same in the Decide tab; deleting a session removes it from the
   store and its pictures no other session names. (the owner's, live: the hook
   glue has no effect-running tests)

## As built (2026-10-19, `main`)

- No IndexedDB schema bump: `docs` was already keyed by collection, so the
  database stays at version 1 and the Image view's store is read as before.
- The chat's stored pictures are read back into memory for every session when
  the page opens (`withPictures`), not when a session is opened: simple, and a
  cost that grows with the pictures kept. A store-side reference read on use is
  the step if it shows.
- A picture the model made (`make_image`) stays inline in the session, as the
  data URI the chat shows: a session with several is several MB, rewritten on
  each write while a reply in it streams.
- Two tabs of one origin write distinct keys, but each holds the list it read:
  a session deleted in one tab is written again by the other if it changes
  there, and a picture deletion in one tab does not see sessions only the other
  has. The Image view has the same limits.
- A reply stopped by a reload carries no figures and no mark of its own: it
  reads as a reply that ended there.
- Settling is not written back: a session is written again when it next
  changes, and read settled each time until then.
- A picture is stored at send a moment before the message names it; an Image
  view delete in that moment can remove it, and the image keeps its redraw.
  While the other view has not read its sessions, a delete leaves its
  pictures in the store, unnamed; nothing sweeps them.
- The Image view keeps its own save timers (`useImageSessions`): it writes at
  its own moments (a step's end at once), and moving it onto
  `useStoredSessions` is a change of its own.
- Open: export and import (JSON, maybe Markdown); a store on ignis.
