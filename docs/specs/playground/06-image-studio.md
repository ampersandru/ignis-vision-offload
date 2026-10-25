# playground 06 — the image studio: image sessions, marks, the picture tool and the next message's model

GitHub: #326

## Problem Statement

The Image view (spec image-model/05 §Playground, #322) was one form and a
gallery. A picture worth keeping takes several edits, each going on from the
last; the form kept no history of them, each edit meant picking the last
picture again by hand, and a reload lost everything. Qwen-Image-2.1 can edit a
marked area ("circles, painted annotations, or separate masks", its card) and
make transparent pictures; the view offered neither. The chat could not make a
picture at all, and with the image model loaded it could not send: the owner
had to switch back from the Server view first.

## Solution

All of it is web only: the server is unchanged.

### Image sessions (the Image view)

The Image view takes the chat's shape: **image sessions** on the left, the
next step's settings on the right, the session's steps between them, the
prompt box at the foot (GLOSSARY.md: image session, image pool, image input).

- A session holds an **image pool** — every picture uploaded into it, made by
  it, marked in it or brought from the chat, each named `#N` for good — and its
  steps, each the prompt, the **image inputs** it was sent with (in order:
  image 1, image 2…) and the pictures it made.
- The next step's inputs default to the last step's first picture; the owner
  adds inputs from the pool (`+`), by upload, or by dropping pictures on the
  prompt box, and drops them. No inputs: the step makes a new picture.
- The last step's pictures show large on the kiln, as the stage; a picture
  opens full-size in the viewer (arrows walk the pool), with "Use as input",
  "Mark", "Reuse prompt and seed", "Download PNG" and "Send to the chat".
- The settings (shape, count, seed, background, enhancement, progress) belong
  to the session's draft. The seed field clears after each step, and an edit
  is never sent a seed behind its inputs: the same noise again builds up into
  grain (finding 2026-10-16-image-edit-same-seed-grain). "Same seed between
  edits" turns that off on purpose, to reproduce a picture: off by default,
  and while on the settings and the prompt box say plainly that it is not
  advised and why.
- "Enhance the prompt" starts off, and a session that never set it takes the
  value last set in any session (stored in the browser). With the image model
  loaded it is a visit to the enhancer and back, which the prompt box says.
- Sending a step empties the prompt box, as the chat's does; a step that makes
  nothing gives its prompt back.
- With a text model loaded, Generate and Edit switch to the image model on
  their own (#324). The explicit switch band is not shown for now (owner's
  call); `image/SwitchBand.tsx` keeps it, tested.
- A step left running by a page that went away reads as stopped.

### Marks

The markup editor (`markup/`) draws circles and brush strokes in four colours
on a picture, at the picture's own resolution, with undo and clear. Saving
makes a new pool picture (`origin: marked`) that takes its source's place among
the inputs. The marks travel one of two ways:

- **drawn on the picture** — the model reads them;
- **as a mask** — the clean picture, then a mask image (the marked area white
  on black) as one more condition image; the prompt names it by number. The
  edits route takes no `mask` field (spec image-model/05): the mask is a
  condition image, as the card has it.

"Change only the marked area" lays each result over the original through the
marked area, feathered (about 0.6 % of the short side), so nothing outside it
changes. A first mark on an empty prompt starts the sentence ("In image 1,
change what is marked in red: ").

### Transparent pictures

The background control sends `background` (`auto`, `transparent`, `opaque`).
Transparent wraps the prompt in the card's wording ("This is an RGBA image
with transparency. … The image has alpha channel and the background is
transparent."); the step shows the owner's own words.

### The image store

`store/imageStore.ts` is the seam: images as bytes named by their SHA-256 (the
same picture is kept once), each with a WebP thumbnail, and JSON documents by
collection and id. Adapters: IndexedDB (`store/indexeddb.ts`, the page's) and
memory (tests, and the fallback without IndexedDB). One contract test file
runs on every adapter (`store/store.test.ts`, IndexedDB on fake-indexeddb); a
local folder (File System Access) or the server are adapters to come. Image
sessions are documents of `image-sessions`. Deleting a session deletes the
images no remaining session names; there is no sweep at start (two open
pages would delete each other's work). IndexedDB belongs to an origin: the
Vite dev server and the Playground the server embeds keep separate stores.

### The picture tool (`make_image`)

A chat tool, on by default. The model passes a prompt, optional references by
name (`user-N`, the Nth picture the user sent; `made-N`, the Nth the tool
made), a shape, transparency, a count (1-4) and a seed. The page switches to
the image model, makes every picture the reply asked for, and switches back to
the text model before the next request — stopped or not. Picture calls run
after the reply's other tool calls, which need the text model; agents do not
get the tool. The user's pictures are sent at their own size (the chat keeps
the original beside the ≤1568 px redraw the vision tower gets). A reference
marked with "change only the marked area" has each result laid over its
original. The tool result tells the model each picture's name, size and seed;
the pictures show in the reply.

### Marks in the chat, and the way between the two views

- A picture of the prompt box, one the user sent, or one the tool made can be
  marked (drawn marks only: the text model sees them too); the marked picture
  goes into the prompt box, ready to name in a message.
- "Image view" on a chat picture opens it in a new image session; "Send to the
  chat" in the viewer puts a pool picture in the chat's prompt box.
- The chat's pictures stay in memory and reach the store only through the
  Image view, so nothing piles up without a way to delete it.

### The next message's model

Under the prompt box, a picker names the model the next message goes to and
what sending does first (nothing, a load, or a switch), with the catalog's
text models to choose from. With the image model loaded the message goes back
to the text model the page last saw (else the one the server last switched
away from, else the first on disk). The page switches with
`POST /v1/models/switch` and waits for the model before the first request;
the prompt shows the switch under way, done, stopped or refused. A choice is
the next message's only: once its model serves the chat follows the server
again, so no chat request ever names a model (#305's ping-pong).

## Seams

- `web/src/store/` — `imageStore.ts` (the interface), `indexeddb.ts`,
  `memory.ts`, `index.ts` (the page's store, chosen at first use),
  `pictures.ts` (thumbnails, browser only).
- `web/src/image/` — `sessions.ts` (the model, pure), `useImageSessions.ts`,
  `Studio.tsx`, `StudioComposer.tsx`, `StudioPanels.tsx`, `Viewer.tsx`,
  `controls.tsx`, `pictures.tsx` (stored pictures as object URLs),
  `SwitchBand.tsx`, `studio.ts` (requests, shapes, the RGBA wording).
- `web/src/markup/` — `markup.ts` (geometry, rendering, keeping to the area),
  `MarkupEditor.tsx`.
- `web/src/tools/image/` — `image.ts` (the tool, the runner), `ImageStrip.tsx`.
- `web/src/api/switching.ts` — a switch the page waits out; used by the chat
  and the tool.
- `web/src/conversation/` — `nextModel.ts`, `useNextModel.ts`,
  `ModelPicker.tsx`, `images.ts` (`full`, `markup` on a prompt image).

## Acceptance

1. The Image view lists image sessions, newest first, each by its last
   picture, with the store's picture count and size at the foot; New session
   reuses an empty one. (Studio.test, sessions.test)
2. A step records its prompt, inputs and settings, and its pictures land in
   the pool and on the step in the order they come. (sessions.test)
3. Sending a step empties the prompt box; after a step that made pictures,
   its first is the only input and the seed is empty (with "Same seed between
   edits", the picture's); a step that made none gives its prompt back to an
   empty box. Stop ends the step and starts nothing. (sessions.test; Stop
   checked live, 2026-10-16)
4. Inputs are shown as the prompt names them; a mask-mode input shows its
   mask's number, and the count against `image.max_condition_images` counts
   masks. (Studio.test)
5. A marked picture is a new pool picture that replaces its source (or its
   earlier marking) among the inputs; removing the marks gives the source its
   place back. (sessions.test)
6. A request sends a drawn input's drawn picture, and a mask-mode input's
   source then its mask. (sessions.test)
7. "Change only the marked area" lays each result over the original through
   the area. (the tool runner: image.test, with an injected compositor; the
   Image view's path and the canvas are browser-only, not under test)
8. `background` is sent only when chosen; transparent carries the card's
   wording and the step shows the owner's words. (images.test, studio.test)
9. Every ImageStore adapter passes the contract tests: content addressing,
   one copy of the same bytes, thumbnails, deletion, documents by collection.
   (store.test, memory and IndexedDB)
10. Sessions survive a reload; a step left running reads as stopped; deleting
    a session deletes only the images no other session names. (sessions.test
    for the stopped step, store.test for what can go; the hook's delete and the
    reload are the owner's to see)
11. `make_image` is declared with the tools (not to agents), switches to the
    image model once per reply, makes every call's pictures, and switches
    back — after a refusal and after a stop too; a turn stopped before its
    calls switches nothing. (image.test)
12. `make_image` names pictures `user-N` and `made-N` across the turn's
    rounds, sends the user's pictures at their own size, refuses a name no
    picture has, and tells the model names, sizes and seeds. (image.test)
13. The prompt box's pictures, the user's sent pictures and the tool's
    pictures can be marked into the prompt box, and opened in the Image view.
    (pictures.test)
14. The picker names the next message's model: the loaded one, the owner's
    choice, or the way back from the image model; a choice clears once its
    model serves. (nextModel.test, ModelPicker.test)
15. With the image model loaded, Send switches to the text model first and the
    turn records the switch on its prompt; a refused switch sends nothing.
    (switching.test for the switch waited out, ModelPicker.test for the line
    on the prompt; the turn itself, in useConversation, is the owner's to see)
16. `cargo test` is untouched (no Rust); `npm run typecheck` and the web
    suite pass.
17. A seed is the server's u64 whole: the page keeps seeds as digits, reads
    them from the server's JSON and the stream without rounding, sends them
    back as numbers of any size, and adds an image's index as the server does
    (wrapping); a session kept with numeric seeds reads them as digits.
    (seed.test, images.test, studio.test, sessions.test, image.test)
18. An edit is never sent a seed behind its inputs — an input's own, or that of
    a picture it was made or marked from: the seed is left to the server and
    the step says so; "Reuse prompt and seed" brings back the step that made
    the picture, its inputs included; `make_image` drops a seed that made one
    of its references and tells the model. A draft kept with the old "Keep the
    last seed" reads with a blank seed. "Same seed between edits" sends it
    anyway, says so on the step, and shows a warning while on; it is off in a
    new session. (sessions.test, image.test, Studio.test)
19. "Enhance the prompt" is off in a new browser, and otherwise starts at the
    value last set. (Studio.test)

## As built (2026-10-16, branch `playground-image-studio`)

- Not run live on the GPU in the session: a `make_image` call (two reloads),
  a send that switches back from the image model, and the canvas paths (marks
  rendered, laid over, thumbnails). The owner watched the page on :5173; the
  console stayed clean on the Image view and the model picker.
- The explicit switch band is hidden, not removed.
- Seeds (2026-10-16, owner's report): a drawn seed such as
  5936113633719656123 came back as …656000 from `JSON.parse` and the seed field
  refused it. Seeds are now digits end to end (`web/src/api/seed.ts`); the
  seeds of pictures made before the fix stay as they were shown, rounded.
- Grain (2026-10-16, owner's report): ten chained edits, the last five with
  "Keep the last seed", turned the picture to grain; reproduced from a clean
  picture in three steps with a fixed seed and not with fresh ones (finding
  2026-10-16-image-edit-same-seed-grain). The option is gone; AC 18 guards.
- Stop (2026-10-16, owner's report): Stop and Generate were one button that
  changed type; the run ended while Stop's click was still being dispatched,
  and the button, a submit again by then, sent the form: a new step started at
  once (and, mid enhancement visit, failed on "a model switch is under way").
  Two buttons now, and Stop prevents the click's default. Reproduced and
  checked live.
- `fake-indexeddb` joined the web dev dependencies (the IndexedDB contract
  tests); `make build` runs `npm ci` once for it.
