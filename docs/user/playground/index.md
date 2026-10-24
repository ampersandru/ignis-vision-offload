---
title: "The Playground"
description: "The browser UI at /ui: chat, Decide, Image, Monitor and Server views."
sidebar:
  order: 1
---

The Playground (`web/`, React + Vite) is served at `/ui/` unless `--server-ui false`, but
it is embedded into `ignis-server` only if `web/dist` exists when cargo builds
it — cargo never runs npm. Build the frontend first (needs Node.js), then the
server:

```
npm --prefix web ci
npm --prefix web run build
cargo build --release -p ignis-server
```

A server built without `web/dist` still serves `/ui/`, as a page with these
instructions.

The **Server** view changes the running server from the page (GitHub #316). It
lists the models a switch can go to (`GET /v1/models/catalog`); **Load**
switches to one by id, whatever `--switch-allow-implicit` says, and a model not
on disk shows the `ignis-server model download <id>` command instead. Below,
every field `GET /v1/config` shows is a form (`GET /v1/config/fields` says
which can change and which reload the model); **Apply** sends the changed
fields as one `PATCH /v1/config`, which writes them to the config file in use,
and reads **Apply and reload** when one of them reloads the model.

The **Image** view is a studio of **image sessions** (GitHub #326; web only,
the server is unchanged). A session keeps every picture uploaded into it, made
by it or marked in it, each named `#N`, and its steps: the prompt, the input
images it was sent with and the pictures it made. The next step's input
defaults to the last step's first picture, so a picture is edited step after
step; no input makes a new one. A picture can be **marked** — circles and
brush strokes drawn on it, or sent as a mask image — and "Change only the
marked area" lays each result over the original through the marked area, so
nothing outside it changes. **Transparent** pictures use the card's own
wording. Sessions are stored in the browser (IndexedDB) and survive a reload.
An edit is never sent the seed behind its inputs unless "Same seed between
edits" is on: the same noise again builds up into grain
([finding](../../findings/2026-10-16-image-edit-same-seed-grain.md)).

In the **chat**, the `make_image` tool (on by default, not for agents) lets the
model make pictures: the page switches to the image model, makes them, and
switches back to the text model before the next request. The pictures the
user sent and the ones the tool made can be referenced, marked, and opened in
an image session. A picker under the prompt box names the model the next
message goes to and what sending does first (nothing, a load or a switch);
with the image model loaded, the next message goes back to a text model on
its own.

The **chat's sessions** and the **Decide tab's decisions** are stored in the
browser too (IndexedDB, GitHub #171), with the image sessions, and come back
after a reload with the session that was open: every message with its
reasoning, tool runs, figures and log, the attached files, and the decisions'
drafts, evidence and answers. A reply that was streaming comes back stopped
with the text it had got, and a call it left running is answered as stopped. A
picture sent in the chat is stored at its own size (with its marks), once,
beside the Image view's pictures; deleting a session deletes the pictures no
other session — chat or image — still names. The store belongs to the browser
and the origin: the dev server on :5173 and the page on `/ui/` keep separate
ones, and nothing goes to the server. When the browser refuses the space, the
session list says so and the page goes on without saving.

A model [unloaded while idle](../api/idle-unload.md) shows as **unloaded** under its id
— in the header and the Server view, not as an error — and the page does not
poll it; **Load** on it loads it now, and a message sent meanwhile loads it and
is answered once it is warm (the header reads **Loading** until it serves).

For frontend work, `npm --prefix web run dev` proxies `/v1` and `/ui/metrics` to
a running engine (`IGNIS_URL`, default `http://127.0.0.1:8000`);
`npm --prefix web run dev:mock` serves a fake engine instead, so no GPU is
needed.
