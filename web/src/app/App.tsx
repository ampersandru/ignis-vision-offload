import { useCallback, useEffect, useRef, useState } from "react";
import { forgetKey, useAuth } from "../api/auth.ts";
import { canChat, canSend, loadUnderWay, type ModelState, useModel } from "../api/model.ts";
import { Composer } from "../conversation/Composer.tsx";
import { type ChatPicture, chatImageFromFile, type PictureAction, type PromptImage } from "../conversation/images.ts";
import { ModelPicker } from "../conversation/ModelPicker.tsx";
import { useNextModel } from "../conversation/useNextModel.ts";
import { DecideView, type Drawer as DecideDrawer } from "../decide/DecideView.tsx";
import { type OpenAgent, Transcript } from "../conversation/Transcript.tsx";
import { contextUsage } from "../metrics/context.ts";
import { Readout } from "../metrics/Readout.tsx";
import { MonitorView } from "../monitor/MonitorView.tsx";
import { useMonitor, useMonitorVisible } from "../monitor/useMonitor.ts";
import { SessionLog } from "../sessions/SessionLog.tsx";
import { SessionsPanel } from "../sessions/SessionsPanel.tsx";
import { StorePictures } from "../image/pictures.tsx";
import { imagesOf, pictureName, type PoolImage } from "../image/sessions.ts";
import { fileFromDataUrl } from "../image/studio.ts";
import type { Mark, MarkedPicture } from "../markup/markup.ts";
import { MarkupEditor } from "../markup/MarkupEditor.tsx";
import { ImageStudio, type StudioDrawer } from "../image/Studio.tsx";
import { useImageSessions } from "../image/useImageSessions.ts";
import { ServerView } from "../server/ServerView.tsx";
import { useServer, useTick } from "../server/useServer.ts";
import { DEFAULT_SETTINGS, type PlaygroundSettings } from "../settings/defaults.ts";
import type { ImageId } from "../store/imageStore.ts";
import { pageStore } from "../store/index.ts";
import { SettingsPanel } from "../settings/SettingsPanel.tsx";
import { dictationContext } from "../speech/dictation.ts";
import { useDictation } from "../speech/useDictation.ts";
import { AgentReader } from "../tools/agents/AgentReader.tsx";
import { MemoryManager } from "../tools/local/MemoryManager.tsx";
import { AGENT_SYSTEM_PROMPT, type AgentRun } from "../tools/agents/agents.ts";
import { ALL_TOOLS, type ToolsState } from "../tools/index.ts";
import { Header, type View } from "./Header.tsx";
import { KeyPage } from "./KeyPage.tsx";
import { TransportNotice } from "./TransportNotice.tsx";
import { STORED_FLAGS, useStoredFlag } from "./storedFlag.ts";
import { useConversation } from "./useConversation.ts";

// The Playground (GitHub #164): a streaming chat against ignis — over its
// Responses WebSocket, or /v1/chat/completions (GitHub #283) — with
// per-request figures measured in the browser.
// The chat's, the Decide tab's and the Image view's sessions are kept in this
// browser's store and come back on a reload; settings live in memory, apart
// from the switches this browser stores (`storedFlag.ts`).
// This is the page's layout; the conversation loop is useConversation.
// The header switches between the chat, the Decide tab (GitHub #247), —
// when ignis serves metrics — the Monitor (GitHub #165), and the Server view
// (GitHub #316). Whichever is showing, the chat stays mounted underneath, so
// a streaming reply carries on.

type Drawer = "sessions" | "settings" | null;

/** A picture of the chat open in the markup editor, and what saving the marks does with them. */
type ChatMarking = { url: string; title: string; marks?: Mark[]; keepOutside?: boolean; apply: (marked: MarkedPicture | null) => Promise<void> };

/** A picture of the chat at its own size: the one kept beside the redraw, else the redraw itself. */
const blobOf = (picture: { full?: Blob; url: string }, name: string): Blob => picture.full ?? fileFromDataUrl(picture.url, name);

const stem = (name: string) => name.replace(/\.[^.]+$/, "");

export function App() {
  const auth = useAuth();
  const [model, refreshModel] = useModel();
  const monitorVisible = useMonitorVisible();
  const [view, setView] = useState<View>("chat");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [tools, setTools] = useState<ToolsState>(ALL_TOOLS);
  const [markdown, setMarkdown] = useState(true);
  // Off: the page runs one turn at a time, whichever session it is in. Stored,
  // so turning it on survives a reload rather than looking like it never took.
  const [parallel, setParallel] = useStoredFlag(STORED_FLAGS.parallel, false);
  const [input, setInput] = useState("");
  // The images the next prompt will carry. They belong to the box, not to a
  // session: once sent they live on the user message, and the box is empty again.
  const [images, setImages] = useState<PromptImage[]>([]);
  const [imageError, setImageError] = useState<string | null>(null);
  const [logOpen, setLogOpen] = useState(false);
  const [reader, setReader] = useState<OpenAgent | null>(null);
  const closeReader = useCallback(() => setReader(null), []);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const closeMemory = useCallback(() => setMemoryOpen(false), []);
  const [drawer, setDrawer] = useState<Drawer>(null);
  // The bench has its own two drawers below `lg`; the header's buttons open
  // whichever view is showing.
  const [decideDrawer, setDecideDrawer] = useState<DecideDrawer>(null);
  // The model the next message goes to: the one loaded, unless the owner picks another or the image model is loaded.
  const nextModel = useNextModel(model);
  // The pictures the Image view's sessions name, for the chat's deletes: the view is held below the chat.
  const studioPictures = useRef<() => Iterable<ImageId> | null>(() => null);
  const chat = useConversation({ model, next: nextModel.next, refreshModel, settings, tools, parallel, live: () => studioPictures.current() });
  const { active } = chat;
  // Dictation into the prompt (spec speech/04): opened with the last reply and the prompt as it stands, in that order.
  const dictation = useDictation({
    onChange: setInput,
    context: () => dictationContext([[...active.messages].reverse().find((m) => m.role === "assistant" && m.content.trim() !== "")?.content ?? "", input]),
    speech: model.speech,
    onUnavailable: refreshModel,
  });
  const monitoring = view === "monitor" && monitorVisible;
  const deciding = view === "decide";
  const serving = view === "server";
  const imaging = view === "image";
  // Mounted from its first visit on, like the Decide tab: a draft of changes
  // and a reload being followed survive a look at the chat.
  const [serverOpened, setServerOpened] = useState(false);
  // The Image view the same: its sessions are read once, and a step under way outlives a look at the chat.
  const [imageOpened, setImageOpened] = useState(false);
  const [imageDrawer, setImageDrawer] = useState<StudioDrawer>(null);
  // Held here, not in the view: the chat hands pictures to it.
  // Each keeps the pictures the other names when it deletes a session: the store holds both views' pictures once.
  const [studio, studioActions] = useImageSessions(model, refreshModel, imaging, { live: chat.pictures });
  studioPictures.current = () => (studio.loaded ? studio.sessions.flatMap((s) => [...imagesOf(s)]) : null);
  const [chatMarking, setChatMarking] = useState<ChatMarking | null>(null);
  // A switch or a load runs the header's flame and heat bar as a reply does (spec playground/05).
  const busy = chat.busy || loadUnderWay(model);

  const readerRun: AgentRun | undefined = reader
    ? active.messages.find((m) => m.id === reader.messageId)?.agents?.find((r) => r.callId === reader.callId)
    : undefined;

  // Metrics going away drops back to the chat *from the Monitor*, so their
  // return never flips the page on its own — and a reader in the Decide tab,
  // which needs no metrics, is left where they are.
  useEffect(() => {
    if (!monitorVisible) setView((current) => (current === "monitor" ? "chat" : current));
  }, [monitorVisible]);

  useEffect(() => {
    if (!drawer) return;
    const close = (e: globalThis.KeyboardEvent) => e.key === "Escape" && setDrawer(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [drawer]);

  function selectSession(id: number) {
    chat.selectSession(id);
    setDrawer(null);
    setReader(null);
  }

  function newSession() {
    chat.newSession();
    setDrawer(null);
  }

  function send() {
    const text = input.trim();
    // A dictation under way is never sent: Send waits until it ends (Q4).
    if ((!text && images.length === 0) || !chat.canRun || dictation.controller.active) return;
    setInput("");
    setImages([]);
    setImageError(null);
    chat.send(text, images);
  }

  /** Picked or pasted files as prompt images; the ones that cannot be read are named instead. */
  async function addImages(files: File[]) {
    // The chat keeps each picture at its own size too: the picture tool edits that one.
    const results = await Promise.all(files.map((file) => chatImageFromFile(file)));
    const errors = results.flatMap((r) => (r.ok ? [] : [r.error]));
    setImageError(errors.length > 0 ? errors.join(" ") : null);
    const added = results.flatMap((r) => (r.ok ? [r.image] : []));
    if (added.length > 0) setImages((current) => [...current, ...added]);
  }

  /** The marked picture as a prompt image of the chat: the drawn version on the wire, the marks and the original beside. */
  async function markedImage(source: Blob, name: string, marked: MarkedPicture): Promise<PromptImage | null> {
    const read = await chatImageFromFile(marked.drawn, `${stem(name)}-marked.png`);
    if (!read.ok) {
      setImageError(read.error);
      return null;
    }
    return { ...read.image, markup: { source, marks: marked.marks, keepOutside: marked.keepOutside, region: marked.region } };
  }

  /** Opens `source` in the markup editor; `apply` gets what was saved. */
  function openMarking(source: Blob, title: string, apply: ChatMarking["apply"], earlier?: PromptImage["markup"]) {
    setChatMarking({
      url: URL.createObjectURL(source),
      title,
      ...(earlier ? { marks: earlier.marks, keepOutside: earlier.keepOutside } : {}),
      apply,
    });
  }

  function closeMarking() {
    if (chatMarking) URL.revokeObjectURL(chatMarking.url);
    setChatMarking(null);
  }

  /** A picture of the prompt box marked again from its original; it keeps its place in the box. */
  function markBoxImage(index: number) {
    const image = images[index];
    if (!image) return;
    const source = image.markup?.source ?? blobOf(image, image.name);
    openMarking(
      source,
      `Mark ${image.name}`,
      async (marked) => {
        const next = marked ? await markedImage(source, image.name, marked) : image.markup ? await clean(source, image.name) : null;
        if (next) setImages((current) => current.map((x) => (x === image ? next : x)));
      },
      image.markup,
    );
  }

  async function clean(source: Blob, name: string): Promise<PromptImage | null> {
    const read = await chatImageFromFile(source, name);
    if (read.ok) return read.image;
    setImageError(read.error);
    return null;
  }

  /** A picture of the conversation: marked into the prompt box, or opened in a new session of the Image view. */
  function onPicture(picture: ChatPicture, action: PictureAction) {
    const source = blobOf(picture, picture.name);
    if (action === "image") {
      void studioActions.importPictures([{ blob: source, name: picture.name }], "new");
      setImageOpened(true);
      setView("image");
      return;
    }
    openMarking(source, `Mark ${picture.name}`, async (marked) => {
      const image = marked && (await markedImage(source, picture.name, marked));
      if (image) setImages((current) => [...current, image]);
    });
  }

  /** A picture of the Image view into the chat's prompt box. */
  function sendToChat(picture: PoolImage) {
    void pageStore()
      .getImage(picture.id)
      .then(async (blob) => {
        if (!blob) return;
        const read = await chatImageFromFile(blob, pictureName(picture));
        if (read.ok) setImages((current) => [...current, read.image]);
        else setImageError(read.error);
        setView("chat");
      });
  }

  const set = <K extends keyof PlaygroundSettings>(key: K, value: PlaygroundSettings[K]) =>
    setSettings((s) => ({ ...s, [key]: value }));

  // A 401 anywhere shows the key prompt in place of the page; App stays
  // mounted, so the sessions are still here once the key is in.
  if (auth.needsKey) return <KeyPage rejected={auth.rejected} />;

  return (
    // Stored pictures are shown by URLs read once from the page's store: the Image view's, and the chat's.
    <StorePictures store={pageStore()}>
    <div className="flex h-dvh flex-col overflow-hidden">
      <Header
        model={model}
        busy={busy}
        onOpen={(which) => (deciding ? setDecideDrawer(which) : imaging ? setImageDrawer(which) : setDrawer(which))}
        onForgetKey={auth.key ? forgetKey : undefined}
        onOpenMemory={() => setMemoryOpen(true)}
        view={monitoring ? "monitor" : deciding ? "decide" : serving ? "server" : imaging ? "image" : "chat"}
        onView={(next) => {
          if (next === "server") setServerOpened(true);
          if (next === "image") setImageOpened(true);
          setView(next);
        }}
        monitorAvailable={monitorVisible}
      />

      <TransportNotice />

      {monitoring && <Monitor />}

      {/* Mounted whichever view is showing, like the chat below it: the tab
          holds a whole typed request, and a glance at the Monitor is not a
          reason to lose it. Nothing here fetches on mount. */}
      <div className={deciding ? "contents" : "hidden"}>
        <DecideView ready={canChat(model)} drawer={decideDrawer} onDrawer={setDecideDrawer} speech={model.speech} />
      </div>

      {serverOpened && (
        <div className={serving ? "contents" : "hidden"}>
          <Server model={model} refreshModel={refreshModel} streaming={chat.busy} visible={serving} />
        </div>
      )}

      {imageOpened && (
        <div className={imaging ? "contents" : "hidden"}>
          <ImageStudio model={model} state={studio} actions={studioActions} drawer={imageDrawer} onDrawer={setImageDrawer} onSendToChat={sendToChat} />
        </div>
      )}

      <div className={monitoring || deciding || serving || imaging ? "hidden" : "contents"}>
        <div className="relative flex min-h-0 flex-1">
          {drawer && (
            <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => setDrawer(null)} aria-hidden />
          )}

          <SessionsPanel
            open={drawer === "sessions"}
            list={chat.list}
            activeId={active.id}
            streaming={chat.streaming}
            onNew={newSession}
            onSelect={selectSession}
            onRemove={chat.deleteSession}
            notice={chat.storeError}
          />

          <main className="flex min-w-0 flex-1 flex-col">
            <Transcript
              session={active}
              model={model}
              markdown={markdown}
              canRerun={chat.canRun}
              following={chat.following}
              openAgent={reader}
              onOpenAgent={setReader}
              onRerun={chat.rerun}
              onSave={chat.saveEdit}
              onFork={chat.fork}
              onAnswer={chat.answer}
              onPicture={onPicture}
            />
            <Composer
              value={input}
              onChange={setInput}
              ready={canSend(model) && nextModel.next.id !== null}
              modelPicker={<ModelPicker model={model} next={nextModel.next} rows={nextModel.rows} onChoose={nextModel.choose} />}
              busy={chat.sendBlocked}
              streamingHere={chat.streamingHere}
              usage={contextUsage(active.log, settings.maxTokens, canChat(model) ? model.contextLimit : null)}
              onSend={send}
              onStop={chat.stop}
              canAttach={tools.enabled && tools.readFiles}
              attachments={active.attachments}
              attachError={chat.attachError}
              onAttach={(files) => void chat.attach(files)}
              onDetach={chat.detach}
              images={images}
              imageError={imageError}
              onAddImages={(files) => void addImages(files)}
              onRemoveImage={(index) => setImages((current) => current.filter((_, i) => i !== index))}
              onMarkImage={markBoxImage}
              dictation={dictation}
            />
          </main>

          <SettingsPanel
            open={drawer === "settings"}
            settings={settings}
            set={set}
            tools={tools}
            onToolsChange={setTools}
            markdown={markdown}
            onMarkdownChange={setMarkdown}
            parallel={parallel}
            onParallelChange={setParallel}
            attachments={active.attachments}
            onOpenMemory={() => setMemoryOpen(true)}
          />
        </div>

        <SessionLog rows={active.log} open={logOpen} onToggle={() => setLogOpen((o) => !o)} />
      </div>

      {memoryOpen && <MemoryManager onClose={closeMemory} />}

      {chatMarking && (
        <MarkupEditor
          source={{ url: chatMarking.url, ...(chatMarking.marks ? { marks: chatMarking.marks, mode: "drawn" as const, keepOutside: chatMarking.keepOutside } : {}) }}
          title={chatMarking.title}
          modes={["drawn"]}
          onSave={(marked) => {
            const apply = chatMarking.apply;
            closeMarking();
            void apply(marked);
          }}
          onClose={closeMarking}
        />
      )}

      {readerRun && (
        <>
          <div className="fixed inset-0 z-40 bg-[#1c2026]/40" onClick={closeReader} aria-hidden />
          <AgentReader
            key={readerRun.callId}
            run={readerRun}
            markdown={markdown}
            systemPrompt={readerRun.systemPrompt ?? AGENT_SYSTEM_PROMPT}
            figures={readerRun.figures && <Readout figures={readerRun.figures} />}
            onClose={closeReader}
          />
        </>
      )}
    </div>
    </StorePictures>
  );
}

/** The Monitor subscribes on its own, so a scrape re-renders it and not the chat. */
function Monitor() {
  return <MonitorView state={useMonitor()} />;
}

/** The Server view's state lives here, and its clock ticks only while a switch runs. */
function Server({ model, refreshModel, streaming, visible }: { model: ModelState; refreshModel: () => void; streaming: boolean; visible: boolean }) {
  const [state, actions] = useServer(model, refreshModel, visible);
  const now = useTick(loadUnderWay(model));
  return <ServerView model={model} state={state} actions={actions} streaming={streaming} now={now} />;
}

