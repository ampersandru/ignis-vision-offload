import { type DragEvent, type ReactNode, useEffect, useState } from "react";
import type { SpeechStatus } from "../api/model.ts";
import { usePictureUrl } from "../image/pictures.tsx";
import type { ImageId } from "../store/imageStore.ts";
import { caption, fieldLook } from "../ui/classes.ts";
import { HtmlPreview, isHtmlName } from "../ui/HtmlPreview.tsx";
import { IconClose, IconDownload, IconPencil, IconPlus, IconTrash } from "../ui/icons.tsx";
import { Lightbox } from "../ui/Lightbox.tsx";
import { AudioViewer, Recording } from "./audio.tsx";
import { type Author, KIND_LABEL, latest, present, USER, type Version, type Workspace, type WorkspaceFile } from "./files.ts";
import { BUTTON, Source } from "./looks.tsx";
import { createFile, downloadName, fileTree, refusal, sizeLabel, type TreeNode, uploadFiles } from "./panel.ts";
import { createRecorder } from "./recorder.ts";
import { exportWorkspace, importWorkspace } from "./transfer.ts";
import type { WorkspaceView } from "./useWorkspaces.ts";
import type { ChatBinding, Workspaces } from "./workspaces.ts";

// The Chat view's workspace panel (spec #170): the files of the chat's
// workspace as a tree of their names, each opening in a viewer for its kind,
// and the user's actions on them — a new empty text file, uploads by the
// picker or by a drop, a recording from the microphone (#395), rename, delete
// and download; an audio file plays beside its transcript. Everything goes
// through the workspace module as the user's; a chat with no workspace yet
// gets one with the first file made here. What the browser refuses, the panel
// says.

/** What the panel's controls do; a name a field sends back comes with the refusal, or null once done. */
export type PanelActions = {
  create: (name: string) => Promise<string | null>;
  upload: (files: File[]) => void;
  /** Saves the workspace as a zip. */
  exportZip: () => void;
  /** Makes a new workspace of a zip; the refusal, or what it made. */
  importZip: (zip: File) => void;
  rename: (file: string, name: string) => Promise<string | null>;
  remove: (file: string) => void;
  download: (file: WorkspaceFile) => void;
  /** A version's bytes, for a player. */
  content: (version: Version) => Promise<Blob | null>;
  /** Computes the text the model reads of a file's latest version, a recording's transcript, and keeps it; why it could not, or null. */
  derive: (file: WorkspaceFile) => Promise<string | null>;
};

/** An icon button's shape; its hover colour is the caller's (two `hover:text-*` on one element, and Tailwind's order picks). */
const ICON_SHAPE = "flex size-7 shrink-0 items-center justify-center text-ash";

const ICON_BUTTON = `${ICON_SHAPE} hover:text-ink`;

const pad = (n: number) => String(n).padStart(2, "0");

function when(at: number): string {
  const d = new Date(at);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Who made a version, as the user reads it. */
const byWhom = (author: Author) => (author.by === "user" ? "you" : author.agent ? `the sub-agent ${author.agent.name}` : "the model");

/** The panel for the chat's workspace, reading and changing it through the workspace module. */
export function WorkspacePanel({
  workspaces,
  view,
  chat,
  speech,
  onClose,
}: {
  workspaces: Workspaces;
  /** The chat's workspace and its files, as useWorkspace reads them. */
  view: WorkspaceView;
  /** The chat's hold on its workspace: the first file made here gives it one. */
  chat: () => ChatBinding;
  /** Whether the speech model can serve, from the models listing: a recording's microphone and transcript need it. */
  speech?: SpeechStatus;
  onClose: () => void;
}) {
  const [notice, setNotice] = useState<string | null>(null);
  const [made, setMade] = useState<string | null>(null);
  const id = view.workspace?.id;
  const [recorder] = useState(() => createRecorder());
  // A recording under way when the panel closes is saved, not lost.
  useEffect(() => () => void recorder.stop(), [recorder]);

  /** `job`, with what the store refused said in the panel. */
  async function attempt<T>(job: () => Promise<T>): Promise<T | { refused: string }> {
    setNotice(null);
    setMade(null);
    try {
      return await job();
    } catch (err) {
      const refused = refusal(err);
      setNotice(refused);
      return { refused };
    }
  }

  const fieldError = (out: { ok: true } | { ok: false; error: string } | { refused: string }) =>
    "refused" in out ? out.refused : out.ok ? null : out.error;

  const actions: PanelActions = {
    create: async (name) => fieldError(await attempt(() => createFile(workspaces, chat(), name))),
    upload: (files) => void attempt(() => uploadFiles(workspaces, chat(), files)),
    exportZip: () =>
      void attempt(async () => {
        if (!view.workspace) return;
        const out = await exportWorkspace(workspaces, view.workspace, view.files);
        save(out.zip, out.name);
        if (out.left.length > 0) setNotice(`Left out of the zip, because an older chat kept no bytes of them: ${out.left.join(", ")}.`);
      }),
    importZip: (zip) =>
      void attempt(async () => {
        const out = await importWorkspace(workspaces, zip);
        if (!out.ok) setNotice(out.error);
        else setMade(`Imported ${out.files} ${out.files === 1 ? "file" : "files"} as the new workspace ${out.workspace.name}. Pick it in the Workspace menu of a chat that has none yet.`);
      }),
    rename: async (file, name) => (id ? fieldError(await attempt(() => workspaces.renameFile(id, file, name))) : null),
    remove: (file) => {
      if (!id) return;
      void attempt(() => workspaces.deleteFile(id, file, USER)).then((out) => {
        if (!("refused" in out) && !out.ok) setNotice(out.error);
      });
    },
    download: (file) =>
      void attempt(async () => {
        const blob = await workspaces.content(latest(file));
        if (blob) save(blob, downloadName(file.name));
        else setNotice(`${file.name} came from an older chat, which kept only its text: there are no bytes to download.`);
      }),
    content: (version) => workspaces.content(version),
    derive: async (file) => {
      const out = await attempt(() => workspaces.readable(file, latest(file)));
      return "refused" in out ? out.refused : out.ok ? null : out.error;
    },
  };

  // The chat a recording goes to is the one on screen when Record was pressed, whichever is on screen when it ends.
  const recordInto = () => {
    const binding = chat();
    return async (file: File) => void (await uploadFiles(workspaces, binding, [file]));
  };

  return (
    <PanelView
      workspace={view.workspace}
      files={view.files}
      notice={notice}
      made={made}
      onClose={onClose}
      actions={actions}
      speech={speech}
      recording={<Recording recorder={recorder} speech={speech} save={recordInto} />}
    />
  );
}

function save(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** The panel as it renders: undefined while the workspace is read, null when the chat has none yet. */
export function PanelView({
  workspace,
  files,
  notice,
  made,
  onClose,
  actions,
  speech,
  recording,
}: {
  workspace: Workspace | null | undefined;
  files: WorkspaceFile[];
  notice?: string | null;
  /** What an import made, said beside the notice. */
  made?: string | null;
  onClose: () => void;
  actions: PanelActions;
  speech?: SpeechStatus;
  /** Record, and the recording under way. */
  recording?: ReactNode;
}) {
  const [opened, setOpened] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const open = files.find((f) => f.id === opened && present(f));
  const tree = fileTree(files);

  const drag = (e: DragEvent) => {
    if (![...e.dataTransfer.types].includes("Files")) return;
    e.preventDefault();
    setOver(true);
  };

  return (
    <aside
      aria-label="Workspace"
      className={`fixed inset-y-0 right-0 z-40 flex w-[22rem] max-w-full flex-col border-l bg-ground 2xl:static 2xl:z-auto ${over ? "border-ember" : "border-line"}`}
      onDragOver={drag}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const dropped = Array.from(e.dataTransfer.files);
        if (dropped.length > 0) actions.upload(dropped);
      }}
    >
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <p className={caption}>Workspace</p>
          <p className="truncate font-display text-sm font-semibold text-ink">
            {workspace ? workspace.name : workspace === null ? "No workspace yet" : "Reading…"}
          </p>
        </div>
        <button type="button" aria-label="Close the workspace" title="Close the workspace" className={ICON_BUTTON} onClick={onClose}>
          <IconClose />
        </button>
      </div>

      {open ? (
        <FileViewer file={open} onBack={() => setOpened(null)} actions={actions} speech={speech} />
      ) : (
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-4 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" className={BUTTON} onClick={() => setCreating(true)}>
              <IconPlus />
              New file
            </button>
            <label className={`${BUTTON} cursor-pointer`}>
              Upload
              <input
                type="file"
                name="workspace-files"
                multiple
                className="sr-only"
                onChange={(e) => {
                  const picked = Array.from(e.currentTarget.files ?? []);
                  e.currentTarget.value = "";
                  if (picked.length > 0) actions.upload(picked);
                }}
              />
            </label>
            {recording}
            <label className={`${BUTTON} cursor-pointer`} title="Make a new workspace of a zip">
              Import zip
              <input
                type="file"
                name="workspace-zip"
                accept=".zip,application/zip"
                className="sr-only"
                onChange={(e) => {
                  const picked = e.currentTarget.files?.[0];
                  e.currentTarget.value = "";
                  if (picked) actions.importZip(picked);
                }}
              />
            </label>
            {tree.length > 0 && (
              <button type="button" className={BUTTON} title="Save the workspace as a zip" onClick={actions.exportZip}>
                Export zip
              </button>
            )}
            <span className="text-xs text-ash">or drop files here</span>
          </div>
          {creating && (
            <NameField
              label="Name of the new file"
              initial="notes.md"
              onSave={async (name) => {
                const error = await actions.create(name);
                if (!error) setCreating(false);
                return error;
              }}
              onCancel={() => setCreating(false)}
            />
          )}
          {workspace === null && <p className="text-xs leading-snug text-ash">The first file you add here, attach or the model writes creates it.</p>}
          {workspace && tree.length === 0 && <p className="text-xs text-ash">No files.</p>}
          <FileTree nodes={tree} renaming={renaming} onOpen={setOpened} onRenaming={setRenaming} actions={actions} />
        </div>
      )}

      {made && (
        <p role="status" className="border-t border-line px-4 py-2 text-xs leading-snug text-ash">
          {made}
        </p>
      )}
      {notice && (
        <p role="status" className="border-t border-line px-4 py-2 text-xs leading-snug text-fault">
          {notice}
        </p>
      )}
    </aside>
  );
}

/** The files as a tree: a folder per part of the names before a `/`, open, and a row per file with its actions. */
export function FileTree({
  nodes,
  renaming,
  onOpen,
  onRenaming,
  actions,
}: {
  nodes: TreeNode[];
  /** The file whose name is being edited. */
  renaming: string | null;
  onOpen: (file: string) => void;
  onRenaming: (file: string | null) => void;
  actions: PanelActions;
}) {
  if (nodes.length === 0) return null;
  return (
    <ul className="flex flex-col gap-0.5">
      {nodes.map((node) =>
        node.kind === "folder" ? (
          <li key={`d:${node.path}`}>
            <details open>
              <summary className="cursor-pointer py-1 font-display text-xs font-semibold text-ash hover:text-ink">{node.name}</summary>
              <div className="ml-2 border-l border-line pl-2">
                <FileTree nodes={node.children} renaming={renaming} onOpen={onOpen} onRenaming={onRenaming} actions={actions} />
              </div>
            </details>
          </li>
        ) : (
          <li key={node.file.id}>
            {renaming === node.file.id ? (
              <NameField
                label={`New name for ${node.file.name}`}
                initial={node.file.name}
                onSave={async (name) => {
                  const error = await actions.rename(node.file.id, name);
                  if (!error) onRenaming(null);
                  return error;
                }}
                onCancel={() => onRenaming(null)}
              />
            ) : (
              <FileRow file={node.file} name={node.name} onOpen={() => onOpen(node.file.id)} onRename={() => onRenaming(node.file.id)} actions={actions} />
            )}
          </li>
        ),
      )}
    </ul>
  );
}

function FileRow({ file, name, onOpen, onRename, actions }: { file: WorkspaceFile; name: string; onOpen: () => void; onRename: () => void; actions: PanelActions }) {
  return (
    <div className="flex min-w-0 items-center gap-1 hover:bg-surface">
      <button type="button" title={file.name} className="min-w-0 flex-1 truncate py-1 pl-1 text-left text-sm text-ink" onClick={onOpen}>
        {name}
      </button>
      <span className="shrink-0 font-display text-xs tabular-nums text-ash">{sizeLabel(latest(file))}</span>
      <button type="button" aria-label={`Rename ${file.name}`} title="Rename" className={ICON_BUTTON} onClick={onRename}>
        <IconPencil />
      </button>
      <button type="button" aria-label={`Download ${file.name}`} title="Download" className={ICON_BUTTON} onClick={() => actions.download(file)}>
        <IconDownload />
      </button>
      <button type="button" aria-label={`Delete ${file.name}`} title="Delete" className={`${ICON_SHAPE} hover:text-fault`} onClick={() => actions.remove(file.id)}>
        <IconTrash />
      </button>
    </div>
  );
}

/** A name being typed: Enter or Save sends it, Escape or Cancel drops it; a refusal stays beside it. */
export function NameField({
  label,
  initial,
  error: refused = null,
  onSave,
  onCancel,
}: {
  label: string;
  initial: string;
  error?: string | null;
  onSave: (name: string) => Promise<string | null>;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(initial);
  const [error, setError] = useState<string | null>(refused);
  const submit = async () => setError(await onSave(value));
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-1.5">
        <input
          aria-label={label}
          value={value}
          autoFocus
          spellCheck={false}
          className={`min-w-0 flex-1 ${fieldLook} py-1`}
          onChange={(e) => setValue(e.currentTarget.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void submit();
            if (e.key === "Escape") onCancel();
          }}
        />
        <button type="button" className="shrink-0 font-display text-xs font-semibold text-ember hover:text-ink" onClick={() => void submit()}>
          Save
        </button>
        <button type="button" className="shrink-0 font-display text-xs text-ash hover:text-ink" onClick={onCancel}>
          Cancel
        </button>
      </div>
      {error && <p className="text-xs leading-snug text-fault">{error}</p>}
    </div>
  );
}

/** A file opened from the tree, its latest version in the viewer for its kind. */
export function FileViewer({ file, onBack, actions, speech }: { file: WorkspaceFile; onBack: () => void; actions: PanelActions; speech?: SpeechStatus }) {
  const v = latest(file);
  let body: ReactNode = null;
  if (v.text !== undefined) {
    body = isHtmlName(file.name) ? <HtmlPreview html={v.text} name={file.name} /> : <Source text={v.text} />;
  } else if (file.kind === "pdf" && v.derived !== undefined) {
    body = (
      <>
        <p className={caption}>Its text, as the model reads it</p>
        <Source text={v.derived} />
      </>
    );
  } else if (file.kind === "audio") {
    body = <AudioViewer file={file} version={v} speech={speech} load={actions.content} transcribe={actions.derive} />;
  } else if (file.kind === "picture" && v.bytes?.store === "image") {
    body = <PictureView id={v.bytes.id} name={file.name} width={v.bytes.width} height={v.bytes.height} />;
  } else if (file.kind === "other") {
    body = <p className="text-xs leading-snug text-ash">The model sees its name and size, and cannot read it.</p>;
  }
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-4 py-3">
      <div className="flex items-center gap-2">
        <button type="button" className="font-display text-xs font-semibold text-ash hover:text-ink" onClick={onBack}>
          ← All files
        </button>
        <button type="button" aria-label={`Download ${file.name}`} title="Download" className={`${ICON_BUTTON} ml-auto`} onClick={() => actions.download(file)}>
          <IconDownload />
        </button>
      </div>
      <p className="font-display text-sm font-semibold break-all text-ink">{file.name}</p>
      <p className="text-xs leading-snug text-ash">
        {KIND_LABEL[file.kind]}, {sizeLabel(v)}; version {v.n}, changed {when(v.at)} by {byWhom(v.author)}
      </p>
      {body}
    </div>
  );
}

/** A picture of the workspace from the page's store, at the panel's width; a click opens it full screen. */
function PictureView({ id, name, width, height }: { id: ImageId; name: string; width?: number; height?: number }) {
  const url = usePictureUrl()(id, "full");
  const [full, setFull] = useState(false);
  return (
    <>
      {width && height && (
        <p className="text-xs text-ash tabular-nums">
          {width} × {height} pixels
        </p>
      )}
      <button type="button" aria-label={`Open ${name} full screen`} title="Full screen" className="block cursor-zoom-in bg-surface" onClick={() => setFull(true)} disabled={!url}>
        {url ? (
          <img src={url} alt={name} className="stage-checker block max-h-[60vh] max-w-full object-contain" />
        ) : (
          <span role="img" aria-label={name} className="block h-40 w-full bg-line/60" />
        )}
      </button>
      {full && url && (
        <Lightbox onClose={() => setFull(false)}>
          <img src={url} alt={name} className="stage-checker block max-h-[calc(100dvh-2rem)] max-w-[calc(100vw-2rem)] object-contain" />
        </Lightbox>
      )}
    </>
  );
}
