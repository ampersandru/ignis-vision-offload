import { useState } from "react";
import { useAnchoredMenu } from "../ui/anchoredMenu.ts";
import { IconChevron } from "../ui/icons.tsx";
import { present, type Workspace, type WorkspaceFile, type WorkspaceId } from "./files.ts";
import { NameField } from "./WorkspacePanel.tsx";

// The picker in the chat's header (spec #170): the workspace the chat works
// in. Before the chat binds it offers a new workspace, the default, born
// with the chat's first file, or one of the existing ones, and picking one
// binds the chat to it for good. Once bound, it names the workspace and
// renames or deletes it, a delete only after a confirmation.

export type PickerMode = "choices" | "actions" | "rename" | "confirm";

const MENU_ROW = "flex w-full items-start gap-2.5 px-3 py-2 text-left hover:bg-line";

const pad = (n: number) => String(n).padStart(2, "0");

function when(at: number): string {
  const d = new Date(at);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

const count = (n: number) => `${n.toLocaleString("en-US")} ${n === 1 ? "file" : "files"}`;

/** The chat's header: its workspace's picker, the button that opens and closes the panel, and what the browser refused. */
export function ChatWorkspaceBar({
  open,
  onToggle,
  notice,
  ...picker
}: Parameters<typeof WorkspacePicker>[0] & { open: boolean; onToggle: () => void; notice?: string | null }) {
  const shown = picker.files.filter(present).length;
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-line px-4 py-1.5 md:px-10">
      <WorkspacePicker {...picker} />
      {notice && (
        <p role="status" className="min-w-0 text-xs leading-snug text-fault">
          {notice}
        </p>
      )}
      <button
        type="button"
        aria-pressed={open}
        title={open ? "Close the workspace panel" : "Open the workspace panel"}
        className={`ml-auto shrink-0 font-display text-[13px] font-semibold ${open ? "text-ember" : "text-ash"} hover:text-ink`}
        onClick={onToggle}
      >
        Files{shown > 0 ? ` · ${shown.toLocaleString("en-US")}` : ""}
      </button>
    </div>
  );
}

export function WorkspacePicker({
  workspace,
  files,
  list,
  onPick,
  onRename,
  onDelete,
}: {
  /** The chat's workspace: null before it binds, undefined while it is read. */
  workspace: Workspace | null | undefined;
  files: WorkspaceFile[];
  /** The workspaces there are, as useWorkspaceList reads them. */
  list: Workspace[] | null;
  /** Binds the chat to an existing workspace. */
  onPick: (id: WorkspaceId) => void;
  /** Renames the chat's workspace; the refusal, or null once done. */
  onRename: (name: string) => Promise<string | null>;
  /** Deletes the chat's workspace, once confirmed. */
  onDelete: () => void;
}) {
  const menu = useAnchoredMenu();
  const [mode, setMode] = useState<PickerMode>("choices");
  const label = workspace ? workspace.name : workspace === null ? "New workspace" : "Reading…";
  return (
    <div className="relative min-w-0">
      <button
        ref={menu.button}
        type="button"
        aria-haspopup="menu"
        aria-expanded={menu.open}
        aria-label={`Workspace of this chat: ${label}`}
        title={workspace === null ? "Choose the workspace of this chat" : "Rename or delete the workspace"}
        disabled={workspace === undefined}
        className="flex min-w-0 items-center gap-2 py-1 font-display text-[13px] font-semibold text-ink hover:text-ember disabled:cursor-default disabled:hover:text-ink"
        onClick={() => {
          setMode(workspace ? "actions" : "choices");
          menu.toggle();
        }}
      >
        <span className="shrink-0 font-medium text-ash">Workspace</span>
        <span className="min-w-0 truncate">{label}</span>
        <IconChevron className={`size-3.5 shrink-0 text-ash transition-transform motion-reduce:transition-none ${menu.open ? "rotate-180" : ""}`} />
      </button>
      {menu.open && (
        <div
          ref={menu.menu}
          className="absolute top-full left-0 z-30 mt-1 w-[21rem] max-w-[calc(100vw-2rem)] bg-surface py-1.5 shadow-[0_8px_24px_rgba(28,32,38,0.28)]"
        >
          <PickerMenu
            mode={mode}
            workspace={workspace ?? null}
            files={files}
            list={list}
            onMode={setMode}
            onPick={(id) => {
              menu.close();
              if (id !== null) onPick(id);
            }}
            onRename={async (name) => {
              const error = await onRename(name);
              if (!error) menu.close();
              return error;
            }}
            onDelete={() => {
              menu.close();
              onDelete();
            }}
          />
        </div>
      )}
    </div>
  );
}

/** What the open picker shows: the choices before the chat binds, else the actions, a rename or the confirmation. */
export function PickerMenu({
  mode,
  workspace,
  files,
  list,
  onPick,
  onMode,
  onRename,
  onDelete,
}: {
  mode: PickerMode;
  workspace: Workspace | null;
  files: WorkspaceFile[];
  list: Workspace[] | null;
  /** A workspace picked, or null for a new one. */
  onPick: (id: WorkspaceId | null) => void;
  onMode: (mode: PickerMode) => void;
  onRename: (name: string) => Promise<string | null>;
  onDelete: () => void;
}) {
  if (!workspace || mode === "choices") return <WorkspaceChoices list={list} current={workspace?.id ?? null} onPick={onPick} />;
  if (mode === "rename") {
    return (
      <div className="px-3 py-2">
        <NameField label="New name for the workspace" initial={workspace.name} onSave={onRename} onCancel={() => onMode("actions")} />
      </div>
    );
  }
  if (mode === "confirm") return <DeleteWorkspaceConfirm workspace={workspace} files={files} onConfirm={onDelete} onCancel={() => onMode("actions")} />;
  return (
    <div role="menu" aria-label={`The workspace ${workspace.name}`} className="flex flex-col">
      <button type="button" role="menuitem" className={`${MENU_ROW} font-display text-sm font-semibold text-ink`} onClick={() => onMode("rename")}>
        Rename
      </button>
      <button type="button" role="menuitem" className={`${MENU_ROW} font-display text-sm font-semibold text-fault`} onClick={() => onMode("confirm")}>
        Delete workspace…
      </button>
    </div>
  );
}

/**
 * A new workspace or one of the existing ones, the last changed first: the
 * chat header's choice before the chat binds, and any view's that sends
 * something to a workspace. `current` is the one chosen, null for a new one.
 */
export function WorkspaceChoices({ list, current, onPick }: { list: Workspace[] | null; current: WorkspaceId | null; onPick: (id: WorkspaceId | null) => void }) {
  const row = (id: WorkspaceId | null, name: string, line: string) => (
    <button key={id ?? "new"} type="button" role="menuitemradio" aria-checked={current === id} className={MENU_ROW} onClick={() => onPick(id)}>
      <span className={`cut mt-1 size-2.5 shrink-0 [--cut-size:3px] ${current === id ? "bg-ember" : "border border-line"}`} aria-hidden />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-display text-sm font-semibold text-ink">{name}</span>
        <span className="block text-xs leading-snug text-ash">{line}</span>
      </span>
    </button>
  );
  return (
    <div role="menu" aria-label="Workspace" className="flex max-h-[60vh] flex-col overflow-y-auto">
      {row(null, "New workspace", "Created with the chat's first file, named after the chat.")}
      {list === null ? (
        <p className="px-3 py-2 text-xs text-ash">Reading the workspaces…</p>
      ) : list.length === 0 ? (
        <p className="px-3 py-2 text-xs text-ash">No other workspace yet.</p>
      ) : (
        list.map((w) => row(w.id, w.name, `changed ${when(w.updated)}`))
      )}
    </div>
  );
}

/** The question a workspace's delete asks first. */
export function DeleteWorkspaceConfirm({ workspace, files, onConfirm, onCancel }: { workspace: Workspace; files: WorkspaceFile[]; onConfirm: () => void; onCancel: () => void }) {
  return (
    <div role="alertdialog" aria-label={`Delete the workspace ${workspace.name}`} className="flex flex-col gap-2 px-3 py-2">
      <p className="text-sm leading-snug text-ink">
        Delete “{workspace.name}” and its {count(files.filter(present).length)}, with every version?
      </p>
      <p className="text-xs leading-snug text-ash">
        This cannot be undone. The chats that worked in it keep their messages; their next file starts a new workspace.
      </p>
      <div className="flex gap-2">
        <button
          type="button"
          className="cut bg-fault px-3 py-1 font-display text-xs font-semibold text-ground [--cut-size:6px] hover:opacity-90"
          onClick={onConfirm}
        >
          Delete workspace
        </button>
        <button type="button" className="px-2 py-1 font-display text-xs text-ash hover:text-ink" onClick={onCancel}>
          Keep it
        </button>
      </div>
    </div>
  );
}
