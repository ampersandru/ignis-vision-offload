import { type ReactNode, useState } from "react";
import { useAnchoredMenu } from "../ui/anchoredMenu.ts";
import { IconChevron } from "../ui/icons.tsx";
import { present, timeLabel, type Workspace, type WorkspaceFile, type WorkspaceId } from "./files.ts";
import { NameField } from "./WorkspacePanel.tsx";

// The picker in the chat's header (spec #170): the workspace the chat works
// in. Before the chat binds it offers a new workspace, the default, born
// with the chat's first file, or one of the existing ones, and picking one
// binds the chat to it. Once bound, it names the workspace and renames it,
// moves the chat to another workspace or out of its own (#404), or deletes
// it, a delete only after a confirmation. A chat whose workspace was deleted
// is offered the new one, or a move to an existing one. A chat moves only
// while no reply of its runs.

export type PickerMode = "choices" | "actions" | "rename" | "confirm" | "move";

const MENU_ROW = "flex w-full items-start gap-2.5 px-3 py-2 text-left hover:bg-line";

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
  deleted = false,
  canMove = true,
  onPick,
  onRename,
  onDelete,
}: {
  /** The chat's workspace: null before it binds, or once it is deleted; undefined while it is read. */
  workspace: Workspace | null | undefined;
  files: WorkspaceFile[];
  /** The workspaces there are, as useWorkspaceList reads them. */
  list: Workspace[] | null;
  /** The chat still names a workspace that was deleted. */
  deleted?: boolean;
  /** The chat may move: false while a reply of its runs. */
  canMove?: boolean;
  /** Binds the chat to an existing workspace, or moves it there, or out of its own to none (null). */
  onPick: (id: WorkspaceId | null) => void;
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
        title={workspace === null ? "Choose the workspace of this chat" : "Rename the workspace, move the chat to another, or delete it"}
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
            deleted={deleted}
            canMove={canMove}
            onMode={setMode}
            onPick={(id) => {
              menu.close();
              // The workspace the chat is in already, or none for a chat with none, changes nothing.
              if (id !== (workspace?.id ?? null)) onPick(id);
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

/** What the open picker shows: the choices before the chat binds, else the actions, a rename, a move or the confirmation. */
export function PickerMenu({
  mode,
  workspace,
  files,
  list,
  deleted = false,
  canMove = true,
  onPick,
  onMode,
  onRename,
  onDelete,
}: {
  mode: PickerMode;
  workspace: Workspace | null;
  files: WorkspaceFile[];
  list: Workspace[] | null;
  deleted?: boolean;
  canMove?: boolean;
  /** A workspace picked, or null for a new one. */
  onPick: (id: WorkspaceId | null) => void;
  onMode: (mode: PickerMode) => void;
  onRename: (name: string) => Promise<string | null>;
  onDelete: () => void;
}) {
  if (!workspace || mode === "choices") {
    // A chat whose workspace was deleted moves to an existing one, which waits while a reply of its runs.
    const note = deleted ? "This chat's workspace was deleted: its next file starts a new one, or pick one to move the chat there." : undefined;
    return <WorkspaceChoices list={list} current={workspace?.id ?? null} onPick={onPick} note={note} held={deleted && !canMove} />;
  }
  if (mode === "move") return <MoveChoices workspace={workspace} list={list} onPick={onPick} />;
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
      <button
        type="button"
        role="menuitem"
        disabled={!canMove}
        title={canMove ? undefined : "A chat moves once its reply ends."}
        className={`${MENU_ROW} font-display text-sm font-semibold text-ink disabled:cursor-default disabled:text-ash disabled:hover:bg-transparent`}
        onClick={() => onMode("move")}
      >
        Move to…
      </button>
      <button type="button" role="menuitem" className={`${MENU_ROW} font-display text-sm font-semibold text-fault`} onClick={() => onMode("confirm")}>
        Delete workspace…
      </button>
    </div>
  );
}

/**
 * One choice of workspace: a radio of a choice made (`checked`), or a plain
 * item of a move; `held` while it cannot be taken now.
 */
function Choice({ name, line, checked, held = false, onClick }: { name: string; line: string; checked?: boolean; held?: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      role={checked === undefined ? "menuitem" : "menuitemradio"}
      aria-checked={checked}
      disabled={held}
      title={held ? "A chat moves once its reply ends." : undefined}
      className={`${MENU_ROW} disabled:cursor-default disabled:opacity-50 disabled:hover:bg-transparent`}
      onClick={onClick}
    >
      <span className={`cut mt-1 size-2.5 shrink-0 [--cut-size:3px] ${checked ? "bg-ember" : "border border-line"}`} aria-hidden />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-display text-sm font-semibold text-ink">{name}</span>
        <span className="block text-xs leading-snug text-ash">{line}</span>
      </span>
    </button>
  );
}

/** The existing workspaces, the last changed first, each a choice; what to say while there is none to offer. */
function existing(list: Workspace[] | null, choice: (w: Workspace) => ReactNode) {
  if (list === null) return <p className="px-3 py-2 text-xs text-ash">Reading the workspaces…</p>;
  if (list.length === 0) return <p className="px-3 py-2 text-xs text-ash">No other workspace yet.</p>;
  return list.map(choice);
}

/**
 * A new workspace or one of the existing ones, the last changed first: the
 * chat header's choice before the chat binds, and once its workspace was
 * deleted. `current` is the one chosen, null for a new one; `note`, what the
 * choice means here; `held`, the existing ones wait while a reply runs.
 */
export function WorkspaceChoices({
  list,
  current,
  onPick,
  note,
  held = false,
}: {
  list: Workspace[] | null;
  current: WorkspaceId | null;
  onPick: (id: WorkspaceId | null) => void;
  note?: string;
  held?: boolean;
}) {
  return (
    <div role="menu" aria-label="Workspace" className="flex max-h-[60vh] flex-col overflow-y-auto">
      <Choice name="New workspace" line="Created with the chat's first file, named after the chat." checked={current === null} onClick={() => onPick(null)} />
      {note && <p className="px-3 py-2 text-xs leading-snug text-ash">{note}</p>}
      {existing(list, (w) => (
        <Choice key={w.id} name={w.name} line={`changed ${timeLabel(w.updated)}`} checked={current === w.id} held={held} onClick={() => onPick(w.id)} />
      ))}
    </div>
  );
}

/**
 * Where a bound chat moves (#404): out to a new workspace, born with its next
 * file, or to one of the other existing ones. Only the chat moves: the files
 * stay in the workspace they are in.
 */
export function MoveChoices({ workspace, list, onPick }: { workspace: Workspace; list: Workspace[] | null; onPick: (id: WorkspaceId | null) => void }) {
  const others = list && list.filter((w) => w.id !== workspace.id);
  return (
    <div role="menu" aria-label="Move this chat to" className="flex max-h-[60vh] flex-col overflow-y-auto">
      <Choice name="New workspace" line={`The chat leaves “${workspace.name}”: its next file starts a new workspace, named after the chat.`} onClick={() => onPick(null)} />
      {existing(others, (w) => (
        <Choice key={w.id} name={w.name} line={`changed ${timeLabel(w.updated)}`} onClick={() => onPick(w.id)} />
      ))}
      <p className="px-3 py-2 text-xs leading-snug text-ash">The files stay where they are. The model hears of the move with the next prompt.</p>
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
