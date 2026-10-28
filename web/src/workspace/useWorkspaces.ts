import { useEffect, useState } from "react";
import type { Workspace, WorkspaceFile, WorkspaceId } from "./files.ts";
import type { Workspaces } from "./workspaces.ts";

// The workspaces as the components show them, read from the workspace module
// and read again after each change it announces. A read that lands after a
// newer one began is dropped.

/**
 * Every workspace, the last changed first, or null while being read: what a
 * picker offers to choose from (the chat's header, and any view that sends
 * something to a workspace).
 */
export function useWorkspaceList(workspaces: Workspaces): Workspace[] | null {
  const [list, setList] = useState<Workspace[] | null>(null);
  useEffect(() => {
    let reads = 0;
    const read = () => {
      const n = ++reads;
      void workspaces.list().then(
        (l) => n === reads && setList(l),
        () => n === reads && setList([]),
      );
    };
    read();
    const off = workspaces.subscribe(read);
    return () => {
      reads = -1;
      off();
    };
  }, [workspaces]);
  return list;
}

/** One workspace and its files (deleted ones included): undefined while being read, null when there is none. */
export type WorkspaceView = { workspace: Workspace | null | undefined; files: WorkspaceFile[] };

const NONE: WorkspaceView = { workspace: null, files: [] };

/** The workspace `id` names, with its files; none for no id. */
export function useWorkspace(workspaces: Workspaces, id: WorkspaceId | undefined): WorkspaceView {
  const [view, setView] = useState<{ id: WorkspaceId; view: WorkspaceView } | null>(null);
  useEffect(() => {
    if (!id) return;
    let reads = 0;
    const read = () => {
      const n = ++reads;
      void Promise.all([workspaces.get(id), workspaces.files(id)]).then(
        ([workspace, files]) => n === reads && setView({ id, view: { workspace, files } }),
        () => n === reads && setView({ id, view: NONE }),
      );
    };
    read();
    const off = workspaces.subscribe((change) => change.workspace === id && read());
    return () => {
      reads = -1;
      off();
    };
  }, [workspaces, id]);
  if (!id) return NONE;
  return view?.id === id ? view.view : { workspace: undefined, files: [] };
}
