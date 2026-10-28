import { createContext, type ReactNode, useContext, useEffect, useMemo, useReducer, useRef } from "react";
import type { WorkspaceFile } from "./files.ts";
import type { Workspaces } from "./workspaces.ts";

// How the page shows a workspace file outside the workspace: a reply's card
// reads the file it wrote, and follows it as it changes, so that it knows
// when a newer version supersedes its own. A file is read from the store the
// first time it is asked for and again each time the module announces a
// change to it. Tests hand in plain records instead.

/** The file, null when the workspace no longer has it, or undefined while it is being read. */
export type FileLookup = (workspace: string, file: string) => WorkspaceFile | null | undefined;

const Files = createContext<FileLookup>(() => undefined);

export const useWorkspaceFile = (workspace: string, file: string) => useContext(Files)(workspace, file);

/** Lookups from a plain list: what the components' tests give them. */
export function StaticWorkspaceFiles({ files, children }: { files: WorkspaceFile[]; children: ReactNode }) {
  const lookup: FileLookup = (workspace, id) => files.find((f) => f.workspace === workspace && f.id === id) ?? null;
  return <Files.Provider value={lookup}>{children}</Files.Provider>;
}

/** Lookups read from the workspace module, each file once and again after each change to it. */
export function StoreWorkspaceFiles({ workspaces, children }: { workspaces: Workspaces; children: ReactNode }) {
  const cache = useRef(new Map<string, WorkspaceFile | null | "reading">());
  const [version, landed] = useReducer((n: number) => n + 1, 0);
  const read = useRef((workspace: string, id: string) => {
    const key = `${workspace}/${id}`;
    cache.current.set(key, cache.current.get(key) ?? "reading");
    void workspaces.file(workspace, id).then(
      (file) => {
        cache.current.set(key, file);
        landed();
      },
      () => {
        cache.current.set(key, null);
        landed();
      },
    );
  });
  useEffect(
    () =>
      workspaces.subscribe(({ workspace, file }) => {
        if (file && cache.current.has(`${workspace}/${file}`)) read.current(workspace, file);
      }),
    [workspaces],
  );
  const lookup = useMemo<FileLookup>(() => {
    // `version` makes a new lookup, and so a new render of the readers, each time a file lands.
    void version;
    return (workspace, id) => {
      const hit = cache.current.get(`${workspace}/${id}`);
      if (hit === undefined) read.current(workspace, id);
      return hit === undefined || hit === "reading" ? undefined : hit;
    };
  }, [version]);
  return <Files.Provider value={lookup}>{children}</Files.Provider>;
}
