import type { Workspace } from "./files.ts";
import type { Destination } from "./send.ts";
import { useWorkspaceList } from "./useWorkspaces.ts";
import type { Workspaces } from "./workspaces.ts";

// The small question "which workspace?": a new one named after what is sent,
// or one of the existing ones, the last changed first. Reusable wherever the
// page sends something to a workspace (a transcript, a picture).

const row = "w-full px-3 py-1.5 text-left font-display text-[13px] text-ink hover:bg-line";

/** The question, with the workspaces already read (null while they are being read). */
export function WorkspaceChooser({
  workspaces,
  newName,
  onChoose,
  onCancel,
  above = false,
}: {
  workspaces: Workspace[] | null;
  /** The name a new workspace would have. */
  newName: string;
  onChoose: (destination: Destination) => void;
  onCancel: () => void;
  /** Opens upward, for a button at the foot of the page. */
  above?: boolean;
}) {
  return (
    <div role="dialog" aria-label="Send to a workspace" className={`absolute right-0 z-20 flex w-64 flex-col border border-line bg-surface py-1 shadow-lg ${above ? "bottom-full mb-1" : "mt-1"}`}>
      <button type="button" className={`${row} font-semibold`} onClick={() => onChoose({ kind: "new", name: newName })}>
        New workspace: {newName}
      </button>
      {workspaces === null ? (
        <p className="px-3 py-1.5 text-xs text-ash">Reading the workspaces...</p>
      ) : workspaces.length === 0 ? (
        <p className="px-3 py-1.5 text-xs text-ash">No workspace yet.</p>
      ) : (
        <ul aria-label="Existing workspaces" className="flex max-h-48 flex-col overflow-y-auto border-t border-line">
          {workspaces.map((w) => (
            <li key={w.id}>
              <button type="button" className={row} onClick={() => onChoose({ kind: "existing", id: w.id })}>
                {w.name}
              </button>
            </li>
          ))}
        </ul>
      )}
      <button type="button" className={`${row} border-t border-line text-ash`} onClick={onCancel}>
        Cancel
      </button>
    </div>
  );
}

/** The question over the page's workspaces, read when it opens, as the chat's picker reads them. */
export function StoreWorkspaceChooser({ source, ...rest }: { source: Workspaces } & Omit<Parameters<typeof WorkspaceChooser>[0], "workspaces">) {
  return <WorkspaceChooser workspaces={useWorkspaceList(source)} {...rest} />;
}
