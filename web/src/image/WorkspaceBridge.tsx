import { useState } from "react";
import { pageStore } from "../store/index.ts";
import type { ImageStore } from "../store/imageStore.ts";
import { type Workspace, type WorkspaceFile } from "../workspace/files.ts";
import { type Destination, sentLine } from "../workspace/send.ts";
import { useWorkspace, useWorkspaceList } from "../workspace/useWorkspaces.ts";
import { StoreWorkspaceChooser } from "../workspace/WorkspaceChooser.tsx";
import { pageWorkspaces, type Workspaces } from "../workspace/workspaces.ts";
import { type WorkspacePicture, takeablePictures } from "./fromWorkspace.ts";
import { Picture } from "./pictures.tsx";
import type { ImageSession, PoolImage } from "./sessions.ts";
import { sendPicture, workspaceNameFor } from "./toWorkspace.ts";

// The two ways a picture crosses between the Image view and a workspace (spec
// #170): "Send to workspace" on a pool picture, which asks which workspace,
// and "From a workspace" in the prompt box's pool menu, which adds a
// workspace's pictures as inputs. Neither copies bytes.

/** "Send to workspace" for the viewer's picture: the question, then the send. `tool` is the button's class. */
export function SendToWorkspace({
  session,
  picture,
  tool,
  workspaces,
  store,
}: {
  session: ImageSession;
  picture: PoolImage;
  tool: string;
  workspaces?: Workspaces;
  store?: ImageStore;
}) {
  const [choosing, setChoosing] = useState(false);
  const [line, setLine] = useState<string | null>(null);
  const modules = () => workspaces ?? pageWorkspaces();
  return (
    <div className="relative">
      <button
        type="button"
        className={tool}
        title="Adds the picture to a workspace the model can read. The session keeps it too."
        onClick={() => {
          setLine(null);
          setChoosing(!choosing);
        }}
      >
        Send to workspace
      </button>
      {choosing && (
        <StoreWorkspaceChooser
          above
          source={modules()}
          newName={workspaceNameFor(session)}
          onCancel={() => setChoosing(false)}
          onChoose={(destination: Destination) => {
            setChoosing(false);
            sendPicture(modules(), store ?? pageStore(), picture, destination).then(
              (done) => {
                setLine(sentLine({ ok: true, file: done.written.name, workspace: done.workspace.name }));
              },
              (err: unknown) => setLine(sentLine({ ok: false, error: err instanceof Error ? err.message : String(err) })),
            );
          }}
        />
      )}
      {line && (
        <p role="status" className="absolute right-0 bottom-full z-20 mb-1 w-64 border border-line bg-surface px-3 py-1.5 text-xs text-ink shadow-lg">
          {line}
        </p>
      )}
    </div>
  );
}

/** The pool menu's "From a workspace" section as its parts: the workspaces, then the chosen one's pictures. */
export function WorkspacePictures({
  workspaces,
  chosen,
  pictures,
  onChoose,
  onTake,
}: {
  workspaces: Workspace[] | null;
  chosen: Workspace | null;
  /** The chosen workspace's pictures; null while being read. */
  pictures: WorkspacePicture[] | null;
  onChoose: (workspace: Workspace | null) => void;
  onTake: (pictures: WorkspacePicture[]) => void;
}) {
  if (!chosen) {
    return (
      <div aria-label="From a workspace" className="flex flex-col gap-1">
        <p className="font-display text-[13px] font-medium text-ash">From a workspace</p>
        {workspaces === null ? (
          <p className="text-xs text-ash">Reading the workspaces...</p>
        ) : workspaces.length === 0 ? (
          <p className="text-xs text-ash">No workspace yet.</p>
        ) : (
          <ul className="flex max-h-40 flex-col overflow-y-auto">
            {workspaces.map((w) => (
              <li key={w.id}>
                <button type="button" role="menuitem" className="w-full py-1 text-left font-display text-sm text-ink hover:text-ember" onClick={() => onChoose(w)}>
                  {w.name}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }
  return (
    <div aria-label={`Pictures of ${chosen.name}`} className="flex flex-col gap-1.5">
      <p className="flex items-center gap-2 font-display text-[13px] font-medium text-ash">
        <button type="button" role="menuitem" className="text-ink hover:text-ember" onClick={() => onChoose(null)}>
          Back
        </button>
        <span className="min-w-0 flex-1 truncate">{chosen.name}</span>
        {pictures && pictures.length > 1 && (
          <button type="button" role="menuitem" className="font-semibold text-ink hover:text-ember" onClick={() => onTake(pictures)}>
            Add all
          </button>
        )}
      </p>
      {pictures === null ? (
        <p className="text-xs text-ash">Reading the pictures...</p>
      ) : pictures.length === 0 ? (
        <p className="text-xs text-ash">This workspace holds no picture the pool can take.</p>
      ) : (
        <ul className="grid max-h-60 grid-cols-4 gap-1.5 overflow-y-auto">
          {pictures.map((p) => (
            <li key={p.file}>
              <button
                type="button"
                role="menuitem"
                aria-label={`Add ${p.name}`}
                title={p.name}
                className="checker block aspect-square w-full overflow-hidden border-2 border-transparent hover:border-ember"
                onClick={() => onTake([p])}
              >
                <Picture id={p.id} alt="" className="size-full object-cover" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** The section over the page's workspaces: the list, then the pictures of the one picked. */
export function StoreWorkspacePictures({ source, onTake }: { source: Workspaces; onTake: (pictures: WorkspacePicture[]) => void }) {
  const [chosen, setChosen] = useState<Workspace | null>(null);
  const list = useWorkspaceList(source);
  const view = useWorkspace(source, chosen?.id);
  const files: WorkspaceFile[] | null = chosen && view.workspace === undefined ? null : view.files;
  return <WorkspacePictures workspaces={list} chosen={chosen} pictures={files && takeablePictures(files)} onChoose={setChosen} onTake={onTake} />;
}
