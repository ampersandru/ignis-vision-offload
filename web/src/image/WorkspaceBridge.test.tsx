import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { createMemoryStore } from "../store/memory.ts";
import { WORKSPACE_FORMAT, type Workspace } from "../workspace/files.ts";
import { createWorkspaces } from "../workspace/workspaces.ts";
import { StaticPictures } from "./pictures.tsx";
import { addToPool, newSession } from "./sessions.ts";
import { SendToWorkspace, sentLine, WorkspacePictures } from "./WorkspaceBridge.tsx";

// The two actions in the components (static markup, as the repo tests views):
// "Send to workspace" on a viewer picture, and "From a workspace" among the
// pool menu's choices. What each does is tested in toWorkspace.test.ts.

const none = () => {};
const ws = (id: string, name: string): Workspace => ({ format: WORKSPACE_FORMAT, id, name, created: 1, updated: 1 });
const [session] = addToPool(newSession("s", 0), [{ id: "up", width: 8, height: 6, origin: "upload" }], 1);

describe("Send to workspace", () => {
  it("is a button on the picture, the question closed until it is pressed", () => {
    const workspaces = createWorkspaces(createMemoryStore());
    const html = renderToStaticMarkup(<SendToWorkspace session={session} picture={session.pool[0]} tool="tool" workspaces={workspaces} store={createMemoryStore()} />);
    expect(html).toContain("Send to workspace");
    expect(html).not.toContain("New workspace");
  });

  it("says what was added where, or why not", () => {
    expect(sentLine({ ok: true, file: "ignis-n1-8x6.png", workspace: "Fox" })).toBe("Added ignis-n1-8x6.png to Fox");
    expect(sentLine({ ok: false, error: "That workspace is gone." })).toBe("Not sent: That workspace is gone.");
  });
});

describe("From a workspace", () => {
  it("lists the workspaces, then the chosen one's pictures with the way to add one or all", () => {
    const list = renderToStaticMarkup(<WorkspacePictures workspaces={[ws("a", "Trip"), ws("b", "Budget")]} chosen={null} pictures={null} onChoose={none} onTake={none} />);
    expect(list).toContain("Trip");
    expect(list).toContain("Budget");
    const pictures = [1, 2].map((i) => ({ file: `f${i}`, name: `${i}.png`, id: `i${i}`, width: 8, height: 6 }));
    const html = renderToStaticMarkup(
      <StaticPictures urls={{ i1: "blob:1", i2: "blob:2" }}>
        <WorkspacePictures workspaces={[ws("a", "Trip")]} chosen={ws("a", "Trip")} pictures={pictures} onChoose={none} onTake={none} />
      </StaticPictures>,
    );
    expect(html).toContain('aria-label="Add 1.png"');
    expect(html).toContain('aria-label="Add 2.png"');
    expect(html).toContain("Add all");
    expect(html).toContain("blob:1");
  });

  it("says when a workspace has no picture to take, and while it reads", () => {
    expect(renderToStaticMarkup(<WorkspacePictures workspaces={[]} chosen={ws("a", "Trip")} pictures={[]} onChoose={none} onTake={none} />)).toContain("holds no picture");
    expect(renderToStaticMarkup(<WorkspacePictures workspaces={null} chosen={null} pictures={null} onChoose={none} onTake={none} />)).toContain("Reading the workspaces");
  });
});
