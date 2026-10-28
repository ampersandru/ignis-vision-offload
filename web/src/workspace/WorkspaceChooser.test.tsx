import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { WORKSPACE_FORMAT, type Workspace } from "./files.ts";
import { WorkspaceChooser } from "./WorkspaceChooser.tsx";

const none = () => {};
const ws = (id: string, name: string): Workspace => ({ format: WORKSPACE_FORMAT, id, name, created: 1, updated: 1 });

describe("the workspace chooser", () => {
  it("offers a new workspace named for what is sent, then the existing ones", () => {
    const html = renderToStaticMarkup(<WorkspaceChooser workspaces={[ws("a", "Trip plan"), ws("b", "Budget")]} newName="25 Oct 15:42 · Call" onChoose={none} onCancel={none} />);
    expect(html).toContain("New workspace: 25 Oct 15:42 · Call");
    expect(html.indexOf("Trip plan")).toBeGreaterThan(html.indexOf("New workspace"));
    expect(html).toContain("Budget");
    expect(html).toContain(">Cancel<");
  });

  it("says so while it reads, and when there is no workspace yet", () => {
    expect(renderToStaticMarkup(<WorkspaceChooser workspaces={null} newName="x" onChoose={none} onCancel={none} />)).toContain("Reading the workspaces");
    expect(renderToStaticMarkup(<WorkspaceChooser workspaces={[]} newName="x" onChoose={none} onCancel={none} />)).toContain("No workspace yet");
  });
});
