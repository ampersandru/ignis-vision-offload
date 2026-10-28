import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { USER, type Workspace, type WorkspaceFile } from "./files.ts";
import { ChatWorkspaceBar, DeleteWorkspaceConfirm, PickerMenu, WorkspaceChoices, WorkspacePicker } from "./WorkspacePicker.tsx";

// The picker in the chat's header as it renders: before the chat binds, a new
// workspace or one of the existing ones; after, the workspace's name with
// rename, a move to another workspace (#404) and delete, and the
// confirmation a delete asks for.

const at = new Date(2026, 9, 27, 9, 30).getTime();
const trip: Workspace = { format: 1, id: "w1", name: "Trip plan", created: at, updated: at };
const site: Workspace = { format: 1, id: "w2", name: "Small site", created: at, updated: at + 60_000 };

const file = (name: string, deleted = false): WorkspaceFile => ({
  format: 1,
  id: name,
  workspace: "w1",
  name,
  kind: "text",
  versions: [{ n: 1, author: USER, at, text: "x" }, ...(deleted ? [{ n: 2, author: USER, at, deleted: true as const }] : [])],
});

const noop = () => {};
const menu = (props: Partial<Parameters<typeof PickerMenu>[0]>) =>
  renderToStaticMarkup(
    <PickerMenu mode="choices" workspace={null} files={[]} list={[site, trip]} onPick={noop} onMode={noop} onRename={async () => null} onDelete={noop} {...props} />,
  );

describe("the workspace picker", () => {
  it("names a new workspace before the chat binds", () => {
    const html = renderToStaticMarkup(<WorkspacePicker workspace={null} files={[]} list={[trip]} onPick={noop} onRename={async () => null} onDelete={noop} />);
    expect(html).toContain('aria-label="Workspace of this chat: New workspace"');
    expect(html).toContain(">New workspace<");
  });

  it("names the workspace once the chat is bound to it", () => {
    const html = renderToStaticMarkup(<WorkspacePicker workspace={trip} files={[file("a.md")]} list={[trip]} onPick={noop} onRename={async () => null} onDelete={noop} />);
    expect(html).toContain('aria-label="Workspace of this chat: Trip plan"');
    expect(html).toContain(">Trip plan<");
  });

  it("offers a new workspace or an existing one before the chat binds", () => {
    const html = menu({ mode: "choices" });
    expect(html).toMatch(/role="menuitemradio" aria-checked="true"[^>]*>.*New workspace/);
    expect(html).toContain("Created with the chat&#x27;s first file");
    const order = ["New workspace", "Small site", "Trip plan"].map((s) => html.indexOf(s));
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    expect(html.match(/aria-checked="false"/g)).toHaveLength(2);
    expect(html).toContain("changed 2026-10-27 09:31");
  });

  it("offers a chat whose workspace was deleted a new one, its default, or a move to an existing one, saying so", () => {
    const html = menu({ mode: "choices", deleted: true });
    expect(html).toMatch(/role="menuitemradio" aria-checked="true"[^>]*>.*New workspace/);
    expect(html).toContain("Small site");
    expect(html).toContain("Trip plan");
    expect(html).toContain("This chat&#x27;s workspace was deleted: its next file starts a new one, or pick one to move the chat there.");
    expect(html).not.toMatch(/<button[^>]*disabled=""/);
  });

  it("holds a deleted workspace's chat where it is while a reply runs: the existing workspaces wait", () => {
    const html = menu({ mode: "choices", deleted: true, canMove: false });
    expect(html.match(/<button[^>]*disabled=""[^>]*>/g)).toHaveLength(2);
    expect(html).toMatch(/<button[^>]*aria-checked="true"(?![^>]*disabled="")[^>]*>/);
  });

  it("says when there is no workspace to pick yet", () => {
    expect(menu({ mode: "choices", list: [] })).toContain("No other workspace yet.");
    expect(menu({ mode: "choices", list: null })).toContain("Reading the workspaces…");
  });

  it("offers rename, a move and delete once the chat is bound", () => {
    const html = menu({ mode: "actions", workspace: trip, files: [file("a.md")] });
    expect(html).toMatch(/role="menuitem"[^>]*>Rename<\/button>/);
    expect(html).toMatch(/<button type="button" role="menuitem"[^>]*>Move to…<\/button>/);
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Move to…/);
    expect(html).toMatch(/role="menuitem"[^>]*>Delete workspace…<\/button>/);
    expect(html).not.toContain("menuitemradio");
  });

  it("disables the move while a reply runs, saying why", () => {
    const html = menu({ mode: "actions", workspace: trip, files: [file("a.md")], canMove: false });
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*title="A chat moves once its reply ends."[^>]*>Move to…<\/button>/);
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Rename/);
  });

  it("moves the chat to one of the other workspaces, or out to a new one, saying the files stay", () => {
    const html = menu({ mode: "move", workspace: trip, files: [file("a.md")] });
    expect(html).toContain('aria-label="Move this chat to"');
    const order = ["New workspace", "Small site"].map((s) => html.indexOf(s));
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    expect(order[0]).toBeGreaterThan(-1);
    expect(html.match(/role="menuitem"/g)).toHaveLength(2);
    expect(html).not.toMatch(/>Trip plan</);
    expect(html).toContain("The chat leaves “Trip plan”: its next file starts a new workspace, named after the chat.");
    expect(html).toContain("The files stay where they are. The model hears of the move with the next prompt.");
    expect(menu({ mode: "move", workspace: trip, list: [trip] })).toContain("No other workspace yet.");
    expect(menu({ mode: "move", workspace: trip, list: null })).toContain("Reading the workspaces…");
  });

  it("renames in a field holding the name", () => {
    expect(menu({ mode: "rename", workspace: trip })).toMatch(/<input[^>]*aria-label="New name for the workspace"[^>]*value="Trip plan"/);
  });

  it("asks before deleting, naming what goes and what stays", () => {
    const html = renderToStaticMarkup(<DeleteWorkspaceConfirm workspace={trip} files={[file("a.md"), file("b.md"), file("old.md", true)]} onConfirm={noop} onCancel={noop} />);
    expect(html).toContain("role=\"alertdialog\"");
    expect(html).toContain("Delete “Trip plan” and its 2 files, with every version?");
    expect(html).toContain("This cannot be undone.");
    expect(html).toContain("The chats that worked in it keep their messages");
    expect(html).toMatch(/<button[^>]*>Delete workspace<\/button>/);
    expect(html).toMatch(/<button[^>]*>Keep it<\/button>/);
    expect(menu({ mode: "confirm", workspace: trip, files: [file("a.md")] })).toContain("its 1 file, with every version?");
  });

  it("sits in the chat's header beside the button that opens the panel, with the count of files", () => {
    const bar = (open: boolean, notice?: string) =>
      renderToStaticMarkup(
        <ChatWorkspaceBar open={open} onToggle={noop} notice={notice} workspace={trip} files={[file("a.md"), file("b.md"), file("c.md", true)]} list={[trip]} onPick={noop} onRename={async () => null} onDelete={noop} />,
      );
    expect(bar(false)).toMatch(/<button[^>]*aria-pressed="false"[^>]*>Files · 2<\/button>/);
    expect(bar(true)).toContain('aria-pressed="true"');
    expect(bar(false, "Not saved in this browser: blocked")).toMatch(/role="status"[^>]*>Not saved in this browser: blocked</);
  });

  it("lists the choices for any view that sends something to a workspace", () => {
    const html = renderToStaticMarkup(<WorkspaceChoices list={[trip]} current="w1" onPick={noop} />);
    expect(html).toMatch(/aria-checked="true"[^>]*>.*Trip plan/);
    expect(html).toMatch(/aria-checked="false"[^>]*>.*New workspace/);
  });
});
