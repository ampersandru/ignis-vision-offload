import { describe, expect, it } from "vitest";
import type { Attachment } from "../tools/local/attachments.ts";
import { bindWorkspace, clearAttachments, createSession, forkSession, removeAttachment, type Session } from "./sessions.ts";

// A chat's files: an older chat's attachments, kept with it until its next
// turn makes them files of its workspace, and the workspace it binds to.

const file = (name: string): Attachment => ({ name, size: 1, kind: "text", text: "x" });

/** An older chat, saved with attached files before the workspace. */
const older = (id: number, names: string[]): Session => ({ ...createSession(id), attachments: names.map(file) });

describe("a chat's files", () => {
  it("drops an older chat's attached file, on that chat only", () => {
    let sessions = [older(1, ["a.txt", "b.txt"]), older(2, ["a.txt"])];
    sessions = removeAttachment(sessions, 1, "a.txt");
    expect(sessions[0].attachments.map((a) => a.name)).toEqual(["b.txt"]);
    expect(sessions[1].attachments.map((a) => a.name)).toEqual(["a.txt"]);
    expect(clearAttachments(sessions, 1)[0].attachments).toEqual([]);
  });

  it("binds a chat to its workspace once, and never moves it", () => {
    let sessions = [createSession(1), createSession(2)];
    sessions = bindWorkspace(sessions, 1, "w1");
    expect(sessions.map((s) => s.workspaceId)).toEqual(["w1", undefined]);
    const same = bindWorkspace(sessions, 1, "w2");
    expect(same[0].workspaceId).toBe("w1");
    expect(same[0]).toBe(sessions[0]);
  });

  it("keeps the files and the workspace in a fork", () => {
    const source = {
      ...bindWorkspace([older(1, ["a.txt"])], 1, "w1")[0],
      messages: [{ id: 5, role: "user" as const, content: "q", reasoning: "", streaming: false }],
    };
    const next = forkSession({ sessions: [source], activeId: 1 }, 1, 5, 9);
    expect(next.sessions[0].attachments.map((a) => a.name)).toEqual(["a.txt"]);
    expect(next.sessions[0].workspaceId).toBe("w1");
  });
});
