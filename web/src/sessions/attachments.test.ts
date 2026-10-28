import { describe, expect, it } from "vitest";
import type { Attachment } from "../tools/local/attachments.ts";
import { bindWorkspace, bornInto, clearAttachments, createSession, forkSession, type Message, moveWorkspace, openSession, PICK_WAITS, pickHeld, removeAttachment, type Session, untoldMove } from "./sessions.ts";

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

  it("binds a chat with no workspace to the one picked for it, and leaves a bound chat to a move", () => {
    let sessions = [createSession(1), createSession(2)];
    sessions = bindWorkspace(sessions, 1, "w1");
    expect(sessions.map((s) => s.workspaceId)).toEqual(["w1", undefined]);
    const same = bindWorkspace(sessions, 1, "w2");
    expect(same[0].workspaceId).toBe("w1");
    expect(same[0]).toBe(sessions[0]);
  });
});

/** A prompt and its reply, as a chat's history holds them. */
const exchange = (id: number): Message[] => [
  { id, role: "user", content: "q", reasoning: "", streaming: false },
  { id: id + 1, role: "assistant", content: "a", reasoning: "", streaming: false },
];

/** A chat bound to `workspaceId` that has talked: its model knows that workspace. */
const talked = (id: number, workspaceId: string): Session => ({ ...createSession(id), workspaceId, messages: exchange(10 * id) });

describe("a chat's move to another workspace (#404)", () => {
  it("changes the chat's binding alone, and keeps the move, naming the workspace left, for its next notice", () => {
    const sessions = [talked(1, "w1"), talked(2, "w1")];
    const moved = moveWorkspace(sessions, 1, "w2", { at: 100, name: "Trip plan" });
    expect(moved[0]).toEqual({ ...sessions[0], workspaceId: "w2", moved: { from: "w1", name: "Trip plan", at: 100 } });
    // The workspace's other chat stays in it.
    expect(moved[1]).toBe(sessions[1]);
  });

  it("leaves the chat with no workspace on a move to a new one: its next file starts it, in no one's place", () => {
    const [left] = moveWorkspace([talked(1, "w1")], 1, undefined, { at: 100, name: "Trip plan" });
    expect(left.workspaceId).toBeUndefined();
    expect("workspaceId" in left).toBe(false);
    expect(left.moved).toEqual({ from: "w1", name: "Trip plan", at: 100 });
    expect(bornInto([left], 1, "w3")[0].workspaceId).toBe("w3");
  });

  it("moves a chat whose workspace was deleted to an existing one, the name of the one it left unknown", () => {
    // Deleted here or in another tab, the workspace is still the one the chat names.
    const [moved] = moveWorkspace([talked(1, "gone")], 1, "w2", { at: 100 });
    expect(moved.workspaceId).toBe("w2");
    expect(moved.moved).toEqual({ from: "gone", at: 100 });
  });

  it("keeps the workspace first left while the move is untold, and keeps nothing when the chat moves back to it", () => {
    const [once] = moveWorkspace([talked(1, "w1")], 1, "w2", { at: 100, name: "Trip plan" });
    const [twice] = moveWorkspace([once], 1, "w3", { at: 200, name: "Small site" });
    expect(twice.workspaceId).toBe("w3");
    expect(twice.moved).toEqual({ from: "w1", name: "Trip plan", at: 100 });
    const [back] = moveWorkspace([twice], 1, "w1", { at: 300, name: "Fox" });
    expect(back.workspaceId).toBe("w1");
    expect("moved" in back).toBe(false);
  });

  it("starts a new move from where a told one left the chat", () => {
    const [once] = moveWorkspace([talked(1, "w1")], 1, "w2", { at: 100, name: "Trip plan" });
    expect(untoldMove(once.messages, once.moved)).toEqual(once.moved);
    // The next prompt's notice told the model of it.
    const told: Session = { ...once, messages: [...once.messages, { ...exchange(20)[0], moved: once.moved }, exchange(20)[1]] };
    expect(untoldMove(told.messages, told.moved)).toBeUndefined();
    const [again] = moveWorkspace([told], 1, "w3", { at: 200, name: "Small site" });
    expect(again.moved).toEqual({ from: "w2", name: "Small site", at: 200 });
  });

  it("counts a move told by a prompt the model never read as untold: the prompt dropped with its failed reply", () => {
    const [once] = moveWorkspace([talked(1, "w1")], 1, "w2", { at: 100, name: "Trip plan" });
    const failed: Message[] = [{ ...exchange(20)[0], moved: once.moved }, { ...exchange(20)[1], error: "the engine is full" }];
    expect(untoldMove([...once.messages, ...failed], once.moved)).toEqual(once.moved);
  });

  it("keeps no move for a chat with no message yet, nor for one with no workspace, which simply binds", () => {
    const empty = { ...createSession(1), workspaceId: "w1" };
    expect(moveWorkspace([empty], 1, "w2", { at: 100, name: "Trip plan" })[0]).toEqual({ ...empty, workspaceId: "w2" });
    const [bound] = moveWorkspace([{ ...createSession(2), messages: exchange(20) }], 2, "w2", { at: 100 });
    expect(bound.workspaceId).toBe("w2");
    expect("moved" in bound).toBe(false);
  });

  it("changes nothing for a move to where the chat already is", () => {
    const sessions = [talked(1, "w1"), createSession(2)];
    expect(moveWorkspace(sessions, 1, "w1", { at: 100, name: "Trip plan" })[0]).toBe(sessions[0]);
    expect(moveWorkspace(sessions, 2, undefined, { at: 100 })[1]).toBe(sessions[1]);
  });

  it("is copied by a fork as it stands at the fork, binding and untold move", () => {
    const [moved] = moveWorkspace([talked(1, "w1")], 1, "w2", { at: 100, name: "Trip plan" });
    const fork = forkSession({ sessions: [moved], activeId: 1 }, 1, 10, 9).sessions[0];
    expect(fork.workspaceId).toBe("w2");
    expect(fork.moved).toEqual({ from: "w1", name: "Trip plan", at: 100 });
  });
});

describe("when a chat may pick or move its workspace (#404)", () => {
  it("waits while a reply of the chat runs, saying why, whether the chat has a workspace or not", () => {
    // The rule reads the chats replying alone: a chat bound or not, moved to a new workspace or deleted out of one, waits alike.
    expect(pickHeld(new Set([1]), 1)).toBe(PICK_WAITS);
    expect(PICK_WAITS).toBe("The workspace changes once this chat's reply ends.");
  });

  it("lets the chat pick or move while no reply of its runs, another chat's included", () => {
    expect(pickHeld(new Set(), 1)).toBeNull();
    expect(pickHeld(new Set([2, 3]), 1)).toBeNull();
  });
});

describe("a chat and the workspace born for it", () => {

  it("binds the workspace born with a chat's next file in place of the deleted one it named, and keeps a pick made meanwhile", () => {
    const sessions = [...bindWorkspace([createSession(1)], 1, "gone"), createSession(2), ...bindWorkspace([createSession(3)], 3, "picked")];
    const born = bornInto(bornInto(bornInto(sessions, 1, "new1", "gone"), 2, "new2"), 3, "new3");
    expect(born.map((s) => s.workspaceId)).toEqual(["new1", "new2", "picked"]);
    expect(born[2]).toBe(sessions[2]);
  });

  it("opens a new session rather than reuse an empty one bound to a workspace", () => {
    const bound = { sessions: bindWorkspace([createSession(1)], 1, "w1"), activeId: 1 };
    const next = openSession(bound, 2);
    expect(next.activeId).toBe(2);
    expect(next.sessions.map((s) => [s.id, s.workspaceId])).toEqual([
      [2, undefined],
      [1, "w1"],
    ]);
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
