import { describe, expect, it } from "vitest";
import type { SpeechStatus } from "../api/model.ts";
import type { Settings } from "../api/request.ts";
import { createMemoryStore } from "../store/memory.ts";
import { type LocalRun, runLocalCalls } from "../tools/local/local.ts";
import { latest, USER } from "./files.ts";
import type { WorkspaceAccess } from "./tools.ts";
import { createWorkspaces, type Deriving, type Workspaces } from "./workspaces.ts";

// The text the model reads of a file's bytes (#395): an audio file's
// transcript, from the server's transcription route, and a PDF's text, each
// computed on the first read of a version and kept with it. The route and
// pdf.js are injected, so no network and no browser are needed.

const settings: Settings = {
  systemPrompt: "",
  temperature: 1,
  topP: 0.95,
  maxTokens: 100,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "interactive",
};

const AVAILABLE: SpeechStatus = { available: true, id: "qwen3-asr-1.7b" };

/** The injected route and pdf.js, counting what they were asked; `null` for a server that says nothing of speech. */
function deriving(speech: SpeechStatus | null = AVAILABLE, transcript = "Buy bread.\nCall Anna at five.") {
  const asked = { transcribe: [] as string[], pdf: 0, speech: 0 };
  const derive: Deriving = {
    speech: async () => {
      asked.speech++;
      return speech ?? undefined;
    },
    transcribe: async (audio, name) => {
      asked.transcribe.push(`${name}:${audio.size}`);
      return transcript;
    },
    pdfText: async () => {
      asked.pdf++;
      return "[Page 1]\nThe brief";
    },
  };
  return { derive, asked };
}

function setup(derive: Deriving) {
  let n = 0;
  const workspaces = createWorkspaces(createMemoryStore(), { newId: () => `id${++n}`, derive });
  let bound: string | undefined;
  const chat = workspaces.chat("chat-a", { bound: () => bound, name: () => "Errands", onBorn: (id) => (bound = id) });
  const access: WorkspaceAccess = { workspaces, chat };
  return { workspaces, access, chat };
}

let callN = 0;

async function read(access: WorkspaceAccess, name: string, signal = new AbortController().signal): Promise<LocalRun> {
  const call = { id: `c${++callN}`, name: "read_file", arguments: JSON.stringify({ name }) };
  const [ran] = await runLocalCalls([call], { settings, signal, jsSafetyCheck: false, workspace: access }, () => {});
  return ran;
}

const memo = () => new Blob([new Uint8Array(44)], { type: "audio/wav" });

async function withMemo(workspaces: Workspaces, access: WorkspaceAccess) {
  const id = await access.chat.workspace();
  await workspaces.add(id, [{ name: "memo.wav", kind: "audio", blob: memo() }], USER);
  return id;
}

describe("read_file of an audio file", () => {
  it("returns its transcript from the transcription route, kept with the version so a second read asks nobody", async () => {
    const { derive, asked } = deriving();
    const { workspaces, access } = setup(derive);
    const id = await withMemo(workspaces, access);
    const first = await read(access, "memo.wav");
    expect(first.status).toBe("done");
    expect(first.result).toBe("1\tBuy bread.\n2\tCall Anna at five.\n\n[memo.wav, version 1: lines 1–2 of 2.]");
    expect(asked.transcribe).toEqual(["memo.wav:44"]);
    const [file] = await workspaces.files(id);
    expect(latest(file).derived).toBe("Buy bread.\nCall Anna at five.");
    const second = await read(access, "memo.wav");
    expect(second.result).toBe(first.result);
    expect(asked.transcribe).toHaveLength(1);
    expect(asked.speech).toBe(1);
  });

  it("says so when the models listing shows no speech model available, and asks the route nothing", async () => {
    const off: SpeechStatus = { available: false, reason: "the speech model is turned off (speech.model is off)" };
    const { derive, asked } = deriving(off);
    const { workspaces, access } = setup(derive);
    const id = await withMemo(workspaces, access);
    const ran = await read(access, "memo.wav");
    expect(ran.status).toBe("failed");
    expect(ran.error).toBe("No speech model can transcribe memo.wav now. Speech is turned off on this server (`speech.model = off`).");
    expect(asked.transcribe).toEqual([]);
    expect(latest((await workspaces.files(id))[0]).derived).toBeUndefined();
  });

  it("says so too when the server says nothing of speech", async () => {
    const { derive, asked } = deriving(null);
    const { workspaces, access } = setup(derive);
    await withMemo(workspaces, access);
    expect((await read(access, "memo.wav")).error).toBe("No speech model can transcribe memo.wav now. The server reports no speech model.");
    expect(asked.transcribe).toEqual([]);
  });

  it("passes the route's refusal on, keeps nothing, and asks again at the next read", async () => {
    const { derive } = deriving();
    let calls = 0;
    const refusing: Deriving = {
      ...derive,
      transcribe: async () => {
        calls++;
        throw new Error("400: `file` is longer than speech.max_clip_seconds (1200 s); nothing is split automatically, send a shorter clip");
      },
    };
    const { workspaces, access } = setup(refusing);
    await withMemo(workspaces, access);
    expect((await read(access, "memo.wav")).error).toBe(
      "The transcription of memo.wav failed: 400: `file` is longer than speech.max_clip_seconds (1200 s); nothing is split automatically, send a shorter clip",
    );
    await read(access, "memo.wav");
    expect(calls).toBe(2);
  });

  it("transcribes once for reads of one version at once, and a new version is transcribed of its own", async () => {
    let release = () => {};
    const gate = new Promise<void>((resolve) => (release = resolve));
    const { derive, asked } = deriving();
    const slow: Deriving = { ...derive, transcribe: async (audio, name) => (await gate, derive.transcribe(audio, name)) };
    const { workspaces, access } = setup(slow);
    const id = await withMemo(workspaces, access);
    const both = Promise.all([read(access, "memo.wav"), read(access, "memo.wav")]);
    release();
    const [a, b] = await both;
    expect(a.result).toBe(b.result);
    expect(asked.transcribe).toHaveLength(1);
    // Another version of the file: kept apart, computed for itself.
    const [file] = await workspaces.files(id);
    const second = { ...latest(file), n: 2 };
    delete second.derived;
    expect(await workspaces.readable(file, second)).toEqual({ ok: true, text: "Buy bread.\nCall Anna at five." });
    expect(asked.transcribe).toHaveLength(2);
  });

  it("stops the transcription when the turn is stopped: nothing is kept, and the next read starts again", async () => {
    const seen: AbortSignal[] = [];
    const { derive } = deriving();
    const hanging: Deriving = {
      ...derive,
      transcribe: (_audio, _name, signal) =>
        new Promise((_resolve, reject) => {
          seen.push(signal!);
          signal!.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
    };
    const { workspaces, access } = setup(hanging);
    const id = await withMemo(workspaces, access);
    const stop = new AbortController();
    const ran = read(access, "memo.wav", stop.signal);
    await new Promise((resolve) => setTimeout(resolve, 0));
    stop.abort();
    expect((await ran).error).toBe("Stopped before the transcript of memo.wav was ready.");
    expect(seen[0].aborted).toBe(true);
    expect(latest((await workspaces.files(id))[0]).derived).toBeUndefined();
  });
});

describe("read_file of a PDF", () => {
  it("computes the text of a version with none on its first read, and keeps it", async () => {
    const { derive, asked } = deriving();
    const { workspaces, access } = setup(derive);
    const id = await access.chat.workspace();
    // A PDF whose text pdf.js gave nothing of when it came in.
    await workspaces.add(id, [{ name: "brief.pdf", kind: "pdf", blob: new Blob([new Uint8Array([37, 80, 68, 70])], { type: "application/pdf" }) }], USER);
    const first = await read(access, "brief.pdf");
    expect(first.result).toBe("1\t[Page 1]\n2\tThe brief\n\n[brief.pdf, version 1: lines 1–2 of 2.]");
    expect(latest((await workspaces.files(id))[0]).derived).toBe("[Page 1]\nThe brief");
    expect((await read(access, "brief.pdf")).result).toBe(first.result);
    expect(asked.pdf).toBe(1);
    expect(asked.speech).toBe(0);
  });

  it("reads the text kept when it was attached, and computes nothing", async () => {
    const { derive, asked } = deriving();
    const { workspaces, access } = setup(derive);
    const id = await access.chat.workspace();
    await workspaces.add(id, [{ name: "brief.pdf", kind: "pdf", blob: new Blob([new Uint8Array([1])]), derived: "[Page 1]\nKept" }], USER);
    expect((await read(access, "brief.pdf")).result).toBe("1\t[Page 1]\n2\tKept\n\n[brief.pdf, version 1: lines 1–2 of 2.]");
    expect(asked.pdf).toBe(0);
  });

  it("says what pdf.js could not read", async () => {
    const { derive } = deriving();
    const { workspaces, access } = setup({ ...derive, pdfText: async () => Promise.reject(new Error("Invalid PDF structure.")) });
    const id = await access.chat.workspace();
    await workspaces.add(id, [{ name: "broken.pdf", kind: "pdf", blob: new Blob([new Uint8Array([1])]) }], USER);
    expect((await read(access, "broken.pdf")).error).toBe("broken.pdf could not be read as a PDF: Invalid PDF structure.");
  });
});
