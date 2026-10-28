import type { Settings, ToolDefinition } from "../../api/request.ts";
import type { ToolCall } from "../../api/sse.ts";
import type { streamChat } from "../../api/stream.ts";
import type { Figures } from "../../metrics/figures.ts";
import type { Written } from "../../workspace/workspaces.ts";
import { runWorkspaceTool, WORKSPACE_TOOL_NAMES, type WorkspaceAccess } from "../../workspace/tools.ts";
import { browserMemory, type MemoryNote, type MemoryStore } from "./memory.ts";
import { checkJs, type JsCheck, JS_OUTPUT_LIMIT, JS_TIMEOUT_MS, runJs, type WorkerFactory } from "./runjs.ts";

// The local tools: everything that runs in the browser with no network.
// run_js computes in a sandboxed worker, update_plan shows the model's plan,
// memory_save and memory_delete keep notes across sessions, and list_files,
// read_file, write_file, edit_file and delete_file work on the chat's
// workspace (workspace/tools.ts).

export const RUN_JS = "run_js";
export const UPDATE_PLAN = "update_plan";
export const MEMORY_SAVE = "memory_save";
export const MEMORY_READ = "memory_read";
export const MEMORY_DELETE = "memory_delete";
/** The file tool of older chats, gone since the workspace (#170): their replies still show its cards. */
export const CREATE_FILE = "create_file";
export { DELETE_FILE, EDIT_FILE, LIST_FILES, READ_FILE, WRITE_FILE } from "../../workspace/tools.ts";

const LOCAL_TOOL_NAMES: readonly string[] = [RUN_JS, UPDATE_PLAN, MEMORY_SAVE, MEMORY_READ, MEMORY_DELETE, ...WORKSPACE_TOOL_NAMES];

export const isLocalTool = (name: string) => LOCAL_TOOL_NAMES.includes(name);

const tool = (name: string, description: string, properties: object, required: string[]): ToolDefinition => ({
  type: "function",
  function: { name, description, parameters: { type: "object", properties, required } },
});

export const RUN_JS_TOOL = tool(
  RUN_JS,
  `Run JavaScript in a sandboxed worker with no network, DOM or storage, for up to ${JS_TIMEOUT_MS / 1000} s. Returns the value you return, or the value of a single expression, and everything logged with console.log.`,
  { code: { type: "string", description: "The code. Use `return` for the result, or pass one expression. Top-level await works." } },
  ["code"],
);

export const UPDATE_PLAN_TOOL = tool(
  UPDATE_PLAN,
  "Replace the plan the user sees with this list of steps and their status.",
  {
    steps: {
      type: "array",
      description: "Every step of the plan, in order, each time.",
      items: {
        type: "object",
        properties: { step: { type: "string" }, status: { type: "string", enum: ["pending", "in_progress", "done"] } },
        required: ["step", "status"],
      },
    },
  },
  ["steps"],
);

export const MEMORY_SAVE_TOOL = tool(
  MEMORY_SAVE,
  "Save a note that stays across sessions: a lasting fact about the user, a preference, or project context. Only its title is shown to the chats; the body is read on demand.",
  {
    title: { type: "string", description: "A few words that say what the note is about." },
    body: { type: "string", description: "The full note, self-contained." },
  },
  ["title", "body"],
);

export const MEMORY_READ_TOOL = tool(
  MEMORY_READ,
  "Read the full text of saved notes.",
  { ids: { type: "array", items: { type: "string" }, description: "The notes' ids, like m3." } },
  ["ids"],
);

export const MEMORY_DELETE_TOOL = tool(
  MEMORY_DELETE,
  "Delete a saved note that is wrong or outdated.",
  { id: { type: "string", description: "The note's id, like m3." } },
  ["id"],
);

export function runJsPrompt(safetyCheck: boolean): string {
  return `# Running JavaScript
You can run JavaScript with \`run_js\`, in a sandboxed worker: no network, no DOM, no storage, at most ${JS_TIMEOUT_MS / 1000} s per call.
- Use it instead of working things out in your head: arithmetic, counting, dates, unit conversions, parsing and transforming data, checking what a snippet prints.
- \`return\` the result or pass a single expression; \`console.log\` output comes back too.${safetyCheck ? "\n- Every call is reviewed before it runs; code that does more than compute is blocked." : ""}`;
}

export const PLAN_PROMPT = `# Plan
For a task with several steps, keep a plan the user can follow with \`update_plan\`.
- Write the plan before you start, then update it as you go: a step is \`in_progress\` while you work on it and \`done\` once finished, one step in progress at a time.
- Send every step each time; the list replaces the previous one.
- Skip it for simple requests.`;

/**
 * The memory's part of the ignis system prompt: fixed text, the same whatever
 * notes are saved (ADR 0072). The titles reach the model in the turn notice,
 * after the history (tools/notice.ts).
 */
export const MEMORY_PROMPT = `# Memory
You keep notes that last across chats, each a title and a body. Their titles reach you in a notice after the conversation: at a chat's first turn the title of every saved note, if there is any, and later only the notes the user or another chat saved, changed or deleted since the last notice. Read a note's full text with \`memory_read\` when a title looks relevant, before you rely on it.
- Save a note with \`memory_save\` when you learn something that will still matter later — who the user is, their preferences, their projects — not the details of this conversation. Give it a title that says what it is about.
- Delete a note with \`memory_delete\` when it turns out wrong or outdated; to change one, delete it and save the new version.`;

export type PlanStep = { step: string; status: "pending" | "in_progress" | "done" };

/** A file an older chat's create_file call handed the user, kept in its run. */
export type CreatedFile = { name: string; content: string };

export type LocalStatus = "checking" | "running" | "done" | "failed" | "blocked" | "stopped";

/** One local tool call as the Playground shows it. */
export type LocalRun = {
  callId: string;
  /** The tool's name, as the model called it. */
  tool: string;
  status: LocalStatus;
  code?: string;
  check?: JsCheck;
  output?: string;
  logs?: string[];
  steps?: PlanStep[];
  note?: MemoryNote;
  recalled?: MemoryNote[];
  deletedId?: string;
  file?: CreatedFile;
  /** The version a write_file, edit_file or delete_file call made, which a write's or an edit's card opens. */
  written?: Written;
  /** The version of a file a read_file call showed the model: the chat's reader has seen it (seenIn). */
  seen?: Written;
  /** How many files a list_files call found. */
  listed?: number;
  /** What a read_file call read: lines from and to, or — in an older chat's run — characters. */
  read?: { name: string; from: number; to: number; total: number; lines?: true };
  error?: string;
  /** What the model receives, once the call ends. */
  result?: string;
};

export type LocalContext = {
  settings: Settings;
  signal: AbortSignal;
  /** Review run_js code with the model before it runs. */
  jsSafetyCheck: boolean;
  /** The chat's workspace, and who of the chat calls: the workspace tools fail without it. */
  workspace?: WorkspaceAccess;
  memory?: MemoryStore;
  stream?: typeof streamChat;
  createWorker?: WorkerFactory;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  /** Each safety check's request, for the session log. */
  onCheck?: (figures: Figures | null, error?: string) => void;
};

/** A local call just made, before it runs. */
export const startedRun = (call: ToolCall): LocalRun => ({ callId: call.id, tool: call.name, status: "running" });

/** Runs every call at once; resolves with each run ended, its `result` set. */
export function runLocalCalls(calls: ToolCall[], context: LocalContext, onUpdate: (run: LocalRun) => void): Promise<LocalRun[]> {
  return Promise.all(calls.map((call) => runLocal(call, context, onUpdate)));
}

/** The tool result the model receives for an ended run. */
export const localToolResult = (run: LocalRun) => run.result ?? `The ${run.tool} call did not finish.`;

const cut = (text: string) =>
  text.length > JS_OUTPUT_LIMIT ? `${text.slice(0, JS_OUTPUT_LIMIT)}\n[Cut: ${text.length} characters in all.]` : text;

function parseArgs(call: ToolCall): Record<string, unknown> | null {
  try {
    const args = JSON.parse(call.arguments) as unknown;
    return typeof args === "object" && args !== null && !Array.isArray(args) ? (args as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

async function runLocal(call: ToolCall, context: LocalContext, onUpdate: (run: LocalRun) => void): Promise<LocalRun> {
  let run = startedRun(call);
  const update = (change: Partial<LocalRun>) => {
    run = { ...run, ...change };
    onUpdate(run);
    return run;
  };
  const fail = (error: string) => update({ status: "failed", error, result: `The ${call.name} call failed: ${error}` });
  const args = parseArgs(call);
  if (!args) return fail("The call's arguments are not a JSON object.");
  const text = (key: string) => (typeof args[key] === "string" ? (args[key] as string) : "");
  const memory = context.memory ?? browserMemory;
  const { signal } = context;

  switch (call.name) {
    case RUN_JS: {
      const code = text("code");
      if (!code.trim()) return fail('The call needs non-empty "code".');
      const stopped = () => update({ status: "stopped", result: "Stopped before the code ran." });
      if (signal.aborted) return update({ code, ...stopped() });
      if (context.jsSafetyCheck) {
        update({ code, status: "checking" });
        const check = await checkJs(code, { settings: context.settings, signal, stream: context.stream, sleep: context.sleep });
        context.onCheck?.(check.figures, check.error);
        if (signal.aborted) return stopped();
        const verdict: JsCheck = { verdict: check.verdict, reason: check.reason };
        if (verdict.verdict === "unsafe") {
          return update({ status: "blocked", check: verdict, result: `Not run: the safety check blocked the code. ${verdict.reason}` });
        }
        update({ check: verdict });
      }
      update({ code, status: "running" });
      const outcome = await runJs(code, { signal, createWorker: context.createWorker });
      if (outcome.status === "stopped") return update({ status: "stopped", result: "Stopped while the code ran." });
      if (outcome.status === "timeout") {
        const error = `Stopped after ${JS_TIMEOUT_MS / 1000} s without finishing.`;
        return update({ status: "failed", error, result: error });
      }
      const logText = outcome.logs.length > 0 ? `\n\nConsole:\n${outcome.logs.join("\n")}` : "";
      if (outcome.status === "failed") {
        return update({ status: "failed", error: outcome.error, logs: outcome.logs, result: cut(`Error: ${outcome.error}${logText}`) });
      }
      return update({ status: "done", output: outcome.result, logs: outcome.logs, result: cut(`Result: ${outcome.result ?? "undefined"}${logText}`) });
    }
    case UPDATE_PLAN: {
      const steps = (Array.isArray(args.steps) ? args.steps : []).flatMap((s): PlanStep[] => {
        const item = (typeof s === "object" && s !== null ? s : {}) as { step?: unknown; status?: unknown };
        if (typeof item.step !== "string" || item.step.trim() === "") return [];
        const status = item.status === "in_progress" || item.status === "done" ? item.status : "pending";
        return [{ step: item.step.trim(), status }];
      });
      if (steps.length === 0) return fail('The call needs "steps": a list of { step, status }.');
      const done = steps.filter((s) => s.status === "done").length;
      return update({ status: "done", steps, result: `Plan updated: ${done} of ${steps.length} steps done.` });
    }
    case MEMORY_SAVE: {
      const body = text("body").trim();
      if (!body) return fail('The call needs a non-empty "body", with a short "title".');
      const saved = memory.save(text("title"), body);
      return update({ status: "done", note: saved, result: `Saved as ${saved.id}: "${saved.title}".` });
    }
    case MEMORY_READ: {
      const ids = (Array.isArray(args.ids) ? args.ids : [args.id]).filter((id): id is string => typeof id === "string" && id.trim() !== "");
      if (ids.length === 0) return fail('The call needs "ids": the notes to read, like ["m3"].');
      const wanted = ids.map((id) => id.trim());
      const found = memory.list().filter((n) => wanted.includes(n.id));
      const missing = wanted.filter((id) => !found.some((n) => n.id === id));
      if (found.length === 0) return fail(`No saved note has the id ${missing.join(", ")}.`);
      const texts = found.map((n) => `[${n.id}] ${n.title}\n${n.body}`);
      if (missing.length > 0) texts.push(`No saved note has the id ${missing.join(", ")}.`);
      return update({ status: "done", recalled: found, result: texts.join("\n\n") });
    }
    case MEMORY_DELETE: {
      const id = text("id").trim();
      if (!memory.remove(id)) return fail(`No saved note has the id "${id}".`);
      return update({ status: "done", deletedId: id, result: `Deleted ${id}.` });
    }
    default: {
      if (!WORKSPACE_TOOL_NAMES.includes(call.name)) return fail(`"${call.name}" is not a local tool.`);
      if (!context.workspace) return fail("The workspace is not available here.");
      try {
        const outcome = await runWorkspaceTool(call.name, args, context.workspace);
        return outcome.ok ? update({ status: "done", ...outcome.run }) : fail(outcome.error);
      } catch (err) {
        // The store refused: the reply says so, and the model hears the same.
        return fail(`The browser's store refused the workspace: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }
}
