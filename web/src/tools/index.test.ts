import { describe, expect, it } from "vitest";
import { AGENT_TOOL, AGENTS_IGNIS_PROMPT } from "./agents/agents.ts";
import { ASK_USER_IGNIS_PROMPT, ASK_USER_TOOL } from "./ask/ask.ts";
import { dateTimePrompt } from "./datetime/datetime.ts";
import { IMAGE_IGNIS_PROMPT, IMAGE_TOOL } from "./image/image.ts";
import {
  agentExtras,
  ALL_TOOLS,
  DEFAULT_TOOL_ROUNDS,
  ignisPrompt,
  NO_TOOLS,
  routeCall,
  setAllTools,
  toolExtras,
  toolsInUse,
  turnDateTime,
} from "./index.ts";
import {
  MEMORY_DELETE_TOOL,
  MEMORY_READ_TOOL,
  MEMORY_SAVE_TOOL,
  memoryPrompt,
  PLAN_PROMPT,
  RUN_JS_TOOL,
  runJsPrompt,
  UPDATE_PLAN_TOOL,
} from "./local/local.ts";
import { WEB_FETCH_TOOL, WEB_IGNIS_PROMPT, WEB_SEARCH_TOOL } from "./web/web.ts";
import { LIST_FILES_TOOL, READ_FILE_TOOL, WORKSPACE_PROMPT, WRITE_FILE_TOOL } from "../workspace/tools.ts";
import { withAttached } from "../workspace/attach.ts";
import { buildChatRequest, type Settings, type Turn } from "../api/request.ts";
import { buildResponseCreate, responseItems } from "../api/responses.ts";

const settings: Settings = {
  systemPrompt: "Be brief.",
  temperature: 1,
  topP: 0.95,
  maxTokens: 100,
  reasoningEffort: "none",
  thinkingBudget: null,
  laneTag: "interactive",
};

const now = new Date("2026-09-15T01:52:00+02:00");
const ctx = { now, notes: [] };
const all = {
  enabled: true,
  agents: true,
  web: true,
  askUser: true,
  dateTime: true,
  dateTimeLive: false,
  runJs: true,
  jsSafetyCheck: true,
  plan: true,
  memory: true,
  workspace: true,
  images: true,
  maxRounds: DEFAULT_TOOL_ROUNDS,
};
const names = (extras: { tools?: { function: { name: string } }[] }) => (extras.tools ?? []).map((t) => t.function.name);

describe("tools", () => {
  it("adds nothing while no tool is on", () => {
    expect(ignisPrompt(NO_TOOLS, ctx)).toBe("");
    expect(toolExtras(NO_TOOLS, ctx)).toEqual({ ignisPrompt: "", tools: [] });
    expect(toolsInUse(NO_TOOLS)).toBe(0);
  });

  it("declares agents and adds their part of the ignis prompt", () => {
    expect(toolExtras({ ...NO_TOOLS, agents: true }, ctx)).toEqual({ ignisPrompt: AGENTS_IGNIS_PROMPT, tools: [AGENT_TOOL] });
  });

  it("writes the date and time into the prompt without declaring a tool", () => {
    expect(toolExtras({ ...NO_TOOLS, dateTime: true }, ctx)).toEqual({ ignisPrompt: dateTimePrompt(now), tools: [] });
    expect(turnDateTime({ ...NO_TOOLS, dateTime: true }, now)).toBeUndefined();
  });

  /**
   * Updating every prompt moves the moment out of the system prompt: the two
   * together would say two different times, and the point of the option is to
   * leave the prompt's head — where the system prompt is — still.
   */
  it("sends the moment as a turn's own developer message once it updates every prompt", () => {
    const live = { ...NO_TOOLS, dateTime: true, dateTimeLive: true };
    expect(ignisPrompt(live, ctx)).toBe("");
    expect(turnDateTime(live, now)).toBe(dateTimePrompt(now));
    expect(turnDateTime({ ...live, enabled: false }, now)).toBeUndefined();
    expect(turnDateTime({ ...NO_TOOLS, dateTimeLive: true }, now)).toBeUndefined();
  });

  /**
   * An agent answers one prompt and is gone: it has no history to put a
   * developer message after, so it keeps taking the session's moment in its
   * system prompt — which is also what has every agent of a session open with
   * the same tokens.
   */
  it("keeps the date in an agent's system prompt even while the conversation updates every prompt", () => {
    const extras = agentExtras({ ...NO_TOOLS, dateTime: true, dateTimeLive: true }, ctx);
    expect(extras.ignisPrompt).toBe(dateTimePrompt(now));
  });

  it("puts every tool in a fixed order, the date first", () => {
    expect(toolExtras(all, ctx)).toEqual({
      ignisPrompt: [
        dateTimePrompt(now),
        AGENTS_IGNIS_PROMPT,
        WEB_IGNIS_PROMPT,
        ASK_USER_IGNIS_PROMPT,
        runJsPrompt(true),
        PLAN_PROMPT,
        memoryPrompt([]),
        WORKSPACE_PROMPT,
        IMAGE_IGNIS_PROMPT,
      ].join("\n\n"),
      tools: [
        AGENT_TOOL,
        WEB_SEARCH_TOOL,
        WEB_FETCH_TOOL,
        ASK_USER_TOOL,
        RUN_JS_TOOL,
        UPDATE_PLAN_TOOL,
        MEMORY_SAVE_TOOL,
        MEMORY_READ_TOOL,
        MEMORY_DELETE_TOOL,
        LIST_FILES_TOOL,
        READ_FILE_TOOL,
        WRITE_FILE_TOOL,
        IMAGE_TOOL,
      ],
    });
    expect(toolsInUse(all)).toBe(9);
  });

  it("declares the workspace's tools whenever its switch is on, and tells run_js about the safety check", () => {
    expect(toolExtras({ ...NO_TOOLS, workspace: true }, ctx)).toEqual({
      ignisPrompt: WORKSPACE_PROMPT,
      tools: [LIST_FILES_TOOL, READ_FILE_TOOL, WRITE_FILE_TOOL],
    });
    expect(ignisPrompt({ ...NO_TOOLS, runJs: true, jsSafetyCheck: false }, ctx)).toBe(runJsPrompt(false));
    expect(runJsPrompt(true)).toContain("reviewed before it runs");
    expect(runJsPrompt(false)).not.toContain("reviewed");
  });

  /**
   * ADR 0072: the system prompt and the tools block open every request, and
   * what the chat holds never moves them. A file attached is named in the
   * prompt that brought it, behind the history, and the model reaches the
   * workspace's files through its tools.
   */
  it("keeps the system prompt and the tools block byte-identical whatever the chat attaches or writes", () => {
    const extras = toolExtras(all, ctx);
    const first: Turn[] = [{ role: "user", content: "Plan a trip" }];
    const later: Turn[] = [
      ...first,
      { role: "assistant", content: "", toolCalls: [{ id: "c1", name: "write_file", arguments: '{"name":"plan.md","content":"x"}' }] },
      { role: "tool", toolCallId: "c1", content: "Created plan.md, version 1" },
      { role: "assistant", content: "Done." },
      { role: "user", content: withAttached("Add these", ["brief.pdf", "notes.txt"]) },
    ];
    const [a, b] = [buildChatRequest(settings, first, extras), buildChatRequest(settings, later, extras)];
    expect(JSON.stringify(b.messages[0])).toBe(JSON.stringify(a.messages[0]));
    expect(JSON.stringify(b.tools)).toBe(JSON.stringify(a.tools));
    expect(b.messages.at(-1)).toEqual({ role: "user", content: "Add these\n\n[Attached to the workspace: brief.pdf, notes.txt]" });
    const [x, y] = [first, later].map((turns) => buildResponseCreate({ settings, turns, extras }, "s", responseItems(turns)));
    expect(y.instructions).toBe(x.instructions);
    expect(JSON.stringify(y.tools)).toBe(JSON.stringify(x.tools));
    expect(names(extras)).toEqual(expect.arrayContaining(["list_files", "read_file", "write_file"]));
  });

  it("declares nothing while the switch for all tools is off", () => {
    expect(toolExtras({ ...all, enabled: false }, ctx)).toEqual({ ignisPrompt: "", tools: [] });
    expect(agentExtras({ ...all, enabled: false }, ctx)).toEqual({ ignisPrompt: "", tools: [] });
    expect(toolsInUse({ ...all, enabled: false })).toBe(0);
  });

  it("gives agents every tool but agent, ask_user, update_plan and make_image", () => {
    expect(names(agentExtras(all, ctx))).toEqual([
      "web_search",
      "web_fetch",
      "run_js",
      "memory_save",
      "memory_read",
      "memory_delete",
      "list_files",
      "read_file",
      "write_file",
    ]);
    expect(agentExtras({ ...NO_TOOLS, agents: true, askUser: true, plan: true, images: true }, ctx)).toEqual({ ignisPrompt: "", tools: [] });
  });

  it("keeps each tool's choice under the switch for all, and turns all on when none was chosen", () => {
    const some = { ...NO_TOOLS, web: true };
    expect(setAllTools(some, false)).toEqual({ ...some, enabled: false });
    expect(setAllTools({ ...some, enabled: false }, true)).toEqual(some);
    expect(setAllTools({ ...NO_TOOLS, enabled: false }, true)).toEqual(all);
    expect(ALL_TOOLS).toEqual(all);
  });

  it("gives a turn 16 rounds of tool calls, and keeps the rounds the owner typed across the switch for all", () => {
    expect(DEFAULT_TOOL_ROUNDS).toBe(16);
    expect(ALL_TOOLS.maxRounds).toBe(16);
    expect(NO_TOOLS.maxRounds).toBe(16);
    // Turning every tool on again hands back ALL_TOOLS: the rounds are not a tool, so they come along.
    const chosen = { ...NO_TOOLS, enabled: false, maxRounds: 40 };
    expect(setAllTools(chosen, true)).toEqual({ ...all, maxRounds: 40 });
    expect(setAllTools({ ...all, maxRounds: 40 }, false).maxRounds).toBe(40);
  });

  it("routes declared calls to their runner, and any undeclared tool to unknown", () => {
    const every = names(toolExtras(all, ctx));
    expect(routeCall("web_search", every)).toBe("web");
    expect(routeCall("agent", every)).toBe("agent");
    expect(routeCall("ask_user", every)).toBe("ask");
    expect(routeCall("make_image", every)).toBe("image");
    expect(routeCall("make_image", names(toolExtras({ ...all, images: false }, ctx)))).toBe("unknown");
    for (const local of ["run_js", "update_plan", "memory_save", "memory_read", "memory_delete", "list_files", "read_file", "write_file"]) {
      expect(routeCall(local, every)).toBe("local");
    }
    expect(routeCall("create_file", every)).toBe("unknown");
    expect(routeCall("read_file_x", every)).toBe("unknown");
    expect(routeCall("run_js", ["agent"])).toBe("unknown");
    expect(routeCall("agent", [])).toBe("unknown");
  });
});
