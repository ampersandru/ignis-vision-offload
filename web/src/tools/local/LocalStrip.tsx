import { type ReactNode, useState } from "react";
import { HtmlPreview, isHtmlName } from "../../ui/HtmlPreview.tsx";
import { latest } from "../../workspace/files.ts";
import { useWorkspaceFile } from "../../workspace/WorkspaceFiles.tsx";
import type { Written } from "../../workspace/workspaces.ts";
import type { AgentStatus } from "../agents/agents.ts";
import { StatusMark } from "../agents/status.tsx";
import {
  CREATE_FILE,
  DELETE_FILE,
  EDIT_FILE,
  type LocalRun,
  LIST_FILES,
  MEMORY_DELETE,
  MEMORY_READ,
  MEMORY_SAVE,
  type PlanStep,
  READ_FILE,
  RUN_JS,
  UPDATE_PLAN,
  WRITE_FILE,
} from "./local.ts";

// The local tool calls a reply made, inside the reply: a plan as a checklist,
// a file it wrote or edited with its download (the workspace's version that
// reply made, or an older chat's create_file kept in the run), run_js with its
// code, safety check and output, and a line for the rest.

const MARK: Record<LocalRun["status"], AgentStatus> = {
  checking: "running",
  running: "running",
  done: "done",
  failed: "failed",
  blocked: "failed",
  stopped: "stopped",
};

export function LocalStrip({ runs }: { runs: LocalRun[] }) {
  return (
    <section aria-label="Tool calls" className="flex flex-col gap-2">
      {/* Keyed by position too: an agent's later requests may reuse call ids. */}
      {runs.map((run, i) => (
        <LocalItem key={`${i}:${run.callId}`} run={run} />
      ))}
    </section>
  );
}

function LocalItem({ run }: { run: LocalRun }) {
  if (run.tool === UPDATE_PLAN && run.steps) return <PlanCard steps={run.steps} />;
  if (run.tool === CREATE_FILE && run.file) return <FileCard name={run.file.name} content={run.file.content} />;
  if ((run.tool === WRITE_FILE || run.tool === EDIT_FILE) && run.written) return <WrittenCard written={run.written} />;
  if (run.tool === RUN_JS && run.code !== undefined) return <JsRow run={run} />;
  return <SimpleRow run={run} />;
}

const JS_OUTCOME: Record<LocalRun["status"], string> = {
  checking: "Checking…",
  running: "Running…",
  done: "Done",
  failed: "Failed",
  blocked: "Blocked",
  stopped: "Stopped",
};

function JsRow({ run }: { run: LocalRun }) {
  const firstLine = (run.code ?? "").trim().split("\n")[0];
  const bad = run.status === "failed" || run.status === "blocked";
  return (
    <details className="cut bg-surface [--cut-size:10px]" open={run.status === "blocked"}>
      <summary className="flex min-w-0 cursor-pointer list-none items-center gap-2 px-3 py-2 select-none hover:bg-line/50 [&::-webkit-details-marker]:hidden">
        <StatusMark status={MARK[run.status]} />
        <span className="shrink-0 font-display text-xs font-semibold text-ash">JavaScript</span>
        <span className="min-w-0 truncate font-mono text-xs text-ink">{firstLine}</span>
        <span className={`ml-auto shrink-0 font-display text-xs ${bad ? "text-fault" : run.status === "done" ? "text-ember" : "text-ash"}`}>
          {JS_OUTCOME[run.status]}
        </span>
      </summary>
      <div className="flex flex-col gap-2.5 border-t border-line px-3 py-2.5">
        <pre className="max-h-60 overflow-auto bg-ground px-2.5 py-2 font-mono text-xs leading-relaxed text-ink">{run.code}</pre>
        {run.check && (
          <p className={`border-l-2 pl-3 text-xs leading-snug ${run.check.verdict === "safe" ? "border-ember text-ash" : "border-fault text-fault"}`}>
            Safety check: {run.check.verdict}. {run.check.reason}
          </p>
        )}
        {run.output !== undefined && <Output label="Result" text={run.output} />}
        {run.logs && run.logs.length > 0 && <Output label="Console" text={run.logs.join("\n")} />}
        {run.error && <p className="border-l-2 border-fault pl-3 text-xs leading-snug text-fault">{run.error}</p>}
      </div>
    </details>
  );
}

function Output({ label, text }: { label: string; text: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="font-display text-xs font-semibold text-ash">{label}</span>
      <pre className="max-h-48 overflow-auto font-mono text-xs leading-relaxed whitespace-pre-wrap break-words text-ink">{text}</pre>
    </div>
  );
}

function PlanCard({ steps }: { steps: PlanStep[] }) {
  const done = steps.filter((s) => s.status === "done").length;
  return (
    <div className="cut flex flex-col gap-2 bg-surface px-3 py-2.5 [--cut-size:10px]">
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="font-display text-[13px] font-semibold text-ink">Plan</h3>
        <span className="font-display text-xs tabular-nums text-ash">
          {done} of {steps.length} done
        </span>
      </div>
      <ol className="flex flex-col gap-1.5">
        {steps.map((s, i) => (
          <li key={i} className="flex items-start gap-2 text-sm leading-snug">
            <span
              className={`cut mt-1 size-2.5 shrink-0 [--cut-size:3px] ${s.status === "done" ? "bg-ember" : s.status === "in_progress" ? "animate-pulse bg-ember/50 motion-reduce:animate-none" : "bg-line"}`}
              aria-hidden
            />
            <span className={s.status === "done" ? "text-ash line-through" : s.status === "in_progress" ? "font-semibold text-ink" : "text-ink"}>
              {s.step}
            </span>
            <span className="sr-only">({s.status.replace("_", " ")})</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/**
 * The card of a version a write_file or edit_file call made: that version,
 * read from the workspace, and once a newer one exists a mark saying so with
 * the way to it.
 */
function WrittenCard({ written }: { written: Written }) {
  const file = useWorkspaceFile(written.workspace, written.file);
  const [showLatest, setShowLatest] = useState(false);
  const own = file?.versions.find((v) => v.n === written.version);
  if (!file || !own) {
    return (
      <div className="cut flex min-w-0 items-center gap-3 bg-surface px-3 py-2.5 [--cut-size:10px]">
        <span className="shrink-0 font-display text-xs font-semibold text-ash">File</span>
        <span className="min-w-0 truncate text-sm text-ash">
          {file === undefined ? `${written.name}, version ${written.version}` : `${written.name}, version ${written.version}: no longer in the workspace.`}
        </span>
      </div>
    );
  }
  const newest = latest(file);
  const superseded = newest.n > own.n && !newest.deleted;
  const shown = showLatest && superseded ? newest : own;
  return (
    <FileCard
      name={written.name}
      content={shown.text ?? ""}
      version={shown.n}
      note={
        superseded && (
          <span className="flex min-w-0 items-center gap-2 font-display text-xs text-ash">
            {showLatest ? `The latest, version ${newest.n}; this reply wrote version ${own.n}.` : `Superseded by version ${newest.n}.`}
            <button type="button" onClick={() => setShowLatest((s) => !s)} className="shrink-0 text-ember hover:text-ink">
              {showLatest ? `Open version ${own.n}` : "Open the latest"}
            </button>
          </span>
        )
      }
    />
  );
}

function FileCard({ name, content, version, note }: { name: string; content: string; version?: number; note?: ReactNode }) {
  const html = isHtmlName(name);
  // An HTML file opens on its preview; other files start folded.
  const [open, setOpen] = useState(html);
  const download = () => {
    const url = URL.createObjectURL(new Blob([content], { type: "text/plain;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = name.slice(name.lastIndexOf("/") + 1);
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <div className="cut flex flex-col gap-2 bg-surface px-3 py-2.5 [--cut-size:10px]">
      <div className="flex min-w-0 items-center gap-3">
        <span className="shrink-0 font-display text-xs font-semibold text-ash">File</span>
        <span className="min-w-0 truncate font-display text-sm font-semibold text-ink">{name}</span>
        {version !== undefined && <span className="shrink-0 font-display text-xs tabular-nums text-ash">version {version}</span>}
        <span className="shrink-0 font-display text-xs tabular-nums text-ash">{content.length.toLocaleString()} chars</span>
        <button type="button" onClick={() => setOpen((o) => !o)} className="ml-auto shrink-0 font-display text-xs text-ash hover:text-ink">
          {open ? "Hide" : "Preview"}
        </button>
        <button
          type="button"
          onClick={download}
          className="cut shrink-0 bg-ember px-3 py-1 font-display text-xs font-semibold text-[#1c2026] [--cut-size:6px] hover:bg-[#ff7a45]"
        >
          Download
        </button>
      </div>
      {note}
      {open &&
        (html ? (
          <HtmlPreview html={content} name={name} />
        ) : (
          <pre className="max-h-72 overflow-auto bg-ground px-2.5 py-2 font-mono text-xs leading-relaxed whitespace-pre-wrap break-words text-ink">
            {content}
          </pre>
        ))}
    </div>
  );
}

function SimpleRow({ run }: { run: LocalRun }) {
  const label =
    { [MEMORY_SAVE]: "Remembered", [MEMORY_READ]: "Recalled", [MEMORY_DELETE]: "Forgot", [READ_FILE]: "Read", [LIST_FILES]: "Listed", [WRITE_FILE]: "Write", [EDIT_FILE]: "Edit", [DELETE_FILE]: "Delete" }[run.tool] ??
    run.tool;
  const unit = run.read?.lines ? "lines" : "characters";
  const detail = run.note
    ? `${run.note.title} (${run.note.id})`
    : run.recalled
      ? run.recalled.map((n) => `${n.title} (${n.id})`).join(", ")
      : run.deletedId
      ? run.deletedId
      : run.read
        ? `${run.read.name}, ${unit} ${run.read.from.toLocaleString()}–${run.read.to.toLocaleString()} of ${run.read.total.toLocaleString()}`
        : run.seen
        ? `${run.seen.name}, version ${run.seen.version}`
        : run.written
        ? run.written.name
        : run.listed !== undefined
          ? `${run.listed.toLocaleString()} ${run.listed === 1 ? "file" : "files"}`
          : "";
  return (
    <div className="cut flex min-w-0 flex-col gap-1 bg-surface px-3 py-2 [--cut-size:10px]">
      <span className="flex min-w-0 items-center gap-2">
        <StatusMark status={MARK[run.status]} />
        <span className="shrink-0 font-display text-xs font-semibold text-ash">{label}</span>
        <span className="min-w-0 truncate text-sm text-ink">{detail}</span>
      </span>
      {run.error && <span className="text-xs leading-snug text-fault">{run.error}</span>}
    </div>
  );
}
