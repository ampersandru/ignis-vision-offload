import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import { StaticPictures } from "./pictures.tsx";
import { addMarked, addToPool, endStep, type ImageSession, landOutput, newSession, setDraft, startStep, type StepSettings, storeLine } from "./sessions.ts";
import { ImageStudio } from "./Studio.tsx";
import { DEFAULT_LIMITS } from "./studio.ts";
import type { ImageSessionsActions, ImageSessionsState, StudioRun } from "./useImageSessions.ts";

// The image view in its states (image/Studio.tsx): the sessions beside, the
// steps with their inputs and pictures, the last picture large, the next
// step's inputs named as the prompt calls them, and the run under way.

const text: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 524288, kind: "text" };
const image: ModelState = { state: "ready", id: "ignis-image-2.1-turbo", contextLimit: null, kind: "image" };
const settings: StepSettings = { size: "2752x1536", count: 1, seed: null, background: "auto", enhance: false };

const none = () => {};
const actions: ImageSessionsActions = {
  newSession: none,
  select: none,
  remove: none,
  setDraft: none,
  addFiles: none,
  toggleInput: none,
  run: none,
  stop: none,
  mark: none,
  saveMarks: none,
  view: none,
  reuseStep: none,
  reusePicture: none,
  importPictures: async () => {},
  switchToImage: none,
};

/** A session with an upload, a finished step that made #2 from it, and #2 the next step's input. */
function worked(): ImageSession {
  let s = newSession("s1", 0);
  [s] = addToPool(s, [{ id: "up", width: 800, height: 600, origin: "upload", name: "fox.png" }], 1, true);
  let step: number;
  [s, step] = startStep(s, "a fox in the snow at dusk", [1], settings, 2);
  [s] = landOutput(s, step, { id: "made", width: 2752, height: 1536, seed: 41 }, 3);
  return endStep(s, step, { status: "done" }, 4);
}

function state(change: Partial<ImageSessionsState> = {}): ImageSessionsState {
  const active = change.active ?? worked();
  return {
    loaded: true,
    sessions: [active],
    active,
    limits: DEFAULT_LIMITS,
    imageModel: { id: "ignis-image-2.1-turbo", source: "built-in", on_disk: "complete", family: "qwenimage21", loaded: false, switchable: true },
    run: null,
    notice: null,
    marking: null,
    viewing: null,
    usage: { bytes: 12_400_000, images: 3 },
    storeKind: "indexeddb",
    switching: false,
    switchRefusal: null,
    ...change,
  };
}

const urls = { "thumb:up": "blob:up-thumb", "full:made": "blob:made", "thumb:made": "blob:made-thumb" };
const render = (model: ModelState, st: ImageSessionsState) =>
  renderToStaticMarkup(
    <StaticPictures urls={urls}>
      <ImageStudio model={model} state={st} actions={actions} drawer={null} onDrawer={none} />
    </StaticPictures>,
  );

describe("ImageStudio", () => {
  it("lists the sessions by their last picture, with what the store holds", () => {
    const html = render(image, state());
    expect(html).toContain("a fox in the snow at dusk");
    expect(html).toContain("1 step, 1 made");
    expect(html).toContain("3 pictures, 12 MB, kept in this browser.");
    expect(storeLine({ storeKind: "memory", usage: null })).toContain("the sessions go with the page");
  });

  it("shows each step with its inputs, its prompt and the picture it made, the last one large on the kiln", () => {
    const html = render(image, state());
    expect(html).toContain('aria-label="Step 1"');
    expect(html).toContain("image 1");
    expect(html).toContain("2752 × 1536");
    expect(html).toMatch(/<img src="blob:made" alt="#2" class="stage-checker block max-w-full object-contain max-h-\[min\(60vh,40rem\)\]"/);
    expect(html).toContain("seed 41");
    expect(html).toContain("image 1 next");
    expect(html).toContain('download="ignis-41-2752x1536.png"');
  });

  it("names the next step's inputs as the prompt calls them, and offers Edit", () => {
    const html = render(image, state({ active: setDraft(worked(), { prompt: "make it night" }, 5) }));
    expect(html).toContain('aria-label="Inputs"');
    expect(html).toContain('aria-label="Drop #2 from the inputs"');
    expect(html).toContain("1 of 10");
    expect(html).toMatch(/<button type="submit"[^>]*>Edit<\/button>/);
  });

  it("names a mask-mode input's mask by its own number", () => {
    let s = worked();
    [s] = addMarked(s, 2, { id: "drawn", width: 2752, height: 1536 }, { marks: [], mode: "mask", keepOutside: true, mask: "m", region: "r" }, 6);
    const html = render(image, state({ active: s }));
    expect(html).toContain("+ mask 2");
    expect(html).toContain("#3, marked");
    expect(html).toContain("2 of 10");
  });

  it("with a text model loaded, Generate is the switch to the image model too", () => {
    const html = render(text, state({ active: setDraft(newSession("e", 0), { prompt: "a fox" }, 1) }));
    expect(html).toMatch(/<button type="submit"[^>]*>Switch &amp; generate<\/button>/);
    expect(html).toContain("Unloads qwen3.8-27b and loads the image model first.");
    expect(html).toContain("Start a picture");
  });

  it("shows the frame filling while a step has made nothing yet, and Stop", () => {
    let s = worked();
    [s] = startStep(s, "make it night", [2], settings, 6);
    const run: StudioRun = { sessionId: "s1", step: 2, edit: true, n: 1, switches: false, progress: { phase: "denoising", image: 0, step: 3, total: 8 }, size: "2752x1536", made: 0 };
    const html = render(image, state({ active: s, run }));
    expect(html).toContain("Denoising · step 3/8");
    expect(html.match(/h-1 flex-1 bg-\[#ff5a1f\]/g)).toHaveLength(3);
    expect(html).toContain(">Stop<");
    expect(html).toContain("Making");
  });

  it("says a failed step's error on the step, and waits for the store before anything else", () => {
    let s = worked();
    let step: number;
    [s, step] = startStep(s, "x", [2], settings, 6);
    s = endStep(s, step, { status: "failed", error: "The queue is full" }, 7);
    expect(render(image, state({ active: s }))).toMatch(/role="alert"[^>]*>The queue is full/);
    expect(render(image, state({ loaded: false }))).toContain("Opening your sessions…");
  });
});
