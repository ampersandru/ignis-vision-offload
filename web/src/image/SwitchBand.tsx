import { loadUnderWay, type ModelState } from "../api/model.ts";
import { actionShape } from "../ui/classes.ts";
import type { ImageSessionsState } from "./useImageSessions.ts";

// The explicit switch to the image model, for while anything else serves.
// Not shown for now — the owner's call: Generate and Edit switch on their own
// (GitHub #324). Kept, and tested, so it can come back in one line.

export function SwitchBand({
  model,
  state,
  onSwitch,
}: {
  model: ModelState;
  state: Pick<ImageSessionsState, "imageModel" | "limits" | "switching" | "switchRefusal" | "run">;
  onSwitch: () => void;
}) {
  const under = loadUnderWay(model) || state.switching;
  const loaded = model.state === "ready" || model.state === "unloaded" ? model.id : null;
  const row = state.imageModel;
  return (
    <section aria-label="Image model" className="monitor-kiln cut text-[#eae8e4] [--cut-size:20px]">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 p-5 md:p-6">
        <div className="flex min-w-0 flex-1 basis-72 flex-col gap-1">
          <h2 className="font-display text-xl leading-tight font-semibold text-white [overflow-wrap:anywhere]">
            {under ? "Switching to the image model…" : loaded ? `${loaded} is loaded` : "No model is loaded"}
          </h2>
          <p className="text-sm text-[#b9bec4]">
            {under
              ? "The page follows the switch and opens the form once the image model serves."
              : row
                ? `Images are made by the image model. ${state.limits.allowImplicit ? "Generate switches to it on its own, or switch now: either" : "Switching"} unloads the model serving and loads it. The chat's next message switches back to a text model.`
                : "ignis lists no image model. Fetch it with `ignis-server model download ignis-image-2.1-turbo`, then reload this page."}
          </p>
          {state.switchRefusal && <p className="text-sm text-[#ff7b6b] [overflow-wrap:anywhere]">{state.switchRefusal}</p>}
        </div>
        <button
          type="button"
          className={`${actionShape} bg-[#ff5a1f] px-4 py-2.5 text-[#1c2026] hover:bg-[#ff8a4c]`}
          disabled={under || !row || state.run !== null || model.state === "reading" || model.state === "error"}
          onClick={onSwitch}
        >
          Switch to the image model
        </button>
      </div>
    </section>
  );
}
