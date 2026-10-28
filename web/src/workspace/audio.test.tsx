import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { MIC_BLOCKED, SECURE_CONTEXT, TURNED_OFF } from "../speech/availability.ts";
import { AudioView, RecorderView, transcriptUnavailable } from "./audio.tsx";

// The workspace panel's audio (#395): a player beside the transcript the model
// reads, and the recorder that makes a new audio file from the microphone.
// Clicks are not exercised here; what Record and Transcribe do is
// recorder.test.ts's and derived.test.ts's.

const none = () => {};

describe("the audio player", () => {
  it("plays the recording beside its transcript", () => {
    const html = renderToStaticMarkup(
      <AudioView name="memo.wav" src="blob:memo" transcript={"Buy bread.\nCall Anna."} unavailable={null} transcribing={false} error={null} onTranscribe={none} />,
    );
    expect(html).toMatch(/<audio[^>]*controls=""[^>]*src="blob:memo"/);
    expect(html).toContain('aria-label="Play memo.wav"');
    expect(html).toContain("Its transcript, as the model reads it");
    expect(html).toMatch(/<pre[^>]*>Buy bread.\nCall Anna.<\/pre>/);
    expect(html).not.toContain(">Transcribe<");
  });

  it("offers to transcribe a recording not transcribed yet", () => {
    const html = renderToStaticMarkup(<AudioView name="memo.wav" src="blob:memo" transcript={undefined} unavailable={null} transcribing={false} error={null} onTranscribe={none} />);
    expect(html).toContain("Not transcribed yet: the model&#x27;s first read of it transcribes it.");
    expect(html).toMatch(/<button[^>]*>Transcribe<\/button>/);
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Transcribe/);
    const busy = renderToStaticMarkup(<AudioView name="memo.wav" src="blob:memo" transcript={undefined} unavailable={null} transcribing error={null} onTranscribe={none} />);
    expect(busy).toMatch(/<button[^>]*disabled=""[^>]*>Transcribing…<\/button>/);
    const failed = renderToStaticMarkup(
      <AudioView name="memo.wav" src="blob:memo" transcript={undefined} unavailable={null} transcribing={false} error="The transcription of memo.wav failed: 503: busy" onTranscribe={none} />,
    );
    expect(failed).toContain("The transcription of memo.wav failed: 503: busy");
  });

  it("says why it cannot transcribe while no speech model is available", () => {
    const why = transcriptUnavailable({ available: false, reason: "the speech model is turned off (speech.model is off)" });
    expect(why).toBe(TURNED_OFF);
    expect(transcriptUnavailable({ available: true })).toBeNull();
    expect(transcriptUnavailable(undefined)).toBeNull();
    const html = renderToStaticMarkup(<AudioView name="memo.wav" src="blob:memo" transcript={undefined} unavailable={why} transcribing={false} error={null} onTranscribe={none} />);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Transcribe<\/button>/);
    expect(html).toContain('Speech is turned off on this server (<code class="font-mono text-[11px]">speech.model = off</code>).');
  });

  it("says when the page holds no bytes to play", () => {
    const html = renderToStaticMarkup(<AudioView name="memo.wav" src={null} transcript="Hi." unavailable={null} transcribing={false} error={null} onTranscribe={none} />);
    expect(html).not.toContain("<audio");
    expect(html).toContain("Hi.");
  });
});

describe("the recorder", () => {
  it("offers Record while the microphone can start", () => {
    const html = renderToStaticMarkup(<RecorderView state={{ phase: "idle", blocked: false }} unavailable={null} onStart={none} onStop={none} onCancel={none} />);
    expect(html).toMatch(/<button[^>]*title="Record from the microphone into a new audio file"[^>]*>(<svg.*?<\/svg>)?Record<\/button>/);
    expect(html).not.toContain('aria-disabled="true"');
  });

  it("strikes Record, saying why, while the microphone cannot start; a blocked one stays pressable", () => {
    const insecure = renderToStaticMarkup(
      <RecorderView state={{ phase: "idle", blocked: false }} unavailable={{ sentence: SECURE_CONTEXT, blocked: false }} onStart={none} onStop={none} onCancel={none} />,
    );
    expect(insecure).toMatch(/<button[^>]*disabled=""[^>]*>/);
    expect(insecure).toContain("The microphone needs https or localhost.");
    const blocked = renderToStaticMarkup(
      <RecorderView state={{ phase: "idle", blocked: true }} unavailable={{ sentence: MIC_BLOCKED, blocked: true }} onStart={none} onStop={none} onCancel={none} />,
    );
    expect(blocked).not.toMatch(/<button[^>]*disabled=""/);
    expect(blocked).toContain("Microphone access is blocked for this site.");
  });

  it("shows the time recorded, the level and the limit, with Stop and save and Cancel", () => {
    const html = renderToStaticMarkup(
      <RecorderView state={{ phase: "recording", seconds: 72.4, level: 0.5, limit: 1200 }} unavailable={null} onStart={none} onStop={none} onCancel={none} />,
    );
    expect(html).toContain('role="status"');
    expect(html).toContain("Recording 1:12 of 20:00");
    expect(html).toContain('style="width:50%"');
    expect(html).toMatch(/<button[^>]*>Stop and save<\/button>/);
    expect(html).toMatch(/<button[^>]*>Cancel<\/button>/);
    expect(renderToStaticMarkup(<RecorderView state={{ phase: "recording", seconds: 3, level: 0, limit: null }} unavailable={null} onStart={none} onStop={none} onCancel={none} />)).toContain(
      "Recording 0:03",
    );
  });

  it("says how the last recording ended", () => {
    const html = renderToStaticMarkup(
      <RecorderView state={{ phase: "idle", blocked: false, note: "Not saved in this browser: QuotaExceededError" }} unavailable={null} onStart={none} onStop={none} onCancel={none} />,
    );
    expect(html).toContain("Not saved in this browser: QuotaExceededError");
    expect(renderToStaticMarkup(<RecorderView state={{ phase: "saving" }} unavailable={null} onStart={none} onStop={none} onCancel={none} />)).toContain("Saving the recording…");
  });
});
