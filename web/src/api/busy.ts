// A full engine, waited out: a side request (an agent's, a safety check's,
// the Transcribe view's notes) that the engine answers with no lane free
// sleeps a second and asks again, and stops sleeping the moment it is stopped.

/** The engine had no lane free: `engine_full` over HTTP, "all lanes in use" on the socket. */
export const isEngineFull = (message: string) => /engine_full|all lanes in use/i.test(message);

/** `ms`, or less when `signal` aborts first; never rejects. */
export function abortableSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(timer), resolve()), { once: true });
  });
}
