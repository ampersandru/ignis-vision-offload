import { useCallback, useEffect, useRef, useState } from "react";
import type { ImageProgress } from "../api/images.ts";
import { imageLoaded, type LastSwitch, type ModelState } from "../api/model.ts";
import { keepOutside, type MarkedPicture, promptStub } from "../markup/markup.ts";
import { afterSwitch } from "../server/outcome.ts";
import { type CatalogRow, type ConfigDocument, readCatalog, readConfig } from "../server/routes.ts";
import type { ImageId, ImageStore, StoreUsage } from "../store/imageStore.ts";
import { unreferenced } from "../store/imageStore.ts";
import { pageStore } from "../store/index.ts";
import { keepPicture, keepPlain } from "../store/pictures.ts";
import {
  addMarked,
  addToPool,
  byRecent,
  endStep,
  type ImageSession,
  imagesOf,
  inputNumbers,
  isEmpty,
  keepFor,
  landOutput,
  newSession,
  pictureOf,
  reusePicture,
  reuseStep,
  type SessionDraft,
  SESSIONS,
  setDraft,
  fromStore,
  startStep,
  type StepSettings,
  toggleInput,
  unmark,
  wireImages,
} from "./sessions.ts";
import {
  fileFromDataUrl,
  imageModelRow,
  lastSwitchOf,
  type Limits,
  limitsOf,
  paramsOf,
  performRun,
  pickReferences,
  runPlan,
  settleSwitch,
  switchToImageModel,
} from "./studio.ts";

// The image view's state (image/sessions.ts says what a session is): the
// sessions, read from the page's ImageStore when the page opens and written
// back as they change; the step under way with its progress; the picture open
// in the markup editor or the viewer; what the server allows; and the explicit
// switch to the image model (its band is not shown for now).
//
// The sessions live in a ref as well as in state: a picture landing, a mark
// saved or a step ending is applied to the session as it is *now*, not as the
// render that started the work saw it. Each change is saved a moment later,
// and at once when a step ends.
//
// A `model` is sent only where the owner chose the image model: a run whose
// params name it — Generate or Edit pressed with a text model loaded
// (`runPlan`, GitHub #324) — and the switch button.

export type StudioRun = {
  sessionId: string;
  step: number;
  edit: boolean;
  n: number;
  /** The request names the image model: the server switches to it before the job. */
  switches: boolean;
  progress: ImageProgress | null;
  /** The size asked, for the frame while nothing has arrived. */
  size: string;
  /** Pictures of this run already in the pool. */
  made: number;
};

export type ImageSessionsState = {
  /** The sessions have been read from the store. */
  loaded: boolean;
  /** Newest first. */
  sessions: ImageSession[];
  active: ImageSession;
  limits: Limits;
  imageModel: CatalogRow | null;
  run: StudioRun | null;
  /** What the last add, mark or send could not do. */
  notice: string | null;
  /** The pool picture open in the markup editor. */
  marking: number | null;
  /** The pool picture open in the viewer. */
  viewing: number | null;
  usage: StoreUsage | null;
  storeKind: ImageStore["kind"];
  switching: boolean;
  switchRefusal: string | null;
};

export type ImageSessionsActions = {
  newSession: () => void;
  select: (id: string) => void;
  remove: (id: string) => void;
  setDraft: (change: Partial<SessionDraft>) => void;
  /** Pictures into the active session's pool, as inputs of the next step. */
  addFiles: (files: File[]) => void;
  toggleInput: (n: number) => void;
  run: () => void;
  stop: () => void;
  mark: (n: number | null) => void;
  saveMarks: (n: number, marked: MarkedPicture | null, color: string) => void;
  view: (n: number | null) => void;
  reuseStep: (step: number) => void;
  reusePicture: (n: number) => void;
  /** Pictures from elsewhere (the chat) into a new session, or the active one, as inputs; the view is the caller's to show. */
  importPictures: (pictures: { blob: Blob; name?: string }[], into: "new" | "active") => Promise<void>;
  switchToImage: () => void;
};

const SAVE_DELAY_MS = 400;

const newId = () => (globalThis.crypto?.randomUUID?.() ?? `s-${Date.now()}-${Math.random().toString(36).slice(2)}`);

export function useImageSessions(
  model: ModelState,
  refreshModel: () => void,
  visible: boolean,
  options: { store?: ImageStore; live?: () => Iterable<ImageId> } = {},
): [ImageSessionsState, ImageSessionsActions] {
  const store = options.store ?? pageStore();
  const [config, setConfig] = useState<ConfigDocument | null>(null);
  const [catalog, setCatalog] = useState<CatalogRow[] | null>(null);
  const first = useRef<ImageSession | null>(null);
  first.current ??= newSession(newId(), Date.now());
  const list = useRef<ImageSession[]>([first.current]);
  const [sessions, setSessions] = useState<ImageSession[]>(list.current);
  const [activeId, setActiveId] = useState(first.current.id);
  const activeNow = useRef(activeId);
  activeNow.current = activeId;
  const [loaded, setLoaded] = useState(false);
  const [run, setRun] = useState<StudioRun | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [marking, setMarking] = useState<number | null>(null);
  const [viewing, setViewing] = useState<number | null>(null);
  const [usage, setUsage] = useState<StoreUsage | null>(null);
  const [following, setFollowing] = useState<{ before?: LastSwitch } | null>(null);
  const [switchRefusal, setSwitchRefusal] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const timers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const modelNow = useRef(model);
  modelNow.current = model;
  const limits = limitsOf(config);
  const limitsNow = useRef(limits);
  limitsNow.current = limits;
  const imageModel = imageModelRow(catalog);
  const imageModelNow = useRef(imageModel);
  imageModelNow.current = imageModel;

  const refreshUsage = useCallback(() => void store.usage().then(setUsage, () => {}), [store]);

  // The sessions as the store has them; a step a page left running ended there.
  useEffect(() => {
    let live = true;
    void store.listDocs<ImageSession>(SESSIONS).then(
      (stored) => {
        if (!live) return;
        const settled = stored.map((s) => fromStore(s, Date.now())).sort(byRecent);
        if (settled.length > 0) {
          list.current = settled;
          setSessions(settled);
          setActiveId(settled[0].id);
        }
        setLoaded(true);
        refreshUsage();
      },
      () => live && setLoaded(true),
    );
    return () => {
      live = false;
    };
  }, [store, refreshUsage]);

  // Read on opening the view and on each return to it, and again once the image model serves: the sizes and counts
  // it offers are the running server's.
  const served = imageLoaded(model);
  useEffect(() => {
    if (!visible) return;
    void readConfig().then((answer) => answer.ok && setConfig(answer.value));
    void readCatalog().then((answer) => answer.ok && setCatalog(answer.value));
  }, [visible, served]);

  // After the owner's switch the page follows the model until it serves as an image model, or the switch's record
  // says it did not land (settleSwitch), and then says why.
  useEffect(() => {
    if (!following) return;
    const settled = settleSwitch(following.before, model);
    if (settled.kind === "done") {
      setFollowing(null);
      if (settled.line) setSwitchRefusal(settled.line);
      return;
    }
    const timer = setInterval(refreshModel, 1000);
    return () => clearInterval(timer);
  }, [following, model, refreshModel]);

  useEffect(() => () => abort.current?.abort(), []);

  const find = (id: string) => list.current.find((s) => s.id === id);

  function save(id: string, now = false) {
    const pending = timers.current.get(id);
    if (pending) clearTimeout(pending);
    const write = () => {
      timers.current.delete(id);
      const s = find(id);
      if (s && !isEmpty(s)) void store.putDoc(SESSIONS, s.id, s).catch(() => setNotice("The session could not be saved in this browser."));
    };
    if (now) write();
    else timers.current.set(id, setTimeout(write, SAVE_DELAY_MS));
  }

  /** Applies `change` to session `id` as it is now, shows it, saves it soon, and hands back what the change returned. */
  function apply<R>(id: string, change: (s: ImageSession) => [ImageSession, R], saveNow = false): R | undefined {
    const s = find(id);
    if (!s) return undefined;
    const [next, result] = change(s);
    if (next !== s) {
      list.current = list.current.map((x) => (x.id === id ? next : x));
      setSessions(list.current);
      save(id, saveNow);
    }
    return result;
  }
  const update = (id: string, change: (s: ImageSession) => ImageSession, saveNow = false) => apply(id, (s) => [change(s), undefined], saveNow);

  const active = find(activeId) ?? list.current[0];
  const wireCount = (s: ImageSession, inputs = s.draft.inputs) => wireImages(s, inputs).length;

  async function addFiles(files: File[], sessionId = activeNow.current, origin: "upload" | "chat" = "upload") {
    const s = find(sessionId);
    if (!s) return;
    const picked = pickReferences(files, wireCount(s), limitsNow.current.maxImages);
    setNotice(picked.notice);
    if (picked.taken.length === 0) return;
    try {
      const stored = await Promise.all(picked.taken.map((file) => keepPicture(store, file)));
      apply(sessionId, (x) =>
        addToPool(
          x,
          stored.map((p, i) => ({ id: p.id, width: p.width, height: p.height, origin, ...(picked.taken[i].name ? { name: picked.taken[i].name } : {}) })),
          Date.now(),
          true,
        ),
      );
      refreshUsage();
    } catch (err) {
      setNotice(`The pictures could not be kept: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function start() {
    if (abort.current) return;
    const s = find(activeNow.current);
    if (!s) return;
    const { draft } = s;
    const limitsAt = limitsNow.current;
    const wire = wireImages(s, draft.inputs);
    const plan = runPlan(modelNow.current, limitsAt, imageModelNow.current, draft.enhance ?? limitsAt.enhance, wire.length > 0, false);
    if (!plan.ready) return;
    const controller = new AbortController();
    abort.current = controller;
    let files: File[];
    let keep: { source: Blob; region: Blob } | null = null;
    try {
      files = await Promise.all(
        wire.map(async (w) => {
          const blob = await store.getImage(w.id);
          if (!blob) throw new Error(`${w.name} is no longer in this browser's store.`);
          return new File([blob], w.name, { type: blob.type || "image/png" });
        }),
      );
      const keepIds = keepFor(s, draft.inputs);
      if (keepIds) {
        const [source, region] = await Promise.all([store.getImage(keepIds.source), store.getImage(keepIds.region)]);
        if (source && region) keep = { source, region };
      }
    } catch (err) {
      abort.current = null;
      setNotice(err instanceof Error ? err.message : String(err));
      return;
    }
    const params = paramsOf(draft, files, limitsAt, plan.model);
    if (!params) {
      abort.current = null;
      return;
    }
    const settings: StepSettings = { size: params.size, count: params.n, seed: params.seed, background: draft.background, enhance: params.enhance };
    const stepN = apply(s.id, (x) => startStep(x, draft.prompt.trim(), [...draft.inputs], settings, Date.now()), true);
    if (stepN === undefined) {
      abort.current = null;
      return;
    }
    const switches = params.model !== undefined;
    let made = 0;
    let progress: ImageProgress | null = null;
    let refreshed = false;
    let revised: string | undefined;
    const show = () => setRun({ sessionId: s.id, step: stepN, edit: files.length > 0, n: params.n, switches, progress, size: params.size, made });
    show();
    setNotice(null);
    // Pictures land in the order they came, each kept in the store (laid over its input first when asked).
    let landing: Promise<void> = Promise.resolve();
    void performRun(
      params,
      draft.stream,
      {
        progress: (p) => {
          progress = p;
          show();
          if (switches && !refreshed && (p.phase === "switching" || p.phase === "loading")) {
            refreshed = true;
            refreshModel();
          }
        },
        image: (item) => {
          landing = landing.then(async () => {
            let blob: Blob = fileFromDataUrl(item.url, "made.png");
            let kept = false;
            if (keep) {
              try {
                blob = await keepOutside(blob, keep);
                kept = true;
              } catch {
                // The picture as the model made it, then.
              }
            }
            const stored = await keepPicture(store, blob);
            if (item.revisedPrompt) revised = item.revisedPrompt;
            apply(s.id, (x) =>
              landOutput(
                x,
                stepN,
                {
                  id: stored.id,
                  width: stored.width,
                  height: stored.height,
                  ...(item.seed !== undefined ? { seed: item.seed } : {}),
                  ...(item.revisedPrompt ? { revisedPrompt: item.revisedPrompt } : {}),
                  ...(item.background ? { background: item.background } : {}),
                  ...(kept ? { kept: true as const } : {}),
                },
                Date.now(),
              ),
            );
            made++;
            show();
          });
        },
        done: (end) => {
          void landing
            .catch((err: unknown) => setNotice(`A picture could not be kept: ${err instanceof Error ? err.message : String(err)}`))
            .then(() => {
              abort.current = null;
              const status = end.error ? "failed" : controller.signal.aborted ? "stopped" : "done";
              update(
                s.id,
                (x) => endStep(x, stepN, { status, ...(end.error ? { error: end.error } : {}), notes: end.notes, ...(revised ? { revised } : {}) }, Date.now()),
                true,
              );
              setRun(null);
              refreshUsage();
              if (end.reread) refreshModel();
            });
        },
      },
      (() => {
        let id = 0;
        return () => ++id;
      })(),
      controller.signal,
    );
  }

  async function saveMarks(n: number, marked: MarkedPicture | null, color: string) {
    const sessionId = activeNow.current;
    const s = find(sessionId);
    const p = s && pictureOf(s, n);
    if (!s || !p) return;
    if (!marked) {
      if (p.markup) update(sessionId, (x) => unmark(x, n, Date.now()));
      setMarking(null);
      return;
    }
    try {
      const [drawn, mask, region] = await Promise.all([keepPicture(store, marked.drawn), keepPlain(store, marked.mask), keepPlain(store, marked.region)]);
      const base = p.markup ? p.markup.source : n;
      const markup = { marks: marked.marks, mode: marked.mode, keepOutside: marked.keepOutside, mask: mask.id, region: region.id };
      const added = apply(sessionId, (x) => addMarked(x, base, { id: drawn.id, width: drawn.width, height: drawn.height }, markup, Date.now(), p.markup ? n : base));
      // A first mark on a draft with no words yet starts the sentence the way the marks are sent.
      const after = find(sessionId);
      const numbered = after && added !== undefined ? inputNumbers(after, after.draft.inputs).find((i) => i.n === added) : undefined;
      if (after && numbered && after.draft.prompt.trim() === "") update(sessionId, (x) => setDraft(x, { prompt: promptStub(marked.mode, numbered, color) }, Date.now()));
      setMarking(null);
      refreshUsage();
    } catch (err) {
      setNotice(`The marks could not be kept: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function remove(id: string) {
    if (run?.sessionId === id) return;
    const gone = find(id);
    if (!gone) return;
    const pending = timers.current.get(id);
    if (pending) clearTimeout(pending);
    list.current = list.current.filter((s) => s.id !== id);
    if (list.current.length === 0) list.current = [newSession(newId(), Date.now())];
    setSessions(list.current);
    if (activeNow.current === id) setActiveId(list.current[0].id);
    try {
      await store.deleteDoc(SESSIONS, id);
      // Only what no other session names, and nothing the chat holds now, goes with it.
      const keep = new Set<ImageId>([...list.current.flatMap((s) => [...imagesOf(s)]), ...(options.live?.() ?? [])]);
      for (const image of unreferenced(imagesOf(gone), keep)) await store.deleteImage(image);
    } finally {
      refreshUsage();
    }
  }

  function openSession() {
    const empty = list.current.find(isEmpty);
    if (empty) {
      setActiveId(empty.id);
      return empty.id;
    }
    const s = newSession(newId(), Date.now());
    list.current = [s, ...list.current];
    setSessions(list.current);
    setActiveId(s.id);
    return s.id;
  }

  const switchToImage = useCallback(() => {
    if (!imageModelRow(catalog) || following) return;
    setSwitchRefusal(null);
    setFollowing({ before: lastSwitchOf(modelNow.current) });
    void switchToImageModel(catalog).then((answer) => {
      if (!answer) {
        setFollowing(null);
        return;
      }
      const after = afterSwitch(answer);
      if (!after.follow) {
        setFollowing(null);
        setSwitchRefusal(after.refusal);
      }
      if (after.refreshModel) refreshModel();
    });
  }, [catalog, following, refreshModel]);

  const state: ImageSessionsState = {
    loaded,
    sessions: [...sessions].sort(byRecent),
    active,
    limits,
    imageModel,
    run,
    notice,
    marking,
    viewing,
    usage,
    storeKind: store.kind,
    switching: following !== null,
    switchRefusal,
  };
  const now = () => Date.now();
  const actions: ImageSessionsActions = {
    newSession: () => void openSession(),
    select: (id) => {
      setActiveId(id);
      setMarking(null);
      setViewing(null);
      setNotice(null);
    },
    remove: (id) => void remove(id),
    setDraft: (change) => update(activeNow.current, (s) => setDraft(s, change, now())),
    addFiles: (files) => void addFiles(files),
    toggleInput: (n) => {
      const s = find(activeNow.current);
      if (!s) return;
      if (!s.draft.inputs.includes(n) && wireCount(s, [...s.draft.inputs, n]) > limitsNow.current.maxImages) {
        setNotice(`A request takes at most ${limitsNow.current.maxImages} pictures, masks included.`);
        return;
      }
      setNotice(null);
      update(s.id, (x) => toggleInput(x, n, now()));
    },
    run: () => void start(),
    stop: () => abort.current?.abort(),
    mark: setMarking,
    saveMarks: (n, marked, color) => void saveMarks(n, marked, color),
    view: setViewing,
    reuseStep: (step) => update(activeNow.current, (s) => reuseStep(s, step, now())),
    reusePicture: (n) => update(activeNow.current, (s) => reusePicture(s, n, now())),
    importPictures: async (pictures, into) => {
      const id = into === "new" ? openSession() : activeNow.current;
      await addFiles(
        pictures.map((p, i) => new File([p.blob], p.name ?? `picture-${i + 1}.png`, { type: p.blob.type || "image/png" })),
        id,
        "chat",
      );
    },
    switchToImage,
  };
  return [state, actions];
}
