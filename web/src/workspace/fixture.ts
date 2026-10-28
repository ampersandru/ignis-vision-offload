import type { PanelActions } from "./WorkspacePanel.tsx";

/** The workspace panel's actions for drawing it without a store: each does nothing, or answers as done. */
export const idleActions: PanelActions = {
  create: async () => null,
  upload: () => {},
  exportZip: () => {},
  importZip: () => {},
  rename: async () => null,
  remove: () => {},
  download: () => {},
  save: async () => ({ ok: true, version: 2 }),
  restore: () => {},
  content: async () => null,
  derive: async () => null,
};
