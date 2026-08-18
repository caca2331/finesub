"use client";

/**
 * Front-end preferences, durable in user-data/settings.json.
 *
 * They used to live in five separate localStorage keys, which put them in the
 * webview's profile: a cleared cache, a reinstall or a second portable copy
 * lost them, and `finesub relocate` / the uninstall tiers -- which do protect
 * user-data -- never knew they existed.
 *
 * localStorage stays, demoted to a **mirror**. Every reader here is
 * synchronous (a theme has to be applied before the first paint, not after a
 * bridge round-trip), so the module keeps an in-memory cache seeded from the
 * mirror at import time. When the bootstrap answers, settings.json wins and
 * the mirror is rewritten. Losing the mirror costs one flash; losing
 * settings.json is what used to cost the settings themselves.
 *
 * Sparse throughout: a value is stored only once someone has chosen it.
 */

import { desktopApi } from "./bridge";
import type { Preferences, TaskDefaultsPatch, TaskRequest } from "./types";

const MIRROR_KEY = "finesub-preferences";

/** The five keys this replaces; read once, then deleted. */
const LEGACY_APPEARANCE = "finesub-appearance";
const LEGACY_LANGUAGE = "finesub-language";
const LEGACY_CLOSE_ACTION = "close-window-action";
const LEGACY_CONFIRM_PREFIX = "finesub-confirm-";
const LEGACY_DEVICE = "finesub-processing-device";

const EMPTY: Preferences = { ui: {}, task_defaults: {} };

let cache: Preferences = EMPTY;
const listeners = new Set<() => void>();

function storage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

function readMirror(): Preferences {
  const store = storage();
  if (!store) return EMPTY;
  try {
    const raw = store.getItem(MIRROR_KEY);
    if (!raw) return EMPTY;
    const parsed = JSON.parse(raw) as Partial<Preferences>;
    return {
      ui: typeof parsed.ui === "object" && parsed.ui ? parsed.ui : {},
      task_defaults:
        typeof parsed.task_defaults === "object" && parsed.task_defaults
          ? parsed.task_defaults
          : {},
    };
  } catch {
    return EMPTY;
  }
}

function writeMirror(): void {
  const store = storage();
  if (!store) return;
  try {
    store.setItem(MIRROR_KEY, JSON.stringify(cache));
  } catch {
    // A full or disabled store only costs the pre-paint shortcut.
  }
}

function notify(): void {
  for (const listener of listeners) listener();
}

cache = readMirror();

/** Re-render hook for the values that are read before the bridge answers. */
export function subscribePreferences(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function uiValue<T>(key: string, fallback: T): T {
  const value = cache.ui[key];
  return value === undefined || value === null ? fallback : (value as T);
}

export function taskDefaults(): Partial<TaskRequest> {
  return cache.task_defaults;
}

/** settings.json has answered: it wins over the mirror, which is refreshed. */
export function hydratePreferences(preferences: Preferences | undefined): void {
  if (preferences) {
    cache = {
      ui: preferences.ui ?? {},
      task_defaults: preferences.task_defaults ?? {},
    };
    writeMirror();
  }
  notify();
  void migrateLegacyKeys();
}

function persist(patch: {
  ui?: Record<string, unknown>;
  task_defaults?: TaskDefaultsPatch;
}): Promise<void> {
  // Optimistic: the mirror and every reader see the new value immediately, and
  // the durable write is allowed to be slow. A failed write leaves the mirror
  // ahead of settings.json until the next successful save -- the same exposure
  // localStorage always had, and better than a UI that lags a click behind.
  const merge = (
    current: Record<string, unknown>,
    update: Record<string, unknown>,
  ): Record<string, unknown> => {
    const next = { ...current };
    for (const [key, value] of Object.entries(update)) {
      if (value === null || value === undefined) delete next[key];
      else next[key] = value;
    }
    return next;
  };
  cache = {
    ui: merge(cache.ui, (patch.ui ?? {}) as Record<string, unknown>),
    task_defaults: merge(
      cache.task_defaults as Record<string, unknown>,
      (patch.task_defaults ?? {}) as Record<string, unknown>,
    ) as Partial<TaskRequest>,
  };
  writeMirror();
  notify();
  // Most callers ignore the result: a preference that did not reach disk is
  // one the user can set again, and an error toast for a theme change would be
  // worse than the loss. The migration is the exception -- it deletes the only
  // other copy -- so the promise is returned rather than swallowed here.
  return desktopApi.savePreferences(patch).then(() => undefined);
}

export function saveUi(patch: Record<string, unknown>): void {
  void persist({ ui: patch }).catch(() => undefined);
}

export function saveTaskDefaults(patch: TaskDefaultsPatch): void {
  void persist({ task_defaults: patch }).catch(() => undefined);
}

/**
 * One-way move of the five old keys, then delete them.
 *
 * Runs after hydration so it cannot overwrite a value settings.json already
 * has: whatever the durable store knows wins, and only keys it has never seen
 * are taken from the old location.
 */
export async function migrateLegacyKeys(): Promise<boolean> {
  const store = storage();
  if (!store) return false;
  const ui: Record<string, unknown> = {};
  const defaults: TaskDefaultsPatch = {};

  const appearance = store.getItem(LEGACY_APPEARANCE);
  if (appearance !== null && cache.ui.appearance === undefined) {
    try {
      ui.appearance = JSON.parse(appearance);
    } catch {
      // Unparseable: drop it rather than carry a broken value forward.
    }
  }
  const language = store.getItem(LEGACY_LANGUAGE);
  if (language !== null && cache.ui.language === undefined) {
    ui.language = language === "en" ? "en" : "zh";
  }
  const closeAction = store.getItem(LEGACY_CLOSE_ACTION);
  if (closeAction !== null && cache.ui.closeWindowAction === undefined) {
    ui.closeWindowAction = closeAction === "close" ? "close" : "minimize";
  }

  const confirms: string[] = [];
  for (let index = 0; index < store.length; index += 1) {
    const key = store.key(index);
    if (key?.startsWith(LEGACY_CONFIRM_PREFIX) && store.getItem(key) === "1") {
      confirms.push(key.slice(LEGACY_CONFIRM_PREFIX.length));
    }
  }
  if (confirms.length > 0 && cache.ui.dismissedConfirms === undefined) {
    ui.dismissedConfirms = confirms;
  }

  const device = store.getItem(LEGACY_DEVICE);
  if (device !== null && cache.task_defaults.device === undefined) {
    if (device === "cpu") {
      defaults.device = "cpu";
    } else {
      try {
        const parsed = JSON.parse(device) as { index?: unknown; name?: unknown };
        const gpuIndex = Number(parsed?.index);
        if (Number.isInteger(gpuIndex) && gpuIndex >= 0) {
          defaults.device = "cuda";
          defaults.gpu_index = gpuIndex;
          defaults.gpu_name = typeof parsed.name === "string" ? parsed.name : "";
        }
      } catch {
        // Same as above.
      }
    }
  }

  const moved = Object.keys(ui).length > 0 || Object.keys(defaults).length > 0;
  if (moved) {
    try {
      // Awaited, unlike every other save: the lines below delete the only
      // other copy of these values, so a failed write must abort the move and
      // leave the old keys where they are for the next start to retry.
      await persist({ ui, task_defaults: defaults });
    } catch {
      return false;
    }
  }
  // Removed whether or not anything moved: a key settings.json already knows
  // about is a leftover, and leaving it would make the next start look like a
  // migration again.
  for (const key of [
    LEGACY_APPEARANCE,
    LEGACY_LANGUAGE,
    LEGACY_CLOSE_ACTION,
    LEGACY_DEVICE,
  ]) {
    store.removeItem(key);
  }
  for (let index = store.length - 1; index >= 0; index -= 1) {
    const key = store.key(index);
    if (key?.startsWith(LEGACY_CONFIRM_PREFIX)) store.removeItem(key);
  }
  return moved;
}

/** Test seam: reset the module between cases. */
export function resetPreferencesCache(): void {
  cache = readMirror();
  listeners.clear();
}
