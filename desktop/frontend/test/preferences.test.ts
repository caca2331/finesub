import assert from "node:assert/strict";
import test, { beforeEach } from "node:test";

// The module reads localStorage at import time, so the stub has to exist first.
class MemoryStorage {
  private data = new Map<string, string>();
  get length() {
    return this.data.size;
  }
  key(index: number) {
    return [...this.data.keys()][index] ?? null;
  }
  getItem(key: string) {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string) {
    this.data.set(key, String(value));
  }
  removeItem(key: string) {
    this.data.delete(key);
  }
  clear() {
    this.data.clear();
  }
}

const store = new MemoryStorage();
(globalThis as { localStorage?: unknown }).localStorage = store;

// The native bridge the module talks to. The migration awaits it (it deletes
// the old copy afterwards), so it has to answer for real here.
const saved: unknown[] = [];
let bridgeFails = false;
(globalThis as { window?: unknown }).window = {
  pywebview: {
    api: {
      save_preferences: async (patch: unknown) => {
        if (bridgeFails) return { ok: false, error: { code: "x", message: "no" } };
        saved.push(patch);
        return { ok: true, data: { preferences: { ui: {}, task_defaults: {} } } };
      },
    },
  },
};

import {
  hydratePreferences,
  migrateLegacyKeys,
  resetPreferencesCache,
  saveTaskDefaults,
  saveUi,
  taskDefaults,
  uiValue,
} from "../lib/preferences";

beforeEach(() => {
  store.clear();
  saved.length = 0;
  bridgeFails = false;
  resetPreferencesCache();
});


test("a value survives in the mirror for the next synchronous read", () => {
  saveUi({ language: "en" });

  assert.equal(uiValue("language", "zh"), "en");
  assert.match(store.getItem("finesub-preferences") ?? "", /"language":"en"/);
});


test("settings.json wins over the mirror when the bridge answers", () => {
  saveUi({ language: "en" });

  hydratePreferences({ ui: { language: "zh" }, task_defaults: { gpu_budget_gb: 8 } });

  assert.equal(uiValue("language", "zh"), "zh");
  assert.equal(taskDefaults().gpu_budget_gb, 8);
});


test("null clears a setting instead of storing a default", () => {
  saveTaskDefaults({ gpu_budget_gb: 8, device: "cpu" });
  saveTaskDefaults({ device: null });

  assert.equal(taskDefaults().gpu_budget_gb, 8);
  assert.equal("device" in taskDefaults(), false);
});


test("the five legacy keys move once and are then gone", async () => {
  store.setItem("finesub-language", "en");
  store.setItem("finesub-appearance", JSON.stringify({ theme: "dark" }));
  store.setItem("close-window-action", "close");
  store.setItem("finesub-confirm-delete", "1");
  store.setItem("finesub-processing-device", JSON.stringify({ index: 1, name: "RTX" }));

  assert.equal(await migrateLegacyKeys(), true);

  assert.equal(uiValue("language", "zh"), "en");
  assert.deepEqual(uiValue("appearance", {}), { theme: "dark" });
  assert.equal(uiValue("closeWindowAction", "minimize"), "close");
  assert.deepEqual(uiValue("dismissedConfirms", []), ["delete"]);
  assert.equal(taskDefaults().device, "cuda");
  assert.equal(taskDefaults().gpu_index, 1);
  assert.equal(taskDefaults().gpu_name, "RTX");

  for (const key of [
    "finesub-language",
    "finesub-appearance",
    "close-window-action",
    "finesub-confirm-delete",
    "finesub-processing-device",
  ]) {
    assert.equal(store.getItem(key), null, `${key} should be gone`);
  }
  // Nothing left to move, so a second start is not a migration.
  assert.equal(await migrateLegacyKeys(), false);
});


test("a legacy key never overwrites what settings.json already knows", async () => {
  store.setItem("finesub-language", "en");
  hydratePreferences({ ui: { language: "zh" }, task_defaults: {} });

  await migrateLegacyKeys();

  assert.equal(uiValue("language", "zh"), "zh");
  assert.equal(store.getItem("finesub-language"), null);
});


test("a failed durable write leaves the legacy keys for the next start", async () => {
  store.setItem("finesub-language", "en");
  bridgeFails = true;

  assert.equal(await migrateLegacyKeys(), false);

  // The old copy is still the only durable one, so it must survive.
  assert.equal(store.getItem("finesub-language"), "en");
});


test("an unparseable legacy value is dropped rather than carried forward", async () => {
  store.setItem("finesub-appearance", "{not json");

  await migrateLegacyKeys();

  assert.deepEqual(uiValue("appearance", { theme: "system" }), { theme: "system" });
});
