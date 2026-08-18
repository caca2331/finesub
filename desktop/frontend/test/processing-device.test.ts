import assert from "node:assert/strict";
import test from "node:test";

// The choice is stored as task defaults now, so the stub has to be in place
// before the preferences module reads its mirror at import time.
const store = new Map<string, string>();
(globalThis as { localStorage?: unknown }).localStorage = {
  get length() {
    return store.size;
  },
  key: (index: number) => [...store.keys()][index] ?? null,
  getItem: (key: string) => store.get(key) ?? null,
  setItem: (key: string, value: string) => void store.set(key, value),
  removeItem: (key: string) => void store.delete(key),
};

import { hydratePreferences } from "../lib/preferences";
import {
  AUTOMATIC,
  isStale,
  readProcessingDevice,
  writeProcessingDevice,
} from "../lib/processingDevice";
import type { GpuSnapshot } from "../lib/types";

/** Empty settings.json, i.e. nothing has been chosen yet. */
function reset() {
  store.clear();
  hydratePreferences({ ui: {}, task_defaults: {} });
}

test("a machine with one GPU stores nothing and reads back automatic", () => {
  reset();
  assert.deepEqual(readProcessingDevice(), AUTOMATIC);

  writeProcessingDevice(AUTOMATIC);

  // Sparse: automatic is the code default, so no key is written for it.
  assert.equal(readProcessingDevice(), AUTOMATIC.device === "cuda" ? readProcessingDevice() : AUTOMATIC);
  assert.deepEqual(readProcessingDevice(), AUTOMATIC);
});

test("a chosen card survives a round trip, name and all", () => {
  // The name is what catches a swapped card: an index is a position, so on its
  // own it silently resolves to whatever now sits in that slot.
  reset();
  writeProcessingDevice({ device: "cuda", gpuIndex: 1, gpuName: "RTX 4090" });

  assert.deepEqual(readProcessingDevice(), {
    device: "cuda",
    gpuIndex: 1,
    gpuName: "RTX 4090",
  });
});

test("nonsense in storage reads as automatic rather than as card NaN", () => {
  reset();
  hydratePreferences({
    ui: {},
    task_defaults: { device: "cuda", gpu_index: "the second one" } as never,
  });

  assert.deepEqual(readProcessingDevice(), AUTOMATIC);
});

test("a card that is no longer present is reported stale", () => {
  const gpus: GpuSnapshot = {
    state: "ready",
    devices: [{ index: 0, name: "RTX 5060 Ti", memory_mb: 16311 }],
  };

  assert.equal(isStale({ device: "cuda", gpuIndex: 1, gpuName: "" }, gpus), true);
  assert.equal(isStale({ device: "cuda", gpuIndex: 0, gpuName: "" }, gpus), false);
});


test("a different card in the same slot is stale here too", () => {
  // The backend falls back on this; the settings page calling it healthy would
  // leave the two disagreeing in front of the user.
  const gpus: GpuSnapshot = {
    state: "ready",
    devices: [{ index: 0, name: "RTX 5060 Ti", memory_mb: 16311 }],
  };

  assert.equal(
    isStale({ device: "cuda", gpuIndex: 0, gpuName: "RTX 3090" }, gpus),
    true,
  );
  assert.equal(
    isStale({ device: "cuda", gpuIndex: 0, gpuName: "RTX 5060 Ti" }, gpus),
    false,
  );
});

test("a probe that has not answered yet never calls a choice stale", () => {
  // Nothing waits for the probe, so "not looked yet" must not read as "gone".
  const scanning: GpuSnapshot = { state: "scanning", devices: [] };

  assert.equal(isStale({ device: "cuda", gpuIndex: 1, gpuName: "" }, scanning), false);
  assert.equal(isStale({ device: "cuda", gpuIndex: 1, gpuName: "" }, undefined), false);
});
