import assert from "node:assert/strict";
import test from "node:test";

import {
  BridgeCallError,
  createDesktopApi,
  unwrapEnvelope,
} from "../lib/bridge";


test("error envelopes preserve bridge code and action", () => {
  assert.throws(
    () =>
      unwrapEnvelope({
        ok: false,
        error: {
          code: "api_key_required",
          message: "missing",
          action: "open_settings",
        },
      }),
    (error: unknown) =>
      error instanceof BridgeCallError &&
      error.code === "api_key_required" &&
      error.action === "open_settings",
  );
});


test("browser preview fallback is deterministic", async () => {
  const api = createDesktopApi({ preview: true });

  const first = await api.getBootstrapState();
  const second = await api.getBootstrapState();

  assert.deepEqual(first, second);
  assert.equal(first.capabilities.translation, false);
  assert.equal(first.resources[0]?.state, "ready");
});


test("browser preview exposes update check and release page", async () => {
  const api = createDesktopApi({ preview: true });

  const check = await api.checkUpdates();
  const release = await api.openUpdatePage();

  assert.deepEqual(check, { available: false, version: "preview" });
  assert.deepEqual(release, {
    url: "https://github.com/caca2331/finesub/releases",
  });
});


test("desktop API uses the native Python bridge by default", async () => {
  const previousWindow = globalThis.window;
  const calls: string[] = [];
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      location: { search: "" },
      pywebview: {
        api: {
          minimize_window: async () => {
            calls.push("minimize_window");
            return { ok: true, data: null };
          },
        },
      },
    },
  });

  try {
    await createDesktopApi().minimizeWindow();
    assert.deepEqual(calls, ["minimize_window"]);
  } finally {
    if (previousWindow === undefined) {
      Reflect.deleteProperty(globalThis, "window");
    } else {
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: previousWindow,
      });
    }
  }
});


test("desktop API exposes a distinct minimize-to-tray action", async () => {
  const previousWindow = globalThis.window;
  const calls: string[] = [];
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      location: { search: "" },
      pywebview: {
        api: {
          minimize_to_tray: async () => {
            calls.push("minimize_to_tray");
            return { ok: true, data: null };
          },
        },
      },
    },
  });

  try {
    await createDesktopApi().minimizeToTray();
    assert.deepEqual(calls, ["minimize_to_tray"]);
  } finally {
    if (previousWindow === undefined) {
      Reflect.deleteProperty(globalThis, "window");
    } else {
      Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: previousWindow,
      });
    }
  }
});
