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


test("installUpdate forwards kind and version to the native bridge", async () => {
  const previousWindow = globalThis.window;
  const calls: unknown[][] = [];
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      location: { search: "" },
      pywebview: {
        api: {
          install_update: async (...args: unknown[]) => {
            calls.push(args);
            return {
              ok: true,
              data: {
                version: "0.3.2",
                kind: "app",
                state: "queued",
                phase: "waiting",
                message: "",
                downloaded: 0,
                total: 0,
                bytes_per_second: 0,
                restart_required: false,
                exit_required: false,
                error: "",
                started_at: 0,
                updated_at: 0,
              },
            };
          },
          get_update_install: async () => ({ ok: true, data: null }),
        },
      },
    },
  });

  try {
    const api = createDesktopApi();
    const snapshot = await api.installUpdate("app", "0.3.2");
    const polled = await api.getUpdateInstall();

    // The backend re-derives the kind from the signed manifest and rejects a
    // mismatch, so what the page saw has to reach it verbatim.
    assert.deepEqual(calls, [["app", "0.3.2"]]);
    assert.equal(snapshot.state, "queued");
    assert.equal(polled, null);
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


test("browser preview refuses to install rather than pretending to", async () => {
  const api = createDesktopApi({ preview: true });

  assert.equal(await api.getUpdateInstall(), null);
  await assert.rejects(() => api.installUpdate("app", "0.3.2"));
});
