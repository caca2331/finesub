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


test("browser preview exposes typed no-update results", async () => {
  const api = createDesktopApi({ preview: true });

  const check = await api.checkUpdates();
  const install = await api.installUpdate("app");

  assert.deepEqual(check, { available: false, version: "preview" });
  assert.deepEqual(install, {
    kind: "app",
    version: "preview",
    restartRequired: false,
  });
});
