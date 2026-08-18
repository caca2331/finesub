import assert from "node:assert/strict";
import test from "node:test";

import {
  blockingResources,
  isEnvironmentReady,
  unresolvedDependency,
} from "../lib/resources";


test("optional on-demand tools never block a task", () => {
  // yt-dlp has no system-tool finder by design, so it is permanently "missing"
  // on a healthy machine. Gating on it meant no task could ever start.
  const resources = [
    { id: "uv", version: "1", state: "ready" as const },
    { id: "ffmpeg", version: "1", state: "ready" as const },
    { id: "git", version: "1", state: "missing" as const, optional: true },
    { id: "yt-dlp", version: "1", state: "missing" as const, optional: true },
  ];

  assert.deepEqual(blockingResources(resources), []);
  assert.equal(isEnvironmentReady(resources), true);
});


test("a missing required resource still blocks", () => {
  const resources = [
    { id: "uv", version: "1", state: "missing" as const },
    { id: "git", version: "1", state: "missing" as const, optional: true },
  ];

  assert.deepEqual(
    blockingResources(resources).map((resource) => resource.id),
    ["uv"],
  );
  assert.equal(isEnvironmentReady(resources), false);
});


test("installing a dependency immediately unblocks its resource", () => {
  const models = {
    id: "models",
    version: "on-demand",
    state: "missing" as const,
    optional: true,
    blocked_by: "uv",
  };
  const before = [
    { id: "uv", version: "1", state: "missing" as const },
    models,
  ];
  const after = [
    { id: "uv", version: "1", state: "ready" as const },
    models,
  ];

  assert.equal(unresolvedDependency(models, before), "uv");
  assert.equal(unresolvedDependency(models, after), "");
});
