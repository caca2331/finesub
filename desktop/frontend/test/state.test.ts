import assert from "node:assert/strict";
import test from "node:test";

import { initialState, reduceAppState } from "../lib/state";


test("api_key_required keeps the task editable and opens settings", () => {
  const selected = reduceAppState(initialState, {
    type: "fileSelected",
    path: "D:/media/a.mp4",
  });

  const next = reduceAppState(selected, {
    type: "taskRejected",
    error: {
      code: "api_key_required",
      message: "翻译需要 API Key",
      action: "open_settings",
    },
  });

  assert.equal(next.task.phase, "ready");
  assert.equal(next.route, "settings");
  assert.equal(next.task.selectedFile, "D:/media/a.mp4");
});


test("completed event exposes subtitle actions", () => {
  const next = reduceAppState(initialState, {
    type: "workerEvent",
    event: {
      type: "completed",
      task_id: "task-1",
      timestamp: "2026-07-25T00:00:00Z",
      payload: { rawSrt: "D:/out/a-raw.srt" },
    },
  });

  assert.equal(next.task.phase, "completed");
  assert.equal(next.task.outputs.rawSrt, "D:/out/a-raw.srt");
});


test("runtime_required keeps task editable and opens resources", () => {
  const selected = reduceAppState(initialState, {
    type: "fileSelected",
    path: "D:/media/a.mp4",
  });

  const next = reduceAppState(selected, {
    type: "taskRejected",
    error: {
      code: "runtime_required",
      message: "请安装运行环境",
      action: "open_resources",
    },
  });

  assert.equal(next.task.phase, "ready");
  assert.equal(next.route, "resources");
});


test("log history stays bounded", () => {
  let state = initialState;
  for (let index = 0; index < 220; index += 1) {
    state = reduceAppState(state, {
      type: "workerEvent",
      event: {
        type: "log",
        task_id: "task-1",
        timestamp: "2026-07-25T00:00:00Z",
        payload: { message: `line-${index}` },
      },
    });
  }

  assert.equal(state.task.logs.length, 200);
  assert.equal(state.task.logs[0], "line-20");
});


test("resource install snapshots keep progress and failure visible", () => {
  const bootstrapped = {
    ...initialState,
    resources: [
      { id: "ffmpeg", version: "1", state: "missing" as const },
    ],
  };
  const baseInstall = {
    resource_id: "ffmpeg",
    resource_version: "1",
    state: "running" as const,
    phase: "downloading" as const,
    message: "正在下载资源",
    downloaded: 25,
    total: 100,
    bytes_per_second: 10,
    cache_path: "C:/FineSub/cache/downloads/ffmpeg.zip",
    install_path: "C:/FineSub/runtime/ffmpeg/1",
    logs: [],
    error: "",
    started_at: 1,
    updated_at: 2,
  };

  const downloading = reduceAppState(bootstrapped, {
    type: "resourceInstallChanged",
    install: baseInstall,
  });
  assert.equal(downloading.resourceInstalls[0]?.downloaded, 25);
  assert.equal(downloading.resources[0]?.state, "downloading");

  const failed = reduceAppState(downloading, {
    type: "resourceInstallChanged",
    install: {
      ...baseInstall,
      state: "failed",
      message: "资源安装失败",
      error: "连接超时",
    },
  });
  assert.equal(failed.resources[0]?.state, "failed");
  assert.equal(failed.resources[0]?.detail, "连接超时");
});
