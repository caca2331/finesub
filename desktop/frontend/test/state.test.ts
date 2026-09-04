import assert from "node:assert/strict";
import test from "node:test";

import { initialState, reduceAppState } from "../lib/state";


test("bootstrap restores an active worker task and its progress", () => {
  const next = reduceAppState(initialState, {
    type: "bootstrapLoaded",
    payload: {
      app_version: "0.2.0",
      resources: [],
      resource_installs: [],
      capabilities: {
        raw_srt: true,
        translation: false,
        web_search: false,
      },
      settings: {
        api_keys: {
          gemini: "missing",
          exa: "missing",
          tavily: "missing",
        },
      },
      preferences: { ui: {}, task_defaults: {} },
      shared_settings: { split_length_scale: null },
      config_path: "C:/config.toml",
      task: {
        task_id: "active-task",
        state: "running",
        request: {
          input: "D:/media/active.mp4",
          name: "",
          cleanup_intermediate: false,
          stage: "raw-srt",
          model_name: "large-v3-turbo",
          device: "cuda",
          language: "ja",
          gpu_tier: "entry",
          word: false,
          asr_stabilize_profile: 0,
          llm_media: "video",
          llm_retrieval: "local",
          llm_difficulty: "quality",
          llm_fast: "auto",
          llm_output_scale: 1,
          extra_info: "",
          extra_style: "",
            knowledge: "none",
          postprocess_profile: 0,
        },
        events: [
          {
            type: "stage",
            task_id: "active-task",
            timestamp: "2026-07-25T00:00:00Z",
            payload: {
              stage: "aligned",
              message: "正在识别",
            },
          },
          {
            type: "log",
            task_id: "active-task",
            timestamp: "2026-07-25T00:00:01Z",
            payload: { message: "worker is alive" },
          },
        ],
        created_at: 100,
      },
      tasks: [],
    },
  });

  assert.equal(next.task.phase, "running");
  assert.equal(next.task.taskId, "active-task");
  assert.equal(next.task.selectedFile, "D:/media/active.mp4");
  assert.equal(next.task.currentStage, "aligned");
  assert.equal(next.task.statusMessage, "正在识别");
  assert.deepEqual(next.task.logs, ["worker is alive"]);
});


test("stages reported as reused are remembered as such", () => {
  // A rerun skips whatever is already on disk. Ticking those the same way as
  // work that just happened would tell the user the run redid it.
  const events = [
    { stage: "vocal", reused: true },
    { stage: "aligned", reused: true },
    { stage: "stable", reused: false },
  ];
  const next = events.reduce(
    (state, payload) =>
      reduceAppState(state, {
        type: "workerEvent",
        event: {
          type: "stage",
          task_id: "task-1",
          timestamp: "2026-07-25T00:00:00Z",
          payload: { ...payload, message: "" },
        },
      }),
    initialState,
  );

  assert.deepEqual(next.task.reusedStages, ["vocal", "aligned"]);
  assert.equal(next.task.currentStage, "stable");
});


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

test("new tasks keep intermediates and derive their own name", () => {
  // Both are contract, not cosmetics: cleanup used to be unconditional and
  // deleted stable.json, and a blank name means "derive it from the source".
  const state = reduceAppState(initialState, { type: "resetTask" });

  assert.equal(state.task.request.cleanup_intermediate, false);
  assert.equal(state.task.request.name, "");
});

test("a machine with no completed task is treated as a first run", () => {
  // The notice keys off completion, not on history being empty: a machine
  // whose only previous attempt failed still has no weights cached, and that
  // run is still the slow one.
  const completed = [{ state: "completed" }, { state: "failed" }];
  const failedOnly = [{ state: "failed" }, { state: "cancelled" }];

  const isFirstRun = (history: { state: string }[]) =>
    !history.some((snapshot) => snapshot.state === "completed");

  assert.equal(isFirstRun([]), true);
  assert.equal(isFirstRun(failedOnly), true);
  assert.equal(isFirstRun(completed), false);
});


test("reusing a recognition run pins its directory and switches to final-srt", () => {
  const next = reduceAppState(initialState, {
    type: "reuseAsr",
    snapshot: {
      task_id: "old-run",
      state: "completed",
      events: [],
      request: {
        input: "D:/media/a.mp4",
        output: "C:/tasks/old-run/a.srt",
        name: "",
        cleanup_intermediate: false,
        stage: "raw-srt",
        model_name: "large-v3-turbo",
        device: "cuda",
        language: null,
        gpu_tier: "entry",
        word: false,
        asr_stabilize_profile: 0,
        llm_media: "video",
        llm_retrieval: "local",
        llm_difficulty: "quality",
        llm_fast: "auto",
        llm_output_scale: 1,
        extra_info: "出自某次直播",
        extra_style: "",
        knowledge: "update",
        postprocess_profile: 0,
      },
    },
  });

  assert.equal(next.route, "new-task");
  assert.equal(next.task.selectedFile, "D:/media/a.mp4");
  assert.equal(next.task.request.stage, "final-srt");
  assert.equal(next.task.request.output, "C:/tasks/old-run/a.srt");
  // The old run's context rides along when the form has none of its own.
  assert.equal(next.task.request.extra_info, "出自某次直播");
});


test("selecting a file clears a pinned output directory", () => {
  // A cancelled task leaves its directory pinned so restarting it is cheap;
  // a different file must not inherit it and write into that directory.
  const pinned = reduceAppState(initialState, {
    type: "requestChanged",
    changes: { output: "C:/tasks/old-run/a.srt" },
  });

  const next = reduceAppState(pinned, {
    type: "fileSelected",
    path: "D:/media/b.mp4",
  });

  assert.equal(next.task.request.output, null);
  assert.equal(next.task.selectedFile, "D:/media/b.mp4");
});


test("a rejection about the running task must not tear the running task down", () => {
  // Reached by retrying or cancelling a *history* row while a task runs: the
  // backend refuses with task_already_running, and the reducer used to rewrite
  // the live task's phase. That tore down the event poller, dropped the
  // processing view, and left the history row stuck at "处理中" forever while
  // the run carried on in the backend.
  const selected = reduceAppState(initialState, {
    type: "fileSelected",
    path: "D:/media/a.mp4",
  });
  const running = reduceAppState(selected, {
    type: "taskStarted",
    snapshot: {
      task_id: "demo-260808-1200-abcdef",
      state: "running",
      request: {
        input: "D:/media/a.mp4",
        output: null,
        name: "",
        cleanup_intermediate: false,
        stage: "raw-srt",
        model_name: "large-v3-turbo",
        device: "cuda",
        gpu_index: null,
        language: null,
        gpu_tier: "entry",
        word: false,
        asr_stabilize_profile: 0,
        llm_media: "video",
        llm_retrieval: "local",
        llm_difficulty: "quality",
        llm_fast: "auto",
        llm_output_scale: 1,
        extra_info: "",
        extra_style: "",
        knowledge: "update",
        postprocess_profile: 0,
      },
      events: [],
      outputs: {},
      created_at: 1,
      updated_at: 1,
    },
  });
  assert.equal(running.task.phase, "running");

  const next = reduceAppState(running, {
    type: "taskRejected",
    error: {
      code: "task_already_running",
      message: "已有字幕任务正在运行。",
      action: "show_current_task",
    },
  });

  assert.equal(next.task.phase, "running", "the live task survives");
  assert.equal(next.task.taskId, "demo-260808-1200-abcdef");
  assert.equal(next.task.error?.code, "task_already_running");
});


test("remembered options survive starting a new task", () => {
  // The reducer used to rebuild an empty task from the hard-coded defaults, so
  // finishing a task and clicking "new task" threw away the budget/language the
  // user had just picked -- until the app was restarted.
  const bootstrapped = reduceAppState(initialState, {
    type: "bootstrapLoaded",
    payload: {
      app_version: "0.2.0",
      resources: [],
      resource_installs: [],
      capabilities: { raw_srt: true, translation: false, web_search: false },
      settings: {
        api_keys: { gemini: "missing", exa: "missing", tavily: "missing" },
      },
      preferences: { ui: {}, task_defaults: { gpu_tier: "standard" } },
      shared_settings: { split_length_scale: null },
      config_path: "C:/config.toml",
      task: null,
      tasks: [],
    },
  });
  assert.equal(bootstrapped.task.request.gpu_tier, "standard");

  const changed = reduceAppState(bootstrapped, {
    type: "requestChanged",
    changes: { language: "ja", model_name: "large-v3" },
  });
  const reset = reduceAppState(changed, { type: "resetTask" });

  assert.equal(reset.task.request.gpu_tier, "standard");
  assert.equal(reset.task.request.language, "ja");
  assert.equal(reset.task.request.model_name, "large-v3");
  // Content, not "how": a new task starts clean.
  assert.equal(reset.task.selectedFile, null);
});


test("a null in stored defaults never overwrites a real default", () => {
  // A hand-edited settings.json (or an older backend) can carry explicit
  // nulls; spreading one over the request makes every task fail validation.
  const next = reduceAppState(initialState, {
    type: "bootstrapLoaded",
    payload: {
      app_version: "0.2.0",
      resources: [],
      resource_installs: [],
      capabilities: { raw_srt: true, translation: false, web_search: false },
      settings: {
        api_keys: { gemini: "missing", exa: "missing", tavily: "missing" },
      },
      preferences: {
        ui: {},
        task_defaults: { model_name: null, stage: null, gpu_tier: "high" },
      } as never,
      shared_settings: { split_length_scale: null },
      config_path: "C:/config.toml",
      task: null,
      tasks: [],
    },
  });

  assert.equal(next.task.request.model_name, "large-v3-turbo");
  assert.equal(next.task.request.stage, "raw-srt");
  assert.equal(next.task.request.gpu_tier, "high");
});
