import type {
  BootstrapState,
  GpuSnapshot,
  BridgeError,
  CapabilityState,
  JobSnapshot,
  PipelineStage,
  PublicSettings,
  ResourceInstallSnapshot,
  ResourceStatus,
  Route,
  SharedSettings,
  TaskRequest,
  WorkerEvent,
} from "./types";


export type TaskPhase =
  | "empty"
  | "ready"
  | "checking"
  | "downloading"
  | "running"
  | "completed"
  | "failed";

export interface TaskState {
  phase: TaskPhase;
  selectedFile: string | null;
  request: Omit<TaskRequest, "input">;
  taskId: string | null;
  currentStage: PipelineStage | null;
  /** Stages that were satisfied by an existing artifact instead of running. */
  reusedStages: PipelineStage[];
  statusMessage: string;
  logs: string[];
  outputs: Record<string, string>;
  error: BridgeError | null;
  startedAt: number | null;
}

export interface AppState {
  route: Route;
  bootstrapped: boolean;
  appVersion: string;
  resources: ResourceStatus[];
  resourceInstalls: ResourceInstallSnapshot[];
  history: JobSnapshot[];
  capabilities: CapabilityState;
  settings: PublicSettings;
  /** Remembered task options; what a new task starts from. */
  taskDefaults: Partial<TaskRequest>;
  /** The config.toml slice the panel can write, and where that file is. */
  sharedSettings: SharedSettings;
  configPath: string;
  /** Undefined until the first bootstrap poll answers. */
  gpus?: GpuSnapshot;
  task: TaskState;
}

export type AppAction =
  | { type: "bootstrapLoaded"; payload: BootstrapState }
  | { type: "navigate"; route: Route }
  | { type: "fileSelected"; path: string }
  | { type: "reuseAsr"; snapshot: JobSnapshot }
  | {
      type: "requestChanged";
      changes: Partial<Omit<TaskRequest, "input">>;
    }
  | { type: "taskChecking" }
  | { type: "taskStarted"; snapshot: JobSnapshot }
  | { type: "taskRejected"; error: BridgeError }
  | { type: "workerEvent"; event: WorkerEvent }
  | { type: "resourceChanged"; resource: ResourceStatus }
  | { type: "resourceInstallChanged"; install: ResourceInstallSnapshot }
  | { type: "resourceInstallsChanged"; installs: ResourceInstallSnapshot[] }
  | { type: "settingsChanged"; settings: PublicSettings }
  | {
      type: "sharedSettingsChanged";
      settings: SharedSettings;
      configPath?: string;
    }
  | { type: "resetTask" };


/**
 * Task options carried from one task to the next -- the "how", not the "what".
 * The input, the output name and the per-task notes are content; the device is
 * remembered too, but by processingDevice, which writes the fields it becomes.
 */
export const REMEMBERED_TASK_FIELDS = [
  "stage",
  "model_name",
  "language",
  "gpu_tier",
  "word",
  "asr_stabilize_profile",
  "llm_media",
  "llm_retrieval",
  "llm_difficulty",
  "llm_fast",
  "llm_output_scale",
  "knowledge",
  "postprocess_profile",
  "cleanup_intermediate",
] as const;

/**
 * Only the keys that carry a value.
 *
 * A remembered option that is absent must stay absent: spread over a request,
 * an explicit `null` overwrites a real default and the backend rejects the
 * task. The backend serializes sparsely for the same reason; this is the belt
 * to that suspenders, and it also covers a hand-edited settings.json.
 */
function definedOnly(values: Partial<TaskRequest> | undefined): Partial<TaskRequest> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(values ?? {})) {
    if (value !== null && value !== undefined) out[key] = value;
  }
  return out as Partial<TaskRequest>;
}

const defaultRequest: Omit<TaskRequest, "input"> = {
  output: null,
  name: "",
  cleanup_intermediate: false,
  stage: "raw-srt",
  model_name: "large-v3-turbo",
  // Not "cuda": see TaskRequest.device. "Automatic" is the absence of a
  // choice, and `writeProcessingDevice` already stores it as null.
  device: null,
  gpu_index: null,
  language: null,
  gpu_tier: "auto",
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
};


const emptyTask = (defaults: Partial<TaskRequest> = {}): TaskState => ({
  phase: "empty",
  selectedFile: null,
  request: { ...defaultRequest, ...defaults },
  taskId: null,
  currentStage: null,
  reusedStages: [],
  statusMessage: "",
  logs: [],
  outputs: {},
  error: null,
  startedAt: null,
});


function restoreRunningTask(
  snapshot: JobSnapshot | null,
  fallback: TaskState,
): TaskState {
  const taskId = snapshot?.task_id ?? snapshot?.taskId ?? null;
  if (snapshot?.state !== "running" || !snapshot.request || !taskId) {
    return fallback;
  }
  const { input, ...request } = snapshot.request;
  let currentStage: PipelineStage | null = null;
  let statusMessage = "";
  // Stages the pipeline entered without doing work, because their output was
  // already on disk. Worth keeping apart: on a rerun most of the list is this,
  // and showing it as freshly finished work is a lie about what just happened.
  const reusedStages: PipelineStage[] = [];
  for (const event of snapshot.events ?? []) {
    if (event.type !== "stage") {
      continue;
    }
    if (typeof event.payload.stage === "string") {
      currentStage = event.payload.stage as PipelineStage;
      if (event.payload.reused === true && !reusedStages.includes(currentStage)) {
        reusedStages.push(currentStage);
      }
    }
    if (typeof event.payload.message === "string") {
      statusMessage = event.payload.message;
    }
  }
  const logs = (snapshot.events ?? [])
    .filter((event) => event.type === "log")
    .map((event) => String(event.payload.message ?? ""))
    .filter(Boolean)
    .slice(-200);
  return {
    phase: "running",
    selectedFile: input,
    request,
    taskId,
    currentStage,
    reusedStages,
    statusMessage,
    logs,
    outputs: snapshot.outputs ?? {},
    error: null,
    startedAt: (snapshot.created_at ?? Date.now() / 1000) * 1000,
  };
}


export const initialState: AppState = {
  route: "new-task",
  bootstrapped: false,
  appVersion: "development",
  resources: [],
  resourceInstalls: [],
  history: [],
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
  taskDefaults: {},
  sharedSettings: { split_length_scale: null },
  configPath: "",
  task: emptyTask(),
};


export function reduceAppState(
  state: AppState,
  action: AppAction,
): AppState {
  switch (action.type) {
    case "bootstrapLoaded": {
      const remembered = definedOnly(action.payload.preferences?.task_defaults);
      return {
        ...state,
        bootstrapped: true,
        appVersion: action.payload.app_version,
        resources: action.payload.resources,
        resourceInstalls: action.payload.resource_installs ?? [],
        history: action.payload.tasks ?? [],
        capabilities: action.payload.capabilities,
        settings: action.payload.settings,
        // Remembered task options seed the form, but never a task that is
        // already running or finished: those carry the request they ran with.
        task: restoreRunningTask(
          action.payload.task,
          state.task.phase === "empty"
            ? {
                ...state.task,
                request: { ...state.task.request, ...remembered },
              }
            : state.task,
        ),
        taskDefaults: remembered,
        sharedSettings: action.payload.shared_settings ?? state.sharedSettings,
        configPath: action.payload.config_path ?? state.configPath,
        gpus: action.payload.gpus ?? state.gpus,
      };
    }
    case "sharedSettingsChanged":
      return {
        ...state,
        sharedSettings: action.settings,
        configPath: action.configPath ?? state.configPath,
      };
    case "navigate":
      return { ...state, route: action.route };
    case "fileSelected":
      return {
        ...state,
        route: "new-task",
        task: {
          ...state.task,
          phase: "ready",
          selectedFile: action.path,
          // A pinned output belongs to the previous file: a cancelled task
          // leaves its directory here (so restarting it is cheap), and a new
          // file must not write into it.
          request: { ...state.task.request, output: null },
          error: null,
          outputs: {},
          logs: [],
        },
      };
    case "reuseAsr": {
      // Point the new task at the finished recognition run's directory: the
      // pipeline skips every stage whose artifact already exists there, so
      // only the LLM pass actually runs. The old run's extra context rides
      // along unless the form already has its own.
      const previous = action.snapshot.request;
      if (!previous?.output) {
        return state;
      }
      return {
        ...state,
        route: "new-task",
        task: {
          ...state.task,
          phase: "ready",
          selectedFile: previous.input,
          request: {
            ...state.task.request,
            stage: "final-srt",
            output: previous.output,
            extra_info:
              state.task.request.extra_info || previous.extra_info || "",
          },
          error: null,
          outputs: {},
          logs: [],
        },
      };
    }
    case "requestChanged": {
      const remembered: Record<string, unknown> = { ...state.taskDefaults };
      for (const field of REMEMBERED_TASK_FIELDS) {
        if (!(field in action.changes)) continue;
        const value = action.changes[field];
        // Kept in step with what the page persists, so "new task" starts from
        // this session's choices and not from what bootstrap happened to read.
        if (value === null || value === undefined) delete remembered[field];
        else remembered[field] = value;
      }
      return {
        ...state,
        taskDefaults: remembered as Partial<TaskRequest>,
        task: {
          ...state.task,
          request: { ...state.task.request, ...action.changes },
          error: null,
        },
      };
    }
    case "taskChecking":
      return {
        ...state,
        task: {
          ...state.task,
          phase: "checking",
          error: null,
          statusMessage: "正在检查运行环境",
        },
      };
    case "taskStarted": {
      const snapshotRequest = action.snapshot.request;
      const snapshotTaskId =
        action.snapshot.task_id ?? action.snapshot.taskId ?? null;
      const request = snapshotRequest
        ? (({ input: _input, ...settings }) => settings)(snapshotRequest)
        : state.task.request;
      return {
        ...state,
        route: "new-task",
        history: [
          action.snapshot,
          ...state.history.filter(
            (task) =>
              (task.task_id ?? task.taskId) !== snapshotTaskId,
          ),
        ],
        task: {
          ...state.task,
          phase: "running",
          selectedFile: snapshotRequest?.input ?? state.task.selectedFile,
          request,
          taskId: snapshotTaskId,
          startedAt: (action.snapshot.created_at ?? Date.now() / 1000) * 1000,
          error: null,
          logs: [],
          outputs: {},
        },
      };
    }
    case "taskRejected": {
      const needsSettings =
        action.error.code === "api_key_required" ||
        action.error.action === "open_settings";
      const needsResources =
        action.error.code === "runtime_required" ||
        action.error.action === "open_resources";
      const route = needsSettings
        ? "settings"
        : needsResources
          ? "resources"
          : state.route;
      // A rejection that says a task is already running is *about* the live
      // task, not a failure of it. Rewriting the phase here tore down the
      // event poller, dropped the processing view and left the history row
      // stuck at "处理中" forever, while the run itself carried on in the
      // backend -- recoverable only by restarting the app. Reached by doing
      // anything to a history row (retry, cancel) while a task is running.
      if (state.task.phase === "running") {
        return {
          ...state,
          route,
          task: { ...state.task, error: action.error },
        };
      }
      return {
        ...state,
        route,
        task: {
          ...state.task,
          phase: state.task.selectedFile ? "ready" : "empty",
          error: action.error,
          statusMessage: "",
        },
      };
    }
    case "workerEvent":
      return applyWorkerEvent(state, action.event);
    case "resourceChanged":
      return {
        ...state,
        resources: state.resources.map((resource) =>
          resource.id === action.resource.id ? action.resource : resource,
        ),
      };
    case "resourceInstallChanged": {
      const exists = state.resourceInstalls.some(
        (install) => install.resource_id === action.install.resource_id,
      );
      return {
        ...state,
        resourceInstalls: exists
          ? state.resourceInstalls.map((install) =>
              install.resource_id === action.install.resource_id
                ? action.install
                : install,
            )
          : [...state.resourceInstalls, action.install],
        resources: applyResourceInstallToResources(
          state.resources,
          action.install,
        ),
      };
    }
    case "resourceInstallsChanged":
      return {
        ...state,
        resourceInstalls: action.installs,
        resources: action.installs.reduce(
          (resources, install) =>
            applyResourceInstallToResources(resources, install),
          state.resources,
        ),
      };
    case "settingsChanged":
      return {
        ...state,
        settings: action.settings,
        capabilities: {
          ...state.capabilities,
          translation: action.settings.api_keys.gemini === "configured",
          web_search:
            action.settings.api_keys.exa === "configured" ||
            action.settings.api_keys.tavily === "configured",
        },
      };
    case "resetTask":
      return { ...state, route: "new-task", task: emptyTask(state.taskDefaults) };
    default:
      return state;
  }
}

function applyResourceInstallToResources(
  resources: ResourceStatus[],
  install: ResourceInstallSnapshot,
): ResourceStatus[] {
  return resources.map((resource) => {
    if (resource.id !== install.resource_id) {
      return resource;
    }
    if (install.state === "ready") {
      return { ...resource, state: "ready", detail: "" };
    }
    if (install.state === "failed") {
      return {
        ...resource,
        state: "failed",
        detail: install.error || install.message,
      };
    }
    if (install.state === "paused") {
      return { ...resource, state: "missing", detail: install.message };
    }
    return { ...resource, state: "downloading", detail: install.message };
  });
}


function applyWorkerEvent(state: AppState, event: WorkerEvent): AppState {
  const payload = event.payload;
  if (event.type === "started") {
    return {
      ...state,
      task: { ...state.task, phase: "running", error: null },
    };
  }
  if (event.type === "stage") {
    const stage =
      typeof payload.stage === "string"
        ? (payload.stage as PipelineStage)
        : state.task.currentStage;
    const reused =
      payload.reused === true && stage && !state.task.reusedStages.includes(stage)
        ? [...state.task.reusedStages, stage]
        : state.task.reusedStages;
    return {
      ...state,
      task: {
        ...state.task,
        phase: "running",
        currentStage: stage,
        reusedStages: reused,
        statusMessage:
          typeof payload.message === "string"
            ? payload.message
            : state.task.statusMessage,
      },
    };
  }
  if (event.type === "log") {
    const message =
      typeof payload.message === "string" ? payload.message : String(payload.message ?? "");
    return {
      ...state,
      task: {
        ...state.task,
        logs: [...state.task.logs, message].slice(-200),
      },
    };
  }
  if (event.type === "completed") {
    const nested = payload.outputs;
    const outputs =
      nested && typeof nested === "object" && !Array.isArray(nested)
        ? nested
        : payload;
    const normalizedOutputs = Object.fromEntries(
      Object.entries(outputs).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      ),
    );
    return {
      ...state,
      history: updateHistorySnapshot(state.history, event.task_id, {
        state: "completed",
        outputs: normalizedOutputs,
        error: null,
      }),
      task: {
        ...state.task,
        phase: "completed",
        statusMessage: "字幕处理完成",
        outputs: normalizedOutputs,
        error: null,
      },
    };
  }
  if (event.type === "failed") {
    return {
      ...state,
      history: updateHistorySnapshot(state.history, event.task_id, {
        state: "failed",
        error:
          typeof payload.message === "string"
            ? payload.message
            : "字幕任务失败。",
      }),
      task: {
        ...state.task,
        phase: "failed",
        error: {
          code: "worker_failed",
          message:
            typeof payload.message === "string"
              ? payload.message
              : "字幕任务失败。",
        },
      },
    };
  }
  return {
    ...state,
    history: updateHistorySnapshot(state.history, event.task_id, {
      state: "cancelled",
    }),
    task: {
      ...state.task,
      phase: state.task.selectedFile ? "ready" : "empty",
      statusMessage: "任务已取消",
    },
  };
}


function updateHistorySnapshot(
  history: JobSnapshot[],
  taskId: string,
  changes: Partial<JobSnapshot>,
): JobSnapshot[] {
  return history.map((snapshot) =>
    (snapshot.task_id ?? snapshot.taskId) === taskId
      ? { ...snapshot, ...changes, updated_at: Date.now() / 1000 }
      : snapshot,
  );
}
