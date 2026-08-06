import type {
  BootstrapState,
  BridgeError,
  CapabilityState,
  JobSnapshot,
  PipelineStage,
  PublicSettings,
  ResourceInstallSnapshot,
  ResourceStatus,
  Route,
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
  task: TaskState;
}

export type AppAction =
  | { type: "bootstrapLoaded"; payload: BootstrapState }
  | { type: "navigate"; route: Route }
  | { type: "fileSelected"; path: string }
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
  | { type: "resetTask" };


const defaultRequest: Omit<TaskRequest, "input"> = {
  output: null,
  name: "",
  cleanup_intermediate: false,
  stage: "raw-srt",
  model_name: "large-v3-turbo",
  device: "cuda",
  language: null,
  gpu_budget_gb: 4,
  word: false,
  asr_stabilize_profile: 0,
  llm_route: "mm",
  llm_level: "high",
  llm_fast: "auto",
  llm_output_scale: 1,
  extra_info: "",
  extra_style: "",
  enable_web_search: true,
  knowledge: "update",
  postprocess_profile: 0,
};


const emptyTask = (): TaskState => ({
  phase: "empty",
  selectedFile: null,
  request: { ...defaultRequest },
  taskId: null,
  currentStage: null,
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
  for (const event of snapshot.events ?? []) {
    if (event.type !== "stage") {
      continue;
    }
    if (typeof event.payload.stage === "string") {
      currentStage = event.payload.stage as PipelineStage;
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
  task: emptyTask(),
};


export function reduceAppState(
  state: AppState,
  action: AppAction,
): AppState {
  switch (action.type) {
    case "bootstrapLoaded":
      return {
        ...state,
        bootstrapped: true,
        appVersion: action.payload.app_version,
        resources: action.payload.resources,
        resourceInstalls: action.payload.resource_installs ?? [],
        history: action.payload.tasks ?? [],
        capabilities: action.payload.capabilities,
        settings: action.payload.settings,
        task: restoreRunningTask(action.payload.task, state.task),
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
          error: null,
          outputs: {},
          logs: [],
        },
      };
    case "requestChanged":
      return {
        ...state,
        task: {
          ...state.task,
          request: { ...state.task.request, ...action.changes },
          error: null,
        },
      };
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
      return {
        ...state,
        route: needsSettings
          ? "settings"
          : needsResources
            ? "resources"
            : state.route,
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
      return { ...state, route: "new-task", task: emptyTask() };
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
    return {
      ...state,
      task: {
        ...state.task,
        phase: "running",
        currentStage:
          typeof payload.stage === "string"
            ? (payload.stage as PipelineStage)
            : state.task.currentStage,
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
