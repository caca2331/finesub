import type {
  ApiEnvelope,
  BootstrapState,
  DesktopApi,
  JobSnapshot,
  PublicSettings,
  ResourceInstallSnapshot,
  TaskRequest,
} from "./types";


type BridgeMethod = (...args: unknown[]) => Promise<ApiEnvelope<unknown>>;
type NativeBridge = Record<string, BridgeMethod>;

declare global {
  interface Window {
    pywebview?: {
      api?: NativeBridge;
    };
  }
}


export class BridgeCallError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly action?: string | null,
  ) {
    super(message);
    this.name = "BridgeCallError";
  }
}


export function unwrapEnvelope<T>(envelope: ApiEnvelope<T>): T {
  if (envelope.ok) {
    return envelope.data;
  }
  throw new BridgeCallError(
    envelope.error.code,
    envelope.error.message,
    envelope.error.action,
  );
}


const previewBootstrap: BootstrapState = {
  resources: [
    { id: "uv", version: "0.11.32", state: "ready" },
    { id: "ffmpeg", version: "N-125752", state: "ready" },
  ],
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
  task: null,
  tasks: [],
};


function previewApi(): DesktopApi {
  let settings = structuredClone(previewBootstrap.settings);
  const installs = new Map<string, ResourceInstallSnapshot>();
  return {
    async getBootstrapState() {
      return structuredClone({ ...previewBootstrap, settings });
    },
    async selectInputFile() {
      return { path: "D:/Media/示例视频.mp4" };
    },
    async startTask(request) {
      return {
        task_id: "preview-task",
        state: "running",
        request: {
          ...requestDefaults,
          ...request,
        } as TaskRequest,
        events: [],
        outputs: {},
      };
    },
    async cancelTask(taskId) {
      return {
        task_id: taskId,
        state: "cancelled",
        events: [],
        outputs: {},
      };
    },
    async retryTask(taskId) {
      return {
        task_id: `${taskId}-retry`,
        state: "running",
        request: { input: "D:/Media/示例视频.mp4", ...requestDefaults },
        events: [],
        outputs: {},
      };
    },
    async resumeTask(taskId) {
      return {
        task_id: taskId,
        state: "running",
        request: { input: "D:/Media/示例视频.mp4", ...requestDefaults },
        events: [],
        outputs: {},
      };
    },
    async getTaskSnapshot() {
      return null;
    },
    async listTasks() {
      return [];
    },
    async pollEvents(cursor) {
      return { events: [], nextCursor: cursor };
    },
    async installResource(resourceId) {
      const now = Date.now() / 1000;
      const snapshot: ResourceInstallSnapshot = {
        resource_id: resourceId,
        resource_version: "preview",
        state: "running",
        phase: "downloading",
        message: "正在下载资源",
        downloaded: 42_000_000,
        total: 100_000_000,
        bytes_per_second: 3_200_000,
        cache_path: "C:\\FineSub Desktop\\cache\\downloads",
        install_path: `C:\\FineSub Desktop\\runtime\\${resourceId}`,
        logs: [],
        error: "",
        started_at: now,
        updated_at: now,
      };
      installs.set(resourceId, snapshot);
      return structuredClone(snapshot);
    },
    async getResourceInstall(resourceId) {
      return structuredClone(installs.get(resourceId) ?? null);
    },
    async listResourceInstalls() {
      return structuredClone([...installs.values()]);
    },
    async pauseResourceInstall(resourceId) {
      const current = installs.get(resourceId);
      if (!current) {
        throw new BridgeCallError("not_found", "没有找到资源下载任务。");
      }
      const paused = {
        ...current,
        state: "paused" as const,
        message: "已暂停，已下载内容会保留",
      };
      installs.set(resourceId, paused);
      return structuredClone(paused);
    },
    async openResourceLocation(resourceId, kind) {
      return {
        path:
          kind === "cache"
            ? "C:\\FineSub Desktop\\cache\\downloads"
            : `C:\\FineSub Desktop\\runtime\\${resourceId}`,
      };
    },
    async saveApiKeys(keys) {
      settings = {
        api_keys: {
          gemini: keys.gemini?.trim() ? "configured" : settings.api_keys.gemini,
          exa: keys.exa?.trim() ? "configured" : settings.api_keys.exa,
          tavily: keys.tavily?.trim() ? "configured" : settings.api_keys.tavily,
        },
      };
      return structuredClone(settings);
    },
    async deleteApiKey(provider) {
      settings = {
        api_keys: { ...settings.api_keys, [provider]: "missing" },
      };
      return structuredClone(settings);
    },
    async checkUpdates() {
      return { available: false, version: "preview" };
    },
    async openUpdatePage() {
      return { url: "https://github.com/caca2331/finesub/releases" };
    },
    async openOutput(path) {
      return { path };
    },
    async minimizeWindow() {},
    async maximizeWindow() {},
    async closeWindow() {},
  };
}


const requestDefaults: Omit<TaskRequest, "input"> = {
  output: null,
  stage: "raw-srt",
  model_name: "large-v3-turbo",
  device: "cuda",
  language: null,
  gpu_budget_gb: 8,
  word: false,
  asr_stabilize_profile: 0,
  llm_route: "mm",
  llm_level: "med",
  llm_fast: "auto",
  llm_output_scale: 1,
  extra_info: "",
  extra_style: "",
  enable_web_search: true,
  knowledge: "none",
  postprocess_profile: 0,
};


async function waitForNativeBridge(): Promise<NativeBridge> {
  if (typeof window === "undefined") {
    throw new BridgeCallError("bridge_unavailable", "桌面桥接不可用。");
  }
  if (window.pywebview?.api) {
    return window.pywebview.api;
  }
  await new Promise<void>((resolve, reject) => {
    const timeout = window.setTimeout(
      () => reject(new BridgeCallError("bridge_timeout", "桌面桥接连接超时。")),
      10_000,
    );
    window.addEventListener(
      "pywebviewready",
      () => {
        window.clearTimeout(timeout);
        resolve();
      },
      { once: true },
    );
  });
  if (!window.pywebview?.api) {
    throw new BridgeCallError("bridge_unavailable", "桌面桥接不可用。");
  }
  return window.pywebview.api;
}


function nativeApi(): DesktopApi {
  const call = async <T>(method: string, ...args: unknown[]): Promise<T> => {
    const bridge = await waitForNativeBridge();
    const implementation = bridge[method];
    if (!implementation) {
      throw new BridgeCallError(
        "method_unavailable",
        `桌面桥接缺少方法：${method}`,
      );
    }
    return unwrapEnvelope((await implementation(...args)) as ApiEnvelope<T>);
  };
  return {
    getBootstrapState: () => call<BootstrapState>("get_bootstrap_state"),
    selectInputFile: () => call<{ path: string | null }>("select_input_file"),
    startTask: (request) => call<JobSnapshot>("start_task", request),
    cancelTask: (taskId) => call<JobSnapshot>("cancel_task", taskId),
    retryTask: (taskId) => call<JobSnapshot>("retry_task", taskId),
    resumeTask: (taskId) => call<JobSnapshot>("resume_task", taskId),
    getTaskSnapshot: () => call<JobSnapshot | null>("get_task_snapshot"),
    listTasks: () => call<JobSnapshot[]>("list_tasks"),
    pollEvents: (cursor) => call("poll_events", cursor),
    installResource: (resourceId) =>
      call<ResourceInstallSnapshot>("install_resource", resourceId),
    getResourceInstall: (resourceId) =>
      call<ResourceInstallSnapshot | null>("get_resource_install", resourceId),
    listResourceInstalls: () =>
      call<ResourceInstallSnapshot[]>("list_resource_installs"),
    pauseResourceInstall: (resourceId) =>
      call<ResourceInstallSnapshot>("pause_resource_install", resourceId),
    openResourceLocation: (resourceId, kind) =>
      call<{ path: string }>("open_resource_location", resourceId, kind),
    saveApiKeys: (keys) => call<PublicSettings>("save_api_keys", keys),
    deleteApiKey: (provider) =>
      call<PublicSettings>("delete_api_key", provider),
    checkUpdates: () => call("check_updates"),
    openUpdatePage: () => call<{ url: string }>("open_update_page"),
    openOutput: (path) => call<{ path: string }>("open_output", path),
    minimizeWindow: () => call("minimize_window"),
    maximizeWindow: () => call("maximize_window"),
    closeWindow: () => call("close_window"),
  };
}


export function createDesktopApi(
  options: { preview?: boolean } = {},
): DesktopApi {
  const queryPreview =
    typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("preview") === "1";
  const preview =
    options.preview ??
    (queryPreview || process.env.NEXT_PUBLIC_DESKTOP_PREVIEW === "1");
  return preview ? previewApi() : nativeApi();
}


export const desktopApi = createDesktopApi();
