export type Route = "new-task" | "history" | "resources" | "settings";

export type PipelineStage =
  | "vocal"
  | "aligned"
  | "stable"
  | "raw-srt"
  | "translated-srt"
  | "final-srt";

export interface BridgeError {
  code: string;
  message: string;
  action?: string | null;
}

export type ApiEnvelope<T> =
  | { ok: true; data: T }
  | { ok: false; error: BridgeError };

export interface CapabilityState {
  raw_srt: boolean;
  translation: boolean;
  web_search: boolean;
}

export interface PublicSettings {
  api_keys: Record<"gemini" | "exa" | "tavily", "configured" | "missing">;
}

export interface RevealedApiKeyEntry {
  /** User-chosen label from the .env container ("" for unnamed entries). */
  name: string;
  key: string;
  masked: string;
}

export type RevealedApiKeys = Record<
  "gemini" | "exa" | "tavily",
  RevealedApiKeyEntry[]
>;

export interface ResourceStatus {
  id: string;
  version: string;
  /** "outdated" is installed and usable, just not the newest version. */
  state: "missing" | "downloading" | "outdated" | "ready" | "failed";
  /** What is on disk when `state` is "outdated"; `version` is what it should be. */
  installed_version?: string;
  detail?: string;
  /** On-demand tools: listed so they are reachable, but not part of "ready". */
  optional?: boolean;
  /** A usable system Python was found, so only AI deps need installing.
   *  Structured because the UI used to sniff `detail` for a Chinese
   *  sentence written in environment.py -- rewording it there silently
   *  removed the affordance here. */
  reuses_system_python?: boolean;
  /** Another resource must be installed first; the action is disabled until then. */
  blocked_by?: string;
}

export interface ResourceInstallSnapshot {
  resource_id: string;
  resource_version: string;
  state: "queued" | "running" | "paused" | "ready" | "failed";
  phase:
    | "waiting"
    | "downloading"
    | "verifying"
    | "extracting"
    | "installing_python"
    | "creating_environment"
    | "installing_dependencies"
    | "activating"
    | "complete";
  message: string;
  downloaded: number;
  total: number;
  bytes_per_second: number;
  cache_path: string;
  install_path: string;
  logs: string[];
  error: string;
  started_at: number;
  updated_at: number;
}

export interface WorkerEvent {
  type: "started" | "stage" | "log" | "completed" | "failed" | "cancelled";
  task_id: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface TaskRequest {
  input: string;
  output?: string | null;
  /** Bare stem, the CLI's --name: produces out/<name>/<name>.srt. */
  name: string;
  /** Off by default: the run directory is what makes a rerun cheap. */
  cleanup_intermediate: boolean;
  stage: PipelineStage;
  model_name: string;
  device: "cuda" | "cpu";
  /** Which card, on a machine with several. null lets CUDA choose. */
  gpu_index?: number | null;
  /** The card that index meant when it was picked; indexes are not identities. */
  gpu_name?: string;
  language?: string | null;
  gpu_budget_gb: 4 | 8 | 12 | 16;
  word: boolean;
  asr_stabilize_profile: -1 | 0 | 1 | 2;
/** One-run override of the shared subtitle-length knob; null follows it. */
  split_length_scale?: number | null;
  llm_media: "text" | "audio" | "video";
  llm_retrieval: "none" | "local" | "native";
  llm_difficulty: "quality" | "intermediate" | "efficiency";
  llm_fast: "auto" | "on" | "off";
  llm_output_scale: number;
  extra_info: string;
  extra_style: string;
  knowledge: "none" | "collect" | "update";
  postprocess_profile: -1 | 0 | 1 | 2 | 3 | 4;
}

export interface JobSnapshot {
  task_id?: string;
  taskId?: string;
  state:
    | "idle"
    | "running"
    | "completed"
    | "failed"
    | "cancelled"
    | "interrupted";
  request?: TaskRequest;
  events: WorkerEvent[];
  outputs?: Record<string, string>;
  error?: string | null;
  created_at?: number;
  updated_at?: number;
}

export interface Gpu {
  index: number;
  name: string;
  memory_mb: number;
}

export interface GpuSnapshot {
  /** "scanning" until the background probe answers; it is never waited on. */
  state: "scanning" | "ready" | "unavailable";
  devices: Gpu[];
}

/**
 * A sparse update to the remembered task options: null clears one back to the
 * code default rather than writing the default out.
 */
export type TaskDefaultsPatch = {
  [K in keyof TaskRequest]?: TaskRequest[K] | null;
};

/** Settings this app remembers for itself (user-data/settings.json). */
export interface Preferences {
  /** Renderer-only state: theme, language, dismissed dialogs. */
  ui: Record<string, unknown>;
  /** What the task form starts with. Absent field = follow the code default. */
  task_defaults: Partial<TaskRequest>;
}

/**
 * The slice of config.toml the settings panel can write. Shared with the CLI
 * and hand-editable, so null means "not set" (the key is removed) rather than
 * "write the default".
 */
export interface SharedSettings {
  split_length_scale: number | null;
}

export interface BootstrapState {
  app_version: string;
  resources: ResourceStatus[];
  resource_installs: ResourceInstallSnapshot[];
  capabilities: CapabilityState;
  settings: PublicSettings;
  preferences: Preferences;
  shared_settings: SharedSettings;
  /** Shown in the panel: the file is meant to be editable by hand. */
  config_path: string;
  gpus?: GpuSnapshot;
  task: JobSnapshot | null;
  tasks?: JobSnapshot[];
}

export interface PollResult {
  events: WorkerEvent[];
  nextCursor: number;
}

export interface UpdateCheck {
  available: boolean;
  version: string;
  kind?: "app" | "full";
  releaseNotes?: string;
  mandatory?: boolean;
  size?: number;
  releaseUrl?: string;
}

export interface UpdateInstallSnapshot {
  version: string;
  kind: "app" | "full";
  state: "queued" | "running" | "ready" | "failed";
  phase: "waiting" | "downloading" | "installing" | "complete";
  message: string;
  downloaded: number;
  total: number;
  bytes_per_second: number;
  /** An app delta only swaps the version pointer: relaunch and it is live. */
  restart_required: boolean;
  /** A full package hands off to an external updater that replaces this
   *  install, so FineSub has to quit before it can proceed. */
  exit_required: boolean;
  error: string;
  started_at: number;
  updated_at: number;
}

export interface DesktopApi {
  getBootstrapState(): Promise<BootstrapState>;
  selectInputFile(): Promise<{ path: string | null }>;
  startTask(request: Partial<TaskRequest> & { input: string }): Promise<JobSnapshot>;
  cancelTask(taskId: string): Promise<JobSnapshot>;
  retryTask(taskId: string): Promise<JobSnapshot>;
  resumeTask(taskId: string): Promise<JobSnapshot>;
  deleteTaskIntermediates(taskId: string): Promise<JobSnapshot>;
  getTaskSnapshot(): Promise<JobSnapshot | null>;
  listTasks(): Promise<JobSnapshot[]>;
  pollEvents(cursor: number): Promise<PollResult>;
  installResource(resourceId: string): Promise<ResourceInstallSnapshot>;
  getResourceInstall(resourceId: string): Promise<ResourceInstallSnapshot | null>;
  listResourceInstalls(): Promise<ResourceInstallSnapshot[]>;
  pauseResourceInstall(resourceId: string): Promise<ResourceInstallSnapshot>;
  openInstallLogs(): Promise<unknown>;
  rescanGpus(): Promise<unknown>;
  openResourceLocation(
    resourceId: string,
    kind: "cache" | "install",
  ): Promise<{ path: string }>;
  saveApiKeys(keys: {
    gemini?: string | null;
    exa?: string | null;
    tavily?: string | null;
  }): Promise<PublicSettings>;
  deleteApiKey(provider: "gemini" | "exa" | "tavily"): Promise<PublicSettings>;
  revealApiKeys(): Promise<RevealedApiKeys>;
  getPreferences(): Promise<{
    preferences: Preferences;
    shared: SharedSettings;
    config_path: string;
  }>;
  savePreferences(patch: {
    ui?: Record<string, unknown> | null;
    task_defaults?: TaskDefaultsPatch | null;
  }): Promise<{ preferences: Preferences }>;
  saveSharedSettings(
    values: SharedSettings,
  ): Promise<{ shared: SharedSettings; config_path: string }>;
  checkUpdates(): Promise<UpdateCheck>;
  installUpdate(kind: "app" | "full", version: string): Promise<UpdateInstallSnapshot>;
  getUpdateInstall(): Promise<UpdateInstallSnapshot | null>;
  openUpdatePage(): Promise<{ url: string }>;
  openTasksDirectory(taskId?: string): Promise<{ path: string }>;
  openOutput(path: string): Promise<{ path: string }>;
  minimizeWindow(): Promise<unknown>;
  minimizeToTray(): Promise<unknown>;
  maximizeWindow(): Promise<unknown>;
  closeWindow(): Promise<unknown>;
  setWindowChrome(background: string, foreground: string): Promise<unknown>;
}
