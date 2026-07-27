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

export interface ResourceStatus {
  id: string;
  version: string;
  state: "missing" | "downloading" | "ready" | "failed";
  detail?: string;
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
  stage: PipelineStage;
  model_name: string;
  device: "cuda" | "cpu";
  language?: string | null;
  gpu_budget_gb: 8 | 12 | 16;
  word: boolean;
  asr_stabilize_profile: -1 | 0 | 1 | 2;
  llm_route: "text" | "mm";
  llm_level: "low" | "med" | "high";
  llm_fast: "auto" | "on" | "off";
  llm_output_scale: number;
  extra_info: string;
  extra_style: string;
  enable_web_search: boolean;
  knowledge: "none" | "collect" | "update";
  postprocess_profile: -1 | 0 | 1 | 2;
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

export interface BootstrapState {
  resources: ResourceStatus[];
  resource_installs: ResourceInstallSnapshot[];
  capabilities: CapabilityState;
  settings: PublicSettings;
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
}

export interface UpdateInstallResult {
  kind: "app" | "full";
  version: string;
  restartRequired?: boolean;
  exitRequired?: boolean;
}

export interface DesktopApi {
  getBootstrapState(): Promise<BootstrapState>;
  selectInputFile(): Promise<{ path: string | null }>;
  startTask(request: Partial<TaskRequest> & { input: string }): Promise<JobSnapshot>;
  cancelTask(taskId: string): Promise<JobSnapshot>;
  retryTask(taskId: string): Promise<JobSnapshot>;
  resumeTask(taskId: string): Promise<JobSnapshot>;
  getTaskSnapshot(): Promise<JobSnapshot | null>;
  listTasks(): Promise<JobSnapshot[]>;
  pollEvents(cursor: number): Promise<PollResult>;
  installResource(resourceId: string): Promise<ResourceInstallSnapshot>;
  getResourceInstall(resourceId: string): Promise<ResourceInstallSnapshot | null>;
  listResourceInstalls(): Promise<ResourceInstallSnapshot[]>;
  pauseResourceInstall(resourceId: string): Promise<ResourceInstallSnapshot>;
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
  checkUpdates(): Promise<UpdateCheck>;
  installUpdate(channel: "app" | "full"): Promise<UpdateInstallResult>;
  openOutput(path: string): Promise<{ path: string }>;
  minimizeWindow(): Promise<unknown>;
  maximizeWindow(): Promise<unknown>;
  closeWindow(): Promise<unknown>;
}
