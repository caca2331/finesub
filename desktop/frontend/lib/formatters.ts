import type {
  CapabilityState,
  PipelineStage,
  UpdateCheck,
} from "./types";


export const stageLabels: Record<PipelineStage, string> = {
  vocal: "分离人声",
  aligned: "识别并对齐语音",
  stable: "稳定字幕片段",
  "raw-srt": "生成原始字幕",
  "translated-srt": "纠错与翻译",
  "final-srt": "整理最终字幕",
};


export function formatCapability(
  capabilities: Pick<CapabilityState, "translation">,
): { tone: "success" | "neutral"; title: string; detail: string } {
  if (capabilities.translation) {
    return {
      tone: "success",
      title: "翻译功能可用",
      detail: "已安全保存 Gemini API Key",
    };
  }
  return {
    tone: "neutral",
    title: "翻译功能未配置",
    detail: "不影响生成原始字幕",
  };
}


export function formatBytes(value?: number): string {
  if (value === undefined || !Number.isFinite(value) || value < 0) {
    return "—";
  }
  if (value < 1024) {
    return `${Math.round(value)} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let size = value / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && size >= 1024; index += 1) {
    size /= 1024;
    unit = units[index];
  }
  const digits = size >= 100 ? 0 : size >= 10 ? 1 : 2;
  return `${size.toFixed(digits)} ${unit}`;
}


export function formatPercent(
  current?: number,
  total?: number,
): string {
  if (
    current === undefined ||
    total === undefined ||
    !Number.isFinite(current) ||
    !Number.isFinite(total) ||
    total <= 0
  ) {
    return "0%";
  }
  const percent = Math.min(100, Math.max(0, (current / total) * 100));
  return `${Math.round(percent)}%`;
}


export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "0:00";
  }
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  if (hours) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}


export function fileName(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? path;
}


/** Save `text` as a download named `filename`.
 *
 * An export beats a clipboard copy for a log: it survives the app closing, it
 * has a name, and it does not depend on the async Clipboard API being usable
 * from a file:// origin.
 */
export function downloadText(text: string, filename: string): void {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}


/** Mirrors TaskRequest.validate_name: the stem names a directory under out/. */
export function invalidOutputName(value: string): boolean {
  const trimmed = value.trim();
  if (!trimmed) {
    return false;
  }
  return (
    trimmed.includes("/") ||
    trimmed.includes("\\") ||
    trimmed === "." ||
    trimmed === ".."
  );
}


/** Mirrors finesub_bootstrap.capabilities.is_url: what makes a source a download. */
export function isUrlSource(source: string): boolean {
  return source.startsWith("http://") || source.startsWith("https://");
}


export function formatUpdateSummary(update: UpdateCheck): string {
  if (!update.available) {
    return "已经是最新版本";
  }
  const kind = update.kind === "full" ? "完整更新" : "轻量补丁";
  return `发现 ${update.version} · ${kind} · ${formatBytes(update.size)}`;
}
