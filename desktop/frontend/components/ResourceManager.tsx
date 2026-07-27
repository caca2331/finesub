"use client";

import {
  AlertCircle,
  Check,
  Download,
  FolderOpen,
  HardDrive,
  LoaderCircle,
  Pause,
  Play,
} from "lucide-react";

import { DownloadProgress } from "@/components/DownloadProgress";
import type {
  ResourceInstallSnapshot,
  ResourceStatus,
} from "@/lib/types";


const resourceCopy: Record<string, { title: string; detail: string }> = {
  uv: {
    title: "Python 运行环境",
    detail: "负责安装并隔离 FineSub 的 Python 与 AI 依赖",
  },
  ffmpeg: {
    title: "FFmpeg 媒体组件",
    detail: "负责音视频读取、转码与音频提取",
  },
};


interface ResourceManagerProps {
  resources: ResourceStatus[];
  installs: ResourceInstallSnapshot[];
  onInstall: (resourceId: string) => void;
  onPause: (resourceId: string) => void;
  onOpenLocation: (
    resourceId: string,
    kind: "cache" | "install",
  ) => void;
}


export function ResourceManager({
  resources,
  installs,
  onInstall,
  onPause,
  onOpenLocation,
}: ResourceManagerProps) {
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="page-kicker">RESOURCES</p>
          <h1>运行资源</h1>
          <p>大型组件在线安装，应用本体保持轻量。</p>
        </div>
      </header>

      <section className="resource-manager-list">
        {resources.map((resource) => {
          const copy = resourceCopy[resource.id] ?? {
            title: resource.id,
            detail: "FineSub 运行资源",
          };
          const install = installs.find(
            (candidate) => candidate.resource_id === resource.id,
          );
          const ready =
            resource.state === "ready" || install?.state === "ready";
          const running =
            install?.state === "queued" || install?.state === "running";
          const paused = install?.state === "paused";
          const failed = install?.state === "failed";
          const systemPythonAvailable =
            resource.id === "uv" &&
            resource.detail?.startsWith("已检测到系统 Python");
          return (
            <article
              className={`resource-card${
                running ? " is-installing" : ""
              }${failed ? " is-failed" : ""}`}
              key={resource.id}
            >
              <span className="resource-large-icon">
                {running ? (
                  <LoaderCircle size={20} className="spin" />
                ) : failed ? (
                  <AlertCircle size={20} />
                ) : (
                  <HardDrive size={20} />
                )}
              </span>
              <div className="resource-card-main">
                <div className="resource-card-heading">
                  <div className="resource-card-copy">
                    <div>
                      <h2>{copy.title}</h2>
                      <span
                        className={`resource-label${
                          ready ? " is-ready" : ""
                        }${running ? " is-busy" : ""}${
                          failed ? " is-failed" : ""
                        }`}
                      >
                        {ready ? (
                          <>
                            <Check size={12} /> 已安装
                          </>
                        ) : running ? (
                          "正在处理"
                        ) : paused ? (
                          "已暂停"
                        ) : failed ? (
                          "安装失败"
                        ) : systemPythonAvailable ? (
                          "系统 Python 可用"
                        ) : (
                          "需要下载"
                        )}
                      </span>
                    </div>
                    <p>{copy.detail}</p>
                    <small>目标版本：{resource.version}</small>
                  </div>
                  <button
                    type="button"
                    className={`button ${
                      ready ? "button-secondary" : "button-primary"
                    }`}
                    onClick={() =>
                      running
                        ? onPause(resource.id)
                        : ready
                          ? onOpenLocation(resource.id, "install")
                          : onInstall(resource.id)
                    }
                  >
                    {ready ? (
                      <FolderOpen size={14} />
                    ) : running ? (
                      <Pause size={14} />
                    ) : paused ? (
                      <Play size={14} />
                    ) : (
                      <Download size={14} />
                    )}
                    {ready
                      ? "打开目录"
                      : running
                        ? "暂停下载"
                        : paused || failed
                          ? "继续下载"
                          : systemPythonAvailable
                            ? "安装 AI 依赖"
                          : "下载并安装"}
                  </button>
                </div>
                {install ? (
                  <div className="resource-install-detail">
                    {running && install.total > 0 ? (
                      <DownloadProgress
                        name={install.message}
                        downloaded={install.downloaded}
                        total={install.total}
                        bytesPerSecond={install.bytes_per_second}
                      />
                    ) : running ? (
                      <div className="resource-indeterminate">
                        <span className="indeterminate-track">
                          <i />
                        </span>
                        <strong>{install.message}</strong>
                      </div>
                    ) : (
                      <strong className="resource-install-message">
                        {install.message}
                      </strong>
                    )}
                    {install.error ? (
                      <p className="resource-error">{install.error}</p>
                    ) : null}
                    {install.logs.length ? (
                      <pre className="resource-install-log">
                        {install.logs.slice(-3).join("\n")}
                      </pre>
                    ) : null}
                    <div className="resource-paths">
                      <span title={install.cache_path}>
                        下载缓存：{install.cache_path}
                      </span>
                      <span title={install.install_path}>
                        安装位置：{install.install_path}
                      </span>
                    </div>
                    <div className="resource-location-actions">
                      <button
                        type="button"
                        className="text-button"
                        onClick={() => onOpenLocation(resource.id, "cache")}
                      >
                        <FolderOpen size={13} /> 打开缓存目录
                      </button>
                      <button
                        type="button"
                        className="text-button"
                        onClick={() => onOpenLocation(resource.id, "install")}
                      >
                        <FolderOpen size={13} /> 打开安装目录
                      </button>
                    </div>
                  </div>
                ) : resource.detail ? (
                  <p className="resource-error">{resource.detail}</p>
                ) : null}
              </div>
            </article>
          );
        })}
      </section>

      <div className="resource-info-note">
        <strong>模型如何管理？</strong>
        <p>
          Whisper 和 BS-RoFormer 权重由 FineSub 在首次使用时下载，并统一写入 models
          目录；更新应用时不会删除。
        </p>
      </div>
    </div>
  );
}
