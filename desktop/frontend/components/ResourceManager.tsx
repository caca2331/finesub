"use client";

import {
  AlertCircle,
  AlertTriangle,
  Check,
  Download,
  FolderOpen,
  HardDrive,
  LoaderCircle,
  Pause,
  Play,
} from "lucide-react";
import { useState } from "react";

import { DownloadProgress } from "@/components/DownloadProgress";
import { RESOURCE_SIZES } from "@/lib/resourceCatalog";
import { isUsable, unresolvedDependency } from "@/lib/resources";
import type {
  ResourceInstallSnapshot,
  ResourceStatus,
} from "@/lib/types";
import { useLanguage } from "./LanguageProvider";


// 格式化字节大小
function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

// 获取资源信息的辅助函数
function getResourceInfo(resourceId: string, t: any): { title: string; detail: string } {
  const known = t.resources.items[resourceId];
  if (known) {
    return { title: known.title, detail: known.detail };
  }
  return { title: resourceId, detail: t.resources.items.unknown.detail };
}


interface ResourceManagerProps {
  resources: ResourceStatus[];
  installs: ResourceInstallSnapshot[];
  onInstall: (resourceId: string) => void;
  onPause: (resourceId: string) => void;
  onOpenLocation: (
    resourceId: string,
    kind: "cache" | "install",
  ) => void;
  onOpenLogs: () => void;
}


export function ResourceManager({
  resources,
  installs,
  onInstall,
  onPause,
  onOpenLocation,
  onOpenLogs,
}: ResourceManagerProps) {
  const { t } = useLanguage();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [pendingResourceId, setPendingResourceId] = useState<string | null>(null);

  // 计算未安装资源的总大小
  // Only what a task actually waits on counts toward "space required"; the
  // on-demand tools are fetched when a run turns out to need them.
  const missingResources = resources.filter(
    (r) =>
      !r.optional &&
      // An upgrade is not a space requirement: the old copy already works, so
      // nothing is waiting on the user to free room for the new one.
      !isUsable(r) &&
      !installs.some((i) => i.resource_id === r.id && i.state === "ready")
  );
  const totalRequiredSpace = missingResources.reduce(
    (sum, r) => sum + (RESOURCE_SIZES[r.id] || 0),
    0
  );

  const handleInstallClick = (resourceId: string) => {
    const resource = resources.find((r) => r.id === resourceId);
    const install = installs.find((i) => i.resource_id === resourceId);
    const isReady = resource?.state === "ready" || install?.state === "ready";
    const isRunning = install?.state === "queued" || install?.state === "running";

    if (isRunning) {
      onPause(resourceId);
    } else if (isReady) {
      onOpenLocation(resourceId, "install");
    } else {
      // 下载前显示确认对话框
      setPendingResourceId(resourceId);
      setConfirmOpen(true);
    }
  };

  const handleConfirmInstall = () => {
    if (pendingResourceId) {
      onInstall(pendingResourceId);
    }
    setConfirmOpen(false);
    setPendingResourceId(null);
  };

  const pendingResourceSize = pendingResourceId ? RESOURCE_SIZES[pendingResourceId] || 0 : 0;
  const pendingResourceInfo = pendingResourceId ? getResourceInfo(pendingResourceId, t) : null;

  return (
    <div className="page">
      <header className="page-header">
        <div>
          {/* <p className="page-kicker">{t.resources.kicker}</p> */}
          <h1>{t.resources.title}</h1>
          <p>{t.resources.description}</p>
        </div>
        <button
          type="button"
          className="button button-secondary button-compact"
          onClick={onOpenLogs}
        >
          <FolderOpen size={14} />
          {t.resources.openLogs}
        </button>
      </header>

      {/* 显示总磁盘空间需求 */}
      {missingResources.length > 0 && (
        <div className="resource-space-warning">
          <AlertTriangle size={16} />
          <div>
            <strong>{t.resources.spaceWarning.title}</strong>
            <p>
              {t.resources.spaceWarning.message.replace(
                "{size}",
                formatBytes(totalRequiredSpace),
              )}
            </p>
          </div>
        </div>
      )}

      <section className="resource-manager-list">
        {resources.map((resource) => {
          const resourceInfo = getResourceInfo(resource.id, t);
          const resourceSize = RESOURCE_SIZES[resource.id] || 0;
          const install = installs.find(
            (candidate) => candidate.resource_id === resource.id,
          );
          // "ready" here drives the finished look and the open-directory
          // action. An outdated copy is installed, so it gets neither the
          // download affordance nor the finished one -- it gets its own.
          const outdated = resource.state === "outdated" && install === undefined;
          const ready =
            (resource.state === "ready" || install?.state === "ready") &&
            !outdated;
          const running =
            install?.state === "queued" || install?.state === "running";
          const paused = install?.state === "paused";
          const failed = install?.state === "failed";
          const systemPythonAvailable =
            resource.id === "uv" && resource.reuses_system_python === true;
          // The model weights are fetched by the managed interpreter, so there
          // is nothing to run before it exists. A button that fails on click
          // would be worse than one that says what is missing.
          const blockedBy = ready
            ? ""
            : unresolvedDependency(resource, resources);
          return (
            <article
              className={`resource-card${running ? " is-installing" : ""
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
                      <h2>{resourceInfo.title}</h2>
                      <span
                        className={`resource-label${ready ? " is-ready" : ""
                          }${running ? " is-busy" : ""}${failed ? " is-failed" : ""
                          }`}
                      >
                        {ready ? (
                          <>
                            <Check size={12} /> {t.resources.status.installed}
                          </>
                        ) : outdated ? (
                          t.resources.status.updateAvailable
                        ) : running ? (
                          t.resources.status.processing
                        ) : paused ? (
                          t.resources.status.paused
                        ) : failed ? (
                          t.resources.status.failed
                        ) : systemPythonAvailable ? (
                          t.resources.status.systemPythonAvailable
                        ) : (
                          t.resources.status.needDownload
                        )}
                      </span>
                    </div>
                    <p>{resourceInfo.detail}</p>
                    <div className="resource-meta">
                      <small>
                        {resource.version === "on-demand"
                          ? // The backend sends a token for rows it does not
                            // version (the model weights); the label is ours to
                            // translate, not its to hardcode in one language.
                            t.resources.meta.onDemand
                          : <>
                            {t.resources.meta.targetVersion}：{resource.version}
                            {outdated && resource.installed_version
                              ? `（${t.resources.meta.installedVersion}：${resource.installed_version}）`
                              : ""}
                          </>}
                      </small>
                      {!ready && resourceSize > 0 && (
                        <small className="resource-size">
                          {t.resources.meta.downloadSize}：{formatBytes(resourceSize)}
                        </small>
                      )}
                    </div>
                  </div>
                  <button
                    type="button"
                    className={`button ${ready ? "button-secondary" : "button-primary"
                      }`}
                    disabled={blockedBy !== ""}
                    title={
                      blockedBy
                        ? t.resources.blockedBy.replace(
                          "{resource}",
                          getResourceInfo(blockedBy, t).title || blockedBy,
                        )
                        : undefined
                    }
                    onClick={() => handleInstallClick(resource.id)}
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
                      ? t.resources.actions.openDirectory
                      : outdated
                        ? t.resources.actions.update
                        : running
                          ? t.resources.actions.pauseDownload
                          : paused || failed
                            ? t.resources.actions.continueDownload
                            : systemPythonAvailable
                              ? t.resources.actions.installAIDeps
                              : t.resources.actions.downloadAndInstall}
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
                        {t.resources.paths.cachePath}：{install.cache_path}
                      </span>
                      <span title={install.install_path}>
                        {t.resources.paths.installPath}：{install.install_path}
                      </span>
                    </div>
                    <div className="resource-location-actions">
                      <button
                        type="button"
                        className="text-button"
                        onClick={() => onOpenLocation(resource.id, "cache")}
                      >
                        <FolderOpen size={13} /> {t.resources.paths.openCacheDir}
                      </button>
                      <button
                        type="button"
                        className="text-button"
                        onClick={() => onOpenLocation(resource.id, "install")}
                      >
                        <FolderOpen size={13} /> {t.resources.paths.openInstallDir}
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
        <strong>{t.resources.modelNote.title}</strong>
        <p>
          {t.resources.modelNote.description}
        </p>
      </div>

      {/* 下载确认对话框 */}
      {confirmOpen && pendingResourceInfo && (
        <div className="dialog-overlay" onClick={() => {
          setConfirmOpen(false);
          setPendingResourceId(null);
        }}>
          <div className="dialog-card resource-confirm-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="resource-confirm-icon">
              <AlertTriangle size={24} />
            </div>
            <h3>{t.resources.confirm.title}</h3>
            <p>
              {t.resources.confirm.message
                .replace("{name}", pendingResourceInfo.title)
                .replace("{size}", formatBytes(pendingResourceSize))}
            </p>
            <p className="confirm-warning-text">
              {t.resources.confirm.warning}
            </p>
            <div className="dialog-actions">
              <button
                type="button"
                className="button button-secondary"
                onClick={() => {
                  setConfirmOpen(false);
                  setPendingResourceId(null);
                }}
              >
                {t.resources.confirm.cancel}
              </button>
              <button
                type="button"
                className="button button-primary"
                onClick={handleConfirmInstall}
              >
                <Download size={14} />
                {t.resources.confirm.startDownload}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
