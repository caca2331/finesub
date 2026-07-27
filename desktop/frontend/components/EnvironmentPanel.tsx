"use client";

import { Check, CircleAlert, Download, LoaderCircle } from "lucide-react";

import type { ResourceStatus } from "@/lib/types";


const resourceNames: Record<string, string> = {
  uv: "Python 运行环境",
  ffmpeg: "FFmpeg 媒体组件",
};


interface EnvironmentPanelProps {
  resources: ResourceStatus[];
  busy?: boolean;
  onInstall: (resourceId: string) => void;
}


export function EnvironmentPanel({
  resources,
  busy,
  onInstall,
}: EnvironmentPanelProps) {
  const ready = resources.filter((resource) => resource.state === "ready").length;
  return (
    <section className="environment-panel" aria-labelledby="environment-title">
      <div className="section-heading compact">
        <div>
          <p className="section-kicker">运行环境</p>
          <h2 id="environment-title">所需组件</h2>
        </div>
        <span className="section-count">
          {ready}/{resources.length || 2} 就绪
        </span>
      </div>
      <div className="resource-list">
        {resources.map((resource) => {
          const isReady = resource.state === "ready";
          const isBusy = resource.state === "downloading";
          return (
            <div className="resource-row" key={resource.id}>
              <span
                className={`resource-state ${
                  isReady ? "is-ready" : isBusy ? "is-busy" : "is-missing"
                }`}
              >
                {isReady ? (
                  <Check size={13} />
                ) : isBusy ? (
                  <LoaderCircle size={13} className="spin" />
                ) : (
                  <CircleAlert size={13} />
                )}
              </span>
              <div>
                <strong>{resourceNames[resource.id] ?? resource.id}</strong>
                <span>
                  {isReady
                    ? `${resource.version} · 已安装`
                    : isBusy
                      ? "正在下载并验证"
                      : "首次使用时在线下载"}
                </span>
              </div>
              {!isReady ? (
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`安装 ${resourceNames[resource.id] ?? resource.id}`}
                  disabled={busy || isBusy}
                  onClick={() => onInstall(resource.id)}
                >
                  <Download size={15} />
                </button>
              ) : null}
            </div>
          );
        })}
      </div>
      <p className="resource-footnote">
        Whisper 与人声分离模型会在首次任务中按需下载到模型目录。
      </p>
    </section>
  );
}
