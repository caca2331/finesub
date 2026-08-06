"use client";

import { Check, CircleAlert, Download, LoaderCircle, Minus } from "lucide-react";

import type { ResourceStatus } from "@/lib/types";
import { useLanguage } from "./LanguageProvider";


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
  const { t } = useLanguage();
  // Optional tools are excluded from the count: a machine with neither is a
  // perfectly complete install, and "2 / 4 ready" would say otherwise.
  const required = resources.filter((resource) => !resource.optional);
  const ready = required.filter((resource) => resource.state === "ready").length;

  const resourceNames: Record<string, string> = {
    uv: t.newTask.env.uv,
    ffmpeg: t.newTask.env.ffmpeg,
    git: t.newTask.env.git,
    "yt-dlp": t.newTask.env.ytDlp,
  };

  return (
    <section className="environment-panel" aria-labelledby="environment-title">
      <div className="section-heading compact">
        <div>
          <p className="section-kicker">{t.newTask.env.kicker}</p>
          <h2 id="environment-title">{t.newTask.env.title}</h2>
        </div>
        <span className="section-count">
          {ready}/{required.length || 2} {t.newTask.env.readyCount}
        </span>
      </div>
      <div className="resource-list">
        {resources.map((resource) => {
          const isReady = resource.state === "ready";
          const isBusy = resource.state === "downloading";
          // A missing optional tool is not a problem to warn about; it is a
          // capability the user has not needed yet.
          const isPending = !isReady && !isBusy && resource.optional;
          return (
            <div className="resource-row" key={resource.id}>
              <span
                className={`resource-state ${
                  isReady
                    ? "is-ready"
                    : isBusy
                      ? "is-busy"
                      : isPending
                        ? "is-optional"
                        : "is-missing"
                }`}
              >
                {isReady ? (
                  <Check size={13} />
                ) : isBusy ? (
                  <LoaderCircle size={13} className="spin" />
                ) : isPending ? (
                  <Minus size={13} />
                ) : (
                  <CircleAlert size={13} />
                )}
              </span>
              <div>
                <strong>{resourceNames[resource.id] ?? resource.id}</strong>
                <span>
                  {isReady
                    ? `${resource.version} · ${t.newTask.env.installed}`
                    : isBusy
                      ? t.newTask.env.downloading
                      : t.newTask.env.willDownload}
                </span>
              </div>
              {!isReady ? (
                <button
                  type="button"
                  className="icon-button"
                  aria-label={`${t.newTask.env.install} ${resourceNames[resource.id] ?? resource.id}`}
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
        {t.newTask.env.footnote}
      </p>
    </section>
  );
}