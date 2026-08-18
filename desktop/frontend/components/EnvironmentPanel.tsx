"use client";

import { Check, CircleAlert, Download, LoaderCircle } from "lucide-react";

import { isUsable, requiredResources } from "@/lib/resources";
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
  // Only what can block a task. Optional rows are excluded from the list as
  // well as the count: this panel answers "is anything standing in my way?",
  // and the model weights -- the one optional resource -- never are. Listing
  // them here also put a one-click 3.4GB download on the page, without the
  // size confirmation the resources page asks for. That page owns them.
  const required = requiredResources(resources);
  // An outdated tool counts as ready here: it works, and the count answers
  // "can I run something?", not "is everything at its newest version?".
  const ready = required.filter(isUsable).length;

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
        {required.map((resource) => {
          const isReady = isUsable(resource);
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
                    ? // What is on disk, not the manifest's target: for an
                      // outdated tool the two differ, and "installed" next to
                      // a version nobody has installed yet is a false claim.
                      `${resource.installed_version || resource.version} · ${t.newTask.env.installed}`
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