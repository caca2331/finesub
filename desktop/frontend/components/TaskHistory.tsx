"use client";

import {
  Ban,
  Clock3,
  FileText,
  FolderOpen,
  Play,
  RotateCcw,
} from "lucide-react";

import { fileName } from "@/lib/formatters";
import { preferredSubtitleOutput } from "@/lib/subtitleOutputs";
import type { JobSnapshot } from "@/lib/types";
import { useLanguage } from "./LanguageProvider";


interface TaskHistoryProps {
  tasks: JobSnapshot[];
  onCancel: (taskId: string) => void;
  onRetry: (taskId: string) => void;
  onResume: (taskId: string) => void;
  onOpenOutput: (path: string) => void;
  onOpenTasksDirectory: () => void;
}


function taskId(snapshot: JobSnapshot): string {
  return snapshot.task_id ?? snapshot.taskId ?? "";
}


function taskTime(snapshot: JobSnapshot): string {
  const seconds = snapshot.updated_at ?? snapshot.created_at;
  if (!seconds) {
    return "";
  }
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(seconds * 1000));
}


export function TaskHistory({
  tasks,
  onCancel,
  onRetry,
  onResume,
  onOpenOutput,
  onOpenTasksDirectory,
}: TaskHistoryProps) {
  const { t } = useLanguage();

  return (
    <div className="page">
      <header className="page-header">
        <div>
          {/* <p className="page-kicker">{t.history.kicker}</p> */}
          <h1>{t.history.title}</h1>
          <p>{t.history.description}</p>
        </div>
        <button
          type="button"
          className="button button-secondary"
          onClick={() => onOpenTasksDirectory()}
        >
          <FolderOpen size={14} />
          {t.history.openTasksDir}
        </button>
      </header>
      {tasks.length ? (
        <section className="history-list">
          {tasks.map((snapshot) => {
            const id = taskId(snapshot);
            const output = preferredSubtitleOutput(snapshot.outputs);
            const stateLabel = t.history.status[snapshot.state];
            return (
              <article className="history-row" key={id}>
                <span className="history-icon">
                  <FileText size={17} />
                </span>
                <div className="history-copy">
                  <strong>{fileName(snapshot.request?.input ?? t.history.unknownFile)}</strong>
                  <span>
                    {stateLabel}
                    {taskTime(snapshot) ? ` · ${taskTime(snapshot)}` : ""}
                  </span>
                  {snapshot.error ? (
                    <small title={snapshot.error}>{snapshot.error}</small>
                  ) : null}
                </div>
                <div className="history-side">
                  <span className={`history-state is-${snapshot.state}`}>
                    {stateLabel}
                  </span>
                  <div className="history-actions">
                    {snapshot.state === "running" ? (
                      <button
                        type="button"
                        className="button button-danger-quiet button-compact"
                        onClick={() => onCancel(id)}
                      >
                        <Ban size={14} /> {t.history.cancelTask}
                      </button>
                    ) : snapshot.state === "interrupted" ? (
                      <button
                        type="button"
                        className="button button-primary button-compact"
                        onClick={() => onResume(id)}
                      >
                        <Play size={14} /> {t.history.resume}
                      </button>
                    ) : snapshot.state === "failed" ||
                      snapshot.state === "cancelled" ? (
                      <button
                        type="button"
                        className="button button-secondary button-compact"
                        onClick={() => onRetry(id)}
                      >
                        <RotateCcw size={14} /> {t.history.retry}
                      </button>
                    ) : snapshot.state === "completed" && output ? (
                      <button
                        type="button"
                        className="button button-secondary button-compact"
                        onClick={() => onOpenOutput(output)}
                      >
                        <FolderOpen size={14} /> {t.history.openResult}
                      </button>
                    ) : null}
                  </div>
                </div>
              </article>
            );
          })}
        </section>
      ) : (
        <section className="empty-state">
          <Clock3 size={24} />
          <h2>{t.history.emptyTitle}</h2>
          <p>{t.history.emptyDescription}</p>
        </section>
      )}
    </div>
  );
}
