"use client";

import {
  Check,
  ChevronDown,
  Circle,
  Download,
  CircleStop,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { downloadText, formatDuration } from "@/lib/formatters";
import type { TaskState } from "@/lib/state";
import type { PipelineStage } from "@/lib/types";
import { useLanguage } from "./LanguageProvider";


const pipelineStages: PipelineStage[] = [
  "vocal",
  "aligned",
  "stable",
  "raw-srt",
  "translated-srt",
  "final-srt",
];


interface ProcessingViewProps {
  task: TaskState;
  /** Nothing has finished on this machine yet: weights and warm-up come first. */
  firstRun?: boolean;
  onCancel: () => void;
  onRetry: () => void;
}


export function ProcessingView({
  task,
  firstRun,
  onCancel,
  onRetry,
}: ProcessingViewProps) {
  const { t } = useLanguage();
  const [logsOpen, setLogsOpen] = useState(false);
  const [now, setNow] = useState(Date.now());
  const logDrawerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (task.phase !== "running") {
      return;
    }
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [task.phase]);
  useEffect(() => {
    if (!logsOpen || !logDrawerRef.current) {
      return;
    }
    logDrawerRef.current.scrollTop = logDrawerRef.current.scrollHeight;
  }, [logsOpen, task.logs]);
  const elapsed = task.startedAt
    ? Math.max(0, (now - task.startedAt) / 1000)
    : 0;
  const targetIndex = pipelineStages.indexOf(task.request.stage);
  const activeIndex = task.currentStage
    ? pipelineStages.indexOf(task.currentStage)
    : 0;
  const visibleStages = pipelineStages.slice(0, Math.max(0, targetIndex) + 1);
  const stageLabels: Record<PipelineStage, string> = {
    vocal: t.processing.stages.vocal,
    aligned: t.processing.stages.aligned,
    stable: t.processing.stages.stable,
    "raw-srt": t.processing.stages.rawSrt,
    "translated-srt": t.processing.stages.translatedSrt,
    "final-srt": t.processing.stages.finalSrt,
  };
  const headline = task.phase === "failed"
    ? t.processing.failedHeadline
    : task.currentStage
      ? stageLabels[task.currentStage]
      : t.processing.starting;

  return (
    <div className="page processing-page">
      <header className="page-header">
        <div>
          {/* <p className="page-kicker">ACTIVE TASK</p> */}
          <h1>
            {task.phase === "failed"
              ? t.processing.failedTitle
              : t.processing.runningTitle}
          </h1>
          <p>{task.selectedFile}</p>
        </div>
        <div className="elapsed">
          <span>{t.processing.elapsed}</span>
          <strong>{formatDuration(elapsed)}</strong>
        </div>
      </header>

      <section className={`processing-card ${task.phase === "failed" ? "is-failed" : ""}`}>
        <div className="processing-hero">
          <div className="processing-mark">
            {task.phase === "failed" ? (
              <CircleStop size={24} />
            ) : (
              <LoaderCircle size={25} className="spin" />
            )}
          </div>
          <div>
            <p>
              {task.phase === "failed"
                ? t.processing.needsAttention
                : t.processing.currentStage}
            </p>
            <h2>{headline}</h2>
            <span>
              {task.phase === "failed"
                ? task.error?.message
                : task.statusMessage || t.processing.initializing}
            </span>
          </div>
        </div>

        {firstRun && task.phase !== "failed" ? (
          <div className="inline-note" role="note">
            {t.newTask.firstRunNotice}
          </div>
        ) : null}

        <ol className="stage-list">
          {visibleStages.map((stage, index) => {
            const done = index < activeIndex;
            const active = index === activeIndex && task.phase !== "failed";
            return (
              <li
                key={stage}
                className={`${done ? "is-done" : ""}${active ? " is-active" : ""}`}
              >
                <span className="stage-symbol">
                  {done ? (
                    <Check size={13} />
                  ) : active ? (
                    <LoaderCircle size={13} className="spin" />
                  ) : (
                    <Circle size={10} />
                  )}
                </span>
                <span>{stageLabels[stage]}</span>
              </li>
            );
          })}
        </ol>

        <div className="processing-actions">
          <button
            type="button"
            className="button button-secondary"
            onClick={() => setLogsOpen((value) => !value)}
          >
            {t.processing.logs}
            <ChevronDown
              size={14}
              className={logsOpen ? "is-rotated" : ""}
            />
          </button>
          {task.phase === "failed" ? (
            <button
              type="button"
              className="button button-primary"
              onClick={onRetry}
            >
              <RotateCcw size={14} />
              {t.processing.retry}
            </button>
          ) : (
            <button
              type="button"
              className="button button-danger-quiet"
              onClick={onCancel}
            >
              <CircleStop size={14} />
              {t.processing.cancel}
            </button>
          )}
        </div>

        {logsOpen ? (
          <>
            <div className="log-toolbar">
              <button
                type="button"
                className="button button-secondary button-compact"
                disabled={!task.logs.length}
                onClick={() => {
                  const stamp = new Date()
                    .toISOString()
                    .slice(0, 16)
                    .replace(/[-:]/g, "")
                    .replace("T", "-");
                  downloadText(
                    task.logs.join("\n"),
                    `finesub-log-${stamp}.txt`,
                  );
                }}
              >
                <Download size={13} />
                {t.processing.exportLogs}
              </button>
            </div>
          <div
            ref={logDrawerRef}
            className="log-drawer"
            role="log"
            aria-label={t.processing.logAria}
            aria-live="polite"
          >
            {task.logs.length ? (
              task.logs.map((line, index) => (
                <div key={`${index}-${line}`}>{line}</div>
              ))
            ) : (
              <span>{t.processing.waitingLogs}</span>
            )}
          </div>
          </>
        ) : null}
      </section>
    </div>
  );
}
