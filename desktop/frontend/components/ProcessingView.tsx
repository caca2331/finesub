"use client";

import {
  Check,
  ChevronDown,
  Circle,
  CircleStop,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { formatDuration, stageLabels } from "@/lib/formatters";
import type { TaskState } from "@/lib/state";
import type { PipelineStage } from "@/lib/types";


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
  onCancel: () => void;
  onRetry: () => void;
}


export function ProcessingView({
  task,
  onCancel,
  onRetry,
}: ProcessingViewProps) {
  const [logsOpen, setLogsOpen] = useState(false);
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (task.phase !== "running") {
      return;
    }
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [task.phase]);
  const elapsed = task.startedAt
    ? Math.max(0, (now - task.startedAt) / 1000)
    : 0;
  const targetIndex = pipelineStages.indexOf(task.request.stage);
  const activeIndex = task.currentStage
    ? pipelineStages.indexOf(task.currentStage)
    : 0;
  const visibleStages = pipelineStages.slice(0, Math.max(0, targetIndex) + 1);
  const headline = useMemo(() => {
    if (task.phase === "failed") {
      return "处理遇到问题";
    }
    return task.currentStage
      ? stageLabels[task.currentStage]
      : "正在启动处理引擎";
  }, [task.currentStage, task.phase]);

  return (
    <div className="page processing-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">ACTIVE TASK</p>
          <h1>{task.phase === "failed" ? "任务未完成" : "正在生成字幕"}</h1>
          <p>{task.selectedFile}</p>
        </div>
        <div className="elapsed">
          <span>已用时间</span>
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
            <p>{task.phase === "failed" ? "需要处理" : "当前步骤"}</p>
            <h2>{headline}</h2>
            <span>
              {task.phase === "failed"
                ? task.error?.message
                : task.statusMessage || "模型初始化可能需要一点时间"}
            </span>
          </div>
        </div>

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
            运行日志
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
              重新尝试
            </button>
          ) : (
            <button
              type="button"
              className="button button-danger-quiet"
              onClick={onCancel}
            >
              <CircleStop size={14} />
              取消任务
            </button>
          )}
        </div>

        {logsOpen ? (
          <div className="log-drawer" role="log" aria-label="任务运行日志">
            {task.logs.length ? (
              task.logs.map((line, index) => (
                <div key={`${index}-${line}`}>{line}</div>
              ))
            ) : (
              <span>等待 worker 输出日志…</span>
            )}
          </div>
        ) : null}
      </section>
    </div>
  );
}
