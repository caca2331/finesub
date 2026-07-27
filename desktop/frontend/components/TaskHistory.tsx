import {
  Ban,
  Clock3,
  FileText,
  FolderOpen,
  Play,
  RotateCcw,
} from "lucide-react";

import { fileName } from "@/lib/formatters";
import type { JobSnapshot } from "@/lib/types";


interface TaskHistoryProps {
  tasks: JobSnapshot[];
  onCancel: (taskId: string) => void;
  onRetry: (taskId: string) => void;
  onResume: (taskId: string) => void;
  onOpenOutput: (path: string) => void;
}


const stateCopy: Record<JobSnapshot["state"], string> = {
  idle: "等待开始",
  running: "处理中",
  completed: "已完成",
  failed: "处理失败",
  cancelled: "已取消",
  interrupted: "等待继续",
};


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
}: TaskHistoryProps) {
  return (
    <div className="page">
      <header className="page-header">
        <div>
          <p className="page-kicker">HISTORY</p>
          <h1>任务记录</h1>
          <p>任务会保存在本机，关闭应用后仍可继续处理。</p>
        </div>
      </header>
      {tasks.length ? (
        <section className="history-list">
          {tasks.map((snapshot) => {
            const id = taskId(snapshot);
            const output = Object.values(snapshot.outputs ?? {})[0];
            return (
              <article className="history-row" key={id}>
                <span className="history-icon">
                  <FileText size={17} />
                </span>
                <div className="history-copy">
                  <strong>{fileName(snapshot.request?.input ?? "未知文件")}</strong>
                  <span>
                    {stateCopy[snapshot.state]}
                    {taskTime(snapshot) ? ` · ${taskTime(snapshot)}` : ""}
                  </span>
                  {snapshot.error ? (
                    <small title={snapshot.error}>{snapshot.error}</small>
                  ) : null}
                </div>
                <div className="history-side">
                  <span className={`history-state is-${snapshot.state}`}>
                    {stateCopy[snapshot.state]}
                  </span>
                  <div className="history-actions">
                    {snapshot.state === "running" ? (
                      <button
                        type="button"
                        className="button button-danger-quiet button-compact"
                        onClick={() => onCancel(id)}
                      >
                        <Ban size={14} /> 取消任务
                      </button>
                    ) : snapshot.state === "interrupted" ? (
                      <button
                        type="button"
                        className="button button-primary button-compact"
                        onClick={() => onResume(id)}
                      >
                        <Play size={14} /> 继续任务
                      </button>
                    ) : snapshot.state === "failed" ||
                      snapshot.state === "cancelled" ? (
                      <button
                        type="button"
                        className="button button-secondary button-compact"
                        onClick={() => onRetry(id)}
                      >
                        <RotateCcw size={14} /> 重试任务
                      </button>
                    ) : snapshot.state === "completed" && output ? (
                      <button
                        type="button"
                        className="button button-secondary button-compact"
                        onClick={() => onOpenOutput(output)}
                      >
                        <FolderOpen size={14} /> 打开结果
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
          <h2>还没有任务记录</h2>
          <p>开始第一个字幕任务后，它会出现在这里。</p>
        </section>
      )}
    </div>
  );
}
