"use client";

import {
  Check,
  ExternalLink,
  FileText,
  FolderOpen,
  RotateCcw,
} from "lucide-react";

import { fileName } from "@/lib/formatters";
import type { TaskState } from "@/lib/state";


const outputLabels: Record<string, string> = {
  rawSrt: "原始字幕",
  translatedSrt: "翻译字幕",
  finalSrt: "最终字幕",
  alignedJson: "对齐数据",
  stableJson: "稳定化数据",
  vocalAudio: "分离后人声",
};


interface CompletedViewProps {
  task: TaskState;
  onOpen: (path: string) => void;
  onReset: () => void;
}


export function CompletedView({
  task,
  onOpen,
  onReset,
}: CompletedViewProps) {
  const outputs = Object.entries(task.outputs).filter(
    ([key, value]) => Boolean(value) && key !== "taskArtifactDir",
  );
  const preferred =
    task.outputs.finalSrt ||
    task.outputs.translatedSrt ||
    task.outputs.rawSrt ||
    outputs[0]?.[1];

  return (
    <div className="page completed-page">
      <header className="page-header">
        <div>
          {/* <p className="page-kicker">COMPLETED</p> */}
          <h1>字幕已经准备好了</h1>
          <p>{task.selectedFile}</p>
        </div>
        <span className="completion-badge">
          <Check size={14} />
          处理完成
        </span>
      </header>

      <section className="completed-card">
        <div className="completed-summary">
          <div className="success-mark">
            <Check size={24} strokeWidth={2} />
          </div>
          <div>
            <p>FineSub 已完成本次任务</p>
            <h2>{preferred ? fileName(preferred) : "字幕输出"}</h2>
            <span>输出文件已保存在任务目录中，可以继续编辑或导入剪辑软件。</span>
          </div>
        </div>

        <div className="output-list">
          {outputs.map(([key, path]) => (
            <button
              type="button"
              className="output-row"
              key={key}
              onClick={() => onOpen(path)}
            >
              <span className="output-icon">
                <FileText size={16} />
              </span>
              <span>
                <strong>{outputLabels[key] ?? key}</strong>
                <small>{fileName(path)}</small>
              </span>
              <ExternalLink size={14} />
            </button>
          ))}
        </div>

        <div className="completed-actions">
          <button
            type="button"
            className="button button-secondary"
            onClick={onReset}
          >
            <RotateCcw size={14} />
            新建任务
          </button>
          {preferred ? (
            <button
              type="button"
              className="button button-primary"
              onClick={() => onOpen(preferred)}
            >
              <FolderOpen size={15} />
              打开输出目录
            </button>
          ) : null}
        </div>
      </section>
    </div>
  );
}
