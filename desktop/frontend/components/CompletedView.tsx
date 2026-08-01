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
import { useLanguage } from "./LanguageProvider";


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
  const { t } = useLanguage();
  const outputLabels: Record<string, string> = t.completed.labels;
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
          <h1>{t.completed.readyTitle}</h1>
          <p>{task.selectedFile}</p>
        </div>
        <span className="completion-badge">
          <Check size={14} />
          {t.completed.done}
        </span>
      </header>

      <section className="completed-card">
        <div className="completed-summary">
          <div className="success-mark">
            <Check size={24} strokeWidth={2} />
          </div>
          <div>
            <p>{t.completed.summary}</p>
            <h2>{preferred ? fileName(preferred) : t.completed.fallbackName}</h2>
            <span>{t.completed.description}</span>
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
            {t.completed.newTask}
          </button>
          {preferred ? (
            <button
              type="button"
              className="button button-primary"
              onClick={() => onOpen(preferred)}
            >
              <FolderOpen size={15} />
              {t.completed.openDirectory}
            </button>
          ) : null}
        </div>
      </section>
    </div>
  );
}
