"use client";

import { ArrowRight, ShieldCheck } from "lucide-react";

import type { AppState } from "@/lib/state";
import type { TaskRequest } from "@/lib/types";

import { DropZone } from "./DropZone";
import { EnvironmentPanel } from "./EnvironmentPanel";
import { TaskSettings } from "./TaskSettings";
import { useLanguage } from "./LanguageProvider";


interface NewTaskProps {
  state: AppState;
  busy: boolean;
  onSelectFile: () => void;
  onDropPath: (path: string) => void;
  onRequestChange: (
    changes: Partial<Omit<TaskRequest, "input">>,
  ) => void;
  onInstallResource: (resourceId: string) => void;
  onStart: () => void;
}


export function NewTask({
  state,
  busy,
  onSelectFile,
  onDropPath,
  onRequestChange,
  onInstallResource,
  onStart,
}: NewTaskProps) {
  const { t } = useLanguage();
  const hasFile = Boolean(state.task.selectedFile);
  return (
    <div className="page page-new-task">
      <header className="page-header">
        <div>
          {/* <p className="page-kicker">{t.newTask.kicker}</p> */}
          <h1>{t.newTask.pageTitle}</h1>
          <p>{t.newTask.pageDescription}</p>
        </div>
        <div className="privacy-note">
          <ShieldCheck size={15} />
          <span>{t.newTask.privacyNote}</span>
        </div>
      </header>

      <div className="task-layout">
        <section className="primary-panel">
          <div className="section-heading">
            <div>

              {/* <p className="section-kicker">{t.newTask.sourceKicker}</p> */}
              <h2>{t.newTask.sourceSection}</h2>
            </div>
          </div>
          <DropZone
            selectedFile={state.task.selectedFile}
            disabled={busy}
            onSelect={onSelectFile}
            onDropPath={onDropPath}
          />

          <div className="panel-divider" />

          <div className="section-heading">
            <div>
              {/* <p className="section-kicker">{t.newTask.outputKicker}</p> */}
              <h2>{t.newTask.outputSection}</h2>
            </div>
            <span className="section-hint">{t.newTask.outputHint}</span>
          </div>
          <TaskSettings
            request={state.task.request}
            capabilities={state.capabilities}
            disabled={busy}
            onChange={onRequestChange}
          />

          {state.task.error ? (
            <div className="error-banner" role="alert">
              <strong>{state.task.error.message}</strong>
              {state.task.error.code === "api_key_required" ? (
                <span>{t.newTask.apiKeyError}</span>
              ) : null}
            </div>
          ) : null}

          <div className="task-actions">
            <div>
              <strong>{hasFile ? t.newTask.canStart : t.newTask.waitingFile}</strong>
              <span>
                {hasFile
                  ? t.newTask.willDownload
                  : t.newTask.supportedFormats}
              </span>
            </div>
            <button
              type="button"
              className="button button-primary"
              disabled={!hasFile || busy}
              onClick={onStart}
            >
              {busy ? t.newTask.preparing : t.newTask.startGenerate}
              <ArrowRight size={15} />
            </button>
          </div>
        </section>

        <EnvironmentPanel
          resources={state.resources}
          busy={busy}
          onInstall={onInstallResource}
        />
      </div>
    </div>
  );
}