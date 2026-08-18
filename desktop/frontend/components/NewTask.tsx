"use client";

import { ArrowRight, ShieldCheck } from "lucide-react";

import { invalidOutputName } from "@/lib/formatters";
import { blockingResources } from "@/lib/resources";
import { reusableAsrTask } from "@/lib/reuse";
import type { AppState } from "@/lib/state";
import type { JobSnapshot, TaskRequest } from "@/lib/types";

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
  onReuse: (snapshot: JobSnapshot) => void;
  onInstallResource: (resourceId: string) => void;
  onOpenResources: () => void;
  onStart: () => void;
}


export function NewTask({
  state,
  busy,
  onSelectFile,
  onDropPath,
  onRequestChange,
  onReuse,
  onInstallResource,
  onOpenResources,
  onStart,
}: NewTaskProps) {
  const { t } = useLanguage();
  const hasFile = Boolean(state.task.selectedFile);
  // Offering "start" when the run would be refused for a missing component
  // wastes a click and answers with an error banner. The model weights are not
  // in here on purpose: `blockingResources` skips optional rows, and a task
  // that finds them missing downloads them itself.
  const missingResources = blockingResources(state.resources);
  const resourcesBlocked = missingResources.length > 0;
  // A pinned output means this task writes into a previous run's directory,
  // so every stage whose artifact is already there is skipped.
  const reusePinned = Boolean(state.task.request.output);
  const reusable =
    !reusePinned &&
    ["translated-srt", "final-srt"].includes(state.task.request.stage)
      ? reusableAsrTask(state.history, state.task.selectedFile)
      : null;
  // The backend rejects a bad stem with a generic "invalid task parameters",
  // which tells the user nothing. TaskSettings already shows what is wrong;
  // this keeps them from submitting it in the first place.
  const nameRejected = invalidOutputName(state.task.request.name);
  const cloudTranslation = ["translated-srt", "final-srt"].includes(
    state.task.request.stage,
  );
  // Nothing has finished on this machine yet, so the first run still has to
  // fetch model weights and warm the compiled separator. Several minutes of
  // that happen before any progress appears, which reads as a hang unless it
  // is said in advance.
  const firstRun = !state.history.some(
    (snapshot) => snapshot.state === "completed",
  );
  return (
    <div className="page page-new-task">
      <header className="page-header">
        <div>
          {/* <p className="page-kicker">{t.newTask.kicker}</p> */}
          <h1>{t.newTask.pageTitle}</h1>
          <p>
            {cloudTranslation
              ? t.newTask.pageDescriptionCloud
              : t.newTask.pageDescription}
          </p>
        </div>
        <div className="privacy-note">
          <ShieldCheck size={15} />
          <span>
            {cloudTranslation
              ? t.newTask.privacyNoteCloud
              : t.newTask.privacyNote}
          </span>
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

          {reusePinned ? (
            <div className="inline-note tab-note" role="note">
              <span>{t.newTask.reuse.active}</span>
              <button
                type="button"
                className="button button-secondary button-compact"
                disabled={busy}
                onClick={() => onRequestChange({ output: null })}
              >
                {t.newTask.reuse.cancel}
              </button>
            </div>
          ) : reusable ? (
            <div className="inline-note tab-note" role="note">
              <span>{t.newTask.reuse.available}</span>
              <button
                type="button"
                className="button button-secondary button-compact"
                disabled={busy}
                onClick={() => onReuse(reusable)}
              >
                {t.newTask.reuse.use}
              </button>
            </div>
          ) : null}

          {state.task.error ? (
            <div className="error-banner" role="alert">
              <strong>{state.task.error.message}</strong>
              {state.task.error.code === "api_key_required" ? (
                <span>{t.newTask.apiKeyError}</span>
              ) : null}
            </div>
          ) : null}

          {firstRun && hasFile ? (
            <div className="inline-note" role="note">
              {t.newTask.firstRunNotice}
            </div>
          ) : null}

          <div className="task-actions">
            <div>
              <strong>
                {resourcesBlocked
                  ? t.newTask.resourcesMissing
                  : hasFile
                    ? t.newTask.canStart
                    : t.newTask.waitingFile}
              </strong>
              <span>
                {resourcesBlocked
                  ? t.newTask.resourcesMissingHint.replace(
                    "{resources}",
                    // Names, not ids: `uv` is the whole managed Python runtime,
                    // and the page this sends the user to calls it that.
                    missingResources
                      .map(
                        (resource) =>
                          t.resources.items[
                            resource.id as keyof typeof t.resources.items
                          ]?.title || resource.id,
                      )
                      .join("、"),
                  )
                  : hasFile
                    ? t.newTask.willDownload
                    : t.newTask.supportedFormats}
              </span>
            </div>
            {resourcesBlocked ? (
              // Not disabled: there is something to do about it, and the page
              // that does it is one click away.
              <button
                type="button"
                className="button button-primary"
                onClick={onOpenResources}
              >
                {t.newTask.goToResources}
                <ArrowRight size={15} />
              </button>
            ) : (
              <button
                type="button"
                className="button button-primary"
                disabled={!hasFile || busy || nameRejected}
                onClick={onStart}
              >
                {busy ? t.newTask.preparing : t.newTask.startGenerate}
                <ArrowRight size={15} />
              </button>
            )}
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
