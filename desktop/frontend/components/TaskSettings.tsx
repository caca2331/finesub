"use client";

import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import { invalidOutputName } from "@/lib/formatters";
import type { CapabilityState, TaskRequest } from "@/lib/types";

import { CustomSelect } from "./CustomSelect";
import { useLanguage } from "./LanguageProvider";


interface TaskSettingsProps {
  request: Omit<TaskRequest, "input">;
  capabilities: CapabilityState;
  disabled?: boolean;
  onChange: (changes: Partial<Omit<TaskRequest, "input">>) => void;
}


export function TaskSettings({
  request,
  capabilities,
  disabled,
  onChange,
}: TaskSettingsProps) {
  const { t } = useLanguage();
  const [advanced, setAdvanced] = useState(false);
  const [nameError, setNameError] = useState("");
  const translationSelected = ["translated-srt", "final-srt"].includes(
    request.stage,
  );

  return (
    <div className="task-settings">
      <div className="field-grid">
        <label className="field">
          <span>{t.newTask.settings.language}</span>
          <CustomSelect
            value={request.language ?? ""}
            disabled={disabled}
            onChange={(value) => onChange({ language: value || null })}
            options={[
              { value: "", label: t.newTask.settings.languageAuto },
              { value: "zh", label: t.newTask.settings.languageZh },
              { value: "ja", label: t.newTask.settings.languageJa },
              { value: "en", label: t.newTask.settings.languageEn },
              { value: "ko", label: t.newTask.settings.languageKo },
            ]}
          />
        </label>

        <label className="field">
          <span>{t.newTask.settings.output}</span>
          <CustomSelect
            value={request.stage === "raw-srt" ? "raw-srt" : "final-srt"}
            disabled={disabled}
            onChange={(value) =>
              onChange({ stage: value as TaskRequest["stage"] })
            }
            options={[
              { value: "raw-srt", label: t.newTask.settings.outputRaw },
              { value: "final-srt", label: t.newTask.settings.outputFinal },
            ]}
          />
        </label>

        <label className="field field-wide">
          <span>{t.newTask.settings.extraInfo}</span>
          <textarea
            value={request.extra_info}
            disabled={disabled}
            rows={3}
            placeholder={t.newTask.settings.extraInfoPlaceholder}
            onChange={(event) => onChange({ extra_info: event.target.value })}
          />
        </label>
      </div>

      {translationSelected && !capabilities.translation ? (
        <div className="inline-note">
          {t.newTask.apiKeyError}
        </div>
      ) : null}

      <button
        type="button"
        className="advanced-toggle"
        aria-expanded={advanced}
        onClick={() => setAdvanced((value) => !value)}
      >
        <SlidersHorizontal size={14} />
        {t.newTask.settings.advanced}
        <ChevronDown
          size={14}
          className={advanced ? "is-rotated" : ""}
          aria-hidden="true"
        />
      </button>

      {advanced ? (
        <div className="advanced-grid">
          <label className="field">
            <span>{t.newTask.settings.whisperModel}</span>
            <input
              value={request.model_name}
              disabled={disabled}
              onChange={(event) => onChange({ model_name: event.target.value })}
            />
          </label>
          <label className="field">
            <span>{t.newTask.settings.gpuBudget}</span>
            <CustomSelect
              value={String(request.gpu_budget_gb)}
              disabled={disabled}
              onChange={(value) =>
                onChange({
                  gpu_budget_gb: Number(value) as 4 | 8 | 12 | 16,
                })
              }
              options={[
                { value: "4", label: "4 GB" },
                { value: "8", label: "8 GB" },
                { value: "12", label: "12 GB" },
                { value: "16", label: "16 GB" },
              ]}
            />
          </label>
          <label className="field">
            <span>{t.newTask.settings.device}</span>
            <CustomSelect
              value={request.device}
              disabled={disabled}
              onChange={(value) =>
                onChange({ device: value as "cuda" | "cpu" })
              }
              options={[
                { value: "cuda", label: t.newTask.settings.deviceGpu },
                { value: "cpu", label: t.newTask.settings.deviceCpu },
              ]}
            />
          </label>
          <label className="field">
            <span>{t.newTask.settings.outputName}</span>
            <input
              value={request.name}
              disabled={disabled}
              onChange={(event) => {
                const value = event.target.value;
                // Explained only when it is wrong: the rule is narrow enough
                // that a permanent hint is noise on every other keystroke.
                setNameError(
                  invalidOutputName(value) ? t.newTask.settings.outputNameError : "",
                );
                onChange({ name: value });
              }}
            />
            {nameError ? <small className="field-error">{nameError}</small> : null}
          </label>
          <label className="field">
            <span>{t.newTask.settings.knowledge}</span>
            <CustomSelect
              value={request.knowledge === "update" ? "update" : "none"}
              disabled={disabled}
              onChange={(value) =>
                onChange({ knowledge: value as "none" | "update" })
              }
              options={[
                { value: "update", label: t.newTask.settings.knowledgeUpdate },
                { value: "none", label: t.newTask.settings.knowledgeNone },
              ]}
            />
          </label>
          <label className="switch-row">
            <input
              type="checkbox"
              checked={request.cleanup_intermediate}
              disabled={disabled}
              onChange={(event) =>
                onChange({ cleanup_intermediate: event.target.checked })
              }
            />
            <span>
              <strong>{t.newTask.settings.cleanup}</strong>
              <small>{t.newTask.settings.cleanupHint}</small>
            </span>
          </label>
        </div>
      ) : null}
    </div>
  );
}
