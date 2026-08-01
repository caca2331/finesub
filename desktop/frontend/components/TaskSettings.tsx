"use client";

import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

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
            value={request.stage}
            disabled={disabled}
            onChange={(value) =>
              onChange({ stage: value as TaskRequest["stage"] })
            }
            options={[
              { value: "raw-srt", label: t.newTask.settings.outputRaw },
              { value: "translated-srt", label: t.newTask.settings.outputTranslated },
              { value: "final-srt", label: t.newTask.settings.outputFinal },
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
          <label className="switch-row">
            <input
              type="checkbox"
              checked={request.enable_web_search}
              disabled={disabled}
              onChange={(event) =>
                onChange({ enable_web_search: event.target.checked })
              }
            />
            <span>
              <strong>{t.newTask.settings.webSearch}</strong>
              <small>{t.newTask.settings.webSearchHint}</small>
            </span>
          </label>
        </div>
      ) : null}
    </div>
  );
}
