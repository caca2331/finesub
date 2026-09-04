"use client";

import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import { invalidOutputName } from "@/lib/formatters";
import type { CapabilityState, GpuTier, TaskRequest } from "@/lib/types";

import { CustomSelect } from "./CustomSelect";
import { useLanguage } from "./LanguageProvider";


interface TaskSettingsProps {
  request: Omit<TaskRequest, "input">;
  capabilities: CapabilityState;
  disabled?: boolean;
  onChange: (changes: Partial<Omit<TaskRequest, "input">>) => void;
}


type SettingsTab = "speech" | "llm";


export function TaskSettings({
  request,
  capabilities,
  disabled,
  onChange,
}: TaskSettingsProps) {
  const { t } = useLanguage();
  const [tab, setTab] = useState<SettingsTab>("speech");
  const [advanced, setAdvanced] = useState(false);
  const [nameError, setNameError] = useState("");
  const translationSelected = ["translated-srt", "final-srt"].includes(
    request.stage,
  );
  // Surfaced on the tab itself: the note explaining the missing key lives
  // inside the LLM panel, which the user may not have open.
  const llmNeedsKey = translationSelected && !capabilities.translation;

  return (
    <div className="task-settings">
      {/* Above the tabs, not inside either: this is the switch that decides
          whether the LLM tab means anything. */}
      <div className="field-grid">
        <div className="field">
          <span>{t.newTask.settings.output}</span>
          <CustomSelect
            value={request.stage === "raw-srt" ? "raw-srt" : "final-srt"}
            disabled={disabled}
            ariaLabel={t.newTask.settings.output}
            onChange={(value) =>
              onChange({ stage: value as TaskRequest["stage"] })
            }
            options={[
              { value: "raw-srt", label: t.newTask.settings.outputRaw },
              { value: "final-srt", label: t.newTask.settings.outputFinal },
            ]}
          />
        </div>
      </div>

      <div className="task-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "speech"}
          className={`task-tab${tab === "speech" ? " is-active" : ""}`}
          onClick={() => setTab("speech")}
        >
          {t.newTask.settings.tabSpeech}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "llm"}
          className={`task-tab${tab === "llm" ? " is-active" : ""}`}
          onClick={() => setTab("llm")}
        >
          {t.newTask.settings.tabLlm}
          {llmNeedsKey ? <span className="tab-dot" aria-hidden="true" /> : null}
        </button>
      </div>

      {tab === "speech" ? (
        <div className="field-grid" role="tabpanel">
          <div className="field">
            <span>{t.newTask.settings.language}</span>
            <CustomSelect
              value={request.language ?? ""}
              disabled={disabled}
              ariaLabel={t.newTask.settings.language}
              onChange={(value) => onChange({ language: value || null })}
              options={[
                { value: "", label: t.newTask.settings.languageAuto },
                { value: "zh", label: t.newTask.settings.languageZh },
                { value: "ja", label: t.newTask.settings.languageJa },
                { value: "en", label: t.newTask.settings.languageEn },
                { value: "ko", label: t.newTask.settings.languageKo },
              ]}
            />
          </div>
          <div className="field">
            <span>{t.newTask.settings.gpuTier}</span>
            <CustomSelect
              value={request.gpu_tier}
              disabled={disabled}
              ariaLabel={t.newTask.settings.gpuTier}
              onChange={(value) => onChange({ gpu_tier: value as GpuTier })}
              options={[
                { value: "auto", label: t.newTask.settings.gpuTierAuto },
                { value: "cpu", label: t.newTask.settings.gpuTierCpu },
                { value: "entry", label: t.newTask.settings.gpuTierEntry },
                { value: "standard", label: t.newTask.settings.gpuTierStandard },
                {
                  value: "standard_large_vram",
                  label: t.newTask.settings.gpuTierStandardLargeVram,
                },
                { value: "high", label: t.newTask.settings.gpuTierHigh },
              ]}
            />
          </div>
        </div>
      ) : (
        <div className="tab-panel" role="tabpanel">
          {!translationSelected ? (
            <div className="inline-note tab-note">
              <span>{t.newTask.settings.llmInactive}</span>
              <button
                type="button"
                className="button button-secondary button-compact"
                disabled={disabled}
                onClick={() => onChange({ stage: "final-srt" })}
              >
                {t.newTask.settings.llmEnable}
              </button>
            </div>
          ) : null}
          {llmNeedsKey ? (
            <div className="inline-note">{t.newTask.apiKeyError}</div>
          ) : null}
          {/* Values are kept, not cleared, while the stage leaves them unused:
              switching back to final-srt must find them where they were. */}
          <div className="field-grid">
            <label className="field field-wide">
              <span>{t.newTask.settings.extraInfo}</span>
              <textarea
                value={request.extra_info}
                disabled={disabled || !translationSelected}
                rows={3}
                placeholder={t.newTask.settings.extraInfoPlaceholder}
                onChange={(event) =>
                  onChange({ extra_info: event.target.value })
                }
              />
            </label>
            <div className="field">
              <span>{t.newTask.settings.knowledge}</span>
              <CustomSelect
                value={request.knowledge === "update" ? "update" : "none"}
                disabled={disabled || !translationSelected}
                ariaLabel={t.newTask.settings.knowledge}
                onChange={(value) =>
                  onChange({ knowledge: value as "none" | "update" })
                }
                options={[
                  { value: "update", label: t.newTask.settings.knowledgeUpdate },
                  { value: "none", label: t.newTask.settings.knowledgeNone },
                ]}
              />
            </div>
          </div>
        </div>
      )}

      {/* Both apply to every run, not just an unusual one, so they sit in the
          open: the naming choice has to be made before the task starts, and
          the cleanup switch decides what survives it. */}
      <div className="advanced-grid">
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
        <div className="advanced-grid advanced-grid-animated">
          <p className="advanced-empty">{t.newTask.settings.advancedEmpty}</p>
        </div>
      ) : null}
    </div>
  );
}
