"use client";

import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import type { CapabilityState, TaskRequest } from "@/lib/types";

import { CustomSelect } from "./CustomSelect";


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
  const [advanced, setAdvanced] = useState(false);
  const translationSelected = ["translated-srt", "final-srt"].includes(
    request.stage,
  );

  return (
    <div className="task-settings">
      <div className="field-grid">
        <label className="field">
          <span>识别语言</span>
          <CustomSelect
            value={request.language ?? ""}
            disabled={disabled}
            onChange={(value) => onChange({ language: value || null })}
            options={[
              { value: "", label: "自动检测" },
              { value: "zh", label: "中文" },
              { value: "ja", label: "日语" },
              { value: "en", label: "英语" },
              { value: "ko", label: "韩语" },
            ]}
          />
        </label>

        <label className="field">
          <span>输出结果</span>
          <CustomSelect
            value={request.stage}
            disabled={disabled}
            onChange={(value) =>
              onChange({ stage: value as TaskRequest["stage"] })
            }
            options={[
              { value: "raw-srt", label: "原始字幕 SRT" },
              { value: "translated-srt", label: "纠错与翻译" },
              { value: "final-srt", label: "最终字幕" },
            ]}
          />
        </label>

        <label className="field">
          <span>处理设备</span>
          <CustomSelect
            value={request.device}
            disabled={disabled}
            onChange={(value) =>
              onChange({ device: value as "cuda" | "cpu" })
            }
            options={[
              { value: "cuda", label: "NVIDIA GPU" },
              { value: "cpu", label: "CPU（较慢）" },
            ]}
          />
        </label>
      </div>

      {translationSelected && !capabilities.translation ? (
        <div className="inline-note">
          翻译尚未配置。开始时会引导你填写 Gemini API Key，也可以继续生成原始字幕。
        </div>
      ) : null}

      <button
        type="button"
        className="advanced-toggle"
        aria-expanded={advanced}
        onClick={() => setAdvanced((value) => !value)}
      >
        <SlidersHorizontal size={14} />
        高级设置
        <ChevronDown
          size={14}
          className={advanced ? "is-rotated" : ""}
          aria-hidden="true"
        />
      </button>

      {advanced ? (
        <div className="advanced-grid">
          <label className="field">
            <span>Whisper 模型</span>
            <input
              value={request.model_name}
              disabled={disabled}
              onChange={(event) => onChange({ model_name: event.target.value })}
            />
          </label>
          <label className="field">
            <span>显存预算</span>
            <CustomSelect
              value={String(request.gpu_budget_gb)}
              disabled={disabled}
              onChange={(value) =>
                onChange({
                  gpu_budget_gb: Number(value) as 8 | 12 | 16,
                })
              }
              options={[
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
              <strong>翻译时使用联网检索</strong>
              <small>需要 Exa 或 Tavily Key；未配置时自动跳过</small>
            </span>
          </label>
        </div>
      ) : null}
    </div>
  );
}
