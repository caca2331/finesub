"use client";

import { ChevronDown, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import type { CapabilityState, TaskRequest } from "@/lib/types";


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
          <div className="select-wrap">
            <select
              value={request.language ?? ""}
              disabled={disabled}
              onChange={(event) =>
                onChange({ language: event.target.value || null })
              }
            >
              <option value="">自动检测</option>
              <option value="zh">中文</option>
              <option value="ja">日语</option>
              <option value="en">英语</option>
              <option value="ko">韩语</option>
            </select>
            <ChevronDown size={14} aria-hidden="true" />
          </div>
        </label>

        <label className="field">
          <span>输出结果</span>
          <div className="select-wrap">
            <select
              value={request.stage}
              disabled={disabled}
              onChange={(event) =>
                onChange({
                  stage: event.target.value as TaskRequest["stage"],
                })
              }
            >
              <option value="raw-srt">原始字幕 SRT</option>
              <option value="translated-srt">纠错与翻译</option>
              <option value="final-srt">最终字幕</option>
            </select>
            <ChevronDown size={14} aria-hidden="true" />
          </div>
        </label>

        <label className="field">
          <span>处理设备</span>
          <div className="select-wrap">
            <select
              value={request.device}
              disabled={disabled}
              onChange={(event) =>
                onChange({ device: event.target.value as "cuda" | "cpu" })
              }
            >
              <option value="cuda">NVIDIA GPU</option>
              <option value="cpu">CPU（较慢）</option>
            </select>
            <ChevronDown size={14} aria-hidden="true" />
          </div>
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
            <div className="select-wrap">
              <select
                value={request.gpu_budget_gb}
                disabled={disabled}
                onChange={(event) =>
                  onChange({
                    gpu_budget_gb: Number(event.target.value) as 8 | 12 | 16,
                  })
                }
              >
                <option value={8}>8 GB</option>
                <option value={12}>12 GB</option>
                <option value={16}>16 GB</option>
              </select>
              <ChevronDown size={14} aria-hidden="true" />
            </div>
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
