"use client";

import {
  ArrowLeft,
  CheckCircle2,
  CircleHelp,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { useState } from "react";

import {
  formatCapability,
  formatUpdateSummary,
} from "@/lib/formatters";
import type { AppState } from "@/lib/state";
import type { UpdateCheck } from "@/lib/types";

import { ApiKeyField } from "./ApiKeyField";


interface SettingsProps {
  state: AppState;
  onSaveKey: (
    provider: "gemini" | "exa" | "tavily",
    value: string,
  ) => Promise<void>;
  onDeleteKey: (provider: "gemini" | "exa" | "tavily") => Promise<void>;
  onUseRawSubtitle: () => void;
  onCheckUpdates: () => Promise<UpdateCheck>;
  onOpenUpdatePage: () => Promise<unknown>;
}


export function Settings({
  state,
  onSaveKey,
  onDeleteKey,
  onUseRawSubtitle,
  onCheckUpdates,
  onOpenUpdatePage,
}: SettingsProps) {
  const [updateMessage, setUpdateMessage] = useState("");
  const [availableUpdate, setAvailableUpdate] = useState<UpdateCheck | null>(null);
  const [updateBusy, setUpdateBusy] = useState(false);
  const capability = formatCapability(state.capabilities);
  const apiError = state.task.error?.code === "api_key_required";

  return (
    <div className="page settings-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">SETTINGS</p>
          <h1>设置</h1>
          <p>API Key 只保存在本机，不会显示或上传。</p>
        </div>
      </header>

      {apiError ? (
        <section className="settings-callout">
          <div className="callout-icon">
            <CircleHelp size={18} />
          </div>
          <div>
            <strong>翻译功能需要 Gemini API Key</strong>
            <p>
              你可以在下方保存 Key，原任务参数会保留；也可以不配置，继续生成原始字幕。
            </p>
          </div>
          <button
            type="button"
            className="button button-secondary"
            onClick={onUseRawSubtitle}
          >
            <ArrowLeft size={14} />
            仅生成原始字幕
          </button>
        </section>
      ) : null}

      <section className="settings-section">
        <div className="settings-section-heading">
          <div>
            <h2>翻译与联网能力</h2>
            <p>所有字段均为写入式；应用不会把已保存的密钥返回给前端。</p>
          </div>
          <span className={`capability-chip is-${capability.tone}`}>
            {capability.tone === "success" ? (
              <CheckCircle2 size={13} />
            ) : (
              <ShieldCheck size={13} />
            )}
            {capability.title}
          </span>
        </div>

        <div className="api-key-list">
          <ApiKeyField
            label="Gemini"
            description="字幕纠错、翻译与风格整理"
            placeholder="AIza…"
            status={state.settings.api_keys.gemini}
            onSave={(value) => onSaveKey("gemini", value)}
            onDelete={() => onDeleteKey("gemini")}
          />
          <ApiKeyField
            label="Exa"
            description="翻译阶段的术语与背景检索"
            placeholder="exa-…"
            status={state.settings.api_keys.exa}
            onSave={(value) => onSaveKey("exa", value)}
            onDelete={() => onDeleteKey("exa")}
          />
          <ApiKeyField
            label="Tavily"
            description="可替代 Exa 的联网检索服务"
            placeholder="tvly-…"
            status={state.settings.api_keys.tavily}
            onSave={(value) => onSaveKey("tavily", value)}
            onDelete={() => onDeleteKey("tavily")}
          />
        </div>
      </section>

      <section className="settings-section update-section">
        <div>
          <h2>应用更新</h2>
          <p>正式版读取签名的 GitHub Release；当前版本仅提供手动下载安装。</p>
          {updateMessage ? <span className="update-message">{updateMessage}</span> : null}
          {availableUpdate?.available && availableUpdate.releaseNotes ? (
            <p className="update-notes">{availableUpdate.releaseNotes}</p>
          ) : null}
        </div>
        <div className="update-actions">
          {availableUpdate?.available && availableUpdate.kind ? (
            <button
              type="button"
              className="button button-primary"
              disabled={updateBusy}
              onClick={async () => {
                setUpdateBusy(true);
                try {
                  await onOpenUpdatePage();
                  setUpdateMessage("已在浏览器中打开下载页面");
                } catch (error) {
                  setUpdateMessage(
                    error instanceof Error ? error.message : "无法打开下载页面",
                  );
                } finally {
                  setUpdateBusy(false);
                }
              }}
            >
              <RefreshCw size={14} />
              打开下载页面
            </button>
          ) : null}
          <button
            type="button"
            className="button button-secondary"
            disabled={updateBusy}
            onClick={async () => {
              setUpdateBusy(true);
              setUpdateMessage("正在检查…");
              try {
                const result = await onCheckUpdates();
                setAvailableUpdate(result);
                setUpdateMessage(formatUpdateSummary(result));
              } catch (error) {
                setAvailableUpdate(null);
                setUpdateMessage(
                  error instanceof Error
                    ? error.message
                    : "当前开发构建未配置更新源",
                );
              } finally {
                setUpdateBusy(false);
              }
            }}
          >
            <RefreshCw size={14} className={updateBusy ? "spin" : ""} />
            检查更新
          </button>
        </div>
      </section>
    </div>
  );
}
