"use client";

import {
  ArrowLeft,
  CheckCircle2,
  CircleHelp,
  Monitor,
  Moon,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Star,
  Sun,
} from "lucide-react";
import { useMemo, useState } from "react";

import {
  formatCapability,
  formatUpdateSummary,
} from "@/lib/formatters";
import { detectAvailableFonts } from "@/lib/fonts";
import type { AppState } from "@/lib/state";
import type { UpdateCheck } from "@/lib/types";
import {
  FONT_SCALE_LABELS,
  type AppearanceSettings,
  type FontScale,
  type ThemeMode,
} from "@/lib/useAppearance";

import { ApiKeyField } from "./ApiKeyField";
import { clearConfirmMemory, listRememberedConfirms } from "./ConfirmDialog";
import { CustomSelect } from "./CustomSelect";


interface SettingsProps {
  state: AppState;
  appearance: AppearanceSettings;
  onAppearanceChange: (changes: Partial<AppearanceSettings>) => void;
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
  appearance: appearanceProp,
  onAppearanceChange,
  onSaveKey,
  onDeleteKey,
  onUseRawSubtitle,
  onCheckUpdates,
  onOpenUpdatePage,
}: SettingsProps) {
  const appearance = appearanceProp ?? { theme: "system" as ThemeMode, fontFamily: "", fontScale: "md" as FontScale };
  const [updateMessage, setUpdateMessage] = useState("");
  const [availableUpdate, setAvailableUpdate] = useState<UpdateCheck | null>(null);
  const [updateBusy, setUpdateBusy] = useState(false);
  const capability = formatCapability(state.capabilities);
  const apiError = state.task.error?.code === "api_key_required";

  const fonts = useMemo(() => detectAvailableFonts(), []);
  const fontOptions = useMemo(
    () => [
      { value: "", label: "默认字体" },
      ...fonts.map((f) => ({ value: f, label: f })),
    ],
    [fonts],
  );

  const scaleOptions = useMemo(
    () =>
      (Object.keys(FONT_SCALE_LABELS) as FontScale[]).map((key) => ({
        value: key,
        label: FONT_SCALE_LABELS[key],
      })),
    [],
  );

  const themeOptions: { value: ThemeMode; label: string; icon: typeof Sun }[] = [
    { value: "light", label: "浅色", icon: Sun },
    { value: "dark", label: "深色", icon: Moon },
    { value: "marisa", label: "魔理沙", icon: Star },
    { value: "reimu", label: "灵梦", icon: Sparkles },
    { value: "system", label: "跟随系统", icon: Monitor },
  ];

  return (
    <div className="page settings-page">
      <header className="page-header">
        <div>
          <p className="page-kicker">SETTINGS</p>
          <h1>设置</h1>
          <p>API Key 只保存在本机，不会显示或上传。</p>
        </div>
      </header>

      <section className="settings-section">
        <div className="settings-section-heading">
          <div>
            <h2>外观</h2>
            <p>主题、字体与字号设置，即时生效并自动保存。</p>
          </div>
        </div>

        <div className="appearance-grid">
          <div className="appearance-item">
            <span className="appearance-label">主题模式</span>
            <div className="theme-switcher">
              {themeOptions.map(({ value, label, icon: Icon }) => (
                <button
                  key={value}
                  type="button"
                  className={`theme-btn${appearance.theme === value ? " is-active" : ""}`}
                  onClick={() => onAppearanceChange({ theme: value })}
                >
                  <Icon size={15} />
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="appearance-item">
            <span className="appearance-label">字体</span>
            <CustomSelect
              ariaLabel="字体"
              value={appearance.fontFamily}
              onChange={(value) => onAppearanceChange({ fontFamily: value })}
              options={fontOptions}
            />
          </div>

          <div className="appearance-item">
            <span className="appearance-label">字体大小</span>
            <CustomSelect
              ariaLabel="字体大小"
              value={appearance.fontScale}
              onChange={(value) =>
                onAppearanceChange({ fontScale: value as FontScale })
              }
              options={scaleOptions}
            />
          </div>
        </div>
      </section>

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

      <section className="settings-section">
        <div className="settings-section-heading">
          <div>
            <h2>弹窗记忆</h2>
            <p>重置后，下次执行相关操作时会再次弹出确认窗口。</p>
          </div>
        </div>
        <button
          type="button"
          className="button button-secondary"
          onClick={() => {
            const remembered = listRememberedConfirms();
            for (const id of remembered) {
              clearConfirmMemory(id);
            }
            setUpdateMessage(`已重置 ${remembered.length} 条弹窗记忆`);
          }}
        >
          重置所有弹窗记忆
        </button>
      </section>
    </div>
  );
}
