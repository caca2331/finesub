"use client";

import {
  ArrowLeft,
  CheckCircle2,
  CircleHelp,
  ExternalLink,
  Github,
  Heart,
  Monitor,
  Moon,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Star,
  Sun,
} from "lucide-react";
import { useMemo, useState } from "react";

import { formatBytes } from "@/lib/formatters";
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
import { CustomSelect } from "./CustomSelect";
import { useLanguage } from "./LanguageProvider";


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
  const appearance = appearanceProp ?? { theme: "system" as ThemeMode, fontFamily: "", fontScale: "md" as FontScale, glassOpacity: 75 };
  const [updateMessage, setUpdateMessage] = useState("");
  const [availableUpdate, setAvailableUpdate] = useState<UpdateCheck | null>(null);
  const [updateBusy, setUpdateBusy] = useState(false);
  const [closeWindowAction, setCloseWindowAction] = useState(
    () => localStorage.getItem("close-window-action") || "minimize"
  );
  const apiError = state.task.error?.code === "api_key_required";
  const { language, setLanguage, t } = useLanguage();
  const capability = {
    tone: state.capabilities.translation ? "success" : "neutral",
    title: state.capabilities.translation
      ? t.sidebar.translationReady
      : t.sidebar.localOnly,
  } as const;

  const fonts = useMemo(() => detectAvailableFonts(), []);
  const fontOptions = useMemo(
    () => [
      { value: "", label: t.settings.appearance.defaultFont },
      ...fonts.map((f) => ({ value: f, label: f })),
    ],
    [fonts, t],
  );

  const scaleOptions = useMemo(
    () =>
      (Object.keys(FONT_SCALE_LABELS) as FontScale[]).map((key) => ({
        value: key,
        label: t.settings.fontScale[key],
      })),
    [t],
  );

  const themeOptions: { value: ThemeMode; label: string; icon: typeof Sun }[] = [
    { value: "light", label: t.settings.theme.light, icon: Sun },
    { value: "dark", label: t.settings.theme.dark, icon: Moon },
    { value: "marisa", label: t.settings.theme.marisa, icon: Star },
    { value: "reimu", label: t.settings.theme.reimu, icon: Sparkles },
    { value: "yanami", label: t.settings.theme.yanami, icon: Heart },
    { value: "system", label: t.settings.theme.system, icon: Monitor },
  ];

  const languageOptions = [
    { value: "zh", label: t.settings.language.zh },
    { value: "en", label: t.settings.language.en },
  ];

  return (
    <div className="page settings-page">
      <header className="page-header">
        <div>
          {/* <p className="page-kicker">{t.settings.kicker}</p> */}
          <h1>{t.settings.title}</h1>
          <p>{t.settings.description}</p>
        </div>
      </header>

      <section className="settings-section">
        <div className="settings-section-heading">
          <div>
            <h2>{t.settings.appearance.title}</h2>
            <p>{t.settings.appearance.description}</p>
          </div>
        </div>

        <div className="appearance-grid">
          <div className="appearance-item appearance-item-vertical">
            <span className="appearance-label">{t.settings.appearance.theme}</span>
            <div className="theme-switcher">
              {themeOptions.map(({ value, label, icon: Icon }) => (
                <button
                  key={value}
                  type="button"
                  data-theme-option={value}
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
            <span className="appearance-label">{t.settings.appearance.fontFamily}</span>
            <CustomSelect
              value={appearance.fontFamily}
              onChange={(value) => onAppearanceChange({ fontFamily: value })}
              options={fontOptions}
            />
          </div>

          <div className="appearance-item">
            <span className="appearance-label">{t.settings.appearance.fontSize}</span>
            <CustomSelect
              value={appearance.fontScale}
              onChange={(value) =>
                onAppearanceChange({ fontScale: value as FontScale })
              }
              options={scaleOptions}
            />
          </div>

          <div className="appearance-item">
            <span className="appearance-label">{t.settings.language.label}</span>
            <CustomSelect
              value={language}
              onChange={(value) => setLanguage(value as "zh" | "en")}
              options={languageOptions}
            />
          </div>

          <div className="appearance-item glass-opacity-item">
            <div className="glass-opacity-label">
              <span className="appearance-label">{t.settings.appearance.glassOpacity}</span>
              <small>{t.settings.appearance.glassOpacityHint}</small>
            </div>
            <div className="glass-opacity-control">
              <input
                type="range"
                min="40"
                max="100"
                step="1"
                value={appearance.glassOpacity}
                onChange={(e) => onAppearanceChange({ glassOpacity: Number(e.target.value) })}
                className="glass-opacity-slider"
              />
              <span className="glass-opacity-value">{appearance.glassOpacity}%</span>
            </div>
          </div>
        </div>
      </section>

      {apiError ? (
        <section className="settings-callout">
          <div className="callout-icon">
            <CircleHelp size={18} />
          </div>
          <div>
            <strong>{t.apiError.title}</strong>
            <p>{t.apiError.description}</p>
          </div>
          <button
            type="button"
            className="button button-secondary"
            onClick={onUseRawSubtitle}
          >
            <ArrowLeft size={14} />
            {t.apiError.rawSubtitleOnly}
          </button>
        </section>
      ) : null}

      <section className="settings-section">
        <div className="settings-section-heading">
          <div>
            <h2>{t.settings.translation.title}</h2>
            <p>{t.settings.translation.description}</p>
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
            label="Gemini Free"
            description={t.settings.translation.gemini}
            placeholder="AIza…"
            status={state.settings.api_keys.gemini}
            onSave={(value) => onSaveKey("gemini", value)}
            onDelete={() => onDeleteKey("gemini")}
          />
          <ApiKeyField
            label="Exa"
            description={t.settings.translation.exa}
            placeholder="exa-…"
            status={state.settings.api_keys.exa}
            onSave={(value) => onSaveKey("exa", value)}
            onDelete={() => onDeleteKey("exa")}
          />
          <ApiKeyField
            label="Tavily"
            description={t.settings.translation.tavily}
            placeholder="tvly-…"
            status={state.settings.api_keys.tavily}
            onSave={(value) => onSaveKey("tavily", value)}
            onDelete={() => onDeleteKey("tavily")}
          />
        </div>
      </section>

      <section className="settings-section update-section">
        <div>
          <h2>{t.settings.updates.title}</h2>
          <p>{t.settings.updates.description}</p>
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
                  setUpdateMessage(t.settings.updates.openedInBrowser);
                } catch (error) {
                  setUpdateMessage(
                    error instanceof Error ? error.message : "Unable to open download page",
                  );
                } finally {
                  setUpdateBusy(false);
                }
              }}
            >
              <RefreshCw size={14} />
              {t.settings.updates.openDownloadPage}
            </button>
          ) : null}
          <button
            type="button"
            className="button button-secondary"
            disabled={updateBusy}
            onClick={async () => {
              setUpdateBusy(true);
              setUpdateMessage(t.settings.updates.checking);
              try {
                const result = await onCheckUpdates();
                setAvailableUpdate(result);
                setUpdateMessage(
                  result.available
                    ? t.settings.updates.available
                        .replace("{version}", result.version)
                        .replace(
                          "{kind}",
                          result.kind === "full"
                            ? t.settings.updates.full
                            : t.settings.updates.patch,
                        )
                        .replace("{size}", formatBytes(result.size))
                    : t.settings.updates.latest,
                );
              } catch (error) {
                setAvailableUpdate(null);
                setUpdateMessage(
                  error instanceof Error
                    ? error.message
                    : t.settings.updates.noUpdateSource,
                );
              } finally {
                setUpdateBusy(false);
              }
            }}
          >
            <RefreshCw size={14} className={updateBusy ? "spin" : ""} />
            {t.settings.updates.checkUpdate}
          </button>
        </div>
      </section>

      <section className="settings-section">
        <div className="settings-section-heading">
          <div>
            <h2>{t.settings.confirmMemory.title}</h2>
            {/* <p>{t.settings.confirm-Memory.description}</p> */}
          </div>
        </div>
        <div className="confirm-memory-list">
          <div className="confirm-memory-row">
            <span className="confirm-memory-label">{t.settings.confirmMemory.closePanel}</span>
            <CustomSelect
              value={closeWindowAction}
              onChange={(value) => {
                const action = value || "minimize";
                localStorage.setItem("close-window-action", action);
                setCloseWindowAction(action);
              }}
              options={[
                { value: "minimize", label: t.settings.confirmMemory.minimizeToTray },
                { value: "close", label: t.settings.confirmMemory.exitApp },
              ]}
            />
          </div>
        </div>
      </section>

      <section className="settings-section">
        <div className="settings-section-heading">
          <div>
            <h2>{t.settings.acknowledgment.title}</h2>
            <p>{t.settings.acknowledgment.description}</p>
          </div>
        </div>
        <div className="acknowledgment-content">
          <div className="acknowledgment-item">
            <Github size={16} />
            <div className="acknowledgment-info">
              <span className="acknowledgment-label">{t.settings.acknowledgment.github}</span>
              <a
                href="https://github.com/caca2331/finesub"
                target="_blank"
                rel="noopener noreferrer"
                className="acknowledgment-link"
              >
                caca2331/finesub
                <ExternalLink size={12} />
              </a>
            </div>
          </div>
          <div className="acknowledgment-item">
            <Heart size={16} />
            <div className="acknowledgment-info">
              <span className="acknowledgment-label">{t.settings.acknowledgment.author}</span>
              <div className="acknowledgment-authors">
                <span className="acknowledgment-value">caca2331</span>
                <span className="acknowledgment-value">tuzibuqiahuluobo</span>
                <span className="acknowledgment-value">星光</span>
              </div>
            </div>
          </div>
          <div className="acknowledgment-item">
            <ExternalLink size={16} />
            <div className="acknowledgment-info">
              <span className="acknowledgment-label">{t.settings.acknowledgment.documentation}</span>
              <a
                href="https://github.com/caca2331/finesub#readme"
                target="_blank"
                rel="noopener noreferrer"
                className="acknowledgment-link"
              >
                {t.settings.acknowledgment.viewDocs}
                <ExternalLink size={12} />
              </a>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
