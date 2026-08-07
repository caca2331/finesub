"use client";

import {
  ArrowLeft,
  BookOpen,
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
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { formatBytes } from "@/lib/formatters";
import { detectAvailableFonts } from "@/lib/fonts";
import type { AppState } from "@/lib/state";
import type { UpdateCheck, UpdateInstallSnapshot } from "@/lib/types";
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
  onInstallUpdate: (
    kind: "app" | "full",
    version: string,
  ) => Promise<UpdateInstallSnapshot>;
  onGetUpdateInstall: () => Promise<UpdateInstallSnapshot | null>;
  onCloseWindow: () => Promise<unknown>;
  onOpenUpdatePage: () => Promise<unknown>;
}

type ViewTransitionDocument = Document & {
  startViewTransition?: (update: () => void) => unknown;
};


export function Settings({
  state,
  appearance: appearanceProp,
  onAppearanceChange,
  onSaveKey,
  onDeleteKey,
  onUseRawSubtitle,
  onCheckUpdates,
  onInstallUpdate,
  onGetUpdateInstall,
  onCloseWindow,
  onOpenUpdatePage,
}: SettingsProps) {
  const appearance = appearanceProp ?? { theme: "system" as ThemeMode, fontFamily: "", fontScale: "md" as FontScale, glassOpacity: 75, animations: true };
  const [updateMessage, setUpdateMessage] = useState("");
  const [availableUpdate, setAvailableUpdate] = useState<UpdateCheck | null>(null);
  const [updateBusy, setUpdateBusy] = useState(false);
  const [install, setInstall] = useState<UpdateInstallSnapshot | null>(null);
  const [docsOpen, setDocsOpen] = useState(false);
  // A download runs in a backend thread, so the page owns no progress of its
  // own -- it polls the snapshot until the install reaches a terminal state.
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollingRef.current !== null) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const pollInstall = useCallback(async () => {
    try {
      const snapshot = await onGetUpdateInstall();
      setInstall(snapshot);
      if (snapshot === null || snapshot.state === "ready" || snapshot.state === "failed") {
        stopPolling();
      }
    } catch {
      // A poll that fails is not itself a failed install; keep the last
      // snapshot on screen and let the next tick decide.
    }
  }, [onGetUpdateInstall, stopPolling]);

  useEffect(() => {
    // Reopening Settings mid-download has to find the install still running.
    void pollInstall();
    return stopPolling;
  }, [pollInstall, stopPolling]);

  const startPolling = useCallback(() => {
    stopPolling();
    pollingRef.current = setInterval(() => {
      void pollInstall();
    }, 500);
  }, [pollInstall, stopPolling]);

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

  const selectTheme = (theme: ThemeMode) => {
    if (theme === appearance.theme) return;
    const commit = () => onAppearanceChange({ theme });
    const startViewTransition = (document as ViewTransitionDocument).startViewTransition;
    if (!appearance.animations || !startViewTransition) {
      commit();
      return;
    }
    startViewTransition.call(document, () => flushSync(commit));
  };

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
                  onClick={() => selectTheme(value)}
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

          <div className="appearance-item motion-preference-item">
            <div className="motion-preference-copy">
              <span className="appearance-label">{t.settings.appearance.animations}</span>
              <small>{t.settings.appearance.animationsHint}</small>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={appearance.animations}
              aria-label={t.settings.appearance.animations}
              className={`motion-toggle${appearance.animations ? " is-on" : ""}`}
              onClick={() => onAppearanceChange({ animations: !appearance.animations })}
            >
              <span aria-hidden="true" />
            </button>
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
        {install ? (
          <div className="update-install" role="status" aria-live="polite">
            {install.state === "running" || install.state === "queued" ? (
              <>
                <div className="update-progress">
                  <div
                    className="update-progress-bar"
                    style={{
                      width: install.total
                        ? `${Math.min(100, (install.downloaded / install.total) * 100)}%`
                        : "100%",
                    }}
                  />
                </div>
                <span className="update-message">
                  {install.phase === "downloading" && install.total
                    ? t.settings.updates.downloading
                        .replace("{done}", formatBytes(install.downloaded))
                        .replace("{total}", formatBytes(install.total))
                    : t.settings.updates.installing}
                </span>
              </>
            ) : null}
            {install.state === "ready" ? (
              <span className="update-message">
                {install.exit_required
                  ? t.settings.updates.exitRequired
                  : t.settings.updates.restartRequired}
              </span>
            ) : null}
            {install.state === "failed" ? (
              <span className="update-message update-message-error">
                {t.settings.updates.installFailed.replace("{error}", install.error)}
              </span>
            ) : null}
          </div>
        ) : null}
        <div className="update-actions">
          {install?.state === "ready" ? (
            <button
              type="button"
              className="button button-primary"
              onClick={() => {
                // Both paths end the process. An app delta is already staged on
                // disk, so the next launch picks it up; a full update needs this
                // one gone before the external updater can replace it.
                void onCloseWindow();
              }}
            >
              <RefreshCw size={14} />
              {install.exit_required
                ? t.settings.updates.exitNow
                : t.settings.updates.restartNow}
            </button>
          ) : null}
          {availableUpdate?.available &&
          availableUpdate.kind &&
          install?.state !== "ready" &&
          install?.state !== "running" &&
          install?.state !== "queued" ? (
            <button
              type="button"
              className="button button-primary"
              disabled={updateBusy}
              onClick={async () => {
                const kind = availableUpdate.kind;
                if (!kind) {
                  return;
                }
                setUpdateBusy(true);
                setUpdateMessage("");
                try {
                  setInstall(await onInstallUpdate(kind, availableUpdate.version));
                  startPolling();
                } catch (error) {
                  setUpdateMessage(
                    error instanceof Error ? error.message : "Unable to install update",
                  );
                } finally {
                  setUpdateBusy(false);
                }
              }}
            >
              <RefreshCw size={14} />
              {install?.state === "failed"
                ? t.settings.updates.retryInstall
                : t.settings.updates.install}
            </button>
          ) : null}
          {availableUpdate?.available && availableUpdate.kind ? (
            <button
              type="button"
              className="button button-secondary"
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
              <ExternalLink size={14} />
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
                <span className="acknowledgment-value">回不去的星光</span>
              </div>
            </div>
          </div>
          <div className="acknowledgment-item">
            <BookOpen size={16} />
            <div className="acknowledgment-info">
              <span className="acknowledgment-label">{t.settings.acknowledgment.documentation}</span>
              <button
                type="button"
                className="acknowledgment-link"
                onClick={() => setDocsOpen(true)}
              >
                {t.settings.acknowledgment.viewDocs}
                <BookOpen size={12} />
              </button>
            </div>
          </div>
        </div>
      </section>

      {docsOpen ? (
        <div className="dialog-overlay" onClick={() => setDocsOpen(false)}>
          <article
            className="dialog-card docs-dialog"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="docs-dialog-header">
              <div>
                <span className="eyebrow">FineSub Desktop</span>
                <h3>{t.settings.acknowledgment.docs.title}</h3>
              </div>
              <button
                type="button"
                className="icon-button"
                aria-label={t.settings.acknowledgment.docs.close}
                onClick={() => setDocsOpen(false)}
              >
                <X size={18} />
              </button>
            </div>
            <p>{t.settings.acknowledgment.docs.intro}</p>
            <div className="docs-dialog-content">
              {t.settings.acknowledgment.docs.sections.map((section) => (
                <section key={section.title}>
                  <h4>{section.title}</h4>
                  <p>{section.body}</p>
                </section>
              ))}
            </div>
            <div className="dialog-actions">
              <button
                type="button"
                className="button button-primary"
                onClick={() => setDocsOpen(false)}
              >
                {t.settings.acknowledgment.docs.close}
              </button>
            </div>
          </article>
        </div>
      ) : null}
    </div>
  );
}
