"use client";

import { ExternalLink, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { formatBytes } from "@/lib/formatters";
import { saveUi, uiValue } from "@/lib/preferences";
import type { UpdateCheck, UpdateInstallSnapshot } from "@/lib/types";

import { useLanguage } from "./LanguageProvider";


export interface UpdateSectionProps {
  /** What the automatic startup check found, when it ran and found something. */
  startupUpdate: UpdateCheck | null;
  onCheckUpdates: () => Promise<UpdateCheck>;
  onInstallUpdate: (
    kind: "app" | "full",
    version: string,
  ) => Promise<UpdateInstallSnapshot>;
  onGetUpdateInstall: () => Promise<UpdateInstallSnapshot | null>;
  onCloseWindow: () => Promise<unknown>;
  onOpenUpdatePage: () => Promise<unknown>;
}

/**
 * The settings page's update panel: what is available, what is downloading,
 * and the one switch that decides whether we look on startup.
 *
 * Its own component because none of that state is shared with the rest of the
 * page -- five pieces of state, a polling timer and three effects that exist
 * only to keep a progress bar honest.
 */
export function UpdateSection({
  startupUpdate,
  onCheckUpdates,
  onInstallUpdate,
  onGetUpdateInstall,
  onCloseWindow,
  onOpenUpdatePage,
}: UpdateSectionProps) {
  const { t } = useLanguage();
  const [updateMessage, setUpdateMessage] = useState("");
  const [availableUpdate, setAvailableUpdate] = useState<UpdateCheck | null>(
    startupUpdate,
  );
  // The startup check can land after this page is already open -- the initial
  // state above would miss it, and the sidebar dot would be pointing at a
  // panel that says nothing. `?? startupUpdate` so a manual check the user
  // just ran is not overwritten by it.
  useEffect(() => {
    if (startupUpdate !== null) {
      setAvailableUpdate((current) => current ?? startupUpdate);
    }
  }, [startupUpdate]);
  const [updateBusy, setUpdateBusy] = useState(false);
  // Same shape as the close-window choice on the page around this: preferences
  // are hydrated before this page can be reached, so the initial read is
  // already the durable one.
  const [autoCheck, setAutoCheck] = useState(
    () => uiValue<boolean>("autoUpdateCheck", true),
  );
  const [install, setInstall] = useState<UpdateInstallSnapshot | null>(null);
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

  const startPolling = useCallback(() => {
    stopPolling();
    pollingRef.current = setInterval(() => {
      void pollInstall();
    }, 500);
  }, [pollInstall, stopPolling]);

  // Self-starting, driven by the install's own state rather than by the click
  // that began it. Polling used to start only from the install button, while
  // the mount effect's cleanup depended on `pollInstall` -> `onGetUpdateInstall`
  // -- an inline arrow rebuilt on every render of the page. So the timer was
  // cleared by any re-render of the parent (changing the theme was enough) and
  // nothing ever restarted it: the progress bar froze, `ready` never arrived,
  // and the restart button never appeared. The comment this replaces --
  // "reopening Settings mid-download has to find the install still running" --
  // is the behaviour it was meant to provide.
  const installIsActive =
    install !== null && (install.state === "queued" || install.state === "running");

  useEffect(() => {
    void pollInstall();
  }, [pollInstall]);

  useEffect(() => {
    if (!installIsActive) {
      return;
    }
    startPolling();
    return stopPolling;
  }, [installIsActive, startPolling, stopPolling]);

  return (
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
      {/* Its own row under the buttons: the section is a space-between flex
          row, and a third item between the text and the actions squeezed
          both. */}
      <label className="switch-row update-auto-check">
        <input
          type="checkbox"
          checked={autoCheck}
          onChange={(event) => {
            setAutoCheck(event.target.checked);
            saveUi({ autoUpdateCheck: event.target.checked });
          }}
        />
        <span>
          <strong>{t.settings.updates.autoCheck}</strong>
          <small>{t.settings.updates.autoCheckHint}</small>
        </span>
      </label>
    </section>
  );
}
