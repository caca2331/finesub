"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { BootstrapScreen } from "@/components/BootstrapScreen";
import { CompletedView } from "@/components/CompletedView";
import { ConfirmDialog, isConfirmRemembered } from "@/components/ConfirmDialog";
import { LanguageProvider, useLanguage } from "@/components/LanguageProvider";
import { NewTask } from "@/components/NewTask";
import { ProcessingView } from "@/components/ProcessingView";
import { ResourceManager } from "@/components/ResourceManager";
import { Settings } from "@/components/Settings";
import { TaskHistory } from "@/components/TaskHistory";
import {
  BridgeCallError,
  desktopApi,
} from "@/lib/bridge";
import { hydratePreferences, saveTaskDefaults, uiValue } from "@/lib/preferences";
import {
  readProcessingDevice,
  requestDeviceFields,
} from "@/lib/processingDevice";
import { blockingResources, hasActiveInstall } from "@/lib/resources";
import {
  REMEMBERED_TASK_FIELDS,
  initialState,
  reduceAppState,
} from "@/lib/state";
import type {
  BridgeError,
  Route,
  TaskRequest,
  UpdateCheck,
} from "@/lib/types";
import { useAppearance } from "@/lib/useAppearance";


function rememberTaskOptions(changes: Partial<Omit<TaskRequest, "input">>): void {
  const patch: Record<string, unknown> = {};
  for (const field of REMEMBERED_TASK_FIELDS) {
    if (field in changes) patch[field] = changes[field];
  }
  if (Object.keys(patch).length > 0) {
    saveTaskDefaults(patch);
  }
}


function toBridgeError(error: unknown): BridgeError {
  if (error instanceof BridgeCallError) {
    return {
      code: error.code,
      message: error.message,
      action: error.action,
    };
  }
  return {
    code: "internal_error",
    message: error instanceof Error ? error.message : "操作失败，请稍后重试。",
  };
}

/** Not a backend failure: the workspace has to leave "checking" before the
 *  resource page takes over, or it sits on a spinner with no task behind it. */
const RESOURCE_REQUIRED_ERROR: BridgeError = {
  code: "runtime_required",
  message: "还缺少运行所需的组件，正在为你安装。",
  action: "open_resources",
};


export default function Home() {
  const [state, dispatch] = useReducer(reduceAppState, initialState);
  const [busy, setBusy] = useState(false);
  const [bootstrapError, setBootstrapError] = useState<BridgeError | null>(null);
  const eventCursor = useRef(0);
  const { settings: appearance, update: updateAppearance } = useAppearance();
  const [confirmOpen, setConfirmOpen] = useState(false);
  // Which task the cleanup dialog is asking about; null when it is closed.
  const [cleanupTaskId, setCleanupTaskId] = useState<string | null>(null);
  const [startupUpdate, setStartupUpdate] = useState<UpdateCheck | null>(null);

  // One release-feed query per launch, when the user leaves it on. Held here
  // rather than in the settings page so the sidebar can point at it: a check
  // nobody is told about is the same as no check. Fired from `loadBootstrap`
  // rather than its own mount effect so the preference is read *after*
  // hydration -- settings.json is the record, and a mount-time read would ask
  // the localStorage mirror instead. The ref makes "one per launch" literal:
  // a retried bootstrap goes through here again, and dev StrictMode would
  // otherwise double the request.
  const startupCheckFired = useRef(false);
  const maybeCheckForUpdates = useCallback(() => {
    if (startupCheckFired.current || !uiValue("autoUpdateCheck", true)) {
      return;
    }
    startupCheckFired.current = true;
    void (async () => {
      try {
        const result = await desktopApi.checkUpdates();
        if (result.available) {
          setStartupUpdate(result);
        }
      } catch {
        // No release source, no network, a rate limit: none of that is worth
        // an error on a screen the user did not ask anything of.
      }
    })();
  }, []);

  const loadBootstrap = useCallback(async () => {
    setBootstrapError(null);
    try {
      const payload = await desktopApi.getBootstrapState();
      // Before the dispatch: the theme and language readers are synchronous
      // and would otherwise render one frame from the mirror.
      hydratePreferences(payload.preferences);
      dispatch({ type: "bootstrapLoaded", payload });
      maybeCheckForUpdates();
    } catch (error) {
      setBootstrapError(toBridgeError(error));
    }
  }, [maybeCheckForUpdates]);

  useEffect(() => {
    void loadBootstrap();
  }, [loadBootstrap]);

  useEffect(() => {
    if (state.task.phase !== "running" || !state.task.taskId) {
      eventCursor.current = 0;
      return;
    }
    let stopped = false;
    // Without this, a poll slower than the 700ms interval overlaps the next
    // one: two responses land out of order, the older `nextCursor` wins, and
    // the drawer replays log lines it already showed.
    let inFlight = false;
    const poll = async () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      try {
        const result = await desktopApi.pollEvents(eventCursor.current);
        if (stopped) {
          return;
        }
        for (const event of result.events) {
          dispatch({ type: "workerEvent", event });
        }
        eventCursor.current = result.nextCursor;
      } catch {
        // A transient bridge poll failure must not destroy task state. A
        // permanent one is indistinguishable from here, which is why the
        // worker also writes task-log.txt beside the outputs.
      } finally {
        inFlight = false;
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 700);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [state.task.phase, state.task.taskId]);

  const hasActiveResourceInstall = hasActiveInstall(state.resourceInstalls);

  useEffect(() => {
    if (!hasActiveResourceInstall) {
      return;
    }
    let stopped = false;
    const poll = async () => {
      try {
        const installs = await desktopApi.listResourceInstalls();
        if (!stopped) {
          dispatch({ type: "resourceInstallsChanged", installs });
        }
      } catch {
        // Keep the last known progress during a transient bridge failure.
      }
    };
    void poll();
    const timer = window.setInterval(
      () => void poll(),
      document.hidden ? 1500 : 500,
    );
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [hasActiveResourceInstall]);

  const selectFile = async () => {
    try {
      const result = await desktopApi.selectInputFile();
      if (result.path) {
        dispatch({ type: "fileSelected", path: result.path });
      }
    } catch (error) {
      dispatch({ type: "taskRejected", error: toBridgeError(error) });
    }
  };

  const installResource = async (resourceId: string) => {
    const current = state.resources.find((resource) => resource.id === resourceId);
    if (!current) {
      return;
    }
    dispatch({
      type: "resourceChanged",
      resource: { ...current, state: "downloading", detail: "" },
    });
    try {
      const install = await desktopApi.installResource(resourceId);
      dispatch({ type: "resourceInstallChanged", install });
    } catch (error) {
      dispatch({
        type: "resourceChanged",
        resource: {
          ...current,
          state: "failed",
          detail: toBridgeError(error).message,
        },
      });
    }
  };

  const pauseResource = async (resourceId: string) => {
    try {
      const install = await desktopApi.pauseResourceInstall(resourceId);
      dispatch({ type: "resourceInstallChanged", install });
    } catch (error) {
      const current = state.resources.find(
        (resource) => resource.id === resourceId,
      );
      if (current) {
        dispatch({
          type: "resourceChanged",
          resource: {
            ...current,
            state: "failed",
            detail: toBridgeError(error).message,
          },
        });
      }
    }
  };

  const startTask = async () => {
    if (!state.task.selectedFile) {
      return;
    }
    if (!isConfirmRemembered("start-task")) {
      setConfirmOpen(true);
      return;
    }
    await executeStartTask();
  };

  const executeStartTask = async () => {
    if (!state.task.selectedFile) {
      return;
    }
    setBusy(true);
    dispatch({ type: "taskChecking" });
    try {
      // Only what a task genuinely needs. `yt-dlp` has no system-tool finder
      // by design, so it is permanently "missing" on a healthy machine; gating
      // on it meant no task could ever start until both on-demand tools were
      // installed, which is exactly what marking them optional prevents.
      const [missing] = blockingResources(state.resources);
      if (missing) {
        // Leave "checking" before handing off, or the workspace sits on a
        // spinner with no task behind it.
        dispatch({ type: "taskRejected", error: RESOURCE_REQUIRED_ERROR });
        dispatch({ type: "navigate", route: "resources" });
        await installResource(missing.id);
        return;
      }
      // The processing device is one global setting, not a per-task control,
      // but it has to ride along with the request: that is the only thing the
      // backend receives, and it is what retry and resume replay later.
      const processing = readProcessingDevice();
      const snapshot = await desktopApi.startTask({
        input: state.task.selectedFile,
        ...state.task.request,
        ...requestDeviceFields(processing),
      });
      dispatch({ type: "taskStarted", snapshot });
    } catch (error) {
      dispatch({ type: "taskRejected", error: toBridgeError(error) });
    } finally {
      setBusy(false);
    }
  };

  const cancelTask = async () => {
    if (!state.task.taskId) {
      return;
    }
    try {
      await desktopApi.cancelTask(state.task.taskId);
      dispatch({
        type: "workerEvent",
        event: {
          type: "cancelled",
          task_id: state.task.taskId,
          timestamp: new Date().toISOString(),
          payload: {},
        },
      });
    } catch (error) {
      dispatch({ type: "taskRejected", error: toBridgeError(error) });
    }
  };

  const restartHistoryTask = async (
    taskId: string,
    mode: "retry" | "resume",
  ) => {
    setBusy(true);
    try {
      const snapshot =
        mode === "resume"
          ? await desktopApi.resumeTask(taskId)
          : await desktopApi.retryTask(taskId);
      dispatch({ type: "taskStarted", snapshot });
    } catch (error) {
      dispatch({ type: "taskRejected", error: toBridgeError(error) });
    } finally {
      setBusy(false);
    }
  };

  const deleteIntermediates = async (taskId: string) => {
    try {
      await desktopApi.deleteTaskIntermediates(taskId);
    } catch (error) {
      dispatch({ type: "taskRejected", error: toBridgeError(error) });
    } finally {
      setCleanupTaskId(null);
    }
  };

  const cancelHistoryTask = async (taskId: string) => {
    try {
      await desktopApi.cancelTask(taskId);
      dispatch({
        type: "workerEvent",
        event: {
          type: "cancelled",
          task_id: taskId,
          timestamp: new Date().toISOString(),
          payload: {},
        },
      });
    } catch (error) {
      dispatch({ type: "taskRejected", error: toBridgeError(error) });
    }
  };

  const saveKey = async (
    provider: "gemini" | "exa" | "tavily",
    value: string,
  ) => {
    const settings = await desktopApi.saveApiKeys({ [provider]: value });
    dispatch({ type: "settingsChanged", settings });
  };

  const deleteKey = async (provider: "gemini" | "exa" | "tavily") => {
    const settings = await desktopApi.deleteApiKey(provider);
    dispatch({ type: "settingsChanged", settings });
  };



  let content;
  if (state.route === "history") {
    content = (
      <TaskHistory
        tasks={state.history}
        reuseDisabled={
          busy ||
          state.task.phase === "running" ||
          state.task.phase === "checking"
        }
        onCancel={(taskId) => void cancelHistoryTask(taskId)}
        onRetry={(taskId) => void restartHistoryTask(taskId, "retry")}
        onResume={(taskId) => void restartHistoryTask(taskId, "resume")}
        onReuse={(snapshot) => dispatch({ type: "reuseAsr", snapshot })}
        onOpenOutput={(path) => void desktopApi.openOutput(path)}
        onOpenTasksDirectory={() => void desktopApi.openTasksDirectory()}
        onDeleteIntermediates={(taskId) => {
          if (isConfirmRemembered("delete-intermediates")) {
            void deleteIntermediates(taskId);
            return;
          }
          setCleanupTaskId(taskId);
        }}
      />
    );
  } else if (state.route === "resources") {
    content = (
      <ResourceManager
        resources={state.resources}
        installs={state.resourceInstalls}
        onInstall={(resourceId) => void installResource(resourceId)}
        onPause={(resourceId) => void pauseResource(resourceId)}
        onOpenLocation={(resourceId, kind) =>
          void desktopApi.openResourceLocation(resourceId, kind)
        }
        onOpenLogs={() => void desktopApi.openInstallLogs()}
      />
    );
  } else if (state.route === "settings") {
    content = (
      <Settings
        state={state}
        appearance={appearance}
        onAppearanceChange={updateAppearance}
        onSaveKey={saveKey}
        onDeleteKey={deleteKey}
        onRevealKeys={() => desktopApi.revealApiKeys()}
        onUseRawSubtitle={() => {
          dispatch({
            type: "requestChanged",
            changes: { stage: "raw-srt" },
          });
          dispatch({ type: "navigate", route: "new-task" });
        }}
        startupUpdate={startupUpdate}
        onCheckUpdates={() => desktopApi.checkUpdates()}
        onInstallUpdate={(kind, version) =>
          desktopApi.installUpdate(kind, version)
        }
        onGetUpdateInstall={() => desktopApi.getUpdateInstall()}
        onCloseWindow={() => desktopApi.closeWindow()}
        onOpenUpdatePage={() => desktopApi.openUpdatePage()}
        onRescanGpus={() => desktopApi.rescanGpus()}
        onSaveSharedSettings={async (values) => {
          const result = await desktopApi.saveSharedSettings(values);
          dispatch({
            type: "sharedSettingsChanged",
            settings: result.shared,
            configPath: result.config_path,
          });
        }}
      />
    );
  } else if (
    state.task.phase === "running" ||
    // A failed task keeps the workspace only while the user is looking at
    // it. Pinning the view regardless of route left "新建任务" rendering the
    // same error screen from every page, with no drop zone and no way out
    // but retrying the file that just failed.
    (state.task.phase === "failed" && state.route === "new-task")
  ) {
    content = (
      <ProcessingView
        task={state.task}
        firstRun={
          !state.history.some((snapshot) => snapshot.state === "completed")
        }
        onCancel={() => void cancelTask()}
        onRetry={() => void startTask()}
      />
    );
  } else if (state.task.phase === "completed") {
    content = (
      <CompletedView
        task={state.task}
        onOpen={(path) => void desktopApi.openOutput(path)}
        onReset={() => dispatch({ type: "resetTask" })}
      />
    );
  } else {
    content = (
      <NewTask
        state={state}
        busy={busy}
        onSelectFile={() => void selectFile()}
        onDropPath={(path) => dispatch({ type: "fileSelected", path })}
        onRequestChange={(changes: Partial<Omit<TaskRequest, "input">>) => {
          dispatch({ type: "requestChanged", changes });
          rememberTaskOptions(changes);
        }}
        onReuse={(snapshot) => dispatch({ type: "reuseAsr", snapshot })}
        onInstallResource={(resourceId) => void installResource(resourceId)}
        onOpenResources={() =>
          dispatch({ type: "navigate", route: "resources" })
        }
        onStart={() => void startTask()}
      />
    );
  }

  return (
    <LanguageProvider>
      {!state.bootstrapped ? (
        <BootstrapScreen error={bootstrapError} onRetry={loadBootstrap} />
      ) : (
        <AppShell
          state={state}
          api={desktopApi}
          onNavigate={(route: Route) => dispatch({ type: "navigate", route })}
          updateAvailable={startupUpdate?.available === true}
        >
          {content}
          <StartTaskConfirmDialog
            open={confirmOpen}
            onConfirm={() => {
              setConfirmOpen(false);
              void executeStartTask();
            }}
            onCancel={() => setConfirmOpen(false)}
          />
          <DeleteIntermediatesConfirmDialog
            open={cleanupTaskId !== null}
            onConfirm={() => {
              if (cleanupTaskId !== null) {
                void deleteIntermediates(cleanupTaskId);
              }
            }}
            onCancel={() => setCleanupTaskId(null)}
          />
        </AppShell>
      )}
    </LanguageProvider>
  );
}

function DeleteIntermediatesConfirmDialog({
  open,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const { t } = useLanguage();

  return (
    <ConfirmDialog
      config={{
        id: "delete-intermediates",
        title: t.history.deleteIntermediatesConfirmTitle,
        message: t.history.deleteIntermediatesConfirmBody,
        confirmLabel: t.history.deleteIntermediatesConfirmAction,
      }}
      open={open}
      onConfirm={onConfirm}
      onCancel={onCancel}
    />
  );
}

function StartTaskConfirmDialog({
  open,
  onConfirm,
  onCancel,
}: {
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const { t } = useLanguage();

  return (
    <ConfirmDialog
      config={{
        id: "start-task",
        title: t.startTaskConfirm.title,
        message: t.startTaskConfirm.message,
        confirmLabel: t.startTaskConfirm.confirm,
        cancelLabel: t.startTaskConfirm.cancel,
      }}
      open={open}
      onConfirm={onConfirm}
      onCancel={onCancel}
    />
  );
}
