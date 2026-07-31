"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import { AppShell } from "@/components/AppShell";
import { BootstrapScreen } from "@/components/BootstrapScreen";
import { CompletedView } from "@/components/CompletedView";
import { ConfirmDialog, isConfirmRemembered } from "@/components/ConfirmDialog";
import { KnowledgeBase } from "@/components/KnowledgeBase";
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
import {
  initialState,
  reduceAppState,
} from "@/lib/state";
import type {
  BridgeError,
  Route,
  TaskRequest,
} from "@/lib/types";
import { useAppearance } from "@/lib/useAppearance";


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


export default function Home() {
  const [state, dispatch] = useReducer(reduceAppState, initialState);
  const [busy, setBusy] = useState(false);
  const [bootstrapError, setBootstrapError] = useState<BridgeError | null>(null);
  const eventCursor = useRef(0);
  const { settings: appearance, update: updateAppearance } = useAppearance();
  const [confirmOpen, setConfirmOpen] = useState(false);

  const loadBootstrap = useCallback(async () => {
    setBootstrapError(null);
    try {
      const payload = await desktopApi.getBootstrapState();
      dispatch({ type: "bootstrapLoaded", payload });
    } catch (error) {
      setBootstrapError(toBridgeError(error));
    }
  }, []);

  useEffect(() => {
    void loadBootstrap();
  }, [loadBootstrap]);

  useEffect(() => {
    if (state.task.phase !== "running" || !state.task.taskId) {
      eventCursor.current = 0;
      return;
    }
    let stopped = false;
    const poll = async () => {
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
        // A transient bridge poll failure should not destroy task state.
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 700);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [state.task.phase, state.task.taskId]);

  const hasActiveResourceInstall = state.resourceInstalls.some(
    (install) => install.state === "queued" || install.state === "running",
  );

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
      const missing = state.resources.find(
        (resource) => resource.state !== "ready",
      );
      if (missing) {
        dispatch({ type: "navigate", route: "resources" });
        await installResource(missing.id);
        return;
      }
      const snapshot = await desktopApi.startTask({
        input: state.task.selectedFile,
        ...state.task.request,
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
        onCancel={(taskId) => void cancelHistoryTask(taskId)}
        onRetry={(taskId) => void restartHistoryTask(taskId, "retry")}
        onResume={(taskId) => void restartHistoryTask(taskId, "resume")}
        onDelete={(taskId) => dispatch({ type: "deleteTask", taskId })}
        onOpenOutput={(path) => void desktopApi.openOutput(path)}
      />
    );
  } else if (state.route === "knowledge") {
    content = <KnowledgeBase />;
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
        onUseRawSubtitle={() => {
          dispatch({
            type: "requestChanged",
            changes: { stage: "raw-srt" },
          });
          dispatch({ type: "navigate", route: "new-task" });
        }}
        onCheckUpdates={() => desktopApi.checkUpdates()}
        onOpenUpdatePage={() => desktopApi.openUpdatePage()}
      />
    );
  } else if (state.task.phase === "running" || state.task.phase === "failed") {
    content = (
      <ProcessingView
        task={state.task}
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
        onRequestChange={(
          changes: Partial<Omit<TaskRequest, "input">>,
        ) => dispatch({ type: "requestChanged", changes })}
        onInstallResource={(resourceId) => void installResource(resourceId)}
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
        </AppShell>
      )}
    </LanguageProvider>
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
