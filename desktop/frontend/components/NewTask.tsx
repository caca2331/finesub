"use client";

import { ArrowRight, ShieldCheck } from "lucide-react";

import type { AppState } from "@/lib/state";
import type { TaskRequest } from "@/lib/types";

import { DropZone } from "./DropZone";
import { EnvironmentPanel } from "./EnvironmentPanel";
import { TaskSettings } from "./TaskSettings";


interface NewTaskProps {
  state: AppState;
  busy: boolean;
  onSelectFile: () => void;
  onDropPath: (path: string) => void;
  onRequestChange: (
    changes: Partial<Omit<TaskRequest, "input">>,
  ) => void;
  onInstallResource: (resourceId: string) => void;
  onStart: () => void;
}


export function NewTask({
  state,
  busy,
  onSelectFile,
  onDropPath,
  onRequestChange,
  onInstallResource,
  onStart,
}: NewTaskProps) {
  const hasFile = Boolean(state.task.selectedFile);
  return (
    <div className="page page-new-task">
      <header className="page-header">
        <div>
          <p className="page-kicker">NEW TASK</p>
          <h1>生成一份干净的字幕</h1>
          <p>选择媒体文件，FineSub 会在本机完成分离、识别与字幕整理。</p>
        </div>
        <div className="privacy-note">
          <ShieldCheck size={15} />
          <span>媒体文件仅在本机处理</span>
        </div>
      </header>

      <div className="task-layout">
        <section className="primary-panel">
          <div className="section-heading">
            <div>
              <p className="section-kicker">01 · SOURCE</p>
              <h2>输入文件</h2>
            </div>
          </div>
          <DropZone
            selectedFile={state.task.selectedFile}
            disabled={busy}
            onSelect={onSelectFile}
            onDropPath={onDropPath}
          />

          <div className="panel-divider" />

          <div className="section-heading">
            <div>
              <p className="section-kicker">02 · OUTPUT</p>
              <h2>处理选项</h2>
            </div>
            <span className="section-hint">已保存为默认值</span>
          </div>
          <TaskSettings
            request={state.task.request}
            capabilities={state.capabilities}
            disabled={busy}
            onChange={onRequestChange}
          />

          {state.task.error ? (
            <div className="error-banner" role="alert">
              <strong>{state.task.error.message}</strong>
              {state.task.error.code === "api_key_required" ? (
                <span>前往设置填写 Key，或把输出结果改为“原始字幕 SRT”。</span>
              ) : null}
            </div>
          ) : null}

          <div className="task-actions">
            <div>
              <strong>{hasFile ? "可以开始" : "等待选择文件"}</strong>
              <span>
                {hasFile
                  ? "缺失的运行组件会先自动下载"
                  : "支持常见音视频格式"}
              </span>
            </div>
            <button
              type="button"
              className="button button-primary"
              disabled={!hasFile || busy}
              onClick={onStart}
            >
              {busy ? "正在准备…" : "开始生成"}
              <ArrowRight size={15} />
            </button>
          </div>
        </section>

        <EnvironmentPanel
          resources={state.resources}
          busy={busy}
          onInstall={onInstallResource}
        />
      </div>
    </div>
  );
}
