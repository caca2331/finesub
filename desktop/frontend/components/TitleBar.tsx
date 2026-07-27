"use client";

import { Minus, Square, X } from "lucide-react";

import type { DesktopApi } from "@/lib/types";


export function TitleBar({ api }: { api: DesktopApi }) {
  return (
    <header className="titlebar">
      <div className="titlebar-brand" aria-hidden="true">
        <img
          className="brand-icon"
          src="./icon.png"
          alt=""
          draggable={false}
        />
        <span>FineSub Desktop</span>
      </div>
      <div className="titlebar-drag pywebview-drag-region">
        <span>字幕工作台</span>
      </div>
      <div className="window-actions" aria-label="窗口控制">
        <button
          type="button"
          aria-label="最小化"
          onClick={() => void api.minimizeWindow()}
        >
          <Minus size={14} strokeWidth={1.8} />
        </button>
        <button
          type="button"
          aria-label="最大化"
          onClick={() => void api.maximizeWindow()}
        >
          <Square size={12} strokeWidth={1.7} />
        </button>
        <button
          type="button"
          className="window-close"
          aria-label="关闭"
          onClick={() => void api.closeWindow()}
        >
          <X size={15} strokeWidth={1.8} />
        </button>
      </div>
    </header>
  );
}
