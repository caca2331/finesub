"use client";

import { Minus, Square, X } from "lucide-react";

import type { DesktopApi } from "@/lib/types";
import { useLanguage } from "./LanguageProvider";


export function TitleBar({ api }: { api: DesktopApi }) {
  const { t } = useLanguage();
  const toggleMaximize = () => void api.maximizeWindow();

  return (
    <header className="titlebar">
      <div
        className="titlebar-brand pywebview-drag-region"
        aria-hidden="true"
        onDoubleClick={toggleMaximize}
      >
        <img
          className="brand-icon"
          src="./icon.png"
          alt=""
          draggable={false}
        />
        <span>FineSub Desktop</span>
      </div>
      <div
        className="titlebar-drag pywebview-drag-region"
        onDoubleClick={toggleMaximize}
      >
        {/* <span>{t.titleBar.brand}</span> */}
      </div>
      <div className="window-actions" aria-label={t.titleBar.controls}>
        <button
          type="button"
          aria-label={t.titleBar.minimize}
          onClick={() => void api.minimizeWindow()}
        >
          <Minus size={14} strokeWidth={1.8} />
        </button>
        <button
          type="button"
          aria-label={t.titleBar.maximize}
          onClick={toggleMaximize}
        >
          <Square size={12} strokeWidth={1.7} />
        </button>
        <button
          type="button"
          className="window-close"
          aria-label={t.titleBar.close}
          onClick={() => {
            const action = localStorage.getItem("close-window-action");
            if (action === "close") {
              void api.closeWindow();
            } else {
              void api.minimizeToTray();
            }
          }}
        >
          <X size={15} strokeWidth={1.8} />
        </button>
      </div>
    </header>
  );
}
