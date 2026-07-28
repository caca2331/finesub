"use client";

import { useCallback, useState } from "react";


const STORAGE_PREFIX = "finesub-confirm-";

export interface ConfirmDialogConfig {
  id: string;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
}

interface ConfirmDialogProps {
  config: ConfirmDialogConfig;
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}


export function isConfirmRemembered(id: string): boolean {
  if (typeof window === "undefined") {
    return false;
  }
  return localStorage.getItem(`${STORAGE_PREFIX}${id}`) === "1";
}

export function clearConfirmMemory(id: string): void {
  localStorage.removeItem(`${STORAGE_PREFIX}${id}`);
}

export function listRememberedConfirms(): string[] {
  const keys: string[] = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key?.startsWith(STORAGE_PREFIX) && localStorage.getItem(key) === "1") {
      keys.push(key.slice(STORAGE_PREFIX.length));
    }
  }
  return keys;
}


export function ConfirmDialog({
  config,
  open,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const [remember, setRemember] = useState(false);

  const handleConfirm = useCallback(() => {
    if (remember) {
      localStorage.setItem(`${STORAGE_PREFIX}${config.id}`, "1");
    }
    onConfirm();
  }, [remember, config.id, onConfirm]);

  if (!open) {
    return null;
  }

  return (
    <div className="dialog-overlay" onClick={onCancel}>
      <div className="dialog-card" onClick={(e) => e.stopPropagation()}>
        <h3>{config.title}</h3>
        <p>{config.message}</p>
        <label className="dialog-remember">
          <input
            type="checkbox"
            checked={remember}
            onChange={(e) => setRemember(e.target.checked)}
          />
          记住选择，以后不再提示
        </label>
        <div className="dialog-actions">
          <button
            type="button"
            className="button button-secondary"
            onClick={onCancel}
          >
            {config.cancelLabel ?? "取消"}
          </button>
          <button
            type="button"
            className="button button-primary"
            onClick={handleConfirm}
          >
            {config.confirmLabel ?? "确认"}
          </button>
        </div>
      </div>
    </div>
  );
}
