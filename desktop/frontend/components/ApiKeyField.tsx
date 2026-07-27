"use client";

import { Check, Eye, EyeOff, KeyRound, Trash2 } from "lucide-react";
import { useState } from "react";


interface ApiKeyFieldProps {
  label: string;
  description: string;
  status: "configured" | "missing";
  placeholder: string;
  onSave: (value: string) => Promise<void>;
  onDelete: () => Promise<void>;
}


export function ApiKeyField({
  label,
  description,
  status,
  placeholder,
  onSave,
  onDelete,
}: ApiKeyFieldProps) {
  const [value, setValue] = useState("");
  const [visible, setVisible] = useState(false);
  const [saving, setSaving] = useState(false);

  return (
    <div className="api-key-row">
      <div className="api-key-heading">
        <span className="api-key-icon">
          <KeyRound size={15} />
        </span>
        <div>
          <strong>{label}</strong>
          <span>{description}</span>
        </div>
        <span className={`key-status ${status === "configured" ? "is-ready" : ""}`}>
          {status === "configured" ? (
            <>
              <Check size={11} /> 已配置
            </>
          ) : (
            "未配置"
          )}
        </span>
      </div>
      <div className="api-key-controls">
        <div className="secret-input">
          <input
            type={visible ? "text" : "password"}
            value={value}
            placeholder={
              status === "configured" ? "输入新值以替换" : placeholder
            }
            autoComplete="off"
            spellCheck={false}
            onChange={(event) => setValue(event.target.value)}
          />
          <button
            type="button"
            aria-label={visible ? "隐藏 API Key" : "显示 API Key"}
            onClick={() => setVisible((shown) => !shown)}
          >
            {visible ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
        {status === "configured" ? (
          <button
            type="button"
            className="icon-button danger"
            aria-label={`删除 ${label}`}
            disabled={saving}
            onClick={async () => {
              setSaving(true);
              try {
                await onDelete();
                setValue("");
              } finally {
                setSaving(false);
              }
            }}
          >
            <Trash2 size={14} />
          </button>
        ) : null}
        <button
          type="button"
          className="button button-secondary button-compact"
          disabled={!value.trim() || saving}
          onClick={async () => {
            setSaving(true);
            try {
              await onSave(value);
              setValue("");
            } finally {
              setSaving(false);
            }
          }}
        >
          {saving ? "保存中" : "保存"}
        </button>
      </div>
    </div>
  );
}
