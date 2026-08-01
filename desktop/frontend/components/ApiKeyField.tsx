"use client";

import { Check, Eye, EyeOff, KeyRound, Trash2 } from "lucide-react";
import { useState } from "react";
import { useLanguage } from "./LanguageProvider";


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
  const { t } = useLanguage();
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
              <Check size={11} /> {t.apiKey.configured}
            </>
          ) : (
            t.apiKey.missing
          )}
        </span>
      </div>
      <div className="api-key-controls">
        <div className="secret-input">
          <input
            type={visible ? "text" : "password"}
            value={value}
            placeholder={
              status === "configured" ? t.apiKey.replace : placeholder
            }
            autoComplete="off"
            spellCheck={false}
            onChange={(event) => setValue(event.target.value)}
          />
          <button
            type="button"
            aria-label={visible ? t.apiKey.hide : t.apiKey.show}
            onClick={() => setVisible((shown) => !shown)}
          >
            {visible ? <EyeOff size={14} /> : <Eye size={14} />}
          </button>
        </div>
        {status === "configured" ? (
          <button
            type="button"
            className="icon-button danger"
            aria-label={`${t.apiKey.delete} ${label}`}
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
          {saving ? t.apiKey.saving : t.apiKey.save}
        </button>
      </div>
    </div>
  );
}
