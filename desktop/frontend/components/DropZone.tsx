"use client";

import { FileVideo2, FolderOpen, Link2, RefreshCw, UploadCloud } from "lucide-react";
import { useEffect, useState } from "react";

import { fileName, isUrlSource } from "@/lib/formatters";
import { useLanguage } from "./LanguageProvider";


interface DropZoneProps {
  selectedFile: string | null;
  disabled?: boolean;
  onSelect: () => void;
  onDropPath: (path: string) => void;
}


export function DropZone({
  selectedFile,
  disabled,
  onSelect,
  onDropPath,
}: DropZoneProps) {
  const { t } = useLanguage();
  const [dragging, setDragging] = useState(false);
  const [url, setUrl] = useState("");
  const [urlError, setUrlError] = useState("");

  useEffect(() => {
    const receiveNativeDrop = (event: Event) => {
      const detail = (event as CustomEvent<{ path?: string }>).detail;
      if (detail?.path) {
        onDropPath(detail.path);
      }
    };
    window.addEventListener("finesub:file-drop", receiveNativeDrop);
    return () => {
      window.removeEventListener("finesub:file-drop", receiveNativeDrop);
    };
  }, [onDropPath]);

  const submitUrl = () => {
    const trimmed = url.trim();
    if (!isUrlSource(trimmed)) {
      setUrlError(t.newTask.dropZone.invalidUrl);
      return;
    }
    setUrlError("");
    setUrl("");
    // A URL travels the same path as a picked file: the pipeline accepts either
    // as its input, and yt-dlp is fetched on demand when it turns out to be a
    // link.
    onDropPath(trimmed);
  };

  if (selectedFile) {
    const fromWeb = isUrlSource(selectedFile);
    return (
      <div className="selected-file">
        <div className="file-icon">
          {fromWeb ? (
            <Link2 size={22} strokeWidth={1.7} />
          ) : (
            <FileVideo2 size={22} strokeWidth={1.7} />
          )}
        </div>
        <div className="file-copy">
          <strong title={selectedFile}>
            {fromWeb ? t.newTask.dropZone.urlSource : fileName(selectedFile)}
          </strong>
          <span title={selectedFile}>{selectedFile}</span>
        </div>
        <button
          type="button"
          className="button button-secondary button-compact"
          disabled={disabled}
          onClick={onSelect}
        >
          <RefreshCw size={14} />
          {t.newTask.dropZone.change}
        </button>
      </div>
    );
  }

  return (
    <div className="drop-area">
      <div
        className={`drop-zone${dragging ? " is-dragging" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (event.currentTarget === event.target) {
            setDragging(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
        }}
      >
        <div className="drop-icon">
          <UploadCloud size={24} strokeWidth={1.6} />
        </div>
        <div className="drop-copy">
          <strong>{t.newTask.dropZone.title}</strong>
          <span>{t.newTask.dropZone.formats}</span>
        </div>
        <button
          type="button"
          className="button button-secondary"
          disabled={disabled}
          onClick={onSelect}
        >
          <FolderOpen size={15} />
          {t.newTask.dropZone.selectFile}
        </button>
      </div>

      <div className="drop-divider">
        <span>{t.newTask.dropZone.or}</span>
      </div>

      <div className="url-source">
        <label className="url-field">
          <span>{t.newTask.dropZone.pasteUrl}</span>
          <div className="url-row">
            <input
              type="url"
              value={url}
              disabled={disabled}
              placeholder={t.newTask.dropZone.urlPlaceholder}
              onChange={(event) => {
                setUrl(event.target.value);
                if (urlError) {
                  setUrlError("");
                }
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  submitUrl();
                }
              }}
            />
            <button
              type="button"
              className="button button-secondary"
              disabled={disabled || !url.trim()}
              onClick={submitUrl}
            >
              <Link2 size={15} />
              {t.newTask.dropZone.useUrl}
            </button>
          </div>
        </label>
        {urlError ? <small className="url-error">{urlError}</small> : null}
      </div>
    </div>
  );
}
