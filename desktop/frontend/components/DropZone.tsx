"use client";

import { FileVideo2, FolderOpen, RefreshCw, UploadCloud } from "lucide-react";
import { useEffect, useState } from "react";

import { fileName } from "@/lib/formatters";
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

  if (selectedFile) {
    return (
      <div className="selected-file">
        <div className="file-icon">
          <FileVideo2 size={22} strokeWidth={1.7} />
        </div>
        <div className="file-copy">
          <strong title={selectedFile}>{fileName(selectedFile)}</strong>
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
  );
}
