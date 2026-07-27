"use client";

import { FileVideo2, FolderOpen, RefreshCw, UploadCloud } from "lucide-react";
import { useState } from "react";

import { fileName } from "@/lib/formatters";


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
  const [dragging, setDragging] = useState(false);

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
          更换
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
        const file = event.dataTransfer.files[0] as
          | (File & { path?: string })
          | undefined;
        if (file) {
          onDropPath(file.path || file.name);
        }
      }}
    >
      <div className="drop-icon">
        <UploadCloud size={24} strokeWidth={1.6} />
      </div>
      <div className="drop-copy">
        <strong>拖入音频或视频</strong>
        <span>MP4、MKV、MOV、MP3、WAV、FLAC</span>
      </div>
      <button
        type="button"
        className="button button-secondary"
        disabled={disabled}
        onClick={onSelect}
      >
        <FolderOpen size={15} />
        选择文件
      </button>
    </div>
  );
}
