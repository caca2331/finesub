import { formatBytes, formatPercent } from "@/lib/formatters";


interface DownloadProgressProps {
  name: string;
  downloaded?: number;
  total?: number;
  bytesPerSecond?: number;
}


export function DownloadProgress({
  name,
  downloaded,
  total,
  bytesPerSecond,
}: DownloadProgressProps) {
  const numericPercent =
    total && downloaded ? Math.min(100, (downloaded / total) * 100) : 0;
  return (
    <div className="download-progress">
      <div className="progress-copy">
        <strong>{name}</strong>
        <span>
          {formatBytes(downloaded)} / {formatBytes(total)}
          {bytesPerSecond ? ` · ${formatBytes(bytesPerSecond)}/s` : ""}
        </span>
        <b>{formatPercent(downloaded, total)}</b>
      </div>
      <div className="progress-track">
        <span style={{ width: `${numericPercent}%` }} />
      </div>
    </div>
  );
}
