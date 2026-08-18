// Download sizes, not installed footprints. Keep manifest-backed entries exact
// so the confirmation dialog never silently falls back to 0 B.
export const MODEL_DOWNLOAD_ESTIMATE = 3_700_000_000; // ~3.4 GiB

export const RESOURCE_SIZES: Record<string, number> = {
  uv: 3_034_000_000, // ~2.83 GiB: uv + the locked wheels
  ffmpeg: 146_688_582,
  git: 38_791_206,
  "yt-dlp": 3_184_705,
  tokcount: 9_303_247,
  models: MODEL_DOWNLOAD_ESTIMATE,
};
