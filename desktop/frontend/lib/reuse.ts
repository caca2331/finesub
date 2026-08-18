import type { JobSnapshot } from "./types";

/**
 * The completed recognition run whose artifacts a new LLM pass on `input`
 * could pick up.
 *
 * Reuse works by pointing the new task's `output` at the old task's directory:
 * the pipeline skips any stage whose artifact already exists there, so the
 * recognition stages fall through and only correction/translation runs. Only
 * raw-srt runs qualify — a completed final-srt run would also satisfy the LLM
 * stage's existence check, and the new task would silently republish the old
 * subtitle instead of doing the work it was asked for.
 */
export function reusableAsrTask(
  history: JobSnapshot[],
  input: string | null,
): JobSnapshot | null {
  if (!input) {
    return null;
  }
  let latest: JobSnapshot | null = null;
  for (const snapshot of history) {
    if (
      snapshot.state !== "completed" ||
      snapshot.request?.input !== input ||
      snapshot.request?.stage !== "raw-srt" ||
      !snapshot.request?.output
    ) {
      continue;
    }
    if ((snapshot.created_at ?? 0) > (latest?.created_at ?? -1)) {
      latest = snapshot;
    }
  }
  return latest;
}
