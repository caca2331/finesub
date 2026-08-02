const SUBTITLE_OUTPUT_KEYS = [
  "finalSrt",
  "translatedSrt",
  "rawSrt",
] as const;


export function subtitleOutputEntries(
  outputs: Record<string, string> | undefined,
): [string, string][] {
  if (!outputs) {
    return [];
  }
  return SUBTITLE_OUTPUT_KEYS.flatMap((key) => {
    const path = outputs[key];
    return path ? [[key, path] as [string, string]] : [];
  });
}


export function preferredSubtitleOutput(
  outputs: Record<string, string> | undefined,
): string | undefined {
  return subtitleOutputEntries(outputs)[0]?.[1];
}
