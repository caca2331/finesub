import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";


test("translation mode discloses Gemini media uploads", () => {
  const component = readFileSync(
    new URL("../components/NewTask.tsx", import.meta.url),
    "utf8",
  );
  const translations = readFileSync(
    new URL("../lib/translations.ts", import.meta.url),
    "utf8",
  );

  assert.match(component, /cloudTranslation/);
  assert.match(component, /privacyNoteCloud/);
  assert.match(translations, /翻译会向 Gemini 上传必要的媒体片段/);
  assert.match(translations, /Translation uploads only the required media clips to Gemini/);
});
