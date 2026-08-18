import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { translations } from "../lib/translations";


test("translation mode discloses Gemini media uploads", () => {
  const component = readFileSync(
    new URL("../components/NewTask.tsx", import.meta.url),
    "utf8",
  );

  assert.match(component, /cloudTranslation/);
  assert.match(component, /privacyNoteCloud/);
  // Read through the module rather than the file: the copy lives in one file
  // per language now, and what has to carry the disclosure is the value the UI
  // renders, not whichever file happens to hold it.
  assert.match(
    translations.zh.newTask.privacyNoteCloud,
    /翻译会向 Gemini 上传必要的媒体片段/,
  );
  assert.match(
    translations.en.newTask.privacyNoteCloud,
    /Translation uploads only the required media clips to Gemini/,
  );
});
