import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";


test("translation mode discloses Gemini media uploads", () => {
  const source = readFileSync(
    new URL("../components/NewTask.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /翻译会向 Gemini 上传必要的媒体片段/);
  assert.match(source, /cloudTranslation/);
});
