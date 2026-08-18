import assert from "node:assert/strict";
import test from "node:test";

import { RESOURCE_SIZES } from "../lib/resourceCatalog";
import { translations } from "../lib/translations";


test("tokcount has exact download metadata and localized copy", () => {
  assert.equal(RESOURCE_SIZES.tokcount, 9_303_247);
  assert.equal(translations.zh.resources.items.tokcount.title, "Gemini 本地分词器");
  assert.equal(translations.en.resources.items.tokcount.title, "Gemini Local Tokenizer");
  assert.ok(translations.zh.resources.items.tokcount.detail.length > 0);
  assert.ok(translations.en.resources.items.tokcount.detail.length > 0);
});
