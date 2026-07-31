import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";


const source = readFileSync(
  new URL("../components/CustomSelect.tsx", import.meta.url),
  "utf8",
);


test("custom select exposes a listbox with basic keyboard support", () => {
  assert.match(source, /aria-haspopup="listbox"/);
  assert.match(source, /aria-expanded=\{open\}/);
  assert.match(source, /role="listbox"/);
  assert.match(source, /role="option"/);
  assert.match(source, /aria-selected=\{/);
  assert.match(source, /event\.key === "Escape"/);
});
