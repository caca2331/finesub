import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";


const source = readFileSync(
  new URL("../components/CustomSelect.tsx", import.meta.url),
  "utf8",
);


test("custom select exposes complete keyboard navigation hooks", () => {
  for (const key of [
    "ArrowDown",
    "ArrowUp",
    "Home",
    "End",
    "Enter",
    " ",
    "Escape",
    "Tab",
  ]) {
    assert.match(source, new RegExp(`case [\"']${key === " " ? " " : key}[\"']`));
  }
  assert.match(source, /role="combobox"/);
  assert.match(source, /aria-activedescendant/);
  assert.match(source, /aria-controls/);
  assert.match(source, /aria-labelledby/);
});
