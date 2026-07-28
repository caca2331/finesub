import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { FONT_SCALE_LABELS } from "../lib/useAppearance";


const css = readFileSync(
  new URL("../app/globals.css", import.meta.url),
  "utf8",
);
const layout = readFileSync(
  new URL("../app/layout.tsx", import.meta.url),
  "utf8",
);
const titleBar = readFileSync(
  new URL("../components/TitleBar.tsx", import.meta.url),
  "utf8",
);
const page = readFileSync(
  new URL("../app/page.tsx", import.meta.url),
  "utf8",
);


test("the complete UI supports selectable fonts and five size levels", () => {
  assert.match(
    css,
    /font-family:\s*var\(--user-font,\s*"Microsoft YaHei UI"\)/,
  );
  assert.match(css, /--base-font-size:\s*15px/);
  assert.deepEqual(Object.values(FONT_SCALE_LABELS), [
    "最小",
    "小",
    "标准",
    "大",
    "最大",
  ]);
});


test("the UI does not load bundled web fonts from CSS", () => {
  const fontFaces = [...css.matchAll(/@font-face\s*\{([\s\S]*?)\}/g)].map(
    (match) => match[1],
  );
  assert.equal(fontFaces.length, 0);
  assert.doesNotMatch(css, /\.woff2/);
});


test("FineSub Desktop metadata and title bar use the supplied icon", () => {
  assert.match(layout, /title:\s*"FineSub Desktop"/);
  assert.match(layout, /href="\.\/icon\.png"/);
  assert.match(titleBar, /src="\.\/icon\.png"/);
  assert.doesNotMatch(titleBar, /brand-glyph/);
  assert.match(page, /className="brand-icon"/);
  assert.match(page, /<strong>FineSub Desktop<\/strong>/);
  assert.doesNotMatch(page, /className="brand-glyph"/);
});
