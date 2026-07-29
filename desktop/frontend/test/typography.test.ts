import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  FONT_SCALE_LABELS,
  fontSizeForScale,
} from "../lib/useAppearance";


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
  assert.match(css, /html[\s\S]*font-size:\s*var\(--base-font-size\)/);
  assert.doesNotMatch(css, /^\s*font-size:\s*\d+(?:\.\d+)?px/m);
  assert.deepEqual(Object.values(FONT_SCALE_LABELS), [
    "最小",
    "小",
    "标准",
    "大",
    "最大",
  ]);
  assert.deepEqual(
    (["xs", "sm", "md", "lg", "xl"] as const).map(fontSizeForScale),
    ["12.75px", "13.875px", "15px", "16.5px", "18px"],
  );
});


test("the UI does not load bundled web fonts from CSS", () => {
  const fontFaces = [...css.matchAll(/@font-face\s*\{([\s\S]*?)\}/g)].map(
    (match) => match[1],
  );
  assert.equal(fontFaces.length, 0);
  assert.doesNotMatch(css, /\.woff2/);
});


test("dark mode surfaces use semantic theme tokens", () => {
  for (const selector of [
    ".output-row",
    ".resource-info-note",
    ".resource-install-log",
    ".settings-callout",
  ]) {
    const start = css.indexOf(`${selector} {`);
    assert.notEqual(start, -1, `${selector} is present`);
    const declaration = css.slice(start, css.indexOf("}", start) + 1);
    assert.match(declaration, /var\(--(?:surface|panel|text|border)/);
    assert.doesNotMatch(declaration, /(?:background|color):\s*#[0-9a-f]{3,8}/i);
  }
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
