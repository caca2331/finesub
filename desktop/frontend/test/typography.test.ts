import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";


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


test("the complete UI uses Windows system fonts at readable sizes", () => {
  assert.match(css, /font-family:\s*var\(--user-font,\s*"Microsoft YaHei UI"\)/);
  assert.match(css, /html\s*\{[\s\S]*font-size:\s*var\(--base-font-size\)/);
  assert.doesNotMatch(css, /^\s*font-size:\s*\d+(?:\.\d+)?px;/m);
  assert.ok([...css.matchAll(/font-size:\s*[\d.]+rem/g)].length > 30);
});


test("the UI does not bundle web fonts", () => {
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
  assert.match(titleBar, /className="brand-icon"/);
  assert.match(titleBar, /<span>FineSub Desktop<\/span>/);
  assert.doesNotMatch(page, /className="brand-glyph"/);
});


test("character themes include the requested Marisa and Yanami palettes", () => {
  assert.match(
    css,
    /\[data-accent="marisa"\][\s\S]*?#dcaa35[\s\S]*?#ffffff/,
  );
  assert.match(css, /\[data-accent="yanami"\]/);
  assert.match(css, /#102e59[\s\S]*?#4674aa[\s\S]*?#75a85b/);
});
