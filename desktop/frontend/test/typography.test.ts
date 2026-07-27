import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
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


test("the complete UI uses embedded Maple Mono at readable sizes", () => {
  assert.match(css, /font-family:\s*"Maple Mono NL NF CN"/);
  const pixelSizes = [...css.matchAll(/font-size:\s*(\d+)px/g)].map(
    (match) => Number(match[1]),
  );
  assert.ok(pixelSizes.length > 30);
  assert.deepEqual(
    pixelSizes.filter((size) => size < 12),
    [],
  );
});


test("three portable Maple Mono weights are declared and present", () => {
  const fontFaces = [...css.matchAll(/@font-face\s*\{([\s\S]*?)\}/g)].map(
    (match) => match[1],
  );
  assert.equal(fontFaces.length, 3);
  const weights = fontFaces
    .map((block) => block.match(/font-weight:\s*(\d+)/)?.[1])
    .sort();
  assert.deepEqual(weights, ["400", "600", "700"]);
  for (const filename of [
    "maple-mono-nl-nf-cn-regular.woff2",
    "maple-mono-nl-nf-cn-semibold.woff2",
    "maple-mono-nl-nf-cn-bold.woff2",
  ]) {
    assert.equal(
      existsSync(new URL(`../public/fonts/${filename}`, import.meta.url)),
      true,
    );
    assert.match(css, new RegExp(filename.replaceAll(".", String.raw`\.`)));
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
