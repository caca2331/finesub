import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) =>
  readFileSync(new URL(path, import.meta.url), "utf8");

// These check agreements that span two files, where neither side can tell it
// has been broken: a class the stylesheet styles but nothing renders, a switch
// with no stylesheet behind it. Assertions about a single file's own wording
// belong nowhere -- they only restate the file back to itself.

test("the moving pill the sidebar renders is the one the stylesheet drives", () => {
  const sidebar = read("../components/Sidebar.tsx");
  const css = read("../app/globals.css");

  assert.match(sidebar, /className="nav-active-pill"/);
  assert.match(sidebar, /"--active-index"/);
  assert.match(css, /\.nav-active-pill\b/);
  assert.match(css, /var\(--active-index\)/);
});

test("motion can be turned off, by the user and by the system", () => {
  const appearance = read("../lib/useAppearance.ts");
  const css = read("../app/globals.css");

  // The setting writes an attribute; the stylesheet is what honours it.
  assert.match(appearance, /animations: boolean/);
  assert.match(appearance, /data-motion/);
  assert.match(css, /\[data-motion="off"\]/);
  assert.match(css, /prefers-reduced-motion: reduce/);
});

test("the animations stay native to the platform", () => {
  // View transitions and CSS keyframes only -- an animation library would be
  // several hundred KB in a window that ships its own runtime already.
  const packageJson = read("../package.json");

  assert.doesNotMatch(packageJson, /framer-motion|motion-one|gsap/);
});
