import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";


const read = (path: string) =>
  readFileSync(new URL(path, import.meta.url), "utf8");


test("sidebar motion is native, optional, and accessibility-aware", () => {
  const css = read("../app/globals.css");
  const shell = read("../components/AppShell.tsx");
  const sidebar = read("../components/Sidebar.tsx");
  const settings = read("../components/Settings.tsx");
  const appearance = read("../lib/useAppearance.ts");
  const packageJson = read("../package.json");

  assert.match(sidebar, /className="nav-active-pill"/);
  assert.match(css, /translate3d\(0, calc\(var\(--active-index\) \* 48px\), 0\)/);
  assert.match(shell, /className="workspace-view" key=\{state\.route\}/);
  assert.match(settings, /role="switch"/);
  assert.match(appearance, /animations: boolean/);
  assert.match(appearance, /data-motion/);
  assert.match(css, /\[data-motion="off"\]/);
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.doesNotMatch(packageJson, /framer-motion|motion-one|gsap/);
});
