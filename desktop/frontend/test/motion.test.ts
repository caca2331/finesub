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
  const taskSettings = read("../components/TaskSettings.tsx");
  const titleBar = read("../components/TitleBar.tsx");
  const translations = read("../lib/translations.ts");
  const packageJson = read("../package.json");

  assert.match(sidebar, /className="nav-active-pill"/);
  assert.match(css, /translate3d\(0, calc\(var\(--active-index\) \* 48px\), 0\)/);
  assert.match(shell, /className="workspace-view" key=\{state\.route\}/);
  assert.match(settings, /role="switch"/);
  assert.match(appearance, /animations: boolean/);
  assert.match(appearance, /data-motion/);
  assert.match(css, /\[data-motion="off"\]/);
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.match(taskSettings, /advanced-grid-animated/);
  assert.match(css, /@keyframes advanced-grid-in/);
  assert.match(settings, /startViewTransition/);
  assert.match(settings, /flushSync/);
  assert.match(css, /view-transition-name: active-theme-choice/);
  assert.match(css, /::view-transition-group\(active-theme-choice\)/);
  assert.match(titleBar, /onDoubleClick=\{toggleMaximize\}/);
  assert.doesNotMatch(titleBar, /beginWindowResize/);
  assert.match(translations, /theme: "主题色"/);
  assert.doesNotMatch(packageJson, /framer-motion|motion-one|gsap/);
});
