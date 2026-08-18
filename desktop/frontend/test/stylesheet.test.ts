import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { importedStylesheets } from "./stylesheet";


test("every file in app/styles is imported, and in one order", () => {
  // The sheet is split across files now, and `globals.css` is the only thing
  // that knows they exist. A new file nobody imports contributes no styles at
  // all -- and the failure is a component that quietly renders unstyled, not
  // an error anyone sees. The reverse (an import naming a file that is gone)
  // breaks the build, so only this direction needs watching.
  const imported = importedStylesheets();
  const onDisk = readdirSync(new URL("../app/styles/", import.meta.url))
    .filter((name) => name.endsWith(".css"))
    .sort();

  assert.deepEqual(
    imported.map((path) => path.replace("./styles/", "")).sort(),
    onDisk,
  );
  // Order is the cascade: same specificity, later wins. Dark overrides only
  // work because they come after the components they repaint.
  const dark = imported.indexOf("./styles/dark.css");
  const appearance = imported.indexOf("./styles/appearance.css");
  // Both explicitly: `indexOf` answers -1 for a file that is gone, and
  // `15 > -1` would let this assertion pass while testing nothing.
  assert.ok(dark >= 0 && appearance >= 0);
  assert.ok(dark > appearance);
  assert.equal(imported.at(-1), "./styles/fallbacks.css");
});
