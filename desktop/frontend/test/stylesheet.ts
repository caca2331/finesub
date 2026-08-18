import { readFileSync } from "node:fs";

const ENTRY = new URL("../app/globals.css", import.meta.url);

/** The stylesheets `globals.css` imports, in cascade order. */
export function importedStylesheets(): string[] {
  const source = readFileSync(ENTRY, "utf8");
  return [...source.matchAll(/@import\s+"([^"]+)"\s*;/g)].map(
    (match) => match[1],
  );
}

/**
 * The whole stylesheet, as the build sees it.
 *
 * `app/globals.css` is now an import list; the tests that assert about the
 * styles still have to read all of them, in order. Resolving the imports here
 * rather than globbing `styles/` also keeps the list itself under test: a file
 * nobody imports contributes nothing, and would silently stop being checked.
 */
export function readStylesheet(): string {
  const imports = importedStylesheets();
  if (imports.length === 0) {
    return readFileSync(ENTRY, "utf8");
  }
  return imports
    .map((path) => readFileSync(new URL(path, ENTRY), "utf8"))
    .join("\n");
}
