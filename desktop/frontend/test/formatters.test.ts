import assert from "node:assert/strict";
import test from "node:test";

import {
  formatBytes,
  formatCapability,
  formatDuration,
  formatPercent,
  formatUpdateSummary,
  invalidOutputName,
  isUrlSource,
} from "../lib/formatters";


test("missing translation key is neutral capability copy", () => {
  assert.deepEqual(formatCapability({ translation: false }), {
    tone: "neutral",
    title: "翻译功能未配置",
    detail: "不影响生成原始字幕",
  });
});


test("bytes and progress never display NaN", () => {
  assert.equal(formatBytes(undefined), "—");
  assert.equal(formatBytes(Number.NaN), "—");
  assert.equal(formatPercent(0, 0), "0%");
  assert.equal(formatPercent(undefined, undefined), "0%");
});


test("duration remains compact for long tasks", () => {
  assert.equal(formatDuration(65), "1:05");
  assert.equal(formatDuration(3_725), "1:02:05");
});


test("update summary distinguishes app and full packages", () => {
  assert.equal(
    formatUpdateSummary({
      available: true,
      version: "1.2.0",
      kind: "app",
      releaseNotes: "",
      mandatory: false,
      size: 18 * 1024 * 1024,
    }),
    "发现 1.2.0 · 轻量补丁 · 18.0 MB",
  );
  assert.equal(
    formatUpdateSummary({ available: false, version: "1.1.0" }),
    "已经是最新版本",
  );
});


test("URL sources are recognised the same way the backend recognises them", () => {
  // finesub_bootstrap.capabilities.is_url decides whether yt-dlp gets fetched;
  // if the UI disagreed it would accept an input the backend then refuses.
  assert.equal(isUrlSource("https://example.test/v"), true);
  assert.equal(isUrlSource("http://example.test/v"), true);
  assert.equal(isUrlSource("C:/media/a.mp4"), false);
  assert.equal(isUrlSource("/home/me/a.mp4"), false);
  assert.equal(isUrlSource("ftp://example.test/v"), false);
  assert.equal(isUrlSource(""), false);
});

test("output names are rejected exactly where the backend rejects them", () => {
  // Mirrors TaskRequest.validate_name. The start button is disabled on this, so
  // a mismatch would either block a legal name or let the backend answer with
  // its generic "invalid task parameters".
  assert.equal(invalidOutputName(""), false);
  assert.equal(invalidOutputName("   "), false);
  assert.equal(invalidOutputName("my-clip"), false);
  assert.equal(invalidOutputName("片段 01"), false);
  assert.equal(invalidOutputName("nested/name"), true);
  assert.equal(invalidOutputName("nested\\name"), true);
  assert.equal(invalidOutputName("."), true);
  assert.equal(invalidOutputName(".."), true);
  assert.equal(invalidOutputName("  ..  "), true);
});
