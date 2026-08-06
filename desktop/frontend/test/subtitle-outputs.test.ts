import assert from "node:assert/strict";
import test from "node:test";

import {
  preferredSubtitleOutput,
  subtitleOutputEntries,
} from "../lib/subtitleOutputs";


test("completed tasks expose subtitles only and prefer the final result", () => {
  const outputs = {
    vocalAudio: "D:/private/a-vocal.ogg",
    rawSrt: "D:/media/a-raw.srt",
    finalSrt: "D:/media/a.srt",
    metadataJson: "D:/private/a-run.json",
  };

  assert.deepEqual(subtitleOutputEntries(outputs), [
    ["finalSrt", "D:/media/a.srt"],
    ["rawSrt", "D:/media/a-raw.srt"],
  ]);
  assert.equal(preferredSubtitleOutput(outputs), "D:/media/a.srt");
});
