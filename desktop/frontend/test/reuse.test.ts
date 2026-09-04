import assert from "node:assert/strict";
import test from "node:test";

import { reusableAsrTask } from "../lib/reuse";
import type { JobSnapshot, TaskRequest } from "../lib/types";


function snapshot(overrides: {
  input: string;
  stage: TaskRequest["stage"];
  output?: string | null;
  state?: JobSnapshot["state"];
  created_at?: number;
}): JobSnapshot {
  return {
    task_id: `${overrides.input}-${overrides.created_at ?? 0}`,
    state: overrides.state ?? "completed",
    created_at: overrides.created_at ?? 0,
    events: [],
    request: {
      input: overrides.input,
      output:
        overrides.output === undefined ? "C:/tasks/a/a.srt" : overrides.output,
      name: "",
      cleanup_intermediate: false,
      stage: overrides.stage,
      model_name: "large-v3-turbo",
      device: "cuda",
      language: null,
      gpu_tier: "entry",
      word: false,
      asr_stabilize_profile: 0,
      llm_media: "video",
      llm_retrieval: "local",
      llm_difficulty: "quality",
      llm_fast: "auto",
      llm_output_scale: 1,
      extra_info: "",
      extra_style: "",
      knowledge: "update",
      postprocess_profile: 0,
    },
  };
}


test("the newest completed recognition run for the same input is offered", () => {
  const older = snapshot({ input: "D:/a.mp4", stage: "raw-srt", created_at: 1 });
  const newer = snapshot({ input: "D:/a.mp4", stage: "raw-srt", created_at: 2 });

  assert.equal(reusableAsrTask([older, newer], "D:/a.mp4"), newer);
  assert.equal(reusableAsrTask([newer, older], "D:/a.mp4"), newer);
});


test("runs that cannot seed an LLM pass are not offered", () => {
  const tasks = [
    // A finished final-srt directory would satisfy the LLM stage's existence
    // check too -- "reusing" it would republish the old subtitle.
    snapshot({ input: "D:/a.mp4", stage: "final-srt" }),
    snapshot({ input: "D:/other.mp4", stage: "raw-srt" }),
    snapshot({ input: "D:/a.mp4", stage: "raw-srt", state: "failed" }),
    snapshot({ input: "D:/a.mp4", stage: "raw-srt", output: null }),
  ];

  assert.equal(reusableAsrTask(tasks, "D:/a.mp4"), null);
  assert.equal(reusableAsrTask(tasks, ""), null);
});
