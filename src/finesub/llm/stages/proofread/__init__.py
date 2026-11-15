"""Three passes' pass 3: proofreading the finished subtitles (`docs/llm_harness_behavior.md`「三个 pass」).

Runs once pass 2's windows are stitched and the final SRT's timeline is post-processed, on the subtitles a
viewer sees: per correction window, one call that reads the corrected text, the translation and how long each
line stays on screen, and answers only the lines it changes -- content, never the split. `prompt` builds what
the model reads, `answers` reads its reply (the agent path's validator reads it the same way), `run` makes the
calls, keeps each window's answer for a rerun, and writes the changes back into the rows.

Ported from `tools/diarization_explore/proofread.py`, as trialled on the test windows.
"""
