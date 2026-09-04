"""Two ASR models over the same audio, scored without a ground truth.

The question this answers -- "is the Japanese finetune better than turbo on
Japanese material" -- has no transcript to score against: the only human
reference this repository holds is *translated* subtitles (docs/data-index.md),
which cannot adjudicate a Japanese spelling. So the protocol is:

1. **Run both arms off the same upstream.** The products already carry
   `-vocal.ogg` and `-vad.json`, so separation and VAD are shared byte for
   byte and the model is the only difference. Language is forced for both, so
   detection noise is not part of the comparison either.
2. **Stabilize both**, because that is the artifact the question is about.
3. **Pair segments by time and keep only where they disagree.** Everywhere the
   two agree there is nothing to judge, and that is most of it.
4. **Adjudicate the disagreements with a third model.** The Qwen referee is
   already the project's second-model evidence path, it is local and free, and
   crucially it is *neutral*: it is neither arm, so "which arm does the judge
   agree with" is symmetric. It is not ground truth -- it is a third opinion,
   and the report says so by also counting the ties.

**Two phases, two interpreters.** Nothing on this machine has both
faster-whisper and transformers 5.x: `envs/asr` runs the arms, `envs/qwen-asr`
runs the referee. So phase 1 writes `report.json` and phase 2 (`--adjudicate`)
reads it back and fills in the verdicts.

What this deliberately does NOT do is score against `disfluency_gold` or the
refined subtitles. Both were produced by editing a turbo run's output, so a
turbo arm is structurally advantaged (docs/data-index.md says as much for the
first, and docs/knowledge.md's "精修字幕怎么读" for the second).

    PYTHONPATH="$(pwd)/src:$(pwd)" python -m tools.bench.probe_asr_model_ab \\
        --arms large-v3-turbo,TransWithAI/whisper-ja-1.5B-ct2 \\
        --language ja --out tmp/bench/ja-ab out/ts-baseline-20260831-hq/*
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finesub.reporting import NullReporter, reporting_to  # noqa: E402
from finesub.speech.postprocessing.stabilization import stabilize_json_file  # noqa: E402
from finesub.speech.recognition import vad_asr_stage  # noqa: E402
from tools.wt_refine_validation.artifact_survey import match_stable  # noqa: E402
from tools.wt_refine_validation.run import edit_similarity  # noqa: E402

#: Below this the two arms are saying different things, not spelling the same
#: thing differently. Same threshold as the decode-batch gate (第十节 10.2), so
#: "disagreement" means the same thing in both reports.
DISAGREE_BELOW = 0.98
#: Clips shorter than this read as "no speech heard" to the referee -- the same
#: floor `apply_verification` uses.
MIN_CLIP_SEC = 0.05


def arm_key(model: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "-", model).strip("-").lower()


def run_arm(product: Path, model: str, out_dir: Path, gpu_tier: str, language: str) -> dict:
    """One model over one product: aligned JSON, then the stable artifact."""

    stem = product.name
    key = arm_key(model)
    aligned = out_dir / f"{stem}-{key}-aligned.json"
    vad_copy = out_dir / f"{stem}-{key}-aligned-vad.json"
    shutil.copy(product / f"{stem}-vad.json", vad_copy)
    if aligned.exists():
        aligned.unlink()
    began = time.perf_counter()
    with reporting_to(NullReporter()):
        vad_asr_stage.run_vad_asr(
            str(product / f"{stem}-vocal.ogg"),
            output_path=str(aligned),
            model_name=model,
            device="cuda",
            gpu_tier=gpu_tier,
            language=language,
            # Both off on purpose: the referee comes back later as an
            # independent judge, and it must not have touched either arm's
            # output first. The redecode is off for the same reason, and
            # because the language is forced anyway.
            qwen_verify="off",
            lang_redecode="off",
        )
    wall = time.perf_counter() - began
    stable, _report = stabilize_json_file(aligned)
    data = json.loads(aligned.read_text(encoding="utf-8"))
    meta = data["metadata"]["asr_align"]
    recovery = meta.get("recovery", {})
    stable_segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    return {
        "model": model,
        "wall_sec": round(wall, 2),
        "asr_align_sec": round(float(meta["timing"]["asr_align_sec"]), 2),
        "aligned_segments": len(data["segments"]),
        "stable_segments": len(stable_segments),
        "stable_chars": sum(len(str(s.get("text", ""))) for s in stable_segments),
        "recovery": {
            k: int(v)
            for k, v in recovery.items()
            if k in ("dropped_groups", "abnormal_groups", "beam_rescue_attempted")
        },
        "aligned": str(aligned),
        "stable": str(stable),
    }


def _segments(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["segments"]


def interval_compare(vad_path: Path, a_path: str, b_path: str) -> dict:
    """Compare per shared VAD interval, not per segment.

    Segment boundaries are a *model output*: the two arms cut the same speech
    into 88 and 62 pieces on one product, and pairing those by overlap scores a
    merge as a text disagreement. The VAD intervals are the same bytes for both
    arms (the products carry one `-vad.json` and both runs read it), so they
    are the one bucket neither model chose. Concatenated text per interval is
    therefore the honest text comparison; the segment-level numbers below stay
    because segmentation *is* part of the deliverable -- they just answer a
    different question.
    """

    intervals = json.loads(vad_path.read_text(encoding="utf-8"))["segments"]
    a, b = _segments(a_path), _segments(b_path)

    def bucket(segments: list[dict]) -> list[list[str]]:
        """Words, not segments.

        Bucketing whole segments by their midpoint reads as a **model
        difference** when it is only a segment-length difference: the finetune
        emits fewer, longer segments, one segment lands wholly in the interval
        holding its midpoint, and the neighbouring interval comes out empty
        even though that speech was transcribed. Measured on this corpus that
        was 390 of 429 apparent silences -- enough to invert the headline.

        A word is far shorter than a VAD interval, so a word midpoint lands in
        the interval its audio is actually in, whichever way either model cut
        its segments.
        """

        packed: list[list[str]] = [[] for _ in intervals]
        starts = [float(interval["start"]) for interval in intervals]
        ends = [float(interval["end"]) for interval in intervals]
        for seg in segments:
            words = seg.get("words") or []
            if not words:
                # No word timings (an empty or degenerate segment): fall back
                # to the segment's own midpoint rather than dropping its text.
                words = [
                    {
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "word": seg.get("text", ""),
                    }
                ]
            for word in words:
                middle = (float(word.get("start", 0.0)) + float(word.get("end", 0.0))) / 2.0
                for index in range(len(intervals)):
                    if starts[index] <= middle < ends[index]:
                        packed[index].append(str(word.get("word", "")))
                        break
        return packed

    packed_a, packed_b = bucket(a), bucket(b)
    rows = []
    identical = empty_a = empty_b = 0
    for index, interval in enumerate(intervals):
        text_a = "".join(packed_a[index]).strip()
        text_b = "".join(packed_b[index]).strip()
        if not text_a and not text_b:
            continue
        empty_a += not text_a
        empty_b += not text_b
        sim = edit_similarity(text_a, text_b)
        sim = 1.0 if sim is None else float(sim)
        identical += sim >= 0.9999
        rows.append(
            {
                "start": round(float(interval["start"]), 2),
                "end": round(float(interval["end"]), 2),
                "sim": round(sim, 4),
                "a_text": text_a,
                "b_text": text_b,
            }
        )
    scored = len(rows)
    disagree = [row for row in rows if row["sim"] < DISAGREE_BELOW]
    return {
        "intervals": len(intervals),
        "scored": scored,
        "identical": identical,
        "identical_pct": round(100.0 * identical / max(1, scored), 2),
        "disagree": len(disagree),
        "disagree_pct": round(100.0 * len(disagree) / max(1, scored), 2),
        "silent_in_a": empty_a,
        "silent_in_b": empty_b,
        "rows": disagree,
    }


def compare(a_path: str, b_path: str) -> dict:
    """Symmetric pairing: every disagreement, from both directions.

    Pairing from one side only hides the asymmetry that matters most -- a
    segment one arm produced and the other did not at all.
    """

    a, b = _segments(a_path), _segments(b_path)
    pairs: dict[tuple[int, int], dict] = {}
    unmatched = {"a": 0, "b": 0}
    index_a = {id(seg): i for i, seg in enumerate(a)}
    index_b = {id(seg): i for i, seg in enumerate(b)}

    for side, source, target in (("a", a, b), ("b", b, a)):
        for i, seg in enumerate(source):
            match = match_stable(seg, target)
            if match is None:
                unmatched[side] += 1
                continue
            j = (index_b if side == "a" else index_a)[id(match)]
            key = (i, j) if side == "a" else (j, i)
            if key in pairs:
                continue
            left = a[key[0]]
            right = b[key[1]]
            sim = edit_similarity(str(left.get("text", "")), str(right.get("text", "")))
            pairs[key] = {
                "start": round(float(left.get("start", 0.0)), 2),
                "end": round(float(left.get("end", 0.0)), 2),
                "b_start": round(float(right.get("start", 0.0)), 2),
                "b_end": round(float(right.get("end", 0.0)), 2),
                "sim": 1.0 if sim is None else round(float(sim), 4),
                "a_text": str(left.get("text", "")),
                "b_text": str(right.get("text", "")),
            }

    rows = list(pairs.values())
    disagree = [row for row in rows if row["sim"] < DISAGREE_BELOW]
    identical = sum(1 for row in rows if row["sim"] >= 0.9999)
    return {
        "paired": len(rows),
        "identical": identical,
        "identical_pct": round(100.0 * identical / max(1, len(rows)), 2),
        "disagree": len(disagree),
        "disagree_pct": round(100.0 * len(disagree) / max(1, len(rows)), 2),
        "a_only": unmatched["a"],
        "b_only": unmatched["b"],
        "rows": disagree,
    }


def _referee_similarity(arm_text: str, referee_text: str) -> float:
    """How well one arm matches the judge, with silence counted as an answer.

    `edit_similarity` returns None when both sides are empty; here that is the
    strongest possible agreement, not a missing value. An interval where the
    judge also heard nothing is the case that decides whether an arm's silence
    is a dropped line or a refused hallucination.
    """

    if not arm_text.strip() and not referee_text.strip():
        return 1.0
    value = edit_similarity(arm_text, referee_text)
    return 0.0 if value is None else float(value)


def adjudicate_report(report_path: Path) -> int:
    """Phase 2: fill in the referee's verdicts on an existing report.

    Separate because the referee needs transformers 5.x and the arms need
    faster-whisper, and no interpreter here has both.
    """

    from finesub.speech.verification.qwen_referee import QwenReferee, _SpanReader

    report = json.loads(report_path.read_text(encoding="utf-8"))
    referee = QwenReferee(device="cuda")
    totals = {"a": 0, "b": 0, "tie": 0}
    # Kept apart because it answers a different question: where one arm said
    # nothing at all, "who does the judge agree with" is asking whether that
    # silence dropped a line or refused a hallucination.
    silence = {"a_silent": {"a": 0, "b": 0, "tie": 0}, "b_silent": {"a": 0, "b": 0, "tie": 0}}
    try:
        for name, entry in report["files"].items():
            product = Path(entry["product"])
            verdicts = adjudicate(
                product, entry["interval_diff"]["rows"], referee, _SpanReader
            )
            entry["verdicts"] = verdicts
            tally = {"a": 0, "b": 0, "tie": 0}
            for verdict in verdicts:
                tally[verdict["winner"]] += 1
                totals[verdict["winner"]] += 1
                quiet_a = not verdict["a_text"].strip()
                quiet_b = not verdict["b_text"].strip()
                if quiet_a and not quiet_b:
                    silence["a_silent"][verdict["winner"]] += 1
                elif quiet_b and not quiet_a:
                    silence["b_silent"][verdict["winner"]] += 1
            print(f"{name:16} referee: {tally}  (of {len(verdicts)} judged)")
    finally:
        referee.close()
    report["totals"]["referee"] = totals
    report["totals"]["referee_silence"] = silence
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    a, b = report["arms"]
    print(f"referee pooled -- A={a} B={b} :: {totals}")
    print(f"  where A was silent: {silence['a_silent']}")
    print(f"  where B was silent: {silence['b_silent']}")
    return 0


def adjudicate(product: Path, rows: list[dict], referee, reader_cls) -> list[dict]:
    """Ask the third model what it hears, and score both arms against it."""

    stem = product.name
    reader = reader_cls(str(product / f"{stem}-vocal.ogg"))
    usable = []
    clips = []
    for row in rows:
        start = float(row["start"])
        end = float(row["end"])
        if end - start < MIN_CLIP_SEC:
            continue
        clip = reader.read(start, end)
        if clip.size == 0:
            continue
        usable.append(row)
        clips.append(clip)
    if not clips:
        return []
    heard = referee.transcribe_batch(clips)
    verdicts = []
    for row, (text, _language) in zip(usable, heard):
        # `None` means both sides were empty, and that is agreement, not a
        # miss: the most interesting asymmetry in this comparison is one arm
        # staying silent where the other speaks, and scoring "the judge heard
        # nothing either" as 0.0 would hand that case to the arm that spoke.
        sim_a = _referee_similarity(row["a_text"], text)
        sim_b = _referee_similarity(row["b_text"], text)
        verdicts.append(
            {
                **row,
                "referee_text": text,
                "sim_a_referee": round(sim_a, 4),
                "sim_b_referee": round(sim_b, 4),
                # A margin, not a coin flip: two arms within 0.02 of the
                # judge are not distinguished by it, and saying so is more
                # honest than awarding the point to floating-point noise.
                "winner": (
                    "a" if sim_a - sim_b > 0.02
                    else "b" if sim_b - sim_a > 0.02
                    else "tie"
                ),
            }
        )
    return verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("products", nargs="*")
    parser.add_argument(
        "--arms",
        default="large-v3-turbo,TransWithAI/whisper-ja-1.5B-ct2",
        help="exactly two model names, arm A then arm B",
    )
    parser.add_argument("--language", default="ja")
    parser.add_argument("--gpu-tier", default="standard")
    parser.add_argument("--out", default="tmp/bench/asr-model-ab")
    parser.add_argument(
        "--adjudicate",
        default="",
        help=(
            "phase 2: path to a report.json from phase 1. Needs the "
            "transformers 5.x interpreter, not the faster-whisper one."
        ),
    )
    args = parser.parse_args(argv)

    if args.adjudicate:
        return adjudicate_report(Path(args.adjudicate))
    if not args.products:
        parser.error("give products, or --adjudicate a report.json")

    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    if len(arms) != 2:
        parser.error("--arms takes exactly two models")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    products = [Path(p) for p in args.products]

    report: dict = {"arms": arms, "language": args.language, "files": {}}
    for product in products:
        runs = {}
        # Arm order alternates per product (bench-baselines P4): a warm card
        # is not the same card, and always running A first would give it the
        # cold one every time.
        order = arms if len(report["files"]) % 2 == 0 else list(reversed(arms))
        for model in order:
            runs[model] = run_arm(product, model, out_dir, args.gpu_tier, args.language)
            row = runs[model]
            print(
                f"{product.name:16} {model:34} {row['asr_align_sec']:7.1f}s "
                f"{row['stable_segments']:4d} seg {row['stable_chars']:6d} chars "
                f"{row['recovery']}"
            )
        seg_diff = compare(runs[arms[0]]["stable"], runs[arms[1]]["stable"])
        vad_path = out_dir / f"{product.name}-{arm_key(arms[0])}-aligned-vad.json"
        iv_diff = interval_compare(
            vad_path, runs[arms[0]]["stable"], runs[arms[1]]["stable"]
        )
        print(
            f"{product.name:16} interval  scored {iv_diff['scored']:4d}  identical "
            f"{iv_diff['identical_pct']:5.1f}%  disagree {iv_diff['disagree']:4d} "
            f"({iv_diff['disagree_pct']:.1f}%)  silent-in-A {iv_diff['silent_in_a']}  "
            f"silent-in-B {iv_diff['silent_in_b']}"
        )
        print(
            f"{product.name:16} segment   paired {seg_diff['paired']:4d}  identical "
            f"{seg_diff['identical_pct']:5.1f}%  only-A {seg_diff['a_only']}  "
            f"only-B {seg_diff['b_only']}"
        )
        report["files"][product.name] = {
            "product": str(product),
            "runs": runs,
            "segment_diff": seg_diff,
            "interval_diff": iv_diff,
        }

    pooled = {
        "seg_paired": 0,
        "seg_identical": 0,
        "seg_only_a": 0,
        "seg_only_b": 0,
        "iv_scored": 0,
        "iv_identical": 0,
        "iv_disagree": 0,
        "iv_silent_a": 0,
        "iv_silent_b": 0,
        "chars_a": 0,
        "chars_b": 0,
        "align_sec_a": 0.0,
        "align_sec_b": 0.0,
    }
    for entry in report["files"].values():
        seg, iv = entry["segment_diff"], entry["interval_diff"]
        pooled["seg_paired"] += seg["paired"]
        pooled["seg_identical"] += seg["identical"]
        pooled["seg_only_a"] += seg["a_only"]
        pooled["seg_only_b"] += seg["b_only"]
        pooled["iv_scored"] += iv["scored"]
        pooled["iv_identical"] += iv["identical"]
        pooled["iv_disagree"] += iv["disagree"]
        pooled["iv_silent_a"] += iv["silent_in_a"]
        pooled["iv_silent_b"] += iv["silent_in_b"]
        pooled["chars_a"] += entry["runs"][arms[0]]["stable_chars"]
        pooled["chars_b"] += entry["runs"][arms[1]]["stable_chars"]
        pooled["align_sec_a"] += entry["runs"][arms[0]]["asr_align_sec"]
        pooled["align_sec_b"] += entry["runs"][arms[1]]["asr_align_sec"]
    pooled["iv_identical_pct"] = round(
        100.0 * pooled["iv_identical"] / max(1, pooled["iv_scored"]), 2
    )
    pooled["referee"] = {"a": 0, "b": 0, "tie": 0}
    report["totals"] = pooled
    print("")
    print("== pooled ==")
    print(json.dumps(pooled, ensure_ascii=False))
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
