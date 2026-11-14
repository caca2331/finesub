"""Community acceptance, step 2: score the candidate arm against the reference.

Reads what `run.py` left and writes `acceptance.json` beside it. The numbers are
the review triggers of README_DEV.md (development principles, class A), which
default to docs/bench-baselines.md section 10; the hardware plan adds coverage,
beam-rung exclusion, refine fallback and per-duration reporting. Exceeding a
trigger is **not a failure**: it asks for a written review against the three
questions (same-signed bias beyond the trigger on every file? a proportion that
grows with duration? a mechanism spreading a local difference?).

    python -m tools.hw_acceptance.score tmp/hw-acceptance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.probe_asr_decode_batch import score as score_segments  # noqa: E402
from tools.bench.probe_asr_decode_batch import summarise  # noqa: E402

REFERENCE = "reference"
CANDIDATE = "candidate"

#: What a stage whose artifact does not say where it ran is recorded as.
UNKNOWN_DEVICE = "?"

#: How much of a segment must be covered for the other arm's segment to be its
#: match. Copied from `tools/wt_refine_validation/artifact_survey.match_stable`
#: rather than imported: that directory is a closed experiment kept for
#: hindsight (`tools/README.md`), and this suite is maintained.
MATCH_MIN_RATIO = 0.3


def match_by_overlap(segment: dict, others: list[dict]) -> dict | None:
    """The best time-overlapping segment, or None when nobody kept this one.

    Time only, so two segments may match the same one and a zero-length segment
    matches nothing. Good enough for a reported rate, not for a threshold.
    """

    start, end = float(segment.get("start", 0.0)), float(segment.get("end", 0.0))
    best, best_overlap = None, 0.0
    for other in others:
        overlap = max(
            0.0,
            min(end, float(other.get("end", 0.0)))
            - max(start, float(other.get("start", 0.0))),
        )
        if overlap > best_overlap:
            best, best_overlap = other, overlap
    if best is not None and best_overlap / max(1e-6, end - start) >= MATCH_MIN_RATIO:
        return best
    return None

#: docs/bench-baselines.md section 10, criteria 1-3, as review triggers.
TRIGGERS = {
    "sim_median_min": 1.0,
    "below_098_pct_max": 0.5,
    "over_20ms_pct_max": 1.0,
    "abs_mean_ms_max": 2.0,
    "pos_neg_ratio_range": (0.8, 1.25),
    "rescue_increase_max": 0.20,
    "coverage_diff_max": 0.01,
}

RESCUE_KEYS = ("abnormal_groups", "beam_rescue_attempted")
DURATION_BUCKETS = ((0, 5 * 60, "short <5 min"), (5 * 60, 30 * 60, "medium 5-30 min"), (30 * 60, None, "long >=30 min"))


def _aligned(root: Path, arm: str, stem: str) -> dict:
    return json.loads((root / arm / stem / f"{stem}-aligned.json").read_text(encoding="utf-8"))


def _recovery(aligned: dict) -> dict:
    return dict(aligned["metadata"]["asr_align"].get("recovery") or {})


def _devices(root: Path, arm: str, stem: str, aligned: dict) -> dict:
    """What each stage of this arm actually ran on, as the artifacts recorded it.

    Not what was asked for: an AMD machine with a CUDA-only CTranslate2 falls
    back to the CPU for ASR, and with the shared vocal track both arms would
    then be the same CPU run scoring a perfect, meaningless record.
    """

    devices = {
        "asr": str(aligned["metadata"]["asr_align"].get("device") or UNKNOWN_DEVICE)
    }
    sidecar = root / arm / stem / f"{stem}-metadata.json"
    try:
        workers = json.loads(sidecar.read_text(encoding="utf-8")).get("workers") or {}
    except (OSError, json.JSONDecodeError):
        workers = {}
    separation = (workers.get("vocal_separation") or {}).get("device")
    if separation:
        devices["vocal_separation"] = str(separation)
    # The referee decides whether a subtitle survives, so where it ran is part
    # of what an acceptance record claims -- and on a platform whose ASR stays
    # on the CPU anyway (Windows + AMD today, Intel by design) it can be the
    # only stage that moved, which is what makes the record worth anything.
    # ⚠ Not on Intel: there the tail referee stays on the CPU by design too
    # (its dtype decides subtitles, and no CPU-vs-XPU agreement has been
    # measured), so the moved stage this record can lean on is
    # `vocal_separation`.
    referee = (aligned["metadata"]["asr_align"].get("qwen_verify") or {}).get("device")
    devices["referee"] = str(referee) if referee else UNKNOWN_DEVICE
    return devices


def _coverage_ratio(aligned: dict) -> float | None:
    """ASR coverage: speech the recognised segments cover, over all VAD speech.

    From the per-group `asr_coverage` ledger (after rescue), not `audio_coverage`,
    which is VAD speech over the file and does not move with the ASR backend.
    """

    groups = aligned["metadata"]["asr_align"].get("asr_coverage") or []
    speech = sum(float(group.get("speech_sec") or 0.0) for group in groups)
    if speech <= 0.0:
        return None
    covered = sum(
        float(group.get("rescued_covered_sec", group.get("covered_sec")) or 0.0)
        for group in groups
    )
    return covered / speech


def rescue_comparison(reference: dict, candidate: dict) -> dict:
    """Increase per rescue rung, leaving out a rung one side does not have.

    A backend without beam search skips the beam rung by contract
    (asr_backend.DECODE_OPTIONS); its zero beam attempts are a capability
    difference, not a cascade, so the rung is excluded and noted.
    """

    skipped = int(candidate.get("beam_rescue_skipped", 0)) or int(reference.get("beam_rescue_skipped", 0))
    out: dict = {"dropped_groups": {REFERENCE: int(reference.get("dropped_groups", 0)), CANDIDATE: int(candidate.get("dropped_groups", 0))}}
    for key in RESCUE_KEYS:
        if key == "beam_rescue_attempted" and skipped:
            out[key] = {"excluded": "one arm has no beam rung (beam_rescue_skipped)"}
            continue
        base, arm = int(reference.get(key, 0)), int(candidate.get(key, 0))
        increase = None if base == 0 else (arm - base) / base
        out[key] = {REFERENCE: base, CANDIDATE: arm, "increase": increase}
    out["refine_fallback"] = {
        REFERENCE: int(reference.get("alignment_retries", 0)),
        CANDIDATE: int(candidate.get("alignment_retries", 0)),
    }
    return out


def stabilize_agreement(root: Path, stem: str) -> dict | None:
    """How often the two arms kept the same subtitle, and tagged it the same way.

    The referee is the reason this is measured at all: it runs in bf16 where
    the card has it and float32 where it does not, and its verdicts reach the
    output only through stabilize -- which keeps or deletes a line. None of
    that moves a single number in the aligned artifacts, so without this the
    arm comparison is blind to the one stage whose differences are visible to
    the user. Reported, never scored: there is no pre-registered threshold for
    it -- but any disagreement sets `review_needed`, because "no threshold" was
    never meant to mean "a candidate that deleted every subtitle passes quietly".
    """

    def _segments(arm: str) -> list[dict] | None:
        path = root / arm / stem / f"{stem}-stable.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        segments = payload.get("segments")
        return segments if isinstance(segments, list) else None

    reference, candidate = _segments(REFERENCE), _segments(CANDIDATE)
    if reference is None or candidate is None:
        return {"available": False, "reason": "一臂没有 -stable.json（--stage 停在更早的档？）"}

    def _tags(segment: dict) -> tuple[str, ...]:
        tags = segment.get("tags")
        return tuple(sorted(str(tag) for tag in tags)) if isinstance(tags, list) else ()

    matched = 0
    tag_mismatches = 0
    tag_differs: dict[str, int] = {}
    for segment in reference:
        match = match_by_overlap(segment, candidate)
        if match is None:
            continue
        matched += 1
        if _tags(segment) != _tags(match):
            tag_mismatches += 1
            for tag in set(_tags(segment)).symmetric_difference(_tags(match)):
                tag_differs[tag] = tag_differs.get(tag, 0) + 1
    only_reference = len(reference) - matched
    only_candidate = len(candidate) - sum(
        1 for segment in candidate if match_by_overlap(segment, reference) is not None
    )
    # The denominator is the larger arm: a line only one arm kept is a
    # disagreement, and must not be able to raise the rate by being absent.
    decided = max(len(reference), len(candidate))
    agreed = matched - tag_mismatches
    return {
        "available": True,
        "segments": {REFERENCE: len(reference), CANDIDATE: len(candidate)},
        "kept_only_by": {REFERENCE: only_reference, CANDIDATE: only_candidate},
        "tag_differences": dict(sorted(tag_differs.items())),
        "agreement": round(agreed / decided, 4) if decided else None,
    }


def compared_stages(shared_vocals: bool) -> tuple[str, ...]:
    """With the shared vocal track separation runs once, so it drops out.

    The referee is always compared when both arms ran it; a run without it
    records no device, and an unknown never counts as acceleration.
    """

    return ("asr", "referee") if shared_vocals else ("asr", "vocal_separation", "referee")


def stages_moved(file: dict, stages: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Of the compared stages, those this file ran on another device, and the unknowns.

    Both arms must say where they ran: a missing or empty record is not
    evidence of acceleration. This check exists to catch records that prove
    nothing, so it must not accept its own missing data as proof.
    """

    moved: list[str] = []
    unknown: list[str] = []
    for stage in stages:
        arm = file["devices"][CANDIDATE].get(stage)
        base = file["devices"][REFERENCE].get(stage)
        if not arm or not base or UNKNOWN_DEVICE in (arm, base):
            unknown.append(stage)
        elif arm != base:
            moved.append(stage)
    return moved, unknown


def triggered(summary: dict, rescue: dict, coverage_diff: float | None) -> list[str]:
    """Which review triggers this file's numbers exceed."""

    hits: list[str] = []
    if summary.get("sim_median") is not None and summary["sim_median"] < TRIGGERS["sim_median_min"]:
        hits.append("segment text similarity median below 1.000")
    if (summary.get("below_098_pct") or 0.0) > TRIGGERS["below_098_pct_max"]:
        hits.append("segments with text similarity <0.98 above 0.5%")
    if (summary.get("over_20ms_pct") or 0.0) > TRIGGERS["over_20ms_pct_max"]:
        hits.append("word starts off by >20 ms above 1%")
    if summary.get("mean_ms") is not None and abs(summary["mean_ms"]) >= TRIGGERS["abs_mean_ms_max"]:
        hits.append("signed word-start mean at or beyond 2 ms")
    ratio = summary.get("pos_neg_ratio")
    low, high = TRIGGERS["pos_neg_ratio_range"]
    if ratio is not None and not (low <= ratio <= high):
        hits.append("positive/negative word-start ratio outside 0.8-1.25")
    if rescue["dropped_groups"][CANDIDATE] > rescue["dropped_groups"][REFERENCE]:
        hits.append("the candidate dropped groups the reference kept")
    for key in RESCUE_KEYS:
        increase = rescue.get(key, {}).get("increase")
        if increase is not None and increase > TRIGGERS["rescue_increase_max"]:
            hits.append(f"{key} up by more than 20%")
    if coverage_diff is not None and abs(coverage_diff) > TRIGGERS["coverage_diff_max"]:
        hits.append("speech coverage differs by more than 1%")
    return hits


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    files = []
    buckets: dict[str, dict[str, list[float]]] = {}
    for source in run["sources"]:
        stem = source["stem"]
        reference, candidate = _aligned(root, REFERENCE, stem), _aligned(root, CANDIDATE, stem)
        scored = score_segments(
            str(root / REFERENCE / stem / f"{stem}-aligned.json"),
            str(root / CANDIDATE / stem / f"{stem}-aligned.json"),
        )
        summary = summarise(scored["sims"], scored["deltas"])
        summary["unmatched"] = scored["unmatched"]
        rescue = rescue_comparison(_recovery(reference), _recovery(candidate))
        ref_cov, cand_cov = _coverage_ratio(reference), _coverage_ratio(candidate)
        coverage_diff = None if ref_cov is None or cand_cov is None else round(cand_cov - ref_cov, 4)
        hits = triggered(summary, rescue, coverage_diff)
        stabilize = stabilize_agreement(root, stem)
        # The referee's differences reach the output only here, and a record
        # that says "nothing to review" while every subtitle changed is the
        # failure this whole file exists to prevent. No threshold: any
        # disagreement is an entry for the written review, which is what
        # "reported, never scored" was always supposed to mean.
        if stabilize.get("available") and (stabilize.get("agreement") or 0.0) < 1.0:
            hits.append("the two arms kept or tagged different subtitles (stabilize)")
        devices = {
            REFERENCE: _devices(root, REFERENCE, stem, reference),
            CANDIDATE: _devices(root, CANDIDATE, stem, candidate),
        }
        moved, unknown = stages_moved(
            {"devices": devices}, compared_stages(bool(run.get("shared_vocals", True)))
        )
        files.append(
            {
                "stem": stem,
                "duration_sec": source.get("duration_sec"),
                "devices": devices,
                "accelerated_stages": moved,
                "stages_without_device_record": unknown,
                "stabilize": stabilize,
                "segments": summary,
                "rescue": rescue,
                "coverage_diff": coverage_diff,
                "review_triggers": hits,
            }
        )
        duration = source.get("duration_sec") or 0.0
        for low, high, label in DURATION_BUCKETS:
            if duration >= low and (high is None or duration < high):
                bucket = buckets.setdefault(label, {"sims": [], "deltas": []})
                bucket["sims"].extend(scored["sims"])
                bucket["deltas"].extend(scored["deltas"])
        agreement = stabilize.get("agreement") if stabilize.get("available") else None
        print(f"{stem:28} below0.98={summary.get('below_098_pct')}% over20ms={summary.get('over_20ms_pct')}% "
              f"mean={summary.get('mean_ms')}ms stabilize={agreement} triggers={len(hits)}")
    signs = {
        (file["segments"].get("mean_ms") or 0.0) > 0
        for file in files
        if file["segments"].get("mean_ms") is not None
    }
    beyond = all(
        file["segments"].get("mean_ms") is not None
        and abs(file["segments"]["mean_ms"]) >= TRIGGERS["abs_mean_ms_max"]
        for file in files
    )
    stages = compared_stages(bool(run.get("shared_vocals", True)))
    # Every file must have used the candidate path: one that quietly fell back
    # dilutes the numbers of the ones that did not, and it is the long file --
    # the one the triggers were written for -- that is likeliest to fall back.
    valid = bool(files) and all(file["accelerated_stages"] for file in files)
    record = {
        "environment": run["environment"],
        "arms": run["arms"],
        "compared_stages": list(stages),
        "accelerated_stages": sorted(
            {stage for file in files for stage in file["accelerated_stages"]}
        ),
        "valid_record": valid,
        "files": files,
        "by_duration": {label: summarise(b["sims"], b["deltas"]) for label, b in buckets.items()},
        "question_1_same_signed_bias_beyond_trigger": bool(files) and len(signs) == 1 and beyond,
        "review_needed": any(file["review_triggers"] for file in files),
        "note": (
            "Exceeding a trigger is not a failure. Where review_needed is true, answer the three "
            "questions of README_DEV.md (development principles) in the record; question 2 compares "
            "proportions across by_duration, question 3 needs a mechanism, not a number. "
            "files[].stabilize has no pass mark, but any disagreement sets review_needed: it is "
            "the one place the referee's differences are visible at all."
        ),
    }
    (root / "acceptance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {root / 'acceptance.json'} (review needed: {record['review_needed']})")
    if not valid:
        for file in files:
            if file["accelerated_stages"]:
                continue
            where = {
                stage: file["devices"][CANDIDATE].get(stage, UNKNOWN_DEVICE)
                for stage in stages
            }
            print(
                f"warning: {file['stem']} 的被比较阶段没有一个换了设备（候选臂 {where}），"
                "候选路径没有被证明用上了。"
                "常见原因：装的 CTranslate2 不是本平台的那个构建，识别已回退到 CPU。",
                file=sys.stderr,
            )
        unknown = sorted(
            {stage for file in files for stage in file["stages_without_device_record"]}
        )
        if unknown:
            print(
                f"warning: 产物里没记下这些阶段跑在哪（{'、'.join(unknown)}），"
                "缺数据不算换了设备。",
                file=sys.stderr,
            )
        print("这不能作为验收记录（valid_record=false）。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
