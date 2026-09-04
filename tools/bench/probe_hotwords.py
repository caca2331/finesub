"""P9 question 2 -- does Qwen3-ASR actually listen better when given the terms?

`docs/plans/crispasr-followups.md` -> P9 makes this an *experiment*, not a feature,
and puts two questions before any wiring. Question 1 (can the knowledge base
hand us a term subset that is knowable at task start?) is answered in
`bench-baselines.md`. This probes question 2, and only question 2.

**What the population is, precisely.** The LLM correction layer is the only
place knowledge terms currently act ("先让它错，再修"), so a segment whose
*corrected* text contains a term its *ASR* text does not is a place where the
**downstream corrector inserted that term**.

⚠ **That is `landed`, not ground truth.** The project's own evidence contract
(`knowledge-node-plan.md` §5.1) is explicit: `landed` proves "输出与该节点一致"
and **cannot** prove "改得对不对"; only `confirmed`/`refuted` -- a refined SRT,
a user confirmation, an external citation -- establishes that. The knowledge
base carries adjudication at the **node** level (is this term a real entry),
not per segment, and the human-refined subtitles we have are Chinese
translations with no Japanese source text.

**So this probe measures agreement with the corrector, not accuracy.** A rise
in the injected arm means "context makes Qwen produce the term the corrector
chose". That is exactly what question 2 asks -- does the model respond to
context at all -- but it is **not** evidence that the resulting transcript is
right. Establishing that needs per-segment adjudication we do not have.

**Three arms, and the third is the one that makes the other two mean anything:**

* `none`   -- no system message. Exactly what production does today.
* `oracle` -- the correct term is in the system prompt. This is an **upper
  bound**, not a proposal: no retrieval could do better than being told the
  answer. If the model does not improve here, it cannot improve anywhere, and
  P9 is closed without any retrieval work.
* `decoy`  -- a term the corrector did **not** put in this segment is in the
  system prompt. Without this arm a rise in the oracle arm is uninterpretable:
  a model that simply echoes whatever it was given would score a perfect
  "improvement" while getting the audio wrong. The decoy arm measures that
  false-insertion rate directly.
  ⚠ The decoy is only checked against the corrected text, so "not in this
  audio" is an assumption, not a verified fact -- the same limit as above.

Decoding is greedy and unsampled, same as production, so a repeat of the same
arm is the noise baseline -- and it is measured rather than assumed.

    python -m tools.bench.probe_hotwords --cases tmp/bench/p9-cases.json --limit 30
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import unicodedata
from pathlib import Path

import numpy as np

MAX_NEW_TOKENS = 96
#: Padding around the segment so the term is not clipped at a boundary.
PAD_SEC = 0.30


def _normalize(text: str) -> str:
    return "".join(
        ch.casefold() for ch in unicodedata.normalize("NFKC", text) if ch.isalnum()
    )


def _similarity(left: str, right: str) -> float:
    a, b = _normalize(left), _normalize(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for row, ch_a in enumerate(a, start=1):
        current = [row]
        for column, ch_b in enumerate(b, start=1):
            current.append(
                min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (ch_a != ch_b))
            )
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def _load_clip(path: Path, start: float, end: float) -> np.ndarray | None:
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception:
        return None
    rate = info.samplerate
    begin = max(0, int((start - PAD_SEC) * rate))
    stop = min(int(info.frames), int((end + PAD_SEC) * rate))
    if stop - begin < rate // 10:
        return None
    data, _ = sf.read(str(path), start=begin, stop=stop, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != 16000:
        import librosa

        mono = librosa.resample(mono, orig_sr=rate, target_sr=16000, res_type="soxr_hq")
    return mono


def _context_prompt(terms: list[str]) -> str:
    """The model's native context slot is the system message (see the processor)."""

    joined = "、".join(terms)
    return f"以下の固有名詞・用語が音声に含まれる可能性があります: {joined}"


KB = "file:C:/Users/Carl/Documents/Carl/projects/asr-playground/knowledge/knowledge.sqlite?mode=ro"


def _knowledge_subjects() -> tuple[dict[str, str], dict[str, list[str]], dict[str, str]]:
    """term -> subject, subject -> its terms, subject -> its display name.

    This is the retrieval a task could actually perform *if it knew the
    subject* -- e.g. the game named in the video title. It is the realistic
    middle ground between "told the answer" (oracle) and "told nothing".
    """

    import sqlite3

    db = sqlite3.connect(KB, uri=True)
    owner: dict[str, str] = {}
    terms_of: dict[str, list[str]] = {}
    name_of: dict[str, str] = {}
    for (subject,) in db.execute("select local_id from nodes where kind='subject'"):
        row = db.execute(
            "select payload from node_versions where local_id=? and valid_to_rev is null",
            (subject,),
        ).fetchone()
        if not row:
            continue
        name_of[subject] = str(json.loads(row[0]).get("surface") or "")
        seen, stack = set(), [subject]
        while stack:
            current = stack.pop()
            for (child,) in db.execute(
                "select child_id from membership_versions where parent_id=? and valid_to_rev is null",
                (current,),
            ):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        collected: list[str] = []
        for node in seen:
            kind = db.execute("select kind from nodes where local_id=?", (node,)).fetchone()
            if not kind or kind[0] != "term":
                continue
            value = db.execute(
                "select payload from node_versions where local_id=? and valid_to_rev is null",
                (node,),
            ).fetchone()
            if not value:
                continue
            surface = str(json.loads(value[0]).get("surface") or "").strip()
            if surface:
                collected.append(surface)
                owner[surface] = subject
        # Sorted, because the walk above collects through a `set` and set
        # iteration order varies per process. An unsorted list made the injected
        # prompt differ run to run: the control and oracle arms were bit-identical
        # across two runs while this arm's text differed on 15/60 cases and its
        # hit count moved 20 -> 25. The order of a term list is not a free
        # variable -- see `bench-baselines.md` 13.3.
        terms_of[subject] = sorted(collected)
    return owner, terms_of, name_of


def _transcribe(model, processor, clip: np.ndarray, context: str | None) -> str:
    import torch

    messages = []
    if context:
        messages.append({"role": "system", "content": [{"type": "text", "text": context}]})
    messages.append({"role": "user", "content": [{"type": "audio", "audio": clip}]})
    inputs = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device, model.dtype)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    tail = generated[:, inputs["input_ids"].shape[1] :]
    return str(processor.decode(tail, return_format="transcription_only")[0] or "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--min-alignment",
        type=float,
        default=0.5,
        help=(
            "drop cases whose ASR and corrected text are not about the same "
            "content: the correction layer merges and re-splits segments, so a "
            "`position` row can pair unrelated text. Measuring those would be "
            "measuring alignment noise, not the hotword effect."
        ),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from finesub.speech.runtime.device import cuda_usable

    if not cuda_usable():
        print("FAIL: needs a usable GPU")
        return 2

    raw = [c for c in json.loads(args.cases.read_text(encoding="utf-8")) if c.get("vocal")]
    cases = [
        c for c in raw if _similarity(c["asr"], c["corrected"]) >= args.min_alignment
    ]
    print(
        f"cases with a vocal track: {len(raw)}; "
        f"kept after alignment filter (>={args.min_alignment}): {len(cases)}"
    )
    rng = random.Random(args.seed)
    rng.shuffle(cases)
    cases = cases[: args.limit]
    all_terms = sorted({c["term"] for c in cases})
    print(f"cases: {len(cases)}   distinct terms: {len(all_terms)}")

    owner, terms_of, name_of = _knowledge_subjects()
    print(
        f"knowledge subjects: {len(terms_of)}; "
        f"cases whose term has an owning subject: "
        f"{sum(1 for c in cases if c['term'] in owner)}/{len(cases)}"
    )

    from finesub.speech.verification.qwen_referee import QwenReferee

    referee = QwenReferee(device="cuda")
    referee.warm()
    model, processor = referee._model, referee._processor  # noqa: SLF001

    rows = []
    for index, case in enumerate(cases, 1):
        clip = _load_clip(Path(case["vocal"]), case["start"], case["end"])
        if clip is None:
            continue
        # A decoy must be absent from this segment's reference text, or a "false
        # insertion" could be the model correctly hearing a term that really is
        # there. Picking any other case's term did not check that.
        others = [
            t
            for t in all_terms
            if t != case["term"] and t not in case["corrected"] and t not in case["asr"]
        ]
        if not others:
            continue
        decoy = rng.choice(others)
        # Retrieval returns a *set*, never the answer alone. These two arms are
        # what a real feature would look like: the right term buried among
        # plausible ones. The plan's own worry is dilution ("塞进 context 会
        # 稀释信号"), and only a sized bundle can measure it.
        bundle_small = [case["term"], *rng.sample(others, min(7, len(others)))]
        bundle_large = [case["term"], *rng.sample(others, min(39, len(others)))]
        rng.shuffle(bundle_small)
        rng.shuffle(bundle_large)
        # "We knew the subject" -- the retrieval a task could really do if the
        # video title names the game. Two strengths: the subject's whole term
        # list, and the subject's *name alone* (topic context, no terms).
        subject = owner.get(case["term"])
        subject_terms = terms_of.get(subject, []) if subject else []
        subject_name = name_of.get(subject, "") if subject else ""
        outputs = {
            "none": _transcribe(model, processor, clip, None),
            "oracle": _transcribe(model, processor, clip, _context_prompt([case["term"]])),
            "decoy": _transcribe(model, processor, clip, _context_prompt([decoy])),
            "bundle8": _transcribe(model, processor, clip, _context_prompt(bundle_small)),
            "bundle40": _transcribe(model, processor, clip, _context_prompt(bundle_large)),
            "subject_terms": (
                _transcribe(model, processor, clip, _context_prompt(subject_terms))
                if subject_terms
                else ""
            ),
            "subject_name": (
                _transcribe(
                    model,
                    processor,
                    clip,
                    f"この音声は「{subject_name}」に関する配信です。",
                )
                if subject_name
                else ""
            ),
            # Repeat of the control arm: the noise baseline, measured not assumed.
            "none_repeat": _transcribe(model, processor, clip, None),
        }
        rows.append(
            {
                "asset": case["asset"],
                "term": case["term"],
                "decoy": decoy,
                "asr": case["asr"],
                "corrected": case["corrected"],
                "subject": subject_name,
                "subject_term_count": len(subject_terms),
                "terms_bundle8": bundle_small,
                "terms_bundle40": bundle_large,
                **{f"out_{k}": v for k, v in outputs.items()},
            }
        )
        print(f"  [{index}/{len(cases)}] {case['term']}", flush=True)

    referee.close()

    def hit(row: dict, arm: str) -> bool:
        return row["term"] in row[f"out_{arm}"]

    print("\n" + "=" * 74)
    print(f"P9 question 2 -- n={len(rows)}")
    print("=" * 74)
    print(
        "NOTE: 'target term present' measures agreement with the downstream\n"
        "corrector (a `landed` signal), NOT transcription accuracy. See the\n"
        "module docstring."
    )
    identical = sum(1 for r in rows if r["out_none"] == r["out_none_repeat"])
    print(f"noise baseline: control arm reproduced itself {identical}/{len(rows)} times")
    print()
    print(f"{'arm':<14}{'agrees with corrector':>18}{'similarity to corrected':>26}")
    for arm in ("none", "oracle", "subject_terms", "subject_name", "bundle8", "bundle40", "decoy"):
        usable = [r for r in rows if r[f"out_{arm}"]]
        if not usable:
            print(f"{arm:<14}  (no case had this context available)")
            continue
        hits = sum(1 for r in usable if hit(r, arm))
        sims = [_similarity(r[f"out_{arm}"], r["corrected"]) for r in usable]
        print(
            f"{arm:<14}{hits:>6}/{len(usable):<11}"
            f"{statistics.median(sims):>26.3f}"
        )
    sizes = [r["subject_term_count"] for r in rows if r.get("subject_term_count")]
    if sizes:
        print(
            f"\n  subject_terms list size: median {statistics.median(sizes):.0f}"
            f"  min {min(sizes)}  max {max(sizes)}"
        )
    echoed = sum(1 for r in rows if r["decoy"] in r["out_decoy"])
    print()
    print(f"decoy term echoed into the output: {echoed}/{len(rows)}"
          "   <- false insertion; every gain above is only real above this")
    for arm in ("bundle8", "bundle40"):
        spurious = sum(
            1
            for r in rows
            for term in r.get(f"terms_{arm}", [])
            if term != r["term"] and term in r[f"out_{arm}"]
        )
        print(f"  wrong terms from the {arm} list that reached the output: {spurious}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
