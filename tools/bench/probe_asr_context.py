"""Does the shipped `--asr-context` help the ASR referee, and at what price?

`bench-baselines.md` 第十三节 answered the model side ("模型这一侧成立，卡住的是检索")
with hand-built arms. This file covers the shipped code end to end: `--asr-context`
picks terms out of the extra-info block, which on a URL run is the scraped title.

Four modes, and they are a chain -- each reads the previous one's `--json`:

  cases   build the population from artifacts already on disk: a segment whose
          CORRECTED text contains a knowledge term its ASR text does not is a
          place where the downstream corrector inserted that term. Needs no GPU.
  reach   for each case, is that term present in what `build_asr_context`
          produces from the asset's real title? Reported against two ceilings --
          whether the term is in the knowledge base at all (nothing can retrieve
          what is not stored), and whether a title could be fetched at all
          (bilibili answers 412 under rate limiting, so a large share of a real
          corpus has no title to read). Needs no GPU.
  arms    decode each sampled clip four times -- `off`, `terms`, `full`, and a
          repeat of `off` as the measured noise baseline. **Needs the GPU.**
          The clip is the one production would build: the pad defaults to
          `qwen_referee.SEGMENT_PAD_SEC`, not to 13.2's wider probe pad.
  score   the numbers in 13.6/13.7: agreement per arm, similarity to the
          corrected text, and the false-insertion pass -- which of the names we
          injected turned up in the `terms` output but not in `off`, and whether
          the corrector wrote them too. Needs no GPU.

⚠ **`reach` takes titles as an input file, which is not the production path.**
The pipeline fetches the title itself, subject to the extra-info gate; handing
this probe a pre-collected `--titles` file measures the retrieval in isolation
and therefore reports an upper bound on what a real run reaches.
`p9_titles.json` next to this file is that snapshot -- `{asset: title}` as
`media.source.resolve_video_title` returned it on 2026-08-31, empty string where
bilibili answered 412. It is committed so the chain runs offline and always over
the same input; regenerate it by re-fetching those ids through the pipeline.

⚠⚠ The target is the term the CORRECTOR inserted, so every mode here measures
**agreement with the corrector, not accuracy** -- the same evidence limit §13.2
states at length. `landed` proves "输出与该节点一致", never "改得对不对".

    python -m tools.bench.probe_asr_context cases --out ../asr-playground/out \\
        --knowledge ../asr-playground/knowledge --json tmp/bench/p9-cases.json
    python -m tools.bench.probe_asr_context reach --cases tmp/bench/p9-cases.json \\
        --titles tools/bench/p9_titles.json --knowledge ../asr-playground/knowledge \\
        --json tmp/bench/p9-reach.json
    python -m tools.bench.probe_asr_context arms --reach tmp/bench/p9-reach.json \\
        --titles tools/bench/p9_titles.json --knowledge ../asr-playground/knowledge \\
        --json tmp/bench/p9-arms.json
    python -m tools.bench.probe_asr_context score --arms tmp/bench/p9-arms.json
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sqlite3
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from finesub import pipeline
from finesub.llm.knowledge import base

#: 128 cases pass `reach`, and the arms run costs four decodes each. Taking
#: every SECOND hit rather than the first 60 halves each asset's share instead
#: of dropping the tail assets entirely -- the population is grouped by asset,
#: so a plain head() would have measured whichever assets sort first.
ARM_STRIDE = 2
ARM_LIMIT = 60


# --- cases ---------------------------------------------------------------------
def cmd_cases(out_root: pathlib.Path, knowledge_root: str, out_json) -> int:
    """The population, rebuilt from artifacts rather than kept as a blob.

    The correction layer is the only place knowledge terms currently act
    ("先让它错，再修"), so its insertions are by construction the segments a
    hotword feature would have to fix. The annotated CSV is pipe-delimited and
    carries only the corrected side, so it is joined to `-stable.json` by the
    `position` column (1-based, and may name a merged span like "3,4").
    """

    store = pathlib.Path(knowledge_root) / "knowledge.sqlite"
    db = sqlite3.connect(f"file:{store.as_posix()}?mode=ro", uri=True)
    terms = set()
    for (local_id,) in db.execute("select local_id from nodes where kind='term'"):
        row = db.execute(
            "select payload from node_versions "
            "where local_id=? and valid_to_rev is null",
            (local_id,),
        ).fetchone()
        if not row:
            continue
        surface = str(json.loads(row[0]).get("surface") or "").strip()
        if len(surface) >= 2:
            terms.add(surface)
    print(f"terms in knowledge base: {len(terms)}")

    hits, scanned = [], 0
    for csv_path in sorted(out_root.rglob("*-annotated.csv")):
        stem = csv_path.name.removesuffix("-annotated.csv")
        stable = csv_path.with_name(f"{stem}-stable.json")
        vocal = csv_path.with_name(f"{stem}-vocal.ogg")
        if not stable.exists():
            continue
        try:
            segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
        except Exception:
            continue
        scanned += 1
        for line in csv_path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("|")
            if len(parts) < 6 or parts[0] != "sub":
                continue
            try:
                indices = [int(p) - 1 for p in parts[1].split(",")]
            except ValueError:
                continue
            if any(i < 0 or i >= len(segments) for i in indices):
                continue
            asr = "".join(str(segments[i].get("text") or "") for i in indices).strip()
            corrected = parts[4].strip()
            if not asr or not corrected:
                continue
            for term in sorted(terms):
                if term in corrected and term not in asr:
                    hits.append({
                        "asset": stem,
                        "vocal": str(vocal) if vocal.exists() else None,
                        "term": term,
                        "asr": asr,
                        "corrected": corrected,
                        "start": float(segments[indices[0]].get("start", 0.0)),
                        "end": float(segments[indices[-1]].get("end", 0.0)),
                    })
                    break

    print(f"asset pairs scanned: {scanned}")
    print(f"corrections that INSERTED a knowledge term: {len(hits)}")
    print(f"  ... with the vocal track still on disk: "
          f"{sum(1 for h in hits if h['vocal'])}")
    _write(out_json, hits)
    return 0


# --- reach ---------------------------------------------------------------------
def _contexts(titles, knowledge_root: str) -> dict:
    context = {}
    for asset, title in titles.items():
        for level in ("terms", "full"):
            context[(asset, level)] = (
                pipeline.build_asr_context(
                    level, knowledge_root=knowledge_root, text=title
                )[0]
                if title
                else ""
            )
    return context


def cmd_reach(cases, titles, knowledge_root: str, out_json) -> int:
    in_kb = {}
    for term in {case["term"] for case in cases}:
        hits = base.match_terms(knowledge_root, term, max_hits=50)
        in_kb[term] = any(term in hit.line for hit in hits)

    context = _contexts(titles, knowledge_root)
    titled = [case for case in cases if titles.get(case["asset"])]
    print(f"cases {len(cases)};  with a title {len(titled)};  "
          f"without {len(cases) - len(titled)}")
    print(f"  term present in the knowledge base : "
          f"{sum(in_kb[case['term']] for case in cases)} / {len(cases)}"
          f"   <- the ceiling for ANY retrieval")
    for level in ("terms", "full"):
        reached = sum(
            1 for case in titled if case["term"] in context[(case["asset"], level)]
        )
        print(f"  retrieval reaches it ({level:<5})       : "
              f"{reached} / {len(titled)} = "
              f"{100 * reached / max(1, len(titled)):.1f}%")

    print("\nper asset with a title:")
    grouped = collections.defaultdict(list)
    for case in titled:
        grouped[case["asset"]].append(case)
    for asset, group in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
        reached = sum(
            1 for case in group if case["term"] in context[(asset, "terms")]
        )
        print(f"  {asset:<14} n={len(group):>3}  reached {reached:>3}"
              f"   title={titles.get(asset, '')[:44]!r}")

    _write(out_json, [
        dict(
            case,
            titled=bool(titles.get(case["asset"])),
            in_kb=in_kb[case["term"]],
            terms=case["term"] in context.get((case["asset"], "terms"), ""),
            full=case["term"] in context.get((case["asset"], "full"), ""),
        )
        for case in cases
    ])
    return 0


# --- arms ----------------------------------------------------------------------
#: A `position` row can pair unrelated text: the correction layer merges and
#: re-splits segments, so a row may hold a corrected span that is simply not
#: about the same content as the ASR text it joined to. Measuring those would
#: measure alignment noise, not the context effect. 13.2 filtered them out at
#: exactly this point and this chain has to keep doing it, or "口径与 13.2 相同"
#: is false.
ARM_MIN_ALIGNMENT = 0.5


def select_arm_cases(
    reach,
    *,
    stride: int = ARM_STRIDE,
    limit: int = ARM_LIMIT,
    min_alignment: float = ARM_MIN_ALIGNMENT,
):
    """The 13.6 sample, as a rule rather than a remembered list.

    Three conditions, in this order: retrieval reaches it (averaging in the
    cases it cannot see would dilute the effect with samples where every arm is
    the control), the ASR and corrected text are about the same content, and
    the clip is on disk. Then every `stride`-th, up to `limit`.
    """

    usable = [
        row
        for row in reach
        if row.get("terms")
        and row.get("vocal")
        and _similarity(row["asr"], row["corrected"]) >= min_alignment
    ]
    return usable[::stride][:limit]


def cmd_arms(reach, titles, knowledge_root: str, out_json, stride, limit, pad) -> int:
    from finesub.speech.runtime.device import cuda_usable

    if not cuda_usable():
        print("FAIL: the arms need a usable GPU")
        return 2

    cases = select_arm_cases(reach, stride=stride, limit=limit)
    reached = [row for row in reach if row.get("terms")]
    print(f"reach hits {len(reached)};  after the alignment filter "
          f"(>={ARM_MIN_ALIGNMENT}) and clip check, every {stride} -> {len(cases)}")
    print(f"clip pad: {pad}s per side (production `SEGMENT_PAD_SEC`)")

    context = _contexts(titles, knowledge_root)
    from finesub.speech.verification.qwen_referee import QwenReferee

    referee = QwenReferee(device="cuda")
    referee.warm()
    rows = []
    for index, case in enumerate(cases, 1):
        clip = _load_clip(
            pathlib.Path(case["vocal"]), case["start"], case["end"], pad
        )
        if clip is None:
            continue
        outputs = {}
        # `off` twice, first and last: greedy decoding makes a repeat of the
        # same arm the noise baseline, and running it at both ends means a
        # drift in model state would show up as a difference rather than hide.
        for arm in ("off", "terms", "full", "off (repeat)"):
            level = arm.split()[0]
            referee.set_context(context.get((case["asset"], level), ""))
            outputs[arm] = referee.transcribe_batch([clip])[0][0]
        referee.set_context("")
        rows.append({
            **{k: case[k] for k in ("asset", "term", "asr", "corrected")},
            "injected": sorted(injected_names(context[(case["asset"], "terms")])),
            "arms": outputs,
        })
        print(f"  [{index}/{len(cases)}] {case['term']}", flush=True)
    referee.close()

    _write(out_json, rows)
    return cmd_score(rows)


# --- score ---------------------------------------------------------------------
def injected_names(context_text: str) -> set[str]:
    """Every NAME the `terms` projection offered, one per pipe field.

    A `terms` line is a term row with the description column already dropped,
    so each remaining field is a name the recogniser was invited to output.
    Headings (`#`, `##`) are structure, not names.
    """

    names = set()
    for line in context_text.splitlines():
        line = line.strip().lstrip("- ").strip()
        if not line or line.startswith("#"):
            continue
        for field in line.split("|"):
            field = field.strip()
            if len(field) >= 2:
                names.add(field)
    return names


def _similarity(left: str, right: str) -> float:
    from tools.bench.probe_hotwords import _similarity as impl

    return impl(left, right)


def _load_clip(path: pathlib.Path, start: float, end: float, pad: float):
    """The clip the production verifier would hand the referee.

    ⚠ The pad is not a free parameter. `qwen_referee.SEGMENT_PAD_SEC` is 0.1
    and its comment says why -- "wide pads bleed neighboring speech into the
    clip and dilute the evidence". 13.2's probe used 0.30, which was fine for
    ITS question (does the model respond to context at all) but makes any claim
    about the SHIPPED path false. Defaulted from the production constant rather
    than copied, so the two cannot drift apart.
    """

    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception:
        return None
    rate = info.samplerate
    begin = max(0, int((start - pad) * rate))
    stop = min(int(info.frames), int((end + pad) * rate))
    if stop - begin < rate // 10:
        return None
    data, _ = sf.read(str(path), start=begin, stop=stop, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != 16000:
        import librosa

        mono = librosa.resample(mono, orig_sr=rate, target_sr=16000, res_type="soxr_hq")
    return mono


def cmd_score(rows) -> int:
    arms = ("off", "terms", "full", "off (repeat)")
    n = len(rows)
    print("\n" + "=" * 74)
    print(f"P9 shipped-retrieval arms -- n={n}")
    print("=" * 74)
    print("NOTE: 'target term present' is agreement with the CORRECTOR (a "
          "`landed` signal),\nnot transcription accuracy. See the docstring.")
    for arm in arms:
        hits = sum(1 for row in rows if row["term"] in row["arms"][arm])
        same = sum(1 for row in rows if row["arms"][arm] == row["arms"]["off"])
        similar = statistics.mean(
            _similarity(row["arms"][arm], row["corrected"]) for row in rows
        )
        print(f"  {arm:<14} {hits:>3} / {n} = {100 * hits / max(1, n):>5.1f}%"
              f"   identical to off {same:>3}/{n}"
              f"   similarity to corrected {similar:.3f}")

    # --- false insertions.
    # `terms` offers a whole name table (median ~229 names), so the honest
    # question is not "was a planted decoy echoed" but "how many of the names
    # we handed over turned up in the output when the control did not write
    # them". Both arms are already decoded; this is arithmetic.
    offered = [len(row["injected"]) for row in rows]
    inserted, unjustified = [], []
    for row in rows:
        for name in row["injected"]:
            if name in row["arms"]["terms"] and name not in row["arms"]["off"]:
                inserted.append((row, name))
                if name not in row["corrected"]:
                    unjustified.append((row, name))
    print(f"\n  names offered per case (median)      : {statistics.median(offered):.0f}")
    print(f"  offered name in `terms` but not `off`: {len(inserted)}")
    print(f"    ... the corrector writes it too    : "
          f"{len(inserted) - len(unjustified)}")
    print(f"    ... it does not  (false insertion) : {len(unjustified)}")
    print(f"  cases with >=1 false insertion       : "
          f"{len({id(row) for row, _ in unjustified})} / {n}")
    for row, name in unjustified:
        print(f"      target {row['term']:<12} inserted {name}")
        print(f"        off  : {row['arms']['off']}")
        print(f"        terms: {row['arms']['terms']}")
        print(f"        fixed: {row['corrected']}")

    # --- the rest of the diff, priced.
    # `terms` rewrites far more than the term itself; leaving those unpriced
    # would report the win and hide the churn that paid for it.
    changed = [row for row in rows if row["arms"]["terms"] != row["arms"]["off"]]
    closer = further = level = 0
    for row in changed:
        delta = _similarity(row["arms"]["terms"], row["corrected"]) - _similarity(
            row["arms"]["off"], row["corrected"]
        )
        closer += delta > 1e-9
        further += delta < -1e-9
        level += abs(delta) <= 1e-9
    print(f"\n  `terms` changed {len(changed)}/{n} outputs: "
          f"closer {closer} / further {further} / unchanged {level}")
    return 0


def _write(path, payload) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cases", "reach", "arms", "score"))
    parser.add_argument("--out", type=pathlib.Path, help="cases: the run tree to scan")
    parser.add_argument("--cases", type=pathlib.Path)
    parser.add_argument("--reach", type=pathlib.Path)
    parser.add_argument("--arms", type=pathlib.Path)
    parser.add_argument("--titles", type=pathlib.Path)
    parser.add_argument("--knowledge")
    parser.add_argument("--stride", type=int, default=ARM_STRIDE)
    parser.add_argument("--limit", type=int, default=ARM_LIMIT)
    parser.add_argument(
        "--pad",
        type=float,
        default=None,
        help="clip pad per side; defaults to the production SEGMENT_PAD_SEC.",
    )
    parser.add_argument("--json", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    def read(path, what):
        if path is None:
            parser.error(f"{args.mode} needs {what}")
        return json.loads(path.read_text(encoding="utf-8"))

    if args.mode == "cases":
        if args.out is None or not args.knowledge:
            parser.error("cases needs --out and --knowledge")
        return cmd_cases(args.out, args.knowledge, args.json)
    if args.mode == "score":
        return cmd_score(read(args.arms, "--arms"))
    if not args.knowledge:
        parser.error(f"{args.mode} needs --knowledge")
    titles = read(args.titles, "--titles")
    if args.mode == "reach":
        return cmd_reach(read(args.cases, "--cases"), titles, args.knowledge, args.json)
    if args.pad is None:
        from finesub.speech.verification import qwen_referee

        args.pad = qwen_referee.SEGMENT_PAD_SEC
    return cmd_arms(
        read(args.reach, "--reach"),
        titles,
        args.knowledge,
        args.json,
        args.stride,
        args.limit,
        args.pad,
    )


if __name__ == "__main__":
    raise SystemExit(main())
