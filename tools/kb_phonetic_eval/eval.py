"""Offline check: does phonetic matching find knowledge-base terms that exact
matching misses in raw ASR text, and at what false-positive cost?

No LLM calls. Reads a COPY of the knowledge store (a copy so that nothing here can touch the shared
one) and the run outputs under the main checkout's ``out/``. Gold = rows where the correction pass
wrote a KB term (surface or alias) that the raw ASR rows did not contain. That gold leans towards the
knowledge base: those corrections ran with the same base, which also learned from some of them. A
"false positive" means only that the correction did not write the term. The unit differs from the
experiment switch (``finesub.llm.kb_phonetic_hints``): rows here are annotated rows (several raw
segments joined), tentative items count, and entry names are matched too -- so read its recall and
false positives as a guide to the switch, not as the switch's numbers.

    python tools/kb_phonetic_eval/eval.py --out-root <main>/out --kb <copy.sqlite> --report <dir> \
        [--exclude <clip id>]... [--write-noisy <file>]

``--write-noisy`` calibrates on every source left after ``--exclude``: exclude the clips an A/B will
score, or the calibration has seen the test. The file records what was excluded; ``run_ab.py`` checks it.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

from finesub.llm.kb_phonetic_hints import (  # noqa: E402
    _READINGS,
    loose,
    mora_count,
    reading,
    reading_complete,
    romaji,
)
from finesub.llm.knowledge.node.matching import ExactIndex, scan_normalize  # noqa: E402
from finesub.llm.knowledge.node.store import KnowledgeStore  # noqa: E402


# --- knowledge --------------------------------------------------------------


@dataclass
class Node:
    node_id: str
    subject_id: str
    entry: str
    label: str
    exact_keys: set[str] = field(default_factory=set)  # surface + alias, normalized
    misheard_keys: set[str] = field(default_factory=set)
    misheard_raw: set[str] = field(default_factory=set)
    raw_names: set[str] = field(default_factory=set)  # surface + alias as stored


def load_nodes(kb_path: str) -> tuple[dict[str, Node], dict[str, str]]:
    store = KnowledgeStore(kb_path)
    index = ExactIndex.build(store, include_tentative=True)
    subject_name = {s.local_id: s.payload.get("surface", "") for s in store.subjects()}
    nodes: dict[str, Node] = {}
    for text, owners in index._keys.items():  # noqa: SLF001 -- offline probe
        for key in owners:
            node = nodes.setdefault(
                key.node_id,
                Node(key.node_id, key.subject_id, subject_name.get(key.subject_id, "?"), ""),
            )
            if key.kind == "misheard":
                node.misheard_keys.add(text)
                node.misheard_raw.add(key.raw)
            else:
                node.exact_keys.add(text)
                node.raw_names.add(key.raw)
                if key.kind == "surface":
                    node.label = key.raw
    for node in nodes.values():
        node.label = node.label or sorted(node.raw_names or node.misheard_raw)[0]
    # entry name -> subject id, through every exact key of the subject node itself
    by_name = {}
    for node in nodes.values():
        if node.node_id == node.subject_id:
            for key in node.exact_keys:
                by_name[key] = node.subject_id
    return nodes, by_name


# --- runs -------------------------------------------------------------------


@dataclass
class Row:
    group: str
    run: str
    positions: tuple[int, ...]
    raw: str
    corrected: str


def find_runs(out_root: str) -> list[tuple[str, str, str]]:
    found = []
    for annotated in glob.glob(os.path.join(out_root, "**", "*-annotated.csv"), recursive=True):
        stem = os.path.basename(annotated)[: -len("-annotated.csv")]
        stable = os.path.join(os.path.dirname(annotated), f"{stem}-stable.json")
        if os.path.exists(stable):
            found.append((annotated, stable, stem))
    return found


def research_entries(run_dir: str, stem: str) -> set[str]:
    names: set[str] = set()
    for path in glob.glob(os.path.join(run_dir, "**", f"{stem}-research-context.json"), recursive=True):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for key in ("injected_entries", "keep_entries"):
            for item in data.get(key) or []:
                names.add(item if isinstance(item, str) else str(item.get("key", "")))
    return {n for n in names if n}


def load_rows(
    out_root: str, exclude: tuple[str, ...] = ()
) -> tuple[list[Row], dict[str, set[str]], dict[str, int]]:
    """One run per distinct source (identical stable text = same source). A source whose
    stable text equals one under ``exclude`` (or whose path names it) is left out."""

    excluded_digests = set()
    for annotated, stable, stem in find_runs(out_root):
        if any(token in annotated for token in exclude):
            segs = json.load(open(stable, encoding="utf-8"))["segments"]
            excluded_digests.add(hashlib.sha1("\n".join(s.get("text", "") for s in segs).encode()).hexdigest()[:12])
    chosen: dict[str, tuple[float, str, str, str]] = {}
    for annotated, stable, stem in find_runs(out_root):
        segs = json.load(open(stable, encoding="utf-8"))["segments"]
        digest = hashlib.sha1("\n".join(s.get("text", "") for s in segs).encode()).hexdigest()[:12]
        if digest in excluded_digests or any(token in annotated for token in exclude):
            continue
        mtime = os.path.getmtime(annotated)
        if digest not in chosen or mtime > chosen[digest][0]:
            chosen[digest] = (mtime, annotated, stable, stem)
    rows: list[Row] = []
    scopes: dict[str, set[str]] = {}
    window_rows: dict[str, int] = {}
    for digest, (_m, annotated, stable, stem) in sorted(chosen.items()):
        segs = json.load(open(stable, encoding="utf-8"))["segments"]
        run = os.path.relpath(os.path.dirname(annotated), out_root)
        scopes[digest] = research_entries(os.path.dirname(annotated), stem)
        plans = glob.glob(os.path.join(os.path.dirname(annotated), "**", "correction-window-plan.json"), recursive=True)
        if plans:
            try:
                windows = json.load(open(plans[0], encoding="utf-8")).get("plan", {}).get("windows") or []
                sizes = [len(w.get("segment_ids") or []) for w in windows]
                if sizes:
                    window_rows[digest] = int(sum(sizes) / len(sizes))
            except Exception:
                pass
        for line in open(annotated, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("|")
            if len(cols) < 5:
                continue
            try:
                positions = tuple(int(p) for p in cols[1].split(","))
            except ValueError:
                continue
            raw = "".join(segs[p - 1].get("text", "") for p in positions if 0 < p <= len(segs))
            rows.append(Row(digest, run, positions, raw, cols[4]))
    return rows, scopes, window_rows


# --- matchers ---------------------------------------------------------------


@dataclass
class KeyForm:
    node_id: str
    text: str  # normalized exact form
    kana: str
    loose: str
    romaji: str
    mora: int


def key_forms(nodes: dict[str, Node], *, with_misheard: bool) -> list[KeyForm]:
    forms = []
    for node in nodes.values():
        keys = set(node.exact_keys) | (node.misheard_keys if with_misheard else set())
        for text in keys:
            kana = reading(text)
            if not kana or not reading_complete(text, kana):
                continue
            forms.append(KeyForm(node.node_id, text, kana, loose(kana), romaji(loose(kana)), mora_count(kana)))
    return forms


def exact_hits(text_norm: str, forms: list[KeyForm]) -> set[str]:
    return {f.node_id for f in forms if f.text in text_norm}


def fuzzy_scores(row_romaji: list[str], forms: list[KeyForm]):
    """Matrix [form][row] of partial_ratio over loose romaji (0..100)."""

    return process.cdist(
        [f.romaji for f in forms], row_romaji, scorer=fuzz.partial_ratio, workers=-1, dtype=np.uint8
    )


#: a 3-letter romaji key is "found" in nearly every row by partial_ratio
MIN_FUZZY_ROMAJI = 6


# --- evaluation -------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--kb", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--exclude", action="append", default=[], help="leave out sources whose path names this (repeatable)")
    ap.add_argument("--write-noisy", help="write the nodes calibrated out of fuzzy matching at threshold 90")
    ap.add_argument("--reader", choices=["pyopenjtalk", "pykakasi"], default="pyopenjtalk",
                    help="which library reads kana; give each reader its own --report dir (readings are cached there)")
    args = ap.parse_args()
    report_dir = Path(args.report)
    report_dir.mkdir(parents=True, exist_ok=True)

    import finesub.llm.kb_phonetic_hints as hints_module

    hints_module.READER = args.reader
    cache = report_dir / "readings.json"
    if cache.exists():
        _READINGS.update(json.loads(cache.read_text(encoding="utf-8")))
    nodes, entry_ids = load_nodes(args.kb)
    rows, scopes, window_rows = load_rows(args.out_root, tuple(args.exclude))
    base = key_forms(nodes, with_misheard=False)
    prod = key_forms(nodes, with_misheard=True)
    print(f"nodes={len(nodes)} keys(surface+alias)={len(base)} keys(+misheard)={len(prod)}")
    print(f"sources={len(scopes)} rows={len(rows)} rows/window(median of runs)="
          f"{sorted(window_rows.values())[len(window_rows)//2] if window_rows else '?'}")

    raw_norm = [scan_normalize(r.raw) for r in rows]
    cor_norm = [scan_normalize(r.corrected) for r in rows]
    raw_kana = [reading(r.raw) for r in rows]
    cache.write_text(json.dumps(_READINGS, ensure_ascii=False), encoding="utf-8")
    raw_loose = [loose(k) for k in raw_kana]
    raw_romaji = [romaji(k) for k in raw_loose]

    # gold: node written by the correction, absent (surface/alias) from raw
    gold_miss: set[tuple[int, str]] = set()
    gold_present: set[tuple[int, str]] = set()
    in_corrected: list[set[str]] = []
    for i in range(len(rows)):
        cor = exact_hits(cor_norm[i], base)
        raw = exact_hits(raw_norm[i], base)
        in_corrected.append(cor)
        for node_id in cor:
            (gold_present if node_id in raw else gold_miss).add((i, node_id))
    print(f"gold: misheard instances={len(gold_miss)} present instances={len(gold_present)}")

    scope_ids: dict[str, set[str]] = {}
    for group, names in scopes.items():
        ids = {entry_ids.get(scan_normalize(n)) for n in names}
        scope_ids[group] = {i for i in ids if i}

    # candidate sets per method: list (row index) -> set(node ids)
    methods: dict[str, list[set[str]]] = {}
    methods["exact (surface+alias)"] = [exact_hits(t, base) for t in raw_norm]
    methods["exact (+misheard, production)"] = [exact_hits(t, prod) for t in raw_norm]
    methods["kana reading exact"] = [
        {f.node_id for f in base if f.mora >= 2 and f.kana in k} for k in raw_kana
    ]
    methods["loose kana exact (>=3 mora)"] = [
        {f.node_id for f in base if f.mora >= 3 and f.loose and f.loose in k} for k in raw_loose
    ]
    scores = fuzzy_scores(raw_romaji, base)
    for min_mora in (3, 4):
        for threshold in (95, 90, 85, 80):
            sets = [set() for _ in rows]
            for fi, form in enumerate(base):
                if form.mora < min_mora or len(form.romaji) < MIN_FUZZY_ROMAJI:
                    continue
                hit_rows = (scores[fi] >= threshold).nonzero()[0]
                for ri in hit_rows:
                    # partial_ratio slides the SHORTER string: a row shorter
                    # than the key would be "found" inside the key
                    if len(raw_romaji[ri]) < len(form.romaji) and fuzz.ratio(form.romaji, raw_romaji[ri]) < threshold:
                        continue
                    sets[ri].add(form.node_id)
            methods[f"fuzzy romaji >={threshold} (>={min_mora} mora)"] = sets

    fuzzy_keys = sum(1 for f in base if f.mora >= 3 and len(f.romaji) >= MIN_FUZZY_ROMAJI)
    print(f"keys eligible for fuzzy (>=3 mora, romaji>={MIN_FUZZY_ROMAJI}): {fuzzy_keys}")
    production = methods["exact (+misheard, production)"]
    for threshold in (95, 90, 85):
        fz_sets = methods[f"fuzzy romaji >={threshold} (>=3 mora)"]
        methods[f"production exact + fuzzy>={threshold}"] = [a | b for a, b in zip(production, fz_sets)]
        # split-half calibration: a node whose fuzzy hits in the OTHER half's
        # rows mostly miss the correction is noisy -- drop it from fuzzy there
        groups = sorted({r.group for r in rows})
        half = {g: idx % 2 for idx, g in enumerate(groups)}
        calibrated = [set(p) for p in production]
        for side in (0, 1):
            fired = collections.Counter()
            wrong = collections.Counter()
            for i, c in enumerate(fz_sets):
                if half[rows[i].group] == side:
                    continue
                for node_id in c - production[i]:
                    fired[node_id] += 1
                    wrong[node_id] += node_id not in in_corrected[i]
            noisy = {n for n in fired if wrong[n] >= 3 and wrong[n] / fired[n] > 0.5}
            for i, c in enumerate(fz_sets):
                if half[rows[i].group] == side:
                    calibrated[i] |= c - noisy
        methods[f"production exact + fuzzy>={threshold}, split-half calibrated"] = calibrated

    if args.write_noisy:
        # the split-half rule above, over ALL the given sources: a node whose fuzzy hits
        # mostly miss the correction is calibrated out
        fired, wrong = collections.Counter(), collections.Counter()
        for i, c in enumerate(methods["fuzzy romaji >=90 (>=3 mora)"]):
            for node_id in c - production[i]:
                fired[node_id] += 1
                wrong[node_id] += node_id not in in_corrected[i]
        noisy = sorted(n for n in fired if wrong[n] >= 3 and wrong[n] / fired[n] > 0.5)
        Path(args.write_noisy).write_text(json.dumps(
            {"noisy_nodes": noisy, "labels": {n: f"{nodes[n].label} ({nodes[n].entry})" for n in noisy},
             "sources": len(scopes), "rows": len(rows), "excluded": list(args.exclude)}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {len(noisy)} noisy nodes to {args.write_noisy}")

    def evaluate(cands: list[set[str]], *, scoped: bool) -> dict:
        def allowed(i: int, node_id: str) -> bool:
            if not scoped:
                return True
            return nodes[node_id].subject_id in scope_ids.get(rows[i].group, set())

        gm = {(i, n) for i, n in gold_miss if allowed(i, n)}
        gp = {(i, n) for i, n in gold_present if allowed(i, n)}
        hit_miss = sum(1 for i, n in gm if n in cands[i])
        hit_present = sum(1 for i, n in gp if n in cands[i])
        fp = sum(1 for i, c in enumerate(cands) for n in c if allowed(i, n) and n not in in_corrected[i])
        scoped_rows = sum(1 for r in rows if not scoped or scope_ids.get(r.group))
        return {
            "gold_miss": len(gm), "recall_miss": hit_miss, "gold_present": len(gp),
            "recall_present": hit_present, "fp": fp, "rows": scoped_rows,
            "fp_per_100_rows": round(100 * fp / max(scoped_rows, 1), 2),
        }

    results = {}
    for scoped in (False, True):
        tag = "scoped" if scoped else "global"
        results[tag] = {name: evaluate(c, scoped=scoped) for name, c in methods.items()}
        print(f"\n== {tag}")
        for name, r in results[tag].items():
            print(f"{name:34s} miss {r['recall_miss']:4d}/{r['gold_miss']:<4d} present {r['recall_present']:4d}/"
                  f"{r['gold_present']:<4d} FP {r['fp']:5d} ({r['fp_per_100_rows']}/100 rows over {r['rows']})")

    # leave-one-out: each stored misheard spelling vs surface+alias keys only
    loo = []
    form_romaji = [f.romaji for f in base]
    for node in nodes.values():
        for raw_value in node.misheard_raw:
            kana = reading(raw_value)
            if not kana:
                continue
            rom = romaji(loose(kana))
            row_scores = process.extract(rom, form_romaji, scorer=fuzz.ratio, limit=5)
            best = [(base[idx].node_id, score) for _s, score, idx in row_scores]
            target = [s for n, s in best if n == node.node_id]
            loo.append({
                "node": node.label, "entry": node.entry, "misheard": raw_value, "misheard_kana": kana,
                "names": sorted(node.raw_names),
                "exact": any(k in scan_normalize(raw_value) or scan_normalize(raw_value) in k for k in node.exact_keys),
                "loose_equal": any(loose(reading(k)) == loose(kana) for k in node.exact_keys),
                "top1": best[0][0] == node.node_id if best else False,
                "target_score": round(max(target), 1) if target else 0,
                "top1_score": round(best[0][1], 1) if best else 0,
                "top1_label": nodes[best[0][0]].label if best else "",
            })
    n = len(loo)
    print(f"\n== leave-one-out over {n} stored misheard spellings (whole-string ratio vs surface/alias)")
    print(f"exact substring either way: {sum(r['exact'] for r in loo)}  loose-kana equal: {sum(r['loose_equal'] for r in loo)}  "
          f"top-1 correct: {sum(r['top1'] for r in loo)}")
    for cut in (90, 80, 70, 60):
        print(f"  top-1 correct and score>={cut}: {sum(1 for r in loo if r['top1'] and r['top1_score'] >= cut)}")

    examples = {"gold_miss": [], "fp_fuzzy85": []}
    fz = methods["fuzzy romaji >=85 (>=3 mora)"]
    ex = methods["exact (surface+alias)"]
    for i, node_id in sorted(gold_miss):
        examples["gold_miss"].append({
            "run": rows[i].run, "positions": rows[i].positions, "raw": rows[i].raw, "corrected": rows[i].corrected,
            "node": nodes[node_id].label, "entry": nodes[node_id].entry,
            "exact+misheard": node_id in methods["exact (+misheard, production)"][i],
            "fuzzy85": node_id in fz[i], "kana": node_id in methods["kana reading exact"][i],
        })
    for i, c in enumerate(fz):
        for node_id in c - in_corrected[i] - ex[i]:
            examples["fp_fuzzy85"].append({"run": rows[i].run, "raw": rows[i].raw, "node": nodes[node_id].label,
                                           "entry": nodes[node_id].entry})
    fp_by_key = collections.Counter()
    for i, c in enumerate(fz):
        for node_id in c - in_corrected[i]:
            fp_by_key[(nodes[node_id].label, nodes[node_id].entry)] += 1
    print("\n== fuzzy>=85 (>=3 mora) false positives by node, top 15")
    for (label, entry), count in fp_by_key.most_common(15):
        print(f"  {count:5d} {label} ({entry})")
    (report_dir / "results.json").write_text(json.dumps(
        {"results": results, "leave_one_out": loo, "examples": examples}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
