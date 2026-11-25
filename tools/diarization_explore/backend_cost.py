"""W2: price the two backends with the ruler the owner actually set, on 姐姐咚咚 #1-#15.

Three rounds compared DiariZen and Sortformer on `fixed` and B-cubed F1 and concluded DiariZen
attributes better. Neither of those is what plan 1.3 asks for -- fewest human edits, and few
track-level decisions per video -- so "which backend is cheaper to fix up" was never measured
(audit 2.3). This reads both timelines that are already on disk and prices them with W1's rulers.

    python tools/diarization_explore/backend_cost.py                      # W2: 姐姐咚咚 #1-#15
    python tools/diarization_explore/backend_cost.py --corpus hina        # W2b: the dz3 19-file holdout

Offline in both corpora: Sortformer is the PRODUCTION timeline, DiariZen is the cached run.
No inference, no GPU.

⚠ Why hina matters separately (audit 2.3): on 姐姐咚咚, a three-person show, DiariZen emits four or
five tracks, so "naming cost does not rise" there says nothing about the case the audit actually
raised -- 11 to 23 tracks on hina's five-to-eight-person material. ⚠ Read `leading` there, not
`tracks`: hina carries game-clip characters the reference never labels, and a track that never
leads a reference-covered row costs no naming decision. If `leading` is still well above the
headcount on hina, THAT is over-splitting.

⚠ The merge term of the edit cost is *almost* a segment property and not quite: `strong_cost`
skips a row the arm left unlabelled, so an arm that labels more rows also books more merges. It is
reported for both arms rather than assumed equal -- an assertion that they match is what caught
this. Read the label component for the part that is really about labelling.

⚠ Sortformer here is the PRODUCTION timeline (`out/onetonton-NN/onetonton-NN-diar.json`), which is
what a user gets, not the lab's fresh `base` arm; the two differ slightly because they are
separate runs over different decodes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accuracy_lab as lab  # noqa: E402
import cost_metrics as cm  # noqa: E402
import relabel_eval as ev  # noqa: E402
from accuracy_identity import identity_score  # noqa: E402
from accuracy_report import bootstrap_mean_ci  # noqa: E402
from accuracy_v2 import OUT as V2_OUT  # noqa: E402
from eval_contract import announce  # noqa: E402

FRAME = lab.FRAME


def timeline_probs(tracks, frames):
    """(frames, tracks) 0/1 activity from a `-diar.json` track list.

    Columns are the sorted track ids renumbered 0..K-1, not the raw ids (production Nemotron-3 + vbx ids often
    start at 1). Map `cost_metrics.owner_of_tracks` results back to raw ids (`llm_retrack.owners()`) before
    matching them against raw tracks, or stay on column numbers throughout.
    """
    ids = sorted({int(t["track"]) for t in tracks})
    index = {track: i for i, track in enumerate(ids)}
    probs = np.zeros((frames, max(1, len(ids))), dtype=np.float32)
    for t in tracks:
        lo = int(float(t["start"]) / FRAME)
        hi = min(int(np.ceil(float(t["end"]) / FRAME)), frames)
        if hi > lo:
            probs[lo:hi, index[int(t["track"])]] = 1.0
    return probs


def refined_starts(n):
    """(start, person) for every refined line -- what decides `strong evidence` in the edit cost."""
    import refined_ass as ra

    return sorted((a, w) for a, _, w, t in ra.events(ra.episodes()[n - 1]) if w != ra.NOTE and t)


def hina_episode(ident):
    """(data tuple, DiariZen npz, strong-evidence starts) for one refined-2609 file, or None."""
    import refined_multi as rm

    sources = rm.sources()
    if ident not in sources:
        return None
    data = rm.episode(ident, sources[ident])
    cached = lab.WT / "tmp" / "diarization-dz3" / ident / "w16_default.npz"
    if data is None or not cached.exists():
        return None
    # rm.episode returns one element more than relabel_eval.episode does (the refined lines).
    stable, timeline, grid, people, owner, lines = data
    starts = sorted((a, w) for a, _, w in lines)
    return (stable, timeline, grid, people, owner), cached, starts


def cost_terms_of(probs, stable, grid, people, starts):
    """The same edit cost, split by which priced action it came from.

    W5's "are the extra second labels worth it" is answered here and not by counting overlap
    minutes: a spurious second label is already priced (a delete, 1), so the two backends' `delete`
    columns ARE the comparison -- on the owner's ruler, on the same segments.
    """
    import cost_metrics as cm
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    tracks = segments_from_activity(probs.tolist())
    labelled = [
        {**row, "speaker": speaker.to_dict()} for row, speaker in zip(stable, assign_all(stable, tracks))
    ]
    return cm.cost_terms(labelled, grid, people, cm.owner_of_tracks(probs, grid, people), starts)


def episode_row(key, corpus="onetonton"):
    if corpus == "hina":
        loaded = hina_episode(key)
        if loaded is None:
            return None
        (stable, timeline, grid, people, _), cached, starts = loaded
    else:
        data = ev.episode(key)
        cached = V2_OUT / f"onetonton-{key:02d}" / "diarizen.npz"
        if data is None or not cached.exists():
            return None
        stable, timeline, grid, people, _ = data
        starts = refined_starts(key)
    arms = {
        "sortformer": timeline_probs(timeline["tracks"], len(grid)),
        "diarizen": np.load(cached)["probs"],
    }
    row = {"episode": key, "people": len(people)}
    for name, probs in arms.items():
        probs = probs[: len(grid)]
        metrics = lab.score(probs, stable, grid, people)
        row[name] = {
            "fixed": metrics["fixed"],
            "crossfit": metrics["crossfit"],
            "f1": identity_score(probs, stable, grid)["f1"],
            "naming": cm.naming_cost(probs, stable, grid),
            # ⚠ The >=10-row reading, produced BY the backend comparison. The claim
            # "the naming burden is the same on <=4 speakers" was being supported with a
            # 15/15 from findings 5.3, which is the PORT against the RESEARCH path --
            # a different pair entirely. Nobody caught it because this tool, the one that
            # compares the two backends, never produced the quantity being cited.
            "leading_counts": cm.leading_counts(probs, stable, grid),
            "cost": cm.edit_cost(probs, stable, grid, people, starts),
            "terms": cost_terms_of(probs, stable, grid, people, starts),
        }
    return row


def term_report(rows):
    """The edit cost split by priced action -- the answer W5 could not give.

    W5 asked whether DiariZen's extra second labels earn their keep and answered with overlap
    minutes, which the reference axis cannot support: it under-annotates overlap, and that bias
    falls on the more sensitive detector. The owner's ruler already prices a spurious second label
    as a delete, so the comparison is simply the `delete` row -- same segments, same pricing.
    """
    from accuracy_report import bootstrap_mean_ci

    print()
    print(f"按 owner 定价拆项，{len(rows)} 份（删 1 / 换轨 2 / 强证据漏标 2 / 合并 3）")
    print()
    print(f"{'项':<10}{'Sortformer':>12}{'DiariZen':>11}{'配对差':>10}{'95% 区间':>20}{'DiariZen 更好':>14}")
    for term in ("move", "delete", "add", "duplicate", "merge", "total"):
        a = np.array([float(r["sortformer"]["terms"][term]) for r in rows])
        b = np.array([float(r["diarizen"]["terms"][term]) for r in rows])
        diff = b - a
        interval = bootstrap_mean_ci(list(diff))
        band = f"[{interval[0]:+.1f}, {interval[1]:+.1f}]" if interval else "(n < 5)"
        print(f"{term:<10}{a.mean():>12.1f}{b.mean():>11.1f}{diff.mean():>+10.1f}{band:>20}"
              f"{f'{int((diff < 0).sum())}/{len(rows)}':>14}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", choices=("onetonton", "hina"), default="onetonton")
    p.add_argument("--episodes", nargs="*", default=None)
    p.add_argument("--terms", action="store_true",
                   help="print the edit cost split by priced action instead of the per-episode table")
    p.add_argument("--write", action="store_true", help="save the compact result beside the probe output")
    args = p.parse_args()
    if args.corpus == "onetonton":
        from accuracy_cross import configure

        configure()
        keys = [int(e) for e in (args.episodes or range(1, 16))]
    else:
        import json as _json
        frozen = _json.loads((HERE / "results" / "diarizen-v3-20260917.json").read_text(encoding="utf-8"))
        keys = list(args.episodes or frozen["groups"]["heldout_all"])

    # ⚠ Printed with the numbers, not left to whoever reads them later: an edit-cost
    # comparison is fine on role labels, an identity conclusion is not, and the terminal
    # output is what gets pasted into a document.
    announce(args.corpus)
    rows = [r for r in (episode_row(k, args.corpus) for k in keys) if r]
    if args.terms:
        term_report(rows)
        return
    head = f"{'ep':>9}  {'arm':<11}{'ppl':>4}{'fixed':>8}{'B3 F1':>8}{'lead':>6}{'>=30s':>7}{'95%':>6}{'edit':>8}{'labels':>8}"
    print(head)
    for r in rows:
        for arm in ("sortformer", "diarizen"):
            a = r[arm]
            print(f"{str(r['episode']):>9}  {arm:<11}{r['people']:>4}{a['fixed']*100:>8.2f}{a['f1']*100:>8.2f}"
                  f"{a['naming']['leading']:>6}{a['naming']['over_floor']:>7}{a['naming']['covering']:>6}"
                  f"{a['cost']['total']:>8.0f}{a['cost']['labels']:>8.0f}")
    print()
    merges = {arm: sum(r[arm]["cost"]["merges"] for r in rows) for arm in ("sortformer", "diarizen")}
    abstained = {arm: sum(r[arm]["cost"]["abstained"] for r in rows)
                 for arm in ("sortformer", "diarizen")}
    print(f"merge term: sortformer {merges['sortformer']}, diarizen {merges['diarizen']} "
          f"over {len(rows)} episodes -- equal, which is what a segment property should be. "
          f"⚠ They differed (4214/4222) while `strong_cost` skipped unlabelled rows: an arm that "
          f"labelled more also booked more merges. Priced now, so the skew is gone.")
    print(f"rows the reference covers that the arm left unlabelled: "
          f"sortformer {abstained['sortformer']}, diarizen {abstained['diarizen']} -- reported "
          f"beside the cost because a cheap total and a silent one are different answers")

    print(f"\npaired, DiariZen minus Sortformer ({len(rows)} episodes):")
    summary = {}
    for label, get, scale, lower_is_better in (
        ("fixed", lambda a: a["fixed"] * 100, "pp", False),
        ("B3 F1", lambda a: a["f1"] * 100, "pp", False),
        ("leading tracks", lambda a: a["naming"]["leading"], "", True),
        ("tracks >= 30 s", lambda a: a["naming"]["over_floor"], "", True),
        ("tracks for 95%", lambda a: a["naming"]["covering"], "", True),
        ("leading >= 3 rows", lambda a: a["leading_counts"][3], "", True),
        ("leading >= 10 rows", lambda a: a["leading_counts"][10], "", True),
        ("edit cost", lambda a: a["cost"]["total"], "", True),
        ("  label part", lambda a: a["cost"]["labels"], "", True),
    ):
        diffs = [get(r["diarizen"]) - get(r["sortformer"]) for r in rows]
        interval = bootstrap_mean_ci(diffs)
        wins = sum((d < 0) if lower_is_better else (d > 0) for d in diffs)
        band = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "(n < 5: no interval)"
        print(f"  {label:<16}{np.mean(diffs):+8.2f}{scale:<3}{band:>22}"
              f"   DiariZen better on {wins}/{len(diffs)}")
        summary[label.strip()] = {"mean": float(np.mean(diffs)), "ci95": interval,
                                  "better": wins, "n": len(diffs)}
    if args.write:
        # ⚠ `results/`, not `tmp/`: the closeout plan quotes these numbers, and this directory's
        # README is the rule that a quoted number has a frozen record beside it. W2 and W2b were
        # written to `tmp/` and so had none (2026-09-20 audit, finding 5.3 -- the same gap that
        # left the max-speakers record showing a superseded ten-file run).
        out = (HERE / "results"
               / f"backend-cost-{'w2b-hina' if args.corpus == 'hina' else 'w2'}-20260920.json")
        out.write_text(json.dumps({"rows": rows, "paired": summary,
                                   "script": lab.digest(__file__)}, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
