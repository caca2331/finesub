"""A3: are DiariZen's native 20 ms boundaries a net gain over today's 80 ms grid?

Pre-registered in `docs/archive/diarization-closeout-plan.md` §2d BEFORE this ran (commit
`de36883e`). Owner 2026-09-20: "是净收益就做".

Production today reduces DiariZen's native 20 ms activity to the 80 ms product grid
(`to_frame_grid`, a max) and only then segments it. The finer map is real and already computed --
it is what goes into the evidence sidecar. The question is whether segmenting at 20 ms instead
changes which track wins a subtitle line often enough, and in the right direction, to be worth it.

Both arms come from **one** clustering pass per file on the cached embeddings, so the neural half
is identical and the ONLY difference is the grid the timeline is cut on. The smoothing parameters
are given in seconds (`min_gap`, `min_duration`), so they mean the same thing on either grid.

⚠ The limit that was written down first: the truth axis is itself an 80 ms raster, so 20 ms
precision can only be seen through which track wins a line. A null result may be the truth not
seeing it.

    python tools/diarizen_export/native_boundary_bed.py
    python tools/diarizen_export/native_boundary_bed.py --corpus hina
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))

EXPORT_DIR = REPO / "tmp" / "diarizen-export"
V2 = REPO / "tmp" / "diarization-accuracy-v2"
DZ3 = REPO / "tmp" / "diarization-dz3"

NATIVE = 0.02
GRID = 0.08

#: The pre-registered gate (§2d). Net gain = edit cost significantly lower on the primary corpus
#: AND neither fixed nor B-cubed F1 significantly worse on any corpus.
REGRESSION_FLOOR = -0.5


def episode(corpus: str, key):
    """(stable, grid, people, starts, filtered, embeddings) or None."""
    import backend_cost as bc

    if corpus == "onetonton":
        data = bc.ev.episode(key)
        local = V2 / f"onetonton-{key:02d}" / "diarizen-local.npz"
        embeds = V2 / f"onetonton-{key:02d}" / "embeddings.npy"
        if data is None or not (local.exists() and embeds.exists()):
            return None
        stable, _, grid, people, _ = data
        starts = bc.refined_starts(key)
    else:
        loaded = bc.hina_episode(key)
        local = DZ3 / key / "w16" / "diarizen-local.npz"
        embeds = DZ3 / key / "w16" / "embeddings.npy"
        if loaded is None or not (local.exists() and embeds.exists()):
            return None
        (stable, _, grid, people, _), _, starts = loaded
    return stable, grid, people, starts, np.load(local)["data"], np.load(embeds)


def arms(filtered, embeddings, frames):
    """Both timelines from ONE clustering pass: `(probs, frame_seconds)` per arm."""
    from finesub.speech.diarization import diarizen_assemble as assembler
    from frame_grid import to_frame_grid
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_runner as runner

    smoothed = runner.median_filtered(filtered)
    result = clusterer.cluster(embeddings, smoothed, EXPORT_DIR / "plda")
    native = assembler.assemble(smoothed, result.hard).astype(np.float32)
    grid80 = to_frame_grid(native, target_step=GRID)[:frames]
    # ⚠ The native map is truncated to the SAME wall-clock length as the 80 ms one, so the two
    # arms cover exactly the same audio and neither gains lines at the end of the file.
    native20 = native[: frames * int(round(GRID / NATIVE))]
    return {"grid80": (grid80, GRID), "native20": (native20, NATIVE)}


def measure(corpus: str, keys) -> list[dict]:
    import accuracy_lab as lab
    import cost_metrics as cm
    from accuracy_identity import identity_score

    rows = []
    for key in keys:
        loaded = episode(corpus, key)
        if loaded is None:
            continue
        stable, grid, people, starts, filtered, embeddings = loaded
        row = {"file": str(key)}
        for name, (probs, step) in arms(filtered, embeddings, len(grid)).items():
            cost = cm.edit_cost(probs, stable, grid, people, starts, frame_seconds=step)
            row[name] = {
                "fixed": lab.score(probs, stable, grid, people, frame_seconds=step)["fixed"],
                "f1": identity_score(probs, stable, grid, frame_seconds=step)["f1"],
                "edit": cost["total"], "abstained": cost["abstained"],
                "leading10": cm.leading_counts(probs, stable, grid, frame_seconds=step)[10],
            }
        rows.append(row)
        a, b = row["grid80"], row["native20"]
        print(f"  {row['file']:>14}  edit {a['edit']:6.0f} -> {b['edit']:6.0f}  "
              f"fixed {100 * a['fixed']:5.2f} -> {100 * b['fixed']:5.2f}  "
              f"F1 {100 * a['f1']:5.2f} -> {100 * b['f1']:5.2f}")
    return rows


def verdict(corpus: str, rows: list[dict], primary: bool) -> dict:
    from accuracy_report import bootstrap_mean_ci

    print(f"\n{corpus}: {len(rows)} files, native20 minus grid80")
    out = {}
    for label, key, scale in (("edit cost", "edit", 1), ("fixed", "fixed", 100),
                              ("B3 F1", "f1", 100), ("leading>=10", "leading10", 1),
                              ("abstained", "abstained", 1)):
        deltas = [scale * (r["native20"][key] - r["grid80"][key]) for r in rows]
        interval = bootstrap_mean_ci(deltas)
        band = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "(n < 5)"
        print(f"  {label:<12}{np.mean(deltas):+9.2f} {band:>22}")
        out[label] = {"mean": float(np.mean(deltas)), "ci95": interval}

    edit, fixed, f1 = out["edit cost"], out["fixed"], out["B3 F1"]
    edit_down = bool(edit["ci95"] and edit["ci95"][1] < 0)
    no_regression = all(m["ci95"] and m["ci95"][0] >= REGRESSION_FLOOR for m in (fixed, f1))
    out["edit_significantly_lower"] = edit_down
    out["no_significant_regression"] = no_regression
    print(f"  gate: edit cost significantly lower? {edit_down}"
          + ("" if primary else "  (secondary corpus: only the regression half applies)")
          + f".  fixed/F1 lower bounds >= {REGRESSION_FLOOR}? {no_regression}.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO / "tools" / "diarization_explore" / "results"
                                             / "a3-native-boundary-20260920.json"))
    args = parser.parse_args()

    import backend_cost as bc  # noqa: F401 -- sets up the onetonton loaders it shares

    print("onetonton (primary: truth names people, line-level)")
    onetonton = measure("onetonton", list(range(1, 16)))
    print("\nhina (secondary)")
    frozen = json.loads((REPO / "tools" / "diarization_explore" / "results"
                         / "diarizen-v3-20260917.json").read_text(encoding="utf-8"))
    hina = measure("hina", list(frozen["groups"]["heldout_all"]))

    primary = verdict("onetonton", onetonton, primary=True)
    secondary = verdict("hina", hina, primary=False)
    net = (primary["edit_significantly_lower"] and primary["no_significant_regression"]
           and secondary["no_significant_regression"])
    mixed = primary["edit_significantly_lower"] and not net
    print("\nPRE-REGISTERED VERDICT: "
          + ("NET GAIN -- do it" if net else
             "MIXED -- edit cost down but something regressed; goes to the owner" if mixed else
             "NO -- not a net gain"))

    Path(args.out).write_text(json.dumps(
        {"onetonton": onetonton, "hina": hina, "onetonton_verdict": primary,
         "hina_verdict": secondary, "net_gain": net, "mixed": mixed}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
