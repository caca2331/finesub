"""Q11's two arms on the same ruler: a hard cap, versus merge suggestions that change nothing.

Over-splitting is the known failure -- three people become four or five tracks -- and the pipeline
knows the headcount elsewhere. Two ways to use that:

* **arm 1, cap**: cut the same dendrogram at a count instead of a distance;
* **arm 2, suggest**: leave the partition alone and attach "this small track looks like s1".

⚠ They are not symmetric in risk. A cap is baked into `-diar.json`, which is a cache that is never
recomputed on demand, so undoing it means deleting the timeline; a suggestion is an attribute the
role layer is already allowed to attach. And a cap that is too tight is the expensive mistake:
over-splitting costs one merge, under-splitting costs a per-line relocation.

⚠ Runs on the FROZEN embeddings, so it is CPU-only and re-runs the clustering alone -- the neural
half is not repeated and cannot drift between arms.

    python tools/diarizen_export/max_speakers_bed.py --episodes 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: ⚠ hina, not onetonton: only the dz3 cache kept `embeddings.npy`, and this bed has to
#: rerun the clustering alone. hina is also where over-splitting is worst (5-23 tracks
#: for a handful of people), which is what a cap is for.
CACHE = REPO / "tmp" / "diarization-dz3"
PLDA = REPO / "tmp" / "diarization-models" / "diarizen-v2" / "plda"


def score_partition(probs, stable, grid, people, starts):
    """`(edit cost, tracks with >=10 leading lines)` for one timeline."""
    import cost_metrics as cm

    probs = probs[: len(grid)]
    return (cm.edit_cost(probs, stable, grid, people, starts)["total"],
            cm.leading_counts(probs, stable, grid)[10])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+",
                        default=[f"hina-{n:02d}" for n in range(10, 35)])
    parser.add_argument("--margin", type=int, default=1,
                        help="how much headroom over the true headcount the cap gets")
    parser.add_argument("--out", default=str(REPO / "tmp" / "diarizen-bed" / "max-speakers.json"))
    args = parser.parse_args()

    import backend_cost as bc

    from finesub.speech.diarization import diarizen_assemble as assembler
    from frame_grid import to_frame_grid
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_runner as runner

    rows = []
    for ident in args.ids:
        local_path = CACHE / ident / "w16" / "diarizen-local.npz"
        embed_path = CACHE / ident / "w16" / "embeddings.npy"
        loaded = bc.hina_episode(ident)
        if loaded is None or not (local_path.exists() and embed_path.exists()):
            print(f"{ident}: cache incomplete, skipped")
            continue
        (stable, _, grid, people, _), _, starts = loaded
        filtered = runner.median_filtered(np.load(local_path)["data"])
        embeddings = np.load(embed_path)

        base = clusterer.cluster(embeddings, filtered, PLDA)
        cap = len(people) + args.margin
        capped = clusterer.cluster(embeddings, filtered, PLDA, max_speakers=cap)

        def timeline(result):
            native = assembler.assemble(filtered, result.hard).astype(np.float32)
            return to_frame_grid(native, target_step=0.08)

        base_native = assembler.assemble(filtered, base.hard).astype(np.float32)
        base_cost, base_tracks = score_partition(
            to_frame_grid(base_native, target_step=0.08),
            stable, grid, people, starts)
        cap_cost, cap_tracks = score_partition(timeline(capped), stable, grid, people, starts)
        # ⚠ Voiced frames per track, so "established" means the tracks that actually speak. Track
        # ids come from a renumbering and say nothing about size; ordering by them (which this
        # bed did until 2026-09-20) let a junk track serve as a merge target.
        sizes = (base_native > 0).sum(axis=0)
        hints = clusterer.merge_suggestions(base.centroids, base.centroid_of_track, sizes,
                                            keep=len(people))

        rows.append({"ident": ident, "people": len(people), "cap": cap,
                     "base_cost": base_cost, "cap_cost": cap_cost,
                     "base_tracks": base_tracks, "cap_tracks": cap_tracks,
                     "base_clusters": int(base.centroids.shape[0]),
                     "cap_clusters": int(capped.centroids.shape[0]),
                     "suggestions": hints})
        print(f"{ident} ({len(people)} people, cap {cap}): "
              f"cost {base_cost:8.1f} -> {cap_cost:8.1f} ({cap_cost - base_cost:+7.1f})   "
              f"tracks>=10 lines {base_tracks} -> {cap_tracks}   "
              f"suggestions {len(hints)}", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    if not rows:
        return
    delta = np.array([r["cap_cost"] - r["base_cost"] for r in rows], dtype=float)
    print(f"\narm 1, hard cap (n={len(rows)}): edit cost median {np.median(delta):+.1f}  "
          f"range [{delta.min():+.1f}, {delta.max():+.1f}]  "
          f"signs +{int((delta > 0).sum())} / -{int((delta < 0).sum())} / 0 {int((delta == 0).sum())}")
    print("  ⚠ POSITIVE is worse: the cap cost more hand fixing than leaving it alone.")
    counts = Counter(len(r["suggestions"]) for r in rows)
    print(f"arm 2, suggestions: {sum(counts.elements())} across {len(rows)} episodes, "
          f"per-episode {dict(sorted(counts.items()))}")
    print("  ⚠ Arm 2 changes no output at all; it is scored by whether its suggestions are right,")
    print("    which needs the truth mapping in roles-truth.json -- not by edit cost.")


if __name__ == "__main__":
    main()
