"""Speaker judge acceptance readouts (diar-marks plan): production's judge, text only, on round 9's eleven windows.

Readouts, each a pass of production code, scored round 9's way:

* the first (2026-10-09, plan 「验收结果」): 150-row judge windows with a cast pick each; the window list was
  patched to round 9's windows. Its sidecars (version 1) stay in `tmp/diarization-retrack/accept/` as the baseline;
* the second (item five, plan 「第五件」): one judge window per correction window, one call each. Nothing is patched:
  production's own correction plan (its dry run's `plan.json`) cuts the file, and a sidecar written ahead holds every
  window but the ones covering round 9's window as judged-and-unchanged, so the judge -- reading the sidecar as it
  always does -- judges only those. Output in `tmp/diarization-retrack/accept-aligned/`;
* the same windows on GPT-6 Luna (`--arm luna`, owner 2026-10-09), in `tmp/diarization-retrack/accept-aligned-luna/`,
  scored against the Sol readout (no gate of its own).

Each file's ASR artifacts were copied and production's pipeline re-ran diarization on them (`setup`; the labels come
out as round 9's acoustic layer, checked row by row); production's preparation task ran on the whole file
(`--research-only`) unless production's dry run decides the file goes fast -- `--research-only` itself forces fast
off -- in which case the judge gets the fast run's context and its candidates come from the knowledge base alone.
The later readouts reuse those contexts. The driver is GPT-6.1 Sol via Codex (owner 2026-10-09) unless `--arm`
names another, at production's quality thinking cell.

Scoring is round 9's: the rows labelled with truth in the eleven windows, a row right when its final track maps to one
of its true speakers under the window's one-to-one matching; the main rows are the fresh windows' non-back-channel
ones, with a cluster bootstrap. `score` reports the second readout against the first (the pre-registered gate:
non-inferiority, lower bound > -3) and against the acoustic layer.

    python tools/diarization_explore/window_judge_accept.py setup            # copy artifacts, re-run diarization
    python tools/diarization_explore/window_judge_accept.py prepare [--only hina-05]
    python tools/diarization_explore/window_judge_accept.py run [--only hina-05] [--arm luna]   # an aligned readout
    python tools/diarization_explore/window_judge_accept.py score [--arm luna]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_judge_r9 as r9  # noqa: E402
import window_partition_score as ps  # noqa: E402

r3 = r9.r3
ROOT = r9.lr.OUT / "accept"
ALIGNED = r9.lr.OUT / "accept-aligned"
WT = HERE.parents[1]
TARGET = "local-codex-gpt-6_1-sol"
#: The aligned readout per driver: the target and where its sidecars go. `sol` is the pre-registered one; `luna`
#: re-judges the same windows on GPT-6 Luna (owner 2026-10-09), scored against `sol` -- no gate of its own.
ARMS = {"sol": (TARGET, ALIGNED), "luna": ("local-codex-gpt-6-luna", r9.lr.OUT / "accept-aligned-luna")}
IDS = r9.MAIN / "tmp" / "refined-2609" / "ids.json"
SKIP = ("-diar.json", "-diar-evidence.npz", "-raw.srt")


def folder(ident):
    return ROOT / ident


def stable_path(ident):
    return folder(ident) / f"{ident}-stable.json"


def source(ident):
    return r9.source_audio(ident)


def title(ident):
    """What production is given for a downloaded video: its title."""
    if ident.startswith("hina-"):
        return json.loads(IDS.read_text(encoding="utf-8"))[ident]["title"]
    return source(ident).stem


def _env():
    return {**os.environ, "PYTHONPATH": str(WT / "src")}


def setup():
    for ident, a, b in r9.windows():
        target = folder(ident)
        target.mkdir(parents=True, exist_ok=True)
        for path in r9.run_dir(ident).glob(f"{ident}-*"):
            if path.is_file() and not path.name.endswith(SKIP) and not (target / path.name).exists():
                shutil.copy2(path, target / path.name)
        if not (target / f"{ident}-diar.json").exists():
            subprocess.run([sys.executable, "-m", "finesub.pipeline", str(source(ident)), "-o", str(target / f"{ident}.srt"),
                            "--diarize", "acoustics", "--stage", "raw-srt"], check=True, env=_env(), cwd=WT)
        segments = json.loads(stable_path(ident).read_text(encoding="utf-8"))["segments"]
        m = r9.load(ident)
        differ = sum(1 for i, s in enumerate(segments) if (s.get("speaker") or {}).get("track") != m.leader(i))
        print(f"{ident}: {len(segments)} rows, window {a}-{b}; rows whose track differs from round 9's acoustic layer: {differ}")


def _correction(ident, *extra):
    subprocess.run([sys.executable, "-m", "finesub.llm.correction_translation", str(stable_path(ident)),
                    "--audio", str(source(ident)), "--media", "text", "--speaker-judge", "text", "--llm-model", TARGET,
                    "--extra-info", title(ident), "-o", str(folder(ident) / f"{ident}.srt"), *extra],
                   check=True, env=_env(), cwd=WT)


def fast(ident):
    """Production's own fast decision, from its dry run (`--research-only` would force it off)."""
    plan = folder(ident) / "plan" / "plan.json"
    if not plan.exists():
        _correction(ident, "--prompt-dir", str(plan.parent))
    return bool(json.loads(plan.read_text(encoding="utf-8"))["fast_decision"]["enabled"])


def prepare(only):
    for ident, *_ in r9.windows():
        if (only and ident != only) or context_path(ident):
            continue
        if fast(ident):
            (folder(ident) / "llm").mkdir(parents=True, exist_ok=True)
            print(f"{ident}: production runs it fast -- no preparation task")
            continue
        _correction(ident, "--research-only", "--execute", "--task-artifact-dir", str(folder(ident) / "llm"))
        print(f"{ident}: research context {context_path(ident)}")


def context_path(ident):
    found = sorted((folder(ident) / "llm").glob("*research-context*.json"))
    return found[0] if found else None


def aligned_stable(ident, arm="sol"):
    return ARMS[arm][1] / ident / f"{ident}-stable.json"


def judge_windows(ident):
    """Production's judge windows for this file: one per correction window of its own dry-run plan (fast: the
    whole file), as `split_windows` cuts them."""
    from finesub.llm.stages.speaker_judge.windows import JudgeWindow
    from finesub.speech.diarization.judgments import segment_ids

    segments = json.loads(stable_path(ident).read_text(encoding="utf-8"))["segments"]
    ids = segment_ids(segments)
    fast(ident)  # makes sure production's dry-run plan is there
    plan = json.loads((folder(ident) / "plan" / "plan.json").read_text(encoding="utf-8"))
    position = {key: index for index, key in enumerate(ids)}
    starts = {}
    for window in plan.get("correction_windows") or []:
        first = str(window["source_ids"][len(window.get("overlap_source_ids") or [])])  # the first core row
        if first in position:
            starts.setdefault(position[first], window["chunk_id"])
    cuts = sorted(starts) or [0]
    if cuts[0] != 0:
        starts[0] = starts[cuts[0]]
        cuts = [0, *cuts]
    starts.setdefault(0, "0001")
    return [JudgeWindow(starts[low], starts[low], low, high) for low, high in zip(cuts, [*cuts[1:], len(ids)])]


def run(only, arm="sol"):
    """The second readout: production judges the correction windows covering round 9's window, nothing else."""
    sys.path.insert(0, str(WT / "src"))
    from finesub.llm.agent.agent_session_host import agent_session_scope
    from finesub.llm.client import RoleClient
    from finesub.llm.correction_translation import DEFAULT_KNOWLEDGE_ROOT
    from finesub.llm.research import load_research_context
    from finesub.llm.routing.config import planning_limits_for
    from finesub.llm.routing.model_routes import install_runtime_preferred
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.stages.plan import fast_context_pack
    from finesub.llm.stages.speaker_judge import run as judge
    from finesub.llm.stages.speaker_judge.task import judge_inputs
    from finesub.llm.token_budget import default_token_counter
    from finesub.speech.diarization import judgments as jd

    install_runtime_preferred({"default": ARMS[arm][0]})
    profile = resolve_profile("text", "local", "quality", "serial")
    for ident, a, b in r9.windows():
        if only and ident != only:
            continue
        stable = aligned_stable(ident, arm)
        stable.parent.mkdir(parents=True, exist_ok=True)
        if not stable.exists():
            shutil.copy2(stable_path(ident), stable)
        side = jd.sidecar_path(stable)
        segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
        ids = jd.segment_ids(segments)
        tracks = jd.profile_tracks([jd.speaker_of(s) for s in segments])
        windows = judge_windows(ident)
        if not side.exists():
            entries = []
            for window in windows:
                entry = {"id": window.id, "chunk": window.chunk, "first": ids[window.low], "last": ids[window.high - 1]}
                if window.high <= a or window.low >= b:  # not covering round 9's window: held, unchanged
                    entry.update(status="success", prompt_version=judge.PROMPT_VERSION, results=[], media="text",
                                 key=jd.window_key(segments[window.low:window.high], ids[window.low:window.high], tracks))
                else:
                    entry["status"] = "pending"
                entries.append(entry)
            side.write_text(json.dumps({"version": jd.SIDECAR_VERSION, "route": "separate",
                                        "rows_hash": jd.rows_hash(segments), "windows": entries},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
        if jd.covers(stable):
            print(f"{ident}: judged already")
            continue
        found = context_path(ident)
        if found is None and not (folder(ident) / "llm").exists():
            raise SystemExit(f"{ident}: run `prepare` first")
        context = load_research_context(found) if found else fast_context_pack(title(ident))  # what a fast run has
        inputs = judge_inputs(SimpleNamespace(extra_info=title(ident)), context, knowledge_root=DEFAULT_KNOWLEDGE_ROOT,
                              fallback_title=ident)
        with agent_session_scope():
            report = judge.run_speaker_judge(
                stable_json=stable, correction_windows=[], profile=profile, client=RoleClient(), inputs=inputs,
                media="text", audio=None, task_artifact_dir=stable.parent / "llm", task_id=ident, parallel=1,
                warn=lambda message: print(f"warning: {message}", flush=True),
                limits=planning_limits_for(judge.judge_group("text"), profile.difficulty), counter=default_token_counter())
        covering = [(w.id, w.high - w.low) for w in windows if not (w.high <= a or w.low >= b)]
        print(f"{ident}: {report}; windows judged {covering}; candidates {len(inputs.pool)}", flush=True)


# --- scoring -------------------------------------------------------------------------------------------

def picks(stable_of=stable_path):
    """{(ident, a, row): pick} in round 9's form, from each file's sidecar (either version: a window's results, or
    its parts'): a moved row names its track, a new track its label (scoped to its window or part), an abstention
    `abstain`; a row whose acoustic cell was filled by a mark and that the judge kept is on that track."""
    from finesub.speech.diarization.judgments import segment_ids, sidecar_path
    out, windows = {}, {}
    for ident, a, b in r9.windows():
        stable = stable_of(ident)
        segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
        index = {key: i for i, key in enumerate(segment_ids(segments))}
        payload = json.loads(sidecar_path(stable).read_text(encoding="utf-8"))
        units = [unit for window in payload["windows"] for unit in (window.get("parts") or [window])]
        windows[ident] = units
        for unit in units:
            for result in unit.get("results") or []:
                key = (ident, a, index[str(result["row"])])
                if "track" in result:
                    out[key] = f"s{result['track']}"
                elif "label" in result:
                    out[key] = f"@{unit.get('id')}/{result['label']}"
                elif "carried" in result:
                    out[key] = f"@{result['carried']['window']}/{result['carried']['label']}"
                elif result.get("abstain"):
                    out[key] = "abstain"
        for i in range(a, b):
            origin = (segments[i].get("speaker") or {})
            if origin.get("track") is None and (ident, a, i) not in out and isinstance(origin.get("mark"), dict) \
                    and origin["mark"].get("track") is not None:
                out[(ident, a, i)] = f"s{origin['mark']['track']}"
    return out, windows


def score(arm="sol"):
    rng = np.random.default_rng(r9.SEED)
    r3.OUT = r9.ROOT / "build"
    base = r3.outcome({})
    keys = [k for k, v in base.items() if v["labelled"]]
    clusters = [base[k]["cluster"] for k in keys]
    group = {(w["ident"], w["a"]): w["group"] for w in r9.plan()}
    fresh = np.array([group[k[:2]] in r9.FRESH for k in keys])
    short = np.array([r9.backchannel(r9.load(k[0]), k[2]) for k in keys])
    manosaba_c = np.array([k[0] == "manosaba-c" for k in keys])
    hina_normal = np.array([group[k[:2]] == "hina 普通" for k in keys])

    def labelled_right(chosen):
        right = {}
        for ident, a, b in r9.windows():
            m = r9.load(ident)
            final = ps.final_labels(r3, m, ident, a, b, chosen)
            rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]]
            found = ps.match(rows, final, m.truth)
            right.update({(ident, a, i): found.get(final[i]) in m.truth[i] for i in rows})
        return np.array([right[k] for k in keys], float)

    def interval(d, mask):
        groups = defaultdict(list)
        for j in np.flatnonzero(mask):
            groups[clusters[j]].append(j)
        groups = list(groups.values())
        sims = [d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean() for _ in range(5000)]
        return d[mask].mean(), np.percentile(sims, 2.5), np.percentile(sims, 97.5)

    # sol is scored against the first readout (its gate); another driver against sol's aligned readout
    previous, _ = picks() if arm == "sol" else picks(aligned_stable)
    first, now = ("first", "aligned") if arm == "sol" else ("sol", arm)
    chosen, windows = picks(lambda ident: aligned_stable(ident, arm))
    acoustic, before, judged = labelled_right({}), labelled_right(previous), labelled_right(chosen)
    main = fresh & ~short
    print(f"labelled rows with truth {len(keys)}; main (fresh, not back-channel) {main.sum()}")
    print(f"right (main): acoustic {acoustic[main].mean():.1%}, {first} {before[main].mean():.1%}, {now} {judged[main].mean():.1%}")
    for name, mask in (("MAIN fresh, not back-channel", main), ("fresh, all rows", fresh), ("fresh, back-channel", fresh & short),
                       ("merged windows, not back-channel", ~fresh & ~short), ("魔裁 c (fast), not back-channel", manosaba_c & ~short),
                       ("hina 普通, not back-channel", hina_normal & ~short)):
        if not mask.any():
            continue
        v, lo, hi = interval(100 * (judged - before), mask)
        w, wlo, whi = interval(100 * (judged - acoustic), mask)
        print(f"  {name} ({mask.sum()} rows): {now} - {first} {v:+.1f} [{lo:+.1f}, {hi:+.1f}]; {now} - acoustic {w:+.1f} [{wlo:+.1f}, {whi:+.1f}]")
    if arm == "sol":
        v, lo, hi = interval(100 * (judged - before), main)
        verdict = "not worse (lower bound > -3)" if lo > -3 else "worse (upper bound < -3)" if hi < -3 else "uncertain"
        print(f"  VERDICT (pre-registered, aligned - first on the main rows): {verdict}: back to the owner unless 'not worse'")
    for ident, a, b in r9.windows():
        m = r9.load(ident)
        rows = [k for k in keys if k[0] == ident and not r9.backchannel(m, k[2])]
        at = [keys.index(k) for k in rows]
        print(f"    {ident:11s} {group[(ident, a)]:10s} rows {len(rows):3d} acoustic {acoustic[at].mean():.1%} "
              f"{first} {before[at].mean():.1%} {now} {judged[at].mean():.1%}")
    # from the sidecars' own results: `chosen` also holds kept rows whose acoustic cell a mark filled
    kinds = defaultdict(int)
    for units in windows.values():
        for unit in units:
            for result in unit.get("results") or ():
                kinds["moved" if "track" in result else "new track" if "label" in result or "carried" in result
                      else "abstain" if result.get("abstain") else "other"] += 1
    print(f"answers other than keep ({now}): {dict(kinds)}")
    usage = defaultdict(int)
    for ident, units in windows.items():
        for unit in units:
            if not unit.get("usage"):
                continue
            for k, value in unit["usage"].items():
                usage[k] += int(value or 0)
            print(f"  {ident} {unit.get('id')} {unit.get('first')}-{unit.get('last')}: {unit.get('status')} model "
                  f"{unit.get('model')} usage {unit.get('usage')} names {unit.get('names')}")
    print(f"usage, judged units: {dict(usage)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("setup", "prepare", "run", "score"))
    parser.add_argument("--only")
    parser.add_argument("--arm", choices=tuple(ARMS), default="sol", help="which driver's aligned readout (run, score)")
    args = parser.parse_args()
    if args.step == "setup":
        setup()
    elif args.step == "prepare":
        prepare(args.only)
    elif args.step == "run":
        run(args.only, args.arm)
    else:
        score(args.arm)


if __name__ == "__main__":
    main()
