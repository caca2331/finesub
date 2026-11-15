"""C.5d: hand a model "two people are audible here" and let it decide whether to split.

Owner 2026-09-20: the good answer needs the overlap-separation line; what is available NOW is the
one signal we do have, given to a model -- especially a multimodal one -- with permission to split
those rows. Try it; if it is middling, shelve it with the rest and wait.

The candidate set is rows whose SECOND track is active for >= 1.0 s inside them. That operating
point was swept, not guessed: it marks 14.2% of rows and is right about 44.5% of them, catching
58.5% of everything splittable (the plan's C.5d table has the whole curve).

⚠⚠ **The gate is the curve, not the number.** "The model beats 44.5% precision" is worth nothing:
answering `no` more often buys precision, and so does turning the threshold up. The model has
added something only if its (recall, precision) point sits ABOVE the threshold sweep -- better
precision than the threshold that would have given the same recall.

⚠ **Truth is `turn_shape_bed.classify`, imported, never restated.** The older `turns_bed` ruler
reads the refined subtitle rasterised to 80 ms and was withdrawn: it disagrees with what S4
delivers on four rows out of five.

Two arms, both one call per 60 s window:

  text    the window's rows, with the signal's candidates marked
  audio   the same plus the window's vocal track -- the multimodal arm

⚠ A window, not a clip per row: `complete()` carries ONE `file_ref`, so per-row clips would be
thousands of calls; and turn-taking is a contextual phenomenon, so a model that hears the
surrounding minute is being asked a fairer question than one handed 2.5 seconds.

    python tools/diarizen_export/turn_llm_bed.py --arm text --dry-run
    python tools/diarizen_export/turn_llm_bed.py --arm audio          # spends quota
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

NATIVE = REPO / "tmp" / "diarizen-export" / "port-timelines-native"
OUT = REPO / "tmp" / "diarizen-bed" / "turn-llm"
CLIPS = REPO / "tmp" / "diarizen-bed" / "turn-llm-clips"

#: The predicted timeline's own step (native 20 ms, not the 80 ms product grid).
STEP = 0.02

#: A row is a candidate when its second-most-active track is on for at least this long inside it.
CANDIDATE_SECONDS = 1.0

#: A sweep column marking fewer rows than this has no readable precision and is dropped.
MIN_SWEEP_MARKS = 5

#: Below this recall a detector is not a detector, whatever its precision -- the verdict says so
#: instead of comparing it against an interpolated curve nobody would operate at.
MIN_USEFUL_RECALL = 0.15

#: One call covers this much audio. Long enough to hear the exchange around a row, short enough
#: that a reply stays parseable.
WINDOW_SECONDS = 60.0

#: The agy front this experiment pins BOTH arms to (owner 2026-09-20: 「可以直接用后者」).
#: ⚠ The `agy` preset binds `research/quality` to a group whose FIRST member is
#: `local-agy-opus-4_6`, which is text-only: letting routing choose would run the text arm on Opus
#: and the audio arm on this, so the two arms would differ by MODEL as well as by modality -- the
#: one comparison this bed exists to make.
AGY_TARGET = "local-agy-gemini-3_8-flash"

#: ⚠ Same guard as `roles_run`: the standing authorisation plus what the owner opened on
#: 2026-09-20, and the paid tail stays shut whatever `ALLOWED` says.
ALLOWED = ("3.1-flash-lite", "3.5-flash-lite", "gemini-free-", "local-agy-")
FORBIDDEN = ("-paid-",)

SYSTEM = """你在校对一份日语对谈节目的字幕。每一行字幕都带着开始时间。

有些行被标了 ⚑ —— 声学分析说**这一行的时间跨度内有第二个人也在出声**，持续超过一秒。
但那个信号分不出两种情况：

A. 两个人**先后**说话，这一行把两个人的话并进了一句 ⇒ **应该拆成两行**
B. 两个人**同时**出声（附和、笑声、插话、背景），而这一行的文字**只是其中一个人的**
   ⇒ **不该拆**，没有第二个人的字可拆

你的任务：只针对被标 ⚑ 的行，判断它是 A 还是 B。

⚠ 判据是**文字**：只有当这一行的文字里**确实包含两个人各自的话**时才答 A。
如果第二个人只是「うん」「えー」这类附和、或者根本没被写进这行文字，那就是 B。

按下面的格式回答，每个被标的行一条，不要有别的内容：

<splits>
{"time": 123.4, "answer": "A", "cut_after": "拆点之前的最后几个字"}
{"time": 128.9, "answer": "B"}
</splits>

`cut_after` 只在答 A 时给，写这一行文字里**第一个人说完的最后几个字**（原文照抄）。"""

#: v2 (owner 2026-09-20): 「大致 prompt 是忽略填充词语气词，有实际意义的词拆出来或补出来」.
#: Two changes from v1, and the second is the interesting one.
#:
#: ⚠⚠ **Supplying is not re-transcribing.** `speaker-diarization-experiments.md` §5o measured
#: "re-transcribe the overlap region and REPLACE the subtitle" and shelved it: no re-transcription
#: kept the words production had already got right (57-90% retention). This asks for something
#: with a different failure mode -- production's text is quoted verbatim and the model only ADDS
#: what the row never contained. Nothing it writes can overwrite a word that was already correct.
#: That is worth testing precisely because the shelved line's defect does not reach it.
SYSTEM_V2 = """你在校对一份日语对谈节目的字幕。每一行字幕都带着开始时间。

有些行被标了 ⚑ —— 声学分析说**这一行的时间跨度内有第二个人也在出声**，持续超过一秒。
你要判断这一行到底属于哪种，并在该拆的时候**把它拆开**。

⚠ **只看有实际意义的话。** 「うん」「ええ」「あー」「へえ」「そう」这类**附和、语气词、笑声**
**一律不算**——第二个人只发出这些声音时，这一行仍然是一个人的话，答 B。

A. 第二个人说了**有实际意义的话** ⇒ 拆。给出两半：
   - `first`：第一个人说完的最后几个字，**从这一行的原文里照抄**
   - `second`：第二个人说的话
     · 如果那句话**已经在这一行的文字里** ⇒ 照抄，并标 `"source": "quoted"`
     · 如果这一行的文字**根本没写到它**（转写漏了）⇒ **按你听到的写出来**，
       并标 `"source": "heard"`
B. 其余一切（只有附和/语气词/笑声/背景，或者第二个人的话压根没有实际内容）⇒ 不拆。

⚠ **不要改写第一个人的字。** 原文写对的部分一个字都不要动，你只负责在它后面补上缺的那半。

按下面的格式回答，每个被标的行一条，不要有别的内容：

<splits>
{"time": 123.4, "answer": "A", "first": "原文照抄的最后几个字", "second": "第二个人的话", "source": "quoted"}
{"time": 128.9, "answer": "B"}
</splits>"""

BLOCK = re.compile(r"<splits>(.*?)</splits>", re.S)


def second_track_seconds(stable, native):
    """`{row index: seconds the second track is active inside it}` for EVERY row.

    ⚠ Separate from `candidates()` on purpose. The sweep needs the raw quantity on every row, and
    reusing the candidate dict made the curve silently wrong: rows below the candidate threshold
    were absent, read as 0.0, and so the 0.2 / 0.5 / 1.0 columns came out identical.
    """
    found = {}
    for index, row in enumerate(stable):
        low = int(float(row["start"]) / STEP)
        high = max(low + 1, int(np.ceil(float(row["end"]) / STEP)))
        span = np.asarray(native[low:high]) > 0
        if not span.size:
            continue
        counts = np.sort(span.sum(axis=0))[::-1]
        found[index] = float(counts[1] * STEP) if len(counts) > 1 else 0.0
    return found


def candidates(stable, native):
    """`{row index: seconds the second track is active inside it}` for the signal's picks."""
    found = {}
    for index, row in enumerate(stable):
        low = int(float(row["start"]) / STEP)
        high = max(low + 1, int(np.ceil(float(row["end"]) / STEP)))
        span = np.asarray(native[low:high]) > 0
        if not span.size:
            continue
        counts = np.sort(span.sum(axis=0))[::-1]
        second = float(counts[1] * STEP) if len(counts) > 1 else 0.0
        if second >= CANDIDATE_SECONDS:
            found[index] = second
    return found


def windows(stable, marked, *, seconds: float = WINDOW_SECONDS):
    """Contiguous `seconds`-long windows that contain at least one candidate row."""
    if not stable:
        return []
    end = max(float(row["end"]) for row in stable)
    found = []
    start = 0.0
    while start < end:
        stop = start + seconds
        rows = [i for i, row in enumerate(stable)
                if float(row["start"]) >= start and float(row["start"]) < stop]
        if rows and any(i in marked for i in rows):
            found.append((start, stop, rows))
        start = stop
    return found


def prompt_for(stable, rows, marked) -> str:
    lines = []
    for index in rows:
        row = stable[index]
        flag = " ⚑" if index in marked else ""
        lines.append(f"{float(row['start']):.1f}{flag} | {str(row.get('text', '')).strip()}")
    return ("这一分钟的字幕：\n\n" + "\n".join(lines)
            + f"\n\n被标 ⚑ 的有 {sum(1 for i in rows if i in marked)} 行，逐行判断。")


def clip_for(episode: int, start: float, stop: float) -> Path:
    from finesub.media.ffmpeg import extract_audio_clip

    CLIPS.mkdir(parents=True, exist_ok=True)
    out = CLIPS / f"onetonton-{episode:02d}-{start:.0f}.aac"
    if not out.exists():
        source = REPO / "out" / f"onetonton-{episode:02d}" / f"onetonton-{episode:02d}-vocal.ogg"
        extract_audio_clip(source, start, stop, out)
    return out


def pinned_roles(target_id: str):
    """`{role: config}` whose whole endpoint chain is one target.

    Injected configs beat cell resolution in `RoleClient`, which is the documented way to say
    "this call, this model" without editing the user's `config.toml` or inventing a preset.
    """
    from dataclasses import replace

    from finesub.llm.routing.config import LLMRole, role_config_for

    base = role_config_for("research", preset_id="agy", role=LLMRole.GENERAL_CAPABLE)
    chain = tuple(e for e in base.endpoint_chain if e.target_id == target_id)
    if not chain:
        raise SystemExit(f"⚠ {target_id!r} is not in the agy preset's research group")
    # ⚠⚠ `model_group_id=""` is the whole pin. `ModelRouter.plan` reads
    # `if config.model_group_id: return self._group_plan(...)` and NEVER LOOKS AT
    # `endpoint_chain` on that path -- so narrowing the chain while leaving the group id set
    # does nothing at all. Measured the hard way: the first text arm ran 21 of its 25 windows on
    # `local-agy-opus-4_6` while its own comment claimed the pin stopped exactly that, and the
    # audio-vs-text comparison it fed was modality confounded with model.
    return {LLMRole.GENERAL_CAPABLE: replace(base, endpoint_chain=chain, test_endpoint=chain[0],
                                             model_group_id="")}


def guarded(target_id: str | None = None):
    """`complete(system, user, clip) -> (text, target)`, refusing an unauthorised model."""
    from finesub.llm.client import RoleClient
    from finesub.llm.routing.config import LLMRole

    client = RoleClient(role_configs=pinned_roles(target_id) if target_id else None)

    def complete(system: str, user: str, clip: Path | None):
        reference = None
        if clip is not None:
            from finesub.llm.media_upload import window_media_ref

            reference = window_media_ref(clip)
        reply = client.complete(LLMRole.GENERAL_CAPABLE,
                                [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                                file_ref=reference)
        target = str(getattr(reply, "target_id", ""))
        # ⚠ When a target is pinned the guard checks THAT TARGET, not its family. The family
        # check passed every one of the 21 Opus calls that silently replaced the pinned model.
        if target_id and target != target_id:
            raise SystemExit(f"⚠ pinned to {target_id!r} but the call landed on {target!r}. "
                             f"Stopped after ONE call.")
        if any(bad in target for bad in FORBIDDEN) or not any(ok in target for ok in ALLOWED):
            raise SystemExit(f"⚠ the call landed on {target!r}, which this experiment is not "
                             f"authorised to spend. Stopped after ONE call.")
        return reply.content, target

    return complete


def parse(text: str) -> dict:
    """`{time: answer}` from the reply block; a row the model skipped simply has no entry."""
    match = BLOCK.search(text or "")
    if not match:
        return {}
    found = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            found[round(float(payload["time"]), 1)] = str(payload.get("answer", "")).upper()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return found


def sweep(chosen, thresholds=(0.2, 0.5, 1.0, 1.5, 2.0, 3.0)):
    """The threshold curve ON THE SAMPLED WINDOWS -- the thing the model has to sit above.

    ⚠ Recomputed locally rather than quoted from the plan's corpus-wide table: 25 windows are not
    the corpus, and comparing a model measured here against a curve measured there would let a
    lucky sample decide the verdict.
    """
    from turn_shape_bed import classify

    rows = []
    for _, _, _, indexes, stable, events, _, seconds in chosen:
        for index in indexes:
            rows.append((classify(stable[index], events)[0] == "splittable",
                         seconds.get(index, 0.0)))
    total = sum(1 for positive, _ in rows if positive) or 1
    curve = []
    for threshold in thresholds:
        marked = [positive for positive, value in rows if value >= threshold]
        # ⚠ A threshold that marks a handful of rows has no usable precision -- the 3.0 s column
        # marked ONE row, missed it, and read 0%, which then dragged the interpolated curve down
        # at low recall and let a model finding 3% of the rows be called "above the curve".
        if len(marked) >= MIN_SWEEP_MARKS:
            curve.append((threshold, len(marked), sum(marked) / total,
                          sum(marked) / max(len(marked), 1)))
    return curve, total


def above(curve, recall: float, precision: float) -> bool:
    """Does `(recall, precision)` beat the threshold that would have given the same recall?

    ⚠ The comparison the whole experiment turns on. Linear interpolation between the two
    bracketing sweep points; outside the curve's range the nearest end is used, which is the
    conservative reading (a recall the threshold sweep cannot reach at all is not a free win).
    """
    points = sorted(((r, p) for _, _, r, p in curve), key=lambda pair: pair[0])
    if recall <= points[0][0]:
        return precision > points[0][1]
    if recall >= points[-1][0]:
        return precision > points[-1][1]
    for (r0, p0), (r1, p1) in zip(points, points[1:]):
        if r0 <= recall <= r1:
            share = (recall - r0) / max(r1 - r0, 1e-9)
            return precision > p0 + share * (p1 - p0)
    return False


def matched_threshold(curve, recall: float):
    """The sweep column whose recall is closest to the model's -- the fair thing to compare to."""
    return min(curve, key=lambda row: abs(row[2] - recall)) if curve else None


def precision_gap(per_window, draws: int = 4000, seed: int = 20260920):
    """Percentile CI for `model precision - threshold precision`, resampling WINDOWS.

    ⚠ The pooled ratio is recomputed inside each draw, not averaged over per-window ratios: a
    window where neither side marked anything has no precision, and averaging would either drop
    it (changing the estimand) or read it as zero.
    """
    if not per_window:
        return None
    generator = np.random.default_rng(seed)
    counts = np.asarray(per_window, dtype=float)  # (windows, 4): m_tp, m_called, t_tp, t_called
    gaps = []
    for _ in range(draws):
        drawn = counts[generator.integers(0, len(counts), len(counts))]
        model_called, threshold_called = drawn[:, 1].sum(), drawn[:, 3].sum()
        if model_called <= 0 or threshold_called <= 0:
            continue
        gaps.append(drawn[:, 0].sum() / model_called - drawn[:, 2].sum() / threshold_called)
    if len(gaps) < draws // 2:
        return None
    return float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def score(arm: str, chosen) -> None:
    """One arm's (recall, precision) against the local curve."""
    log = OUT / f"{arm}.jsonl"
    if not log.exists():
        print(f"{arm}: no answers on disk")
        return
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    curve, total_positive = sweep(chosen)

    # Per window, so the bootstrap below can resample the unit that was actually sampled.
    seconds_by_key = {f"{episode}-{start:.0f}": (indexes, stable, events, every)
                      for episode, start, _, indexes, stable, events, _, every in chosen}
    from turn_shape_bed import classify as shape_of

    hit = called = answered = 0
    per_window = []
    for record in records:
        window_hit = window_called = 0
        for time_key, shape in record["truth"].items():
            answer = record["answers"].get(str(time_key))
            answered += answer is not None
            if answer == "A":
                window_called += 1
                window_hit += shape == "splittable"
        hit += window_hit
        called += window_called
        per_window.append([window_hit, window_called, 0, 0])
    recall = hit / max(total_positive, 1)
    precision = hit / max(called, 1)
    print(f"\n{arm}: {len(records)} windows, model answered {answered} marked rows, "
          f"said split on {called}")
    print(f"  recall {recall:.1%} (of all {total_positive} splittable rows in these windows)   "
          f"precision {precision:.1%}")
    # ⚠ Both facts, always. An earlier version printed the recall warning INSTEAD of the curve
    # verdict, which threw away the one comparison the experiment was built to make -- and a
    # high-precision, low-recall point is a real operating mode (a short, mostly-right list for a
    # proofreader), so "not usable" is not the reader's call to have made for them.
    # The threshold column at the model's own recall, filled in per window for the paired CI.
    match = matched_threshold(curve, recall)
    if match is not None:
        threshold = match[0]
        for row, record in zip(per_window, records):
            indexes, stable, events, every = seconds_by_key.get(record["key"], ([], [], [], {}))
            for index in indexes:
                if every.get(index, 0.0) >= threshold:
                    row[3] += 1
                    row[2] += shape_of(stable[index], events)[0] == "splittable"
        marked = sum(row[3] for row in per_window)
        thr_precision = sum(row[2] for row in per_window) / max(marked, 1)
        interval = precision_gap(per_window)
        band = f"[{interval[0]:+.1%}, {interval[1]:+.1%}]" if interval else "(no interval)"
        print(f"  vs the >= {threshold:.1f}s threshold (recall {match[2]:.1%}, precision "
              f"{thr_precision:.1%}, marks {marked}): "
              f"{precision - thr_precision:+.1%} precision {band}")
    print(f"  ⇒ {'ABOVE' if above(curve, recall, precision) else '⚠ ON OR BELOW'} the curve"
          f" -- {'added information beyond the threshold' if above(curve, recall, precision) else 'no information beyond turning the threshold up'}")
    if recall < MIN_USEFUL_RECALL:
        print(f"  ⚠ but recall is under {MIN_USEFUL_RECALL:.0%}: the model declined most of what "
              f"the signal offered, so this is a short high-confidence list, not a detector. "
              f"Read the curve verdict with that in mind -- at this end the sweep has few marks "
              f"too, so both sides are thin.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=("text", "audio"), default="text")
    parser.add_argument("--episodes", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--windows", type=int, default=25, help="total windows across episodes")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score", action="store_true", help="read the logs, call nothing")
    parser.add_argument("--tag", default="agy", help="log suffix; one per model family")
    parser.add_argument("--variant", choices=("v1", "v2"), default="v1",
                        help="v2 ignores fillers and asks for the second half, quoted or heard")
    parser.add_argument("--target", default=AGY_TARGET,
                        help="pin every call to this target id; empty lets routing choose")
    arguments = parser.parse_args()

    from turn_shape_bed import classify, episode_rows

    pool = []
    for episode in arguments.episodes:
        loaded = episode_rows(episode)
        path = NATIVE / f"onetonton-{episode:02d}.npz"
        if loaded is None or not path.exists():
            continue
        stable, _, _, events = loaded
        native = np.load(path)["probs"]
        marked = candidates(stable, native)
        every = second_track_seconds(stable, native)
        for start, stop, rows in windows(stable, marked):
            pool.append((episode, start, stop, rows, stable, events, marked, every))

    # ⚠ Uniform over the windows that hold a candidate, with a fixed seed -- NOT the windows with
    # the most candidates. Picking the busiest minutes would hand the model the material the
    # signal is already best on and report the result as the model's.
    generator = np.random.default_rng(arguments.seed)
    chosen = [pool[i] for i in
              generator.choice(len(pool), size=min(arguments.windows, len(pool)), replace=False)]
    print(f"{len(pool)} windows hold a candidate; sampling {len(chosen)}")

    if arguments.score:
        curve, total = sweep(chosen)
        print(f"\nthreshold sweep on these windows ({total} splittable rows):")
        for threshold, marked, recall, precision in curve:
            print(f"  >= {threshold:.1f}s  marks {marked:>4}  recall {recall:>6.1%}  "
                  f"precision {precision:>6.1%}")
        for name in sorted(q.stem for q in OUT.glob("*.jsonl")):
            score(name, chosen)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if arguments.variant == "v1" else f"-{arguments.variant}"
    log = OUT / f"{arguments.arm}-{arguments.tag}{suffix}.jsonl"
    done = {json.loads(line)["key"] for line in log.read_text(encoding="utf-8").splitlines()} \
        if log.exists() else set()
    complete = None if arguments.dry_run else guarded(arguments.target or None)

    for episode, start, stop, rows, stable, events, marked, _ in chosen:
        key = f"{episode}-{start:.0f}"
        truth = {round(float(stable[i]["start"]), 1): classify(stable[i], events)[0]
                 for i in rows if i in marked}
        user = prompt_for(stable, rows, marked)
        if arguments.dry_run:
            print(f"  ep{episode:02d} {start:6.0f}s  {len(rows):3d} rows, "
                  f"{len(truth):2d} marked, truth {sorted(set(truth.values()))}")
            continue
        if key in done:
            continue
        clip = clip_for(episode, start, stop) if arguments.arm == "audio" else None
        text = target = None
        answers = {}
        for attempt in range(4):
            try:
                text, target = complete(SYSTEM_V2 if arguments.variant == "v2" else SYSTEM,
                                        user, clip)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{key}: {type(error).__name__} {str(error)[:90]}; retry {attempt + 1}")
                time.sleep(15 * (attempt + 1))
                continue
            answers = parse(text)
            # ⚠⚠ An unparseable reply is RETRIED, not counted. agy's answer channel is malformed
            # about 23% of the time (thought leakage, empty, degenerate) independently of the
            # material, and `parse` returns {} for all of those -- which reads as "the model
            # declined every row", which is exactly the quantity this bed measures. Silence from
            # the transport must not become an answer.
            if answers:
                break
            print(f"{key}: reply carried no <splits> rows; retry {attempt + 1}")
        if not answers:
            print(f"{key}: gave up")
            continue
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "episode": episode, "target": target,
                                     "truth": truth, "answers": answers, "raw": text},
                                    ensure_ascii=False) + "\n")
        said = sum(1 for value in answers.values() if value == "A")
        print(f"  {key} [{target}] {len(truth)} marked, model says split on {said}")


if __name__ == "__main__":
    main()
