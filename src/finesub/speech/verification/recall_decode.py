"""The model half of the recall repair (`postprocessing/recall_repair` and
`postprocessing/row_fill` decide): cut the stretches the rules ask about from
the audio and have them transcribed, then hand the texts back to the rules.

`recall_repair(...)` builds the hook `stabilization.stabilize_payload` takes
(`RecallRepair`) from the three switches (`--squeeze-repair`, `--gap-fill`,
`--row-fill`: auto | on | off). The hook settles the episode's
language on the stabilized rows (`recall_repair.episode_language`), checks
what the repairs need, and runs, in this order:

0. where the collapse repair is the ladder (`collapse_ladder`): Whisper, as
   the ASR step loads and decodes it (`_ladder_decode`), over the ladder's
   windows around the squeezed rows; then freed;
1. the collapse repair: the referee (Qwen3-ASR, the default model, no context,
   no compiled path) over `squeeze_windows`, then `repair_squeezed`; for the
   ladder, the stretch's proposals, then the referee's presence check over
   every distinct proposal in one more batch, then the pick;
2. the referee's half of the gap fill, on the rows step 1 left: every gap and,
   in a second batch, every row near one;
3. the referee's half of the row fill, on the same rows: every row its gate
   keeps (`row_fill.candidates`, at the switch's threshold), over its own span;
4. the referee is closed; Whisper over the gaps the gap fill's
   `needs_whisper` picks and the rows the row fill's does, one load for both;
5. `fill_gaps`, then `row_fill.fill_rows` on its result (the lines the gap
   fill added are neighbours its gates read; a row the gap fill moved is
   left out).

Every clip is cut tight from the audio (no padding) and the referee is called
once per batch above, each with every clip of its call as the accepted arms'
decodes were: all the collapse windows (every kind together), all the gaps,
all the rows near a gap; the row fill's batch holds only the rows its gate
keeps, where the accepted arm decoded every row and gated after (a batch's
members change the referee's text, to the degree of a bf16 tie). None is left
out but an empty clip (a span past the audio's end), which reads as nothing
heard. Whisper is plain
faster-whisper, not the refining subclass the ASR stage uses: a second opinion
on the text, not new timing. Greedy at temperature 0, the episode's language,
no conditioning on earlier text, no VAD filter; its segments' texts joined.
It is loaded only when some gap or row needs it, after the referee is closed,
so the two models are never resident together; the ladder's Whisper (step 0)
is freed before the referee loads.
"""

from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from ...reporting import NullReporter, current_reporter, reporting_to
from ..postprocessing import collapse_ladder as cl
from ..postprocessing import recall_repair as rr
from ..postprocessing import row_fill as rf
from ..postprocessing.recall_sounds import readings_available
from ..postprocessing.segmentation import split_params_of
from ..postprocessing.stabilization import RecallRepair
from .qwen_referee import DEFAULT_QWEN_MODEL, QwenReferee, SpanReader

Span = tuple[float, float]
Heard = tuple[str, "str | None"]
#: (start, end) spans in, (text, language) per span out.
Transcribe = Callable[[Sequence[Span]], list[Heard]]
#: The referee's transcribe takes what it is for too (the progress line).
Referee = Callable[[Sequence[Span], str], list[Heard]]

MODES = ("auto", "on", "off")
SQUEEZE_LABEL = "修坍缩行"
GAP_LABEL = "补空档"
ROW_FILL_LABEL = "行内补漏"


def _transcriber(reader: SpanReader, decode: Callable[[list[np.ndarray]], list[Heard]]) -> Transcribe:
    """`decode` over the spans' clips, empty clips left out."""

    def transcribe(spans: Sequence[Span]) -> list[Heard]:
        clips = [reader.read(a, b) for a, b in spans]
        usable = [i for i, clip in enumerate(clips) if len(clip)]
        out: list[Heard] = [("", None)] * len(clips)
        if usable:
            for i, reply in zip(usable, decode([clips[i] for i in usable])):
                out[i] = reply
        return out
    return transcribe


def _whisper_weights(model_name: str):
    """Put Whisper's weights on disk (the ASR step's own prefetch); the plan `_load_whisper` loads with."""

    from ..recognition.vad_asr_stage import ensure_asr_weights

    return ensure_asr_weights(model_name)


def _load_whisper(model_name: str, device: str, lang: str, weights) -> Callable[[list[np.ndarray]], list[Heard]]:
    """Whisper's decode, the model loaded now and freed with the returned function."""

    from faster_whisper import WhisperModel

    from ..runtime import hf_weights

    model = hf_weights.offline_first(
        lambda plan: WhisperModel(model_name, device=device,
                                  compute_type="float16" if device.startswith("cuda") else "float32",
                                  revision=plan.revision, local_files_only=plan.local_files_only),
        weights, what="asr")

    def decode(clips: list[np.ndarray]) -> list[Heard]:
        out = []
        for clip in clips:
            segments, _ = model.transcribe(clip, language=lang, temperature=0.0, beam_size=1,
                                           condition_on_previous_text=False, vad_filter=False)
            out.append(("".join(s.text for s in segments), lang))
        return out
    return decode


def repair(payload: Mapping[str, object], rows: list[dict], *, lang: str, squeeze: bool, gap: bool,
           referee: Referee, whisper: Callable[[], Transcribe], row_gate: float | None = None,
           decoded: Mapping[tuple[str, int], dict] | None = None,
           ) -> tuple[list[dict], dict]:
    """`rows` (the stabilized rows of `payload`, which carries the aligned
    segments and the `vad_timeline`) repaired, and what to record. `referee`
    transcribes with the referee; `whisper()` makes Whisper's transcribe and is
    called at most once, when a gap or a row needs it, after the last `referee`
    call. `decoded` (the ladder's Whisper windows, `_ladder_decode`) makes the
    collapse repair the ladder; None, the stretch alone. `row_gate`: the row
    fill's gate threshold (`row_fill.GATE_THRESHOLD` or `AUTO_THRESHOLD`);
    None, no row fill."""

    record: dict = {"language": lang}
    current = {**payload, "segments": rows}
    if squeeze:
        windows = rr.squeeze_windows(current)
        said = referee([(w["start"], w["end"]) for w in windows], SQUEEZE_LABEL)
        heard_windows = [{**w, "text": text} for w, (text, _) in zip(windows, said)]
        if decoded is None:
            rows, verdicts = rr.repair_squeezed(current, heard_windows, lang)
            rows = rr.shown(rows, rr.overlap_sources(verdicts), payload["vad_timeline"]["intervals"])
            record["stretched"] = sum(v["branch"] in ("stretch", "overlap") for v in verdicts)
        else:
            rows, verdicts = _ladder(payload, rows, heard_windows, decoded, lang, referee)
            record["stretched"] = sum("span" in v for v in verdicts)
        record["squeezed"] = verdicts
        current = {**payload, "segments": rows}
    asked: list[dict] = []  # what Whisper hears: {"start", "end"}, its text written back under "whisper"
    if gap:
        spans = rr.gaps(current)
        heard = [{"start": a, "end": b, "qwen": text, "language": language, "whisper": None}
                 for (a, b), (text, language) in zip(spans, referee(spans, GAP_LABEL))]
        near = rr.near_rows(spans, rows)
        own = [{"index": i, "qwen": text, "language": language}
               for i, (text, language) in zip(near, referee([(rows[i]["start"], rows[i]["end"]) for i in near],
                                                          GAP_LABEL))]
        asked += [g for g in heard if rr.needs_whisper(g, lang)]
    if row_gate is not None:
        # On the rows the gap fill starts from (the gate was fitted on rows no gap fill had touched), so the
        # referee hears them before it is closed for Whisper.
        picked = rf.candidates(rows, payload["vad_timeline"]["intervals"], row_gate)
        said = referee([(rows[k]["start"], rows[k]["end"]) for k in picked], ROW_FILL_LABEL)
        filling = {k: {"start": rows[k]["start"], "end": rows[k]["end"], "qwen": text, "language": language,
                       "whisper": None} for k, (text, language) in zip(picked, said)}
        asked += [h for k, h in filling.items() if rf.needs_whisper(rows[k], h, lang)]
    if asked:
        for h, (text, _) in zip(asked, whisper()([(h["start"], h["end"]) for h in asked])):
            h["whisper"] = text
    before = rows
    if gap:
        intervals = [(float(v["start"]), float(v["end"])) for v in payload["vad_timeline"]["intervals"]]
        rows = rr.fill_gaps(rows, heard, own, lang, intervals)
        record["gaps"] = [{**g, "heard": rr.heard(g, lang)} for g in heard]
        record["inserted"] = sum(1 for r in rows if r.get("gap_fill")) - sum(1 for r in before if r.get("gap_fill"))
        record["moved"] = sum(1 for r in rows if "gap_retime" in r and r not in before)
    if row_gate is not None:
        rows, filled = _fill_rows(before, rows, filling, lang)
        record["row_fill"] = {"gate": row_gate, **filled}
    return rows, record


def _ladder(payload: Mapping[str, object], rows: list[dict], heard_windows: list[dict],
            decoded: Mapping[tuple[str, int], dict], lang: str, referee: Referee,
            ) -> tuple[list[dict], list[dict]]:
    """The collapse ladder (`postprocessing/collapse_ladder`) on `rows`: the
    stretch's proposals from the referee's texts over the collapse windows
    (`heard_windows`), Whisper's from `decoded`, the referee's presence check
    over every distinct proposal in one more batch, then the pick and the move."""
    stretched, stretch_verdicts = rr.repair_squeezed({**payload, "segments": rows}, heard_windows, lang)
    recs = cl.proposals(rows, stretched, stretch_verdicts, decoded, lang)
    asks = cl.to_hear(recs)
    said = referee([span for _, span in asks], SQUEEZE_LABEL)
    recs = cl.settle(rows, recs, {ask: text for ask, (text, _) in zip(asks, said)}, decoded, lang)
    spans, verdicts = cl.decide(rows, recs)
    return cl.moved(payload, rows, spans, decoded, lang), verdicts


def _ladder_prefix(payload: Mapping[str, object], prefix_path: Path | None):
    """The VAD prefix the aligned of `payload` was decoded from (its energy
    track, pause hints and intervals are what the ASR step's chain after the
    decode reads), or the reason it cannot be had."""
    from ..recognition.vad_asr_stage import load_vad_prefix

    if prefix_path is None or not Path(prefix_path).exists():
        return None, "no vad prefix"
    try:
        prefix = load_vad_prefix(Path(prefix_path))
    except Exception as exc:
        return None, f"vad prefix unreadable: {type(exc).__name__}"
    spans = lambda items: [(float(v["start"]), float(v["end"])) for v in items]  # noqa: E731
    if spans(prefix.segments) != spans(payload["vad_timeline"]["intervals"]):
        return None, "vad prefix of other audio"
    return prefix, None


def _ladder_decode(payload: Mapping[str, object], rows: list[dict], prefix, *, audio_path: Path,
                   model_name: str, device: str, lang: str, weights) -> dict[tuple[str, int], dict]:
    """The ladder's Whisper windows (`collapse_ladder.windows`) around the
    squeezed `rows`, each decoded and finished as the ASR step decodes a group
    (its model class and settings, its audio loader, the gap it recorded,
    `decode_intervals`, then `finish_decoded` with the step's VAD prefix):
    (kind, row index) -> the
    window with its "segments". The `pad` windows first, then `iso` and `ctx`
    only for the rows `pad` gave nothing (`collapse_ladder.retried`), the only
    ones the ladder reads them for. The model is loaded only when there is a
    window, and freed before this returns. What the step's chain would warn
    about again is kept quiet: it was said when the step ran."""
    from ..recognition import transcribe as tr
    from ..recognition.fw_refine_backend import FwRefineModelPool
    from ..recognition.vad_asr_stage import finish_decoded

    out: dict[tuple[str, int], dict] = {}
    first = cl.windows(payload, rows, ("pad",))
    later = lambda: cl.windows(payload, rows, cl.KINDS[1:], only=cl.retried(rows, out, lang))  # noqa: E731
    if not first and not later():
        return {}
    reporter = current_reporter()
    params = split_params_of(payload)
    gap = ((payload.get("metadata") or {}).get("asr_align") or {}).get("gap_sec")
    gap = tr.DEFAULT_GAP_SEC if gap is None else float(gap)
    pool = FwRefineModelPool(model_name, device=device, size=1, refine_sec=tr.REFINE_SEC, load=weights)
    loader = tr.AudioBlockLoader(str(audio_path), target_sr=tr.TARGET_SR, block_seconds=600.0, pad_seconds=10.0,
                                 preprocess=False)

    def decode(model, todo: list[dict]) -> None:
        for n, w in enumerate(todo):
            reporter.progress("stable", completed=n, total=len(todo), unit="段", detail=SQUEEZE_LABEL)
            with reporting_to(NullReporter()):
                segments = tr.decode_intervals(w["intervals"], model=model, language=lang, audio_loader=loader,
                                               tail_real_limit_sec=w["tail"], gap_sec=gap)
                segments = finish_decoded(segments, vad_segments=prefix.segments, vad_meta=prefix.vad_meta,
                                          energy_track=prefix.energy_track, split_params=params,
                                          align_meta={"segment_split": {}}, timing={},
                                          audio_duration=prefix.audio_duration)
            out[(w["kind"], w["index"])] = {**w, "segments": segments}
        if todo:
            reporter.progress("stable", completed=len(todo), total=len(todo), unit="段", detail=SQUEEZE_LABEL)
    try:
        with pool.lease() as model:
            decode(model, first)
            decode(model, later())
    finally:
        loader.close()
        pool.close()
    return out


def _fill_rows(before: list[dict], rows: list[dict], filling: dict[int, dict], lang: str) -> tuple[list[dict], dict]:
    """The in-row fill on the rows after the gap fill (`rows`), whose neighbour
    gates read the lines it added; `filling` is keyed by index in `before`, the
    rows it started from. A row the gap fill moved is left out: its span is no
    longer the one heard. The gap fill sorts its output and stabilize's rows are
    not always in order, so each row is found again by its span before the gap
    fill and its text (two rows alike in both are interchangeable)."""
    at: dict[tuple, list[int]] = {}
    for j, r in enumerate(rows):
        if not r.get("gap_fill") and "gap_retime" not in r:
            at.setdefault((r["start"], r["end"], r.get("text")), []).append(j)
    heard = {}
    for k, h in filling.items():
        same = at.get((before[k]["start"], before[k]["end"], before[k].get("text")))
        if same:
            heard[same.pop(0)] = h
    tally: Counter = Counter()
    out = rf.fill_rows(rows, heard, lang, tally)
    return out, {"rows": len(filling), "audio": round(sum(h["end"] - h["start"] for h in filling.values()), 3),
                 "whisper": sum(h["whisper"] is not None for h in filling.values()),
                 "moved": len(filling) - len(heard), "filled": sum(1 for r in out if rf.FIELD in r), **tally}


def _free_gpu() -> None:
    """Let the closed referee's memory go before Whisper loads."""
    gc.collect()
    from ..runtime.device import cuda_usable

    if cuda_usable():
        import torch

        torch.cuda.empty_cache()


def _missing(payload: Mapping[str, object], audio_path: Path | None) -> str | None:
    """The first thing every repair needs that is not there, as a skip reason
    (what only the gap fill and the row fill need, Whisper, is checked apart)."""
    if audio_path is None or not Path(audio_path).exists():
        return "no vocals"
    if "intervals" not in (payload.get("vad_timeline") or {}):
        return "no vad_timeline"
    try:
        from transformers import AutoModelForMultimodalLM  # noqa: F401  (the referee's class, transformers 5.x)
    except Exception:
        return "no transformers"
    if not readings_available():
        return "no reading libraries"
    return None


#: What a missing prerequisite is, as the user reads it, and whether skipping
#: for it under `auto` is warned about (the ones that cost the user the repair;
#: an older aligned without a VAD timeline or a model name is just old).
_REASONS = {
    "no vocals": ("没有人声轨（-vocal.*）", True),
    "no vad_timeline": ("aligned 里没有 VAD 时间线（更老的产物）", False),
    "no asr model": ("aligned 里没有记 Whisper 模型（更老的产物）", False),
    "no transformers": ("没有 transformers 5.x（[asr] 依赖）", True),
    "no reading libraries": ("读音库（pykakasi、pypinyin、opencc）缺失", True),
    "load failed": ("Qwen 或 Whisper 加载失败", True),
}
#: Why the collapse repair fell back from the ladder to the stretch alone, as
#: the user reads it, and whether it is warned about (not for an aligned too
#: old to name its model).
_LADDER_FALLBACK = {
    "no asr model": ("", False),
    "no vad prefix": ("没有 VAD 前缀（-vad.json）", True),
    "vad prefix unreadable": ("VAD 前缀读不了", True),
    "vad prefix of other audio": ("VAD 前缀与 aligned 的 VAD 时间线对不上", True),
    "load failed": ("Whisper 的权重取不到", True),
}
_NAMES = {"squeeze": "坍缩行", "gap": "空档", "row_fill": "行内漏字"}
_IMPACT = {"squeeze": "坍缩行照旧一闪而过", "gap": "漏掉的空档照旧没有字幕", "row_fill": "行里漏掉的字照旧缺着"}
#: How a repair that has to run was asked for, as the error names it.
_FORCED = {"squeeze": "--squeeze-repair on", "gap": "--gap-fill on", "row_fill": "--row-fill on"}
#: Reasons that skip a repair even when it was asked for: nothing to repair in, not something to fix.
_NEVER_RAISE = ("no language",)


def _ladder_inputs(payload: Mapping[str, object], rows: list[dict], *, lang: str, model_name: str, weights,
                   vad_prefix: Path | None, audio_path: Path, device: str) -> tuple[dict | None, dict, object]:
    """Whisper's decodes for the ladder (`_ladder_decode`), or None where the
    collapse repair falls back to the stretch alone (warned when
    `_LADDER_FALLBACK` says so); what to record under `collapse`; Whisper's
    weights, fetched here when nothing before needed them."""
    why = None if model_name else "no asr model"
    prefix = None
    if why is None:
        prefix, why = _ladder_prefix(payload, vad_prefix)
    if why is None:
        try:
            weights = weights or _whisper_weights(model_name)
        except Exception as exc:
            why = f"load failed: {type(exc).__name__}: {exc}"[:300]
    if why is not None:
        what, warn = _LADDER_FALLBACK[why.split(":")[0]]
        if warn:
            current_reporter().warning("collapse-ladder-fallback",
                                       f"{what}，坍缩行只用拉长修（阶梯没跑）" + (f"：{why}" if ":" in why else ""),
                                       impact="坍缩行修得少一些")
        return None, {"repair": "stretch", "fallback": why}, weights
    decoded = _ladder_decode(payload, rows, prefix, audio_path=audio_path, model_name=model_name, device=device,
                             lang=lang, weights=weights)
    _free_gpu()
    return decoded, {"repair": "ladder", "windows": len(decoded)}, weights


def recall_repair(*, audio_path: Path | None, squeeze: str, gap: str, qwen_verify: str,
                  devices: Callable[[], tuple[str, str]], row_fill: str = "off",
                  vad_prefix: Path | None = None) -> RecallRepair | None:
    """The stabilize hook for the three switches (`squeeze`, `gap`, `row_fill`:
    auto | on | off), or None when none runs. `auto` follows `--qwen-verify`:
    off there, off here, since a run that turned the referee off does not want
    it loaded; `on` runs anyway. The row fill's `auto` is also the stricter
    gate (`row_fill.AUTO_THRESHOLD`), `on` the full one (`GATE_THRESHOLD`).
    In the hook a repair that is not off needs the vocal track `audio_path`, the
    aligned VAD timeline, transformers 5.x, the reading libraries and the
    referee's load; the gap fill and the row fill also need the model name
    aligned records and Whisper's weights (missing only those skips them
    alone). Without one, `auto` skips (recorded under `skipped`, warned on
    stderr when it costs the user the repair) and `on` raises. That is the only
    difference: a decode, a decision or Whisper's own load that fails raises
    under both. A run whose rows name no language skips under both, and one
    with nothing to hear loads no model. The collapse repair is the
    ladder when the aligned names its Whisper model, its
    VAD prefix `vad_prefix` (the ASR step's `-vad.json`) matches its VAD
    timeline and Whisper's weights can be had; else the stretch alone, the
    reason recorded under `collapse` (warned unless the aligned is too old to
    name its model). `devices()` answers (Whisper's, the referee's) once a model is
    going to load; either on the CPU warns."""

    modes = {"squeeze": squeeze, "gap": gap, "row_fill": row_fill}
    for name, mode in modes.items():
        if mode not in MODES:
            raise ValueError(f"{name}: expected one of {MODES}, got {mode!r}")
    if qwen_verify == "off":
        modes = {name: "off" if mode == "auto" else mode for name, mode in modes.items()}
    wanted = {name: mode for name, mode in modes.items() if mode != "off"}
    if not wanted:
        return None
    # as before the row fill existed while it is off
    switches = {name: mode for name, mode in modes.items() if name != "row_fill" or mode != "off"}
    row_gate = {"auto": rf.AUTO_THRESHOLD, "on": rf.GATE_THRESHOLD}.get(modes["row_fill"])

    def drop(names: list[str], reason: str, detail: str = "") -> dict[str, str]:
        """Take `names` out of the run for `reason`: raise when one of them is
        `on` (reasons in `_NEVER_RAISE` aside), else warn when the reason costs
        the user the repair; what to record under `skipped`."""
        what, warn = _REASONS.get(reason, ("", False))
        forced = [name for name in names if wanted[name] == "on"]
        if forced and reason not in _NEVER_RAISE:
            flags = "、".join(_FORCED[name] for name in forced)
            raise RuntimeError(f"{flags}：{what}" + (f"：{detail}" if detail else ""))
        if warn:
            current_reporter().warning("recall-repair-skipped",
                                       f"{what}，{'与'.join(_NAMES[n] for n in names)}的补救跳过" + (f"：{detail}" if detail else ""),
                                       impact="；".join(_IMPACT[n] for n in names))
        return {name: reason for name in names}

    def run(payload: Mapping[str, object], rows: list[dict]) -> tuple[list[dict], dict]:
        lang = rr.episode_language(payload, rows)
        todo = [name for name in ("squeeze", "gap", "row_fill") if name in wanted]
        skipped: dict[str, str] = {}
        done = lambda: (rows, {"switches": switches, "language": lang, **({"skipped": skipped} if skipped else {})})
        reason = _missing(payload, audio_path) or (None if lang else "no language")
        if reason:
            skipped.update(drop(todo, reason))
            return done()
        whisper_model = str(((payload.get("metadata") or {}).get("asr_align") or {}).get("model") or "")
        if not whisper_model and {"gap", "row_fill"} & set(todo):
            names = [name for name in ("gap", "row_fill") if name in todo]
            skipped.update(drop(names, "no asr model"))
            todo = [name for name in todo if name not in names]
        current = {**payload, "segments": rows}
        # Nothing to hear, no model loaded: no collapse window for the collapse repair, no gap for the gap fill
        # (the collapse repair only takes unclaimed speech, so it never opens a gap that was not there), no row
        # the row fill's gate keeps (without a collapse window the rows it reads are these).
        hears = {"squeeze": lambda: rr.squeeze_windows(current), "gap": lambda: rr.gaps(current),
                 "row_fill": lambda: rf.candidates(rows, payload["vad_timeline"]["intervals"], row_gate)}
        if not any(hears[name]() for name in todo):
            return done()
        weights = None
        # the row fill's rows are only final after the collapse repair, so its Whisper is made ready whenever it runs
        needing = [name for name in ("gap", "row_fill") if name in todo and (name == "row_fill" or hears[name]())]
        if needing:
            try:  # Whisper's weights, which only the gap fill and the row fill need
                weights = _whisper_weights(whisper_model)
            except Exception as exc:
                skipped.update(drop(needing, "load failed", f"{type(exc).__name__}: {exc}"[:300]))
                todo = [name for name in todo if name not in needing]
        if not todo:
            return done()
        device, referee_device = devices()
        if not str(device).startswith("cuda") or not str(referee_device).startswith("cuda"):
            current_reporter().warning("recall-repair-on-cpu", "stable 这一步的补救在 CPU 上跑（Qwen 与 Whisper 都是 float32）",
                                       impact="只影响 stable 这一步的耗时")
        ran: dict[str, dict] = {}
        decoded, collapse = None, None
        try:  # what all need first: the audio's header (the ladder's decode below reads the audio too)
            reader = SpanReader(str(audio_path))
        except Exception as exc:
            skipped.update(drop(todo, "load failed", f"{type(exc).__name__}: {exc}"[:300]))
            return done()
        if "squeeze" in todo and hears["squeeze"]():
            # Whisper's proposals first, the model freed before the referee loads: never both resident.
            decoded, collapse, weights = _ladder_inputs(payload, rows, lang=lang, model_name=whisper_model,
                                                        weights=weights, vad_prefix=vad_prefix,
                                                        audio_path=Path(audio_path), device=device)
            if decoded:
                ran["ladder_whisper"] = {"model": whisper_model, "device": device}
        referee = QwenReferee(DEFAULT_QWEN_MODEL, device=referee_device, accel="off")
        try:  # the referee's load
            referee.warm()
        except Exception as exc:
            referee.close()
            skipped.update(drop(todo, "load failed", f"{type(exc).__name__}: {exc}"[:300]))
            return done()
        reporter = current_reporter()

        def qwen_for(label: str) -> Callable[[list[np.ndarray]], list[Heard]]:
            def qwen(clips: list[np.ndarray]) -> list[Heard]:
                reporter.progress("stable", completed=0, total=len(clips), unit="段", detail=label)
                out = referee.transcribe_batch(clips, on_batch=lambda done, total: reporter.progress(
                    "stable", completed=done, total=total, unit="段", detail=label))
                ran["referee"] = {"model": DEFAULT_QWEN_MODEL, "device": referee.effective_device}
                return out
            return qwen

        def whisper() -> Transcribe:
            referee.close()
            _free_gpu()
            ran["whisper"] = {"model": whisper_model, "device": device}
            return _transcriber(reader, _load_whisper(whisper_model, device, lang, weights))

        try:
            out, record = repair(payload, rows, lang=lang, squeeze="squeeze" in todo, gap="gap" in todo,
                                 row_gate=row_gate if "row_fill" in todo else None,
                                 referee=lambda spans, label: _transcriber(reader, qwen_for(label))(spans),
                                 whisper=whisper, decoded=decoded)
        finally:
            referee.close()
            gc.collect()  # Whisper, when it was loaded
        return out, {"switches": switches, **record, **({"collapse": collapse} if collapse else {}), **ran,
                     **({"skipped": skipped} if skipped else {})}
    return run
