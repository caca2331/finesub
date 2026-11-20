"""Golden snapshots of the shipped prompts, plus the version ledger.

Generated prompts change six places at once when one material row moves, so the
snapshots put that diff back on the review surface
(``docs/llm_prompts.md``). Three passes' pass 1 (the speaker judge) and pass 3
(proofread) have their own prompt versions and are held the same way, as is the
single pass that also names who speaks (``capableC-tracked``, its own version).
The window's own proofread turn is versioned by nothing (it is sent only after
the answer passed, `docs/llm_prompts.md`), so its snapshot has no ledger entry.

The snapshots alone cannot enforce the *version* discipline: they are tracked
files, so regenerating them makes the comparison pass again and "changed the
prompt without bumping PROMPT_VERSION" has no detection point. ``manifest.json``
is that second, independent载体 -- it records ``PROMPT_VERSION -> {file: sha256}``
and refuses a different hash under a version it has already seen.

Regenerate with::

    python -m pytest test/test_llm_prompt_goldens.py --regenerate-goldens
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from finesub.llm.prompt_compose import (
    GROUPED_PROMPT_VERSION,
    PROMPT_VERSION,
    TRACKED_PROMPT_VERSION,
    compose_correction_system,
    compose_correction_user,
)
from finesub.llm.routing.profiles import resolve_profile

GOLDEN_DIR = Path(__file__).parent / "data" / "prompt_goldens"
MANIFEST = GOLDEN_DIR / "manifest.json"

# Not a full axis sweep: the six vectors the retired presets mapped to,
# plus one representative new combination per axis.
SHIPPED_COMBINATIONS = {
    "text-none-minimum": ("text", "none", "efficiency"),
    "text-none-high": ("text", "none", "quality"),
    "text-native-high": ("text", "native", "quality"),
    "text-local-high": ("text", "local", "quality"),
    "audio-local-high": ("audio", "local", "quality"),
    "video-local-high": ("video", "local", "quality"),
    # representative uncalibrated combinations
    "audio-none-high": ("audio", "none", "quality"),
    "video-local-med": ("video", "local", "intermediate"),
}

# A task with speaker labels: the `turn` column and its merge rule (every
# other combination renders without them).
TURN_COMBINATIONS = {
    "audio-local-high-turns": ("audio", "local", "quality"),
}

VARIANTS = ("capableC", "basicA")

# Three passes' pass 2 (its own version): capableC only, in both media, since `prompt_compose` has
# fragments that only text media gets.
GROUPED_COMBINATIONS = {
    "audio-local-high-grouped": ("audio", "local", "quality"),
    "text-none-high-grouped": ("text", "none", "quality"),
}

# A cloud task's window (its lines carry keys) where two people talk at once: `turn` also reads 同时 and 换回,
# which the input column's description explains and the merge rules name; single pass and pass 2, capableC
# only -- the variants share those fragments.
CLOUD_SWITCHES = ("audio", "local", "quality")

# A correct-only task (a Chinese source, its own sets `<variant>-polish`): each variant once, and pass 2;
# the window's rows are Chinese. (file stem, switches, variant, turns, grouped)
CORRECT_ONLY_CASES = (
    ("audio-local-high-turns", ("audio", "local", "quality"), "capableC-polish", True, False),
    ("text-none-minimum", ("text", "none", "efficiency"), "basicA-polish", False, False),
    ("audio-local-high-grouped", ("audio", "local", "quality"), "capableC-polish", True, True),
)
ZH_ROWS = {"テスト": "大家好", "です": "开始吧"}

# A single-judged task's window (`intermediate`): the speaker material on it serves the cell's set as its tracked
# one. Four rows: s0, s0, s1 (low, alternative s0), s1+s0. (file stem, the cell's set, the rows' text)
TRACKED_SWITCHES = ("audio", "local", "intermediate")
TRACKED_CASES = (
    ("audio-local-med-tracked", "capableC", ("それでさ、昨日の配信で", "ボスに負けちゃって", "えー本当に?", "うんマジで")),
    ("audio-local-med-tracked", "capableC-polish", ("然后啊，昨天直播的时候", "被boss打败了", "诶真的吗?", "嗯真的")),
)


def _factory_style(correct_only: bool = False) -> str:
    """What a task without a user style is shown: the factory default (owner 2026-10-31) -- nothing for a
    correct-only task. Rendering it into every golden puts its template under the same version discipline as the
    prompts it rides in."""
    from finesub.llm.knowledge.style import render_style_block

    return render_style_block("", (), correct_only=correct_only)


def _render(switches, variant: str, turns: bool = False, grouped: bool = False, keyed: bool = False,
            rows: dict | None = None) -> str:
    profile = resolve_profile(*switches)
    system = compose_correction_system(profile, variant=variant, turns=turns, grouped=grouped, keyed=keyed, cloud=keyed,
                                       style_block=_factory_style(variant.endswith("-polish")), length_scale=1.0)
    user = compose_correction_user(
        profile,
        variant=variant,
        general_context_json="{}",
        user_requests=True,
        window_context="（无）",
        entry_details="（无）",
        previous_advice="（无）",
        pre_round_notes="",
        search_results="（无）",
        preceding_context_csv="-1|-1.0|0.5||前文",
        current_asr_csv=_swap((
            "1|0.0|4.0|||テスト\n2|1.0|1.0|0.0|同时|うん\n3|4.2|1.0|2.2|换回|です" if keyed
            else "1|0.0|1.0|||テスト\n2|1.1|1.0|0.1|可并|です" if grouped
            else "1|0.0|1.0|||テスト" if turns else "1|0.0|1.0||テスト"
        ), rows or {}),
        current_asr_row_count=3 if keyed else 2 if grouped else 1,
        turns=turns,
        grouped=grouped,
        keyed=keyed,
        length_scale=1.0,
    )
    return f"=== system ===\n{system}\n=== user ===\n{user}"


def _tracked_window(texts, scratch: Path):
    from finesub.llm.chunking import CorrectionBudget, SubtitleWindow, load_product_segments
    from finesub.llm.prompts import ContextPack
    from finesub.llm.window_speakers import speaker_material, with_speakers

    def speaker(track, *others, mark=None):
        payload = {"track": track, "tracks": [track, *others], "block": 0,
                   "labels": [{"role": "unknown", "conf": "high"}], "coverage": {str(track): 0.9}}
        return {**payload, "mark": mark} if mark else payload

    speakers = (speaker(0), speaker(0), speaker(1, mark={"tier": "low", "alternative": 0}), speaker(1, 0))
    stable = scratch / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [
        {"id": str(i), "start": 2.0 * i, "end": 2.0 * i + 1.5, "text": text, "speaker": who}
        for i, (who, text) in enumerate(zip(speakers, texts), start=1)]}, ensure_ascii=False), encoding="utf-8")
    segments, _layer = load_product_segments(stable)
    window = SubtitleWindow("0001", segments, [], "even", CorrectionBudget(0, 0, 0, "test"), clip_start=0.0,
                            clip_end=10.0, length_scale=1.0)
    return with_speakers([window], speaker_material(stable, ContextPack(), knowledge_root=None, title="clip"))[0]


def _render_tracked(variant: str, texts) -> str:
    import tempfile

    from finesub.llm.prompts import build_correction_csv_messages

    with tempfile.TemporaryDirectory() as scratch:
        window = _tracked_window(texts, Path(scratch))
        system, user = (message["content"] for message in build_correction_csv_messages(
            window=window, profile=resolve_profile(*TRACKED_SWITCHES), variant=variant,
            style_block=_factory_style(variant.endswith("-polish"))))
    return f"=== system ===\n{system}\n=== user ===\n{user}"


def _render_inline_proofread() -> str:
    """The proofread turn on a tracked window (its `spk` sentence included), as an API user message and as a tool
    session's reply to `submit`, then the repair round's message."""
    import tempfile

    from finesub.llm import inline_proofread
    from finesub.llm.prompt_variants import resolve_variant

    variant = resolve_variant("capableC")
    with tempfile.TemporaryDirectory() as scratch:
        window = _tracked_window(TRACKED_CASES[0][2], Path(scratch))
        api = inline_proofread.instruction(variant, window, ["- 2（读速）：跨度 1.5 秒、16 个字，10.7 字/秒"])
        tool = inline_proofread.instruction(variant, window, [], tool_session=True)
        repair = inline_proofread.repair_instruction(
            variant, window, ["<patch> row 1 changes `conf` from `high` to `low`: the proofread turn changes text only."])
    return f"=== api ===\n{api}\n=== tool session ===\n{tool}\n=== repair ===\n{repair}"


def _swap(text: str, rows: dict) -> str:
    for old, new in rows.items():
        text = text.replace(old, new)
    return text


def _cases():
    """(file, a renderer, the version its hash is recorded under)"""

    for combinations, turns in ((SHIPPED_COMBINATIONS, False), (TURN_COMBINATIONS, True)):
        for name, switches in combinations.items():
            for variant in VARIANTS:
                yield (f"{name}__{variant}.txt", lambda s=switches, v=variant, u=turns: _render(s, v, u),
                       PROMPT_VERSION)
    for name, switches in GROUPED_COMBINATIONS.items():
        yield (f"{name}__capableC.txt", lambda s=switches: _render(s, "capableC", True, True),
               GROUPED_PROMPT_VERSION)
    yield ("audio-local-high-cloud__capableC.txt", lambda: _render(CLOUD_SWITCHES, "capableC", True, keyed=True),
           PROMPT_VERSION)
    yield ("audio-local-high-cloud-grouped__capableC.txt",
           lambda: _render(CLOUD_SWITCHES, "capableC", True, True, keyed=True), GROUPED_PROMPT_VERSION)
    for name, switches, variant, turns, grouped in CORRECT_ONLY_CASES:
        yield (f"{name}__{variant}.txt",
               lambda s=switches, v=variant, u=turns, g=grouped: _render(s, v, u, g, rows=ZH_ROWS),
               GROUPED_PROMPT_VERSION if grouped else PROMPT_VERSION)
    for name, variant, texts in TRACKED_CASES:
        yield (f"{name}__{variant}.txt", lambda v=variant, x=texts: _render_tracked(v, x), TRACKED_PROMPT_VERSION)
    yield ("proofread-inline.txt", _render_inline_proofread, None)
    yield from _pass_cases()


def _pass_cases():
    """Three passes' pass 1 (the speaker judge: alone, with the shape section, heard) and pass 3
    (proofread), each under its own version."""
    from dataclasses import replace

    from finesub.llm.prompts import ContextPack
    from finesub.llm.stages.proofread import prompt as proofread
    from finesub.llm.stages.proofread.run import PROMPT_VERSION as PROOFREAD_VERSION
    from finesub.llm.stages.speaker_judge import prompt as judge
    from finesub.llm.stages.speaker_judge.run import PROMPT_VERSION as JUDGE_VERSION
    from finesub.llm.stages.speaker_judge.windows import JudgeWindow

    from .support.speaker_judge import segments_of, speaker

    rows = judge.rows_of(segments_of([(speaker(i % 2), 2, f"台词{i}") for i in range(1, 9)]))
    window = JudgeWindow("0001", "0001", 0, 8)
    profile = judge.profiles(rows, window)
    common = {"title": "clip", "summary": "雑談", "cast": (judge.Candidate("さとう"),)}
    heard = judge.audio_layout(rows, window, profile.samples)

    def joined(pair):
        return f"=== system ===\n{pair[0]}\n=== user ===\n{pair[1]}"

    for name, options in (("alone", {}), ("shape", {"shape": True}), ("shape-heard", {"shape": True, "audio": heard})):
        yield (f"speaker-judge-{name}.txt",
               lambda o=options: joined(judge.judge_prompt(rows, window, marks=False, profile=profile, **common, **o)),
               JUDGE_VERSION)
    # a correct-only task's pass 1: the shape section says what pass 2 weighs without a translation
    yield ("speaker-judge-shape-correct.txt",
           lambda: joined(judge.judge_prompt(rows, window, marks=False, profile=profile, shape=True, correct_only=True,
                                             **common)),
           JUDGE_VERSION)
    # a cloud task's rows: the tracks are its transcription's speakers
    cloud = [replace(row, group_key=f"S0{row.speaker.track + 1}") for row in rows]
    cloud_profile = judge.profiles(cloud, window)
    yield ("speaker-judge-cloud-shape-heard.txt",
           lambda: joined(judge.judge_prompt(cloud, window, marks=False, profile=cloud_profile, shape=True,
                                             audio=judge.audio_layout(cloud, window, cloud_profile.samples), **common)),
           JUDGE_VERSION)
    lines = [proofread.Line(2.0, "そうだね", f"行{n}", f"译{n}", None, "") for n in range(1, 6)]
    context = ContextPack(general_context={"global_summary": "雑談", "entities": [{"name": "佐藤", "zh": "佐藤"}]},
                          user_requests=("歌名用《》括起来",))
    yield ("proofread.txt",
           lambda: joined(proofread.proofread_prompt(proofread.Window("0002", (2, 3), before=(0, 1), after=(4,)), lines,
                                                     title="clip", context_pack=context,
                                                     style=proofread.style_text(_factory_style(), ""),
                                                     kb_terms="# 佐藤\nさとう|佐藤|")),
           PROOFREAD_VERSION)


def test_goldens_match(request) -> None:
    regenerate = request.config.getoption("--regenerate-goldens")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    cases = list(_cases())
    entries = {version: dict(manifest.get(version, {})) for version in dict.fromkeys(case[2] for case in cases)
               if version is not None}
    mismatched = []
    for filename, render, version in cases:
        rendered = render()
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        path = GOLDEN_DIR / filename
        if regenerate:
            path.write_text(rendered, encoding="utf-8", newline="\n")
        elif not path.exists():
            mismatched.append(f"{filename}: no golden yet (run --regenerate-goldens)")
            continue
        elif path.read_text(encoding="utf-8") != rendered:
            mismatched.append(f"{filename}: differs from its golden")
        if version is None:
            continue
        entry = entries[version]
        recorded = entry.get(filename)
        if recorded is not None and recorded != digest:
            mismatched.append(
                f"{filename}: content changed under an existing prompt version "
                f"({version}) -- bump the version rather than rewriting "
                "history"
            )
        entry[filename] = digest
    # The ledger is written only on a clean regenerate. Writing it before the
    # assert would let the discipline be laundered by running the command
    # twice: the first run reports the violation but records the new hashes,
    # so the second run finds them matching and passes.
    if regenerate and not mismatched:
        manifest.update({version: entry for version, entry in entries.items() if entry})
        MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    assert not mismatched, "\n".join(mismatched)


def test_manifest_is_append_only_per_version() -> None:
    """A bump may repeat an unchanged hash; a version may not change one."""

    if not MANIFEST.exists():
        pytest.skip("no manifest yet")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for version in (PROMPT_VERSION, GROUPED_PROMPT_VERSION, TRACKED_PROMPT_VERSION):
        assert version in manifest, f"{version} has no manifest entry -- run --regenerate-goldens"
    for version, entry in manifest.items():
        assert entry, f"{version} has an empty entry"
        for filename, digest in entry.items():
            assert len(digest) == 64, f"{version}/{filename}: not a sha256"


def test_regenerate_does_not_record_a_violating_run(tmp_path, monkeypatch) -> None:
    """A failing --regenerate-goldens must leave the ledger untouched.

    Writing the manifest before the assert let the discipline be laundered:
    the first run reported "bump the version" but recorded the new hashes, so
    an identical second run passed.
    """

    import finesub.llm.prompt_compose as pc
    import test.test_llm_prompt_goldens as mod

    golden_dir = tmp_path / "goldens"
    golden_dir.mkdir()
    manifest = golden_dir / "manifest.json"
    monkeypatch.setattr(mod, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(mod, "MANIFEST", manifest)
    monkeypatch.setattr(
        mod, "SHIPPED_COMBINATIONS", {"text-none-high": ("text", "none", "quality")}
    )
    monkeypatch.setattr(mod, "VARIANTS", ("capableC",))

    class _Config:
        @staticmethod
        def getoption(_name):
            return True

    class _Request:
        config = _Config()

    # First regenerate: clean, records the baseline.
    mod.test_goldens_match(_Request())
    baseline = json.loads(manifest.read_text(encoding="utf-8"))
    assert baseline[pc.PROMPT_VERSION]

    # Now change the rendered content without bumping the version.
    monkeypatch.setattr(mod, "_render", lambda *a, **k: "different content")
    with pytest.raises(AssertionError, match="content changed under an existing"):
        mod.test_goldens_match(_Request())

    # The ledger must still hold the pre-change hashes, so a second identical
    # run reports the same violation instead of silently passing.
    assert json.loads(manifest.read_text(encoding="utf-8")) == baseline
    with pytest.raises(AssertionError, match="content changed under an existing"):
        mod.test_goldens_match(_Request())
