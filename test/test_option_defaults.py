"""Where an option's default lives, and that the layers still agree.

`README_DEV.md` -> 开发原则 states the chain: CLI args -> project config ->
global config -> front-end default -> backend default, with the backend as the
single source of truth and a front end overriding only as a registered
exception.

Today's tree does not fully implement that, and this file is the ratchet rather
than a wish: `_ARGPARSE_CARRIES_A_COPY` lists every option whose argparse entry
still repeats the backend's answer. **That list may only shrink.** A new option
that copies its default fails here, and an option converted to the
`None` -> resolver shape is removed from the list in the same change.

Two things static equality cannot see, so they are tested behaviourally below:
a value can be duplicated *consistently* (which is drift waiting to happen but
not yet wrong), and every layer can carry the same explicit value and thereby
override the config together -- which is what the desktop front end does today.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

from finesub import config as app_config
from finesub import pipeline
from finesub import stages
from finesub.speech.postprocessing import stabilization
from finesub.speech.recognition import vad_asr_stage as vad_asr

#: Options whose argparse entry still repeats the backend default instead of
#: passing `None` and letting a resolver answer. **Empty since 2026-08-31** --
#: all 28 were converted at once, which is why the ratchet below now reads as
#: "nothing may be added" rather than "the list may only shrink".
#:
#: ⚠ Empty does NOT mean the chain is implemented. It means one layer stopped
#: writing a second copy: an absent flag now reaches `_defaults_from_args` as
#: `None`, is omitted from the row, and `run_pipeline`'s signature answers.
#: The two middle layers are still missing -- see the strict xfails below.
_ARGPARSE_CARRIES_A_COPY: frozenset[str] = frozenset()


def _cli_defaults(monkeypatch) -> dict[str, object]:
    monkeypatch.setattr(sys, "argv", ["finesub", "clip.wav"])
    return vars(pipeline.parse_args())


def _backend_defaults() -> dict[str, object]:
    return {
        name: parameter.default
        for name, parameter in inspect.signature(pipeline.run_pipeline).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _shared(monkeypatch) -> list[str]:
    """CLI dests that name a `run_pipeline` parameter, ALIASES INCLUDED.

    ⚠ Intersecting the two name sets was this guard's blind spot: `--model`,
    `--gap` and `--separator-rate` are spelled differently on the two sides
    (`_ROW_ALIASES`), so they were never compared -- and two of them were
    carrying a duplicate default the whole time (found 2026-08-31 by the
    behavioural sweep below, not by this comparison).
    """

    cli, backend = _cli_defaults(monkeypatch), _backend_defaults()
    aliases = pipeline._ROW_ALIASES
    return sorted(name for name in cli if aliases.get(name, name) in backend)


# --- static: the copies must at least agree, and must not multiply -----------


def test_no_layer_disagrees_with_another_about_a_default(monkeypatch) -> None:
    """Duplication is debt; disagreement is a bug that ships two behaviours."""

    cli, backend = _cli_defaults(monkeypatch), _backend_defaults()
    disagreements = {
        name: (cli[name], backend[name])
        for name in _shared(monkeypatch)
        if cli[name] is not None
        and cli[name] != backend[pipeline._ROW_ALIASES.get(name, name)]
    }
    assert disagreements == {}


def test_the_list_of_argparse_copies_only_shrinks(monkeypatch) -> None:
    """A new option must pass `None` and let a backend resolver answer."""

    cli, backend = _cli_defaults(monkeypatch), _backend_defaults()
    copies = {
        name
        for name in _shared(monkeypatch)
        if cli[name] is not None
        and cli[name] == backend[pipeline._ROW_ALIASES.get(name, name)]
    }
    new = sorted(copies - _ARGPARSE_CARRIES_A_COPY)
    assert new == [], (
        "these argparse entries repeat a backend default; pass None and resolve "
        "in the backend (README_DEV -> 开发原则), or add them to the list with a reason"
    )
    stale = sorted(_ARGPARSE_CARRIES_A_COPY - copies)
    assert stale == [], (
        "these are no longer duplicated -- remove them from "
        "_ARGPARSE_CARRIES_A_COPY so the ratchet stays tight"
    )


def test_the_converted_options_really_pass_nothing(monkeypatch) -> None:
    """The shape the rest should move to, pinned so it cannot quietly regress."""

    cli = _cli_defaults(monkeypatch)
    for name in ("split_length_scale", "vad_silero_assist", "knowledge", "language"):
        assert cli[name] is None, f"{name} should reach the backend as 'unset'"


def test_an_unset_option_is_absent_from_the_row_not_present_as_none(
    monkeypatch,
) -> None:
    """The behavioural half of the ratchet above.

    `None` in the namespace is only half the shape: `_defaults_from_args` has
    to DROP it, because a key present with value `None` would still be passed
    to `run_pipeline` and would beat the signature default. Static equality
    cannot see the difference -- this walks every shared option and checks the
    row that a manifest inherits.
    """

    monkeypatch.setattr(sys, "argv", ["finesub", "clip.wav"])
    args = pipeline.parse_args()
    row = pipeline._defaults_from_args(args, single=True)
    backend = _backend_defaults()

    # `stage` and `extra_info` are rules over the row rather than row values
    # (`_defaults_from_args` composes both), so they are written deliberately.
    composed = {"stage", "extra_info"}
    leaked = sorted(
        name
        for name in _shared(monkeypatch)
        if name not in composed and getattr(args, name, "missing") is None
        and name in row
    )
    assert leaked == [], (
        "these reach run_pipeline as an explicit None instead of being left "
        "out, so the signature default never applies"
    )
    # And what does land in the row must be something the user actually typed.
    for name in row:
        if name in composed:
            continue
        assert row[name] is not None, name
        assert name in backend or name in composed, name



def test_the_row_is_never_read_with_a_literal_fallback() -> None:
    """The third copy the ratchet above could not see.

    `_shared` compares argparse against the signature. It never looks at the
    ROW, and the URL branch reads the row directly -- it has to know
    `llm_media` and friends before `run_pipeline` runs, because they decide
    what gets downloaded. Those reads used to spell the default out
    (`opts.get("llm_media") or "audio"`), which is a third copy sitting
    outside every existing check (reviewer 2026-09-01).

    Today's values agreed, so nothing was broken; what this prevents is a
    signature change taking effect on the run but not on the download that
    fed it — the same task downloading under the old default and executing
    under the new one.

    `opt(opts, key)` is the one admissible spelling. Two things are
    deliberately NOT flagged: keys that are not `run_pipeline` parameters
    (`group`, `priority` — scheduling values with no signature to defer to),
    and a fallback that is COMPUTED rather than literal
    (`opts.get("task_artifact_dir") or paths.task_artifact_dir` derives the
    same way the backend does; it is a derivation, not a second copy of a
    default).
    """

    import ast

    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    params = {
        pipeline._ROW_ALIASES.get(name, name): name
        for name in pipeline.ALLOWED_ITEM_KEYS
    }
    known = {
        name for name in pipeline.ALLOWED_ITEM_KEYS
        if pipeline._ROW_ALIASES.get(name, name) in pipeline._PIPELINE_PARAMS
    }

    def _row_get(node) -> str:
        """The row key `node` reads, if it is `opts.get("...")`."""

        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return ""
        if node.func.attr != "get" or not isinstance(node.func.value, ast.Name):
            return ""
        if node.func.value.id != "opts" or not node.args:
            return ""
        first = node.args[0]
        return first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else ""

    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            key = _row_get(node.values[0])
            literal = len(node.values) > 1 and isinstance(node.values[1], ast.Constant)
            if key in known and literal:
                offenders.append(f"line {node.lineno}: opts.get({key!r}) or <literal>")
        if isinstance(node, ast.Call) and len(node.args) > 1:
            key = _row_get(node)
            second = node.args[1]
            if key in known and isinstance(second, ast.Constant) and second.value is not None:
                offenders.append(f"line {node.lineno}: opts.get({key!r}, <literal>)")
    assert offenders == [], (
        "these read a row key with their own default instead of deferring to "
        "`run_pipeline`'s signature; use `opt(opts, key)`: " + "; ".join(offenders)
    )


def test_opt_defers_to_the_signature() -> None:
    """And the resolver really is the signature, not a table beside it."""

    import inspect

    backend = inspect.signature(pipeline.run_pipeline).parameters
    assert pipeline.opt({}, "llm_media") == backend["llm_media"].default
    assert pipeline.opt({}, "download_video_source") == backend["download_video_source"].default
    # an alias resolves to the parameter it names
    assert pipeline.opt({}, "gap") == backend["gap_sec"].default
    # a value the row DOES carry wins, falsey or not
    assert pipeline.opt({"llm_media": "text"}, "llm_media") == "text"
    assert pipeline.opt({"llm_output_scale": 0.5}, "llm_output_scale") == 0.5

# --- behavioural: what static equality cannot see ---------------------------


def _with_config(tmp_path, monkeypatch, body: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    app_config.clear_config_cache()


def test_an_explicit_value_beats_the_config(tmp_path, monkeypatch) -> None:
    _with_config(tmp_path, monkeypatch, "[vad]\nsilero_assist = false\n")
    assert vad_asr.resolve_vad_silero_assist(True) is True


def test_the_config_beats_the_backend_default(tmp_path, monkeypatch) -> None:
    _with_config(tmp_path, monkeypatch, "[vad]\nsilero_assist = false\n")
    assert vad_asr.resolve_vad_silero_assist(None) is False
    assert vad_asr.DEFAULT_VAD_SILERO_ASSIST is True


def test_the_config_can_turn_on_the_veto_level_floor(tmp_path, monkeypatch) -> None:
    """`[stabilize] veto_level_floor` is the only way to reach this switch.

    No front end offers it and `stages` never passes a value, so if the config
    read broke, the resolver would silently keep answering the default and
    nothing else in the suite would notice -- while a documented, deliberately
    dev-only knob quietly stopped working. It changes production output (it
    withdraws 6 of 8 wrong second-model rescues and 4 of 41 right ones), which
    is why it is worth one test even though it stays out of the manual.
    """

    _with_config(tmp_path, monkeypatch, "[stabilize]\nveto_level_floor = true\n")
    assert stabilization.resolve_veto_level_floor(None) is True
    assert stabilization.DEFAULT_VETO_LEVEL_FLOOR is False

    # ... and an explicit argument still wins over it.
    assert stabilization.resolve_veto_level_floor(False) is False


def test_an_unrelated_config_leaves_the_veto_level_floor_alone(
    tmp_path, monkeypatch
) -> None:
    _with_config(tmp_path, monkeypatch, "[providers]\ntavily = false\n")
    assert (
        stabilization.resolve_veto_level_floor(None)
        is stabilization.DEFAULT_VETO_LEVEL_FLOOR
    )


def test_everything_unset_lands_on_the_backend_default(tmp_path, monkeypatch) -> None:
    _with_config(tmp_path, monkeypatch, "[providers]\ntavily = false\n")
    assert (
        vad_asr.resolve_vad_silero_assist(None) is vad_asr.DEFAULT_VAD_SILERO_ASSIST
    )


# The two layers the chain names but the tree does not implement. Both drive
# the *real* resolver and assert the target behaviour, so each fails today for
# its own concrete reason and will XPASS the moment it is implemented -- which
# is what makes `strict` a reminder to delete the marker rather than a label
# that outlives the gap. A body that just raised would xfail forever.


@pytest.mark.xfail(
    reason=(
        "no per-key project/global merge yet: resolve_config_file takes the first "
        "root that exists and uses that file whole, so there is only ever one "
        "config.toml in play. The chain's middle two layers are a target, not "
        "current behaviour -- README_DEV -> 开发原则 carries the same caveat."
    ),
    strict=True,
)
def test_a_project_config_overrides_a_global_one(tmp_path, monkeypatch) -> None:
    global_root = tmp_path / "global"
    project_root = tmp_path / "project"
    for root in (global_root, project_root):
        root.mkdir()
    (global_root / "config.toml").write_text(
        "[vad]\nsilero_assist = false\n", encoding="utf-8"
    )
    (project_root / "config.toml").write_text(
        "[segmentation]\nlength_scale = 0.8\n", encoding="utf-8"
    )
    # The project file is the one in play but says nothing about the assist, so
    # a per-key merge would fall through to the global file's `false`.
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(project_root / "config.toml"))
    monkeypatch.setenv("FINESUB_GLOBAL_CONFIG_FILE", str(global_root / "config.toml"))
    app_config.clear_config_cache()

    assert vad_asr.resolve_vad_silero_assist(None) is False


@pytest.mark.xfail(
    reason=(
        "front-end defaults have no layer of their own: a front end expresses a "
        "preference by passing an explicit value, which enters at the args layer "
        "and therefore outranks config instead of sitting below it. Implementing "
        "it means a resolver parameter for 'the front end prefers X, if nobody "
        "else said' -- which is what this calls."
    ),
    strict=True,
)
def test_a_front_end_preference_sits_below_the_config(tmp_path, monkeypatch) -> None:
    _with_config(tmp_path, monkeypatch, "[vad]\nsilero_assist = true\n")

    # A front end preferring "off" must still lose to a config saying "on".
    assert vad_asr.resolve_vad_silero_assist(None, front_end_default=False) is True


# The packaged CLI records what a run was configured with, so a desktop retry
# replays that run rather than the desktop's own (deliberately different)
# defaults. It cannot import the main package, so its recorder mirrors the
# resolver instead of calling it -- and a mirror is exactly the shape this file
# exists to police. The desktop's own suite can only pin the constant it
# expects; here both sides are importable, so compare them directly.


@pytest.mark.parametrize("difficulty", ["quality", "intermediate", "efficiency"])
def test_the_cli_record_mirrors_the_knowledge_resolver(difficulty: str) -> None:
    from finesub_bootstrap.shell import _RunPlan, Shell

    arguments = ["clip.wav", "--llm-difficulty", difficulty]
    plan = _RunPlan(
        task_id="t",
        source="clip.wav",
        stage="raw-srt",
        arguments=arguments,
        output=Path("out") / "clip" / "clip.srt",
    )

    recorded = Shell._recorded_request(plan)

    assert recorded["knowledge"] == stages.resolve_knowledge_switch(None, difficulty)


def test_that_mirror_still_lets_an_explicit_switch_through() -> None:
    """The resolver's first rule, which a two-branch mirror could drop."""

    from finesub_bootstrap.shell import _RunPlan, Shell

    plan = _RunPlan(
        task_id="t",
        source="clip.wav",
        stage="raw-srt",
        arguments=["clip.wav", "--llm-difficulty", "efficiency", "--knowledge", "update"],
        output=Path("out") / "clip" / "clip.srt",
    )

    assert Shell._recorded_request(plan)["knowledge"] == "update"
