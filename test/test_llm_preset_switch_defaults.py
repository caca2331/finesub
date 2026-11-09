"""Presets carry correction switch defaults (`[llm.presets.<id>.defaults]`).

Precedence (owner 2026-09-30): command line > active preset > the default
preset > the code defaults; no switch pins another (efficiency pinned media,
retrieval and knowledge until 2026-10-30). A value a preset supplies reaches the checkpoints only as the
value it resolves to -- exactly like the same value typed on the command line.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from finesub import config as app_config
from finesub.llm.correction_request import CODE_DEFAULTS, resolve_switches
from finesub.llm.routing import model_routes
from finesub.llm.routing.model_routes import ModelRouteConfigError, load_model_routes

STABLE = {"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}


def _routes(**presets_toml: dict) -> model_routes.ModelRouteCatalog:
    return load_model_routes(user_config=presets_toml)


def _my_preset(defaults: dict, *, active: bool = True, default_defaults: dict | None = None) -> dict:
    user: dict = {
        "presets": {
            "my": {
                "name": "mine",
                "bindings": {"correction-text/quality": "correction-capable"},
                "defaults": defaults,
            }
        }
    }
    if active:
        user["preset"] = "my"
    if default_defaults is not None:
        import tomllib

        packaged_file = Path(model_routes.__file__).with_name(model_routes.ROUTES_FILENAME)
        packaged = tomllib.loads(packaged_file.read_text(encoding="utf-8"))["presets"]["default"]
        user["presets"]["default"] = {**packaged, "defaults": default_defaults}
    return user


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Point the process at a config.toml and drop the memoized routes on both
    sides of the test."""

    def write(body: str) -> None:
        path = tmp_path / "config.toml"
        path.write_text(body, encoding="utf-8")
        monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
        app_config.clear_config_cache()
        model_routes.default_model_routes.cache_clear()

    yield write
    app_config.clear_config_cache()
    model_routes.default_model_routes.cache_clear()


# --- precedence ----------------------------------------------------------------


def test_the_command_line_beats_the_preset() -> None:
    routes = _routes(**_my_preset({"retrieval": "none"}))
    resolved = resolve_switches({"retrieval": "native"}, routes=routes)
    assert (resolved.retrieval, resolved.sources["retrieval"]) == ("native", "explicit")


def test_the_active_preset_beats_the_code_default() -> None:
    routes = _routes(**_my_preset({"retrieval": "none", "continuity": "parallel"}))
    resolved = resolve_switches({}, routes=routes)
    assert (resolved.retrieval, resolved.sources["retrieval"]) == ("none", "preset:my")
    assert (resolved.continuity, resolved.sources["continuity"]) == ("parallel", "preset:my")
    assert (resolved.media, resolved.sources["media"]) == (CODE_DEFAULTS["media"], "default")


def test_a_key_the_active_preset_lacks_falls_back_to_the_default_preset() -> None:
    routes = _routes(**_my_preset({"retrieval": "none"}, default_defaults={"media": "text", "retrieval": "native"}))
    resolved = resolve_switches({}, routes=routes)
    assert (resolved.retrieval, resolved.sources["retrieval"]) == ("none", "preset:my")
    assert (resolved.media, resolved.sources["media"]) == ("text", "preset:default")


def test_without_presets_everything_is_the_code_default() -> None:
    resolved = resolve_switches({}, routes=_routes())
    assert {key: getattr(resolved, key) for key in CODE_DEFAULTS} == CODE_DEFAULTS
    assert set(resolved.sources.values()) == {"default"}


# --- efficiency pins nothing -----------------------------------------------------


def test_efficiency_leaves_the_other_switches_to_the_chain() -> None:
    """owner 2026-10-24 / 10-30: efficiency no longer pins media and retrieval
    nor caps knowledge -- alone it is the code defaults (audio, local, read)."""

    resolved = resolve_switches({"difficulty": "efficiency"}, routes=_routes())
    assert (resolved.media, resolved.retrieval, resolved.knowledge) == ("audio", "local", "read")
    assert {resolved.sources[key] for key in ("media", "retrieval", "knowledge")} == {"default"}
    preset = resolve_switches({"difficulty": "efficiency"},
                              routes=_routes(**_my_preset({"media": "audio", "knowledge": "update"})))
    assert (preset.media, preset.knowledge) == ("audio", "update")
    assert preset.sources["media"] == preset.sources["knowledge"] == "preset:my"


def test_efficiency_with_audio_and_search_is_a_profile_and_may_update_the_base() -> None:
    from finesub.llm.correction_request import CorrectionRequest, resolved_request
    from finesub.llm.routing.profiles import resolve_profile

    profile = resolve_profile("audio", "local", "efficiency")
    assert (profile.correction_media, profile.retrieval, profile.difficulty) == ("audio", "local", "efficiency")
    assert resolved_request(
        CorrectionRequest(stable_json=Path("s.json"), mode="plan", profile=profile, knowledge="update")
    ).knowledge == "update"


def test_a_preset_may_default_efficiency_with_audio_and_update() -> None:
    routes = _routes(**_my_preset({"difficulty": "efficiency", "media": "audio", "knowledge": "update"}))
    assert dict(routes.presets["my"].defaults) == {"difficulty": "efficiency", "media": "audio", "knowledge": "update"}


# --- the preset table itself ---------------------------------------------------


@pytest.mark.parametrize(
    ("defaults", "message"),
    [
        ({"fast": "on"}, "defaults key 'fast'"),
        ({"retrieval": "web"}, "defaults['retrieval']"),
    ],
)
def test_a_bad_defaults_table_is_refused_at_load(defaults, message) -> None:
    with pytest.raises(ModelRouteConfigError, match=message.replace("[", r"\[")):
        _routes(**_my_preset(defaults))


def test_defaults_stay_out_of_the_routing_identity_digest() -> None:
    """A default reaches the checkpoints only as the value it resolves to; in
    the routing digest it would invalidate every session checkpoint -- even of
    a run whose command line overrode it."""

    plain = _routes(**_my_preset({}))
    with_default = _routes(**_my_preset({"retrieval": "none"}))
    assert with_default.routing_identity_digest == plain.routing_identity_digest
    assert with_default.advisory_digest != plain.advisory_digest


# --- front ends -------------------------------------------------------------------


def _plan(tmp_path: Path, name: str, *flags: str) -> Path:
    from finesub.llm import correction_translation

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(STABLE), encoding="utf-8")
    prompts = tmp_path / f"{name}-prompts"
    art = tmp_path / f"{name}-art"
    argv = [str(stable), "--media", "text", "--prompt-dir", str(prompts), "--task-artifact-dir", str(art), *flags]
    old = sys.argv
    sys.argv = ["correction_translation", *argv]
    try:
        assert correction_translation.main() == 0
    finally:
        sys.argv = old
    return tmp_path / name


def test_the_cli_takes_the_preset_value_and_records_where_it_came_from(tmp_path, config) -> None:
    config('[llm]\npreset = "my"\n[llm.presets.my]\nname = "mine"\n'
           '[llm.presets.my.bindings]\n"correction-text/quality" = "correction-capable"\n'
           '[llm.presets.my.defaults]\nretrieval = "none"\n')
    _plan(tmp_path, "viapreset")
    record = json.loads((tmp_path / "viapreset-art" / "resolved-switches.json").read_text(encoding="utf-8"))
    assert record["switches"]["retrieval"] == "none"
    assert record["sources"]["retrieval"] == "preset:my"
    assert record["sources"]["media"] == "explicit"
    assert record["active_preset"] == "my"
    assert (record["switches"]["parallel_windows"], record["sources"]["parallel_windows"]) == (1, "default")


def test_a_preset_value_and_the_same_value_typed_give_the_same_prompts(tmp_path, config) -> None:
    """Acceptance 4: once resolved, nothing downstream can tell the two apart."""

    config('[llm]\npreset = "my"\n[llm.presets.my]\nname = "mine"\n'
           '[llm.presets.my.bindings]\n"correction-text/quality" = "correction-capable"\n'
           '[llm.presets.my.defaults]\nretrieval = "none"\n')
    _plan(tmp_path, "viapreset")
    _plan(tmp_path, "typed", "--retrieval", "none")
    via_preset = sorted((tmp_path / "viapreset-prompts").iterdir())
    typed = sorted((tmp_path / "typed-prompts").iterdir())
    assert [p.name for p in via_preset] == [p.name for p in typed]
    for left, right in zip(via_preset, typed):
        assert left.read_bytes() == right.read_bytes(), left.name


def test_the_pipeline_resolves_unset_switches_through_the_preset(config) -> None:
    from finesub.stages import resolve_llm_switches

    config('[llm]\npreset = "my"\n[llm.presets.my]\nname = "mine"\n'
           '[llm.presets.my.bindings]\n"correction-text/quality" = "correction-capable"\n'
           '[llm.presets.my.defaults]\nmedia = "text"\nretrieval = "none"\n')
    common = dict(llm_media=None, llm_retrieval=None, llm_difficulty=None, llm_continuity=None,
                  knowledge=None, llm_output_scale=1.0)
    switches, profile = resolve_llm_switches("final-srt", **common)
    assert (profile.correction_media, profile.retrieval) == ("text", "none")
    assert switches.sources["media"] == "preset:my"
    # A run that stops before the LLM stage does not read the presets at all.
    switches, profile = resolve_llm_switches("raw-srt", **common)
    assert (profile.correction_media, profile.retrieval) == ("audio", "local")


def test_reference_ingest_does_not_read_preset_switch_defaults(config) -> None:
    """D5: ingest's own workflow presets decide its switches; the routing
    preset still picks the models but not the switch values."""

    from finesub.workflows.reference_ingest import ResolvedSettings, TaskRow, build_task

    config('[llm]\npreset = "my"\n[llm.presets.my]\nname = "mine"\n'
           '[llm.presets.my.bindings]\n"correction-text/quality" = "correction-capable"\n'
           '[llm.presets.my.defaults]\ndifficulty = "quality"\nretrieval = "none"\n')
    task = build_task(TaskRow(srt="clip.srt", media=""), None, ResolvedSettings())
    assert (task.profile.difficulty, task.profile.retrieval) == ("intermediate", "local")


FAST_DEFAULTS = {"difficulty": "efficiency", "media": "text", "retrieval": "none", "continuity": "parallel",
                 "parallel_windows": 4}
EFFICIENCY_CELLS = {("correction-text", "efficiency"), ("research", "efficiency")}


def test_the_shipped_presets_defaults() -> None:
    """owner 2026-09-30: the subscription-only preset searches with agy's own
    tool (`local` needs keys that preset's users do not want to manage);
    owner 2026-10-01: the two rough-pass presets; the rest default nothing."""

    routes = _routes()
    assert dict(routes.presets["agy"].defaults) == {"retrieval": "native"}
    assert dict(routes.presets["default"].defaults) == {}
    assert dict(routes.presets["agy-hybrid"].defaults) == {}
    assert dict(routes.presets["agy-fast"].defaults) == FAST_DEFAULTS
    assert dict(routes.presets["fast"].defaults) == FAST_DEFAULTS


def test_the_fast_presets_leave_the_other_presets_alone() -> None:
    """owner 2026-10-01: the quick path lives only in the new presets. `default`
    binds its correction efficiency cells only to keep them on the lites now
    that intermediate is on the capable group."""

    routes = _routes()
    for preset_id in ("agy", "agy-hybrid"):
        assert not EFFICIENCY_CELLS & set(routes.presets[preset_id].bindings), preset_id
    assert ("research", "efficiency") not in routes.presets["default"].bindings


@pytest.mark.parametrize("preset_id", ["fast", "agy-fast"])
def test_a_fast_preset_alone_still_runs_text_only_without_retrieval(runtime_preset, preset_id) -> None:
    """Efficiency pins nothing any more, so the fast presets write media and
    retrieval themselves; the command line still wins."""

    runtime_preset(preset_id)
    resolved = resolve_switches({}, routes=_routes())
    assert (resolved.difficulty, resolved.media, resolved.retrieval) == ("efficiency", "text", "none")
    assert resolved.sources["media"] == resolved.sources["retrieval"] == f"preset:{preset_id}"
    assert resolve_switches({"media": "audio"}, routes=_routes()).media == "audio"


def test_agy_fast_with_audio_falls_back_to_its_capable_agy_group() -> None:
    """Only its text efficiency cell is bound to the fast group: `--llm-media
    audio` walks up to the intermediate cell's group, still agy-only."""

    group, cell = _routes().resolve_binding("agy-fast", "correction-mm", "efficiency")
    assert group.id == "agy-only-capable" and cell.variant == "basicA"


def test_agy_fast_is_agy_plus_its_efficiency_cells_and_never_an_api() -> None:
    """A cell a preset leaves out falls back to `default`, not to `agy`, so
    `agy-fast` copies `agy`'s bindings -- and must keep them equal. Every
    group it binds is agent-only: a spent subscription stops the run rather
    than moving it to an API (owner 2026-10-01)."""

    routes = _routes()
    fast = dict(routes.presets["agy-fast"].bindings)
    assert {cell: fast.pop(cell) for cell in EFFICIENCY_CELLS} == dict.fromkeys(EFFICIENCY_CELLS, "agy-only-fast")
    assert fast == dict(routes.presets["agy"].bindings)
    assert routes.presets["agy-fast"].test_target_id == routes.presets["agy"].test_target_id
    bound = {
        target
        for group_id in routes.presets["agy-fast"].bindings.values()
        for target in routes.model_groups[group_id].target_ids
    }
    assert all(routes.targets[target].backend == "local_agent" for target in bound), bound


def test_fast_runs_on_the_default_presets_lites() -> None:
    routes = _routes()
    assert dict(routes.presets["fast"].bindings) == {("correction-text", "efficiency"): "correction-basic"}
    assert routes.presets["fast"].test_target_id == routes.presets["default"].test_target_id
    # What `default` itself falls back to at that cell.
    group, _cell = routes.resolve_binding("default", "correction-text", "efficiency")
    assert group.id == "correction-basic"


# --- parallel_windows -------------------------------------------------------------


def test_a_presets_parallel_windows_applies_under_parallel_only() -> None:
    routes = _routes(**_my_preset({"continuity": "parallel", "parallel_windows": 3}))
    resolved = resolve_switches({}, routes=routes)
    assert (resolved.parallel_windows, resolved.sources["parallel_windows"]) == (3, "preset:my")
    # The command line said serial: the preset's value is no reason to fail.
    resolved = resolve_switches({"continuity": "serial"}, routes=routes)
    assert resolved.parallel_windows == 1
    assert resolved.sources["parallel_windows"].startswith("default(preset:my=3")
    # An explicit value always wins.
    assert resolve_switches({"parallel_windows": 2}, routes=routes).parallel_windows == 2


def test_parallel_windows_with_continuity_from_the_default_preset() -> None:
    """The active preset writes the count; continuity comes from `default`."""

    routes = _routes(**_my_preset({"parallel_windows": 3}, default_defaults={"continuity": "parallel"}))
    resolved = resolve_switches({}, routes=routes)
    assert (resolved.continuity, resolved.parallel_windows) == ("parallel", 3)
    serial = _routes(**_my_preset({"parallel_windows": 3}))
    assert resolve_switches({}, routes=serial).parallel_windows == 1


@pytest.mark.parametrize(
    ("defaults", "message"),
    [
        ({"parallel_windows": 0}, "integer >= 1"),
        ({"parallel_windows": "4"}, "integer >= 1"),
        ({"parallel_windows": True}, "integer >= 1"),
        ({"parallel_windows": 4, "continuity": "serial"}, "only applies under"),
    ],
)
def test_a_bad_parallel_windows_is_refused_at_load(defaults, message) -> None:
    with pytest.raises(ModelRouteConfigError, match=message):
        _routes(**_my_preset(defaults))


# --- knowledge read / update ----------------------------------------------------


def test_read_reads_the_base_but_only_update_asks_for_feedback(tmp_path, config) -> None:
    """owner 2026-10-01: `read` (the default, formerly `collect`) reads and
    injects the base and collects nothing; only `update` asks the windows and
    the preparation task for `<task_update_feedback>`."""

    config("")
    _plan(tmp_path, "read", "--knowledge", "read")
    _plan(tmp_path, "update", "--knowledge", "update")
    read_text = "".join(p.read_text(encoding="utf-8") for p in (tmp_path / "read-prompts").iterdir())
    update_prompts = tmp_path / "update-prompts"
    # The feedback contract's own opening line is what asks for the block;
    # elsewhere the prompts only ever mention it as "if asked".
    marker = "本次任务开启了知识库反馈采集"
    assert marker not in read_text
    # Each of the two tasks asks for its own block under update.
    for name in ("research-prepare.txt", "correction-0001.txt"):
        assert marker in (update_prompts / name).read_text(encoding="utf-8"), name


# --- --llm-preset ------------------------------------------------------------------


@pytest.fixture
def runtime_preset():
    from finesub.llm.routing.model_routes import install_runtime_preferred, install_runtime_preset

    yield install_runtime_preset
    install_runtime_preset(None)
    install_runtime_preferred(None)


def test_llm_preset_picks_the_active_preset_over_the_config(runtime_preset) -> None:
    runtime_preset("fast")
    routes = _routes(**_my_preset({"retrieval": "none"}))  # config says `my`
    assert routes.active_preset_id == "fast"
    resolved = resolve_switches({}, routes=routes)
    assert (resolved.difficulty, resolved.knowledge, resolved.active_preset) == ("efficiency", "read", "fast")
    assert resolved.parallel_windows == 4


def test_an_unknown_llm_preset_names_the_known_ones(runtime_preset) -> None:
    runtime_preset("nope")
    with pytest.raises(ModelRouteConfigError, match=r"--llm-preset 'nope' names no known preset; known: .*agy-fast"):
        _routes()


def test_llm_model_applies_on_top_of_the_chosen_preset(runtime_preset) -> None:
    """The preset is chosen first; `--llm-model` then rebinds its cells."""

    from finesub.llm.routing.model_routes import install_runtime_preferred

    runtime_preset("agy-fast")
    install_runtime_preferred({"correction-text": "correction-basic"})
    routes = _routes()
    group, _cell = routes.resolve_binding("agy-fast", "correction-text", "efficiency")
    assert group.id == "correction-basic"


def test_the_cli_takes_llm_preset_and_records_it(tmp_path, config, runtime_preset) -> None:
    config("")
    _plan(tmp_path, "viaflag", "--llm-preset", "fast")
    record = json.loads((tmp_path / "viaflag-art" / "resolved-switches.json").read_text(encoding="utf-8"))
    assert record["active_preset"] == "fast"
    assert record["switches"]["difficulty"] == "efficiency"
    assert record["sources"]["difficulty"] == "preset:fast"
    assert (record["switches"]["parallel_windows"], record["sources"]["parallel_windows"]) == (4, "preset:fast")


def test_a_manifest_row_cannot_choose_a_preset() -> None:
    """Process-level only: the route table is one per process, like
    `--llm-model`. A row naming one is refused, not silently ignored."""

    from finesub.pipeline import merge_item_options

    with pytest.raises(ValueError, match="cannot choose an LLM preset.*--llm-preset on the command line"):
        merge_item_options({"source": "a.wav", "llm_preset": "fast"}, {})


def test_an_explicit_window_count_stands_under_serial() -> None:
    resolved = resolve_switches({"continuity": "serial", "parallel_windows": 3}, routes=_routes())
    assert (resolved.parallel_windows, resolved.sources["parallel_windows"]) == (3, "explicit")


def test_a_caller_given_vector_is_recorded_as_such(tmp_path, runtime_preset) -> None:
    """Reference ingest hands the task a whole vector and no sources."""

    from finesub.llm import correction_translation as ct
    from finesub.llm.correction_request import CorrectionRequest
    from finesub.llm.routing.profiles import resolve_profile

    runtime_preset("fast")
    request = CorrectionRequest(
        stable_json=Path("s.json"), profile=resolve_profile("audio", "local", "quality"), knowledge="read"
    )
    ct._record_switches(request, tmp_path)
    record = json.loads((tmp_path / "resolved-switches.json").read_text(encoding="utf-8"))
    assert set(record["sources"].values()) == {"caller"}
    assert record["active_preset"] == "fast"


class _Stop(Exception):
    pass


def _raise_stop(*_args, **_kwargs):
    raise _Stop


def test_knowledge_update_takes_llm_preset(monkeypatch, runtime_preset) -> None:
    from finesub.llm.knowledge import update
    from finesub.llm.routing import model_routes

    monkeypatch.setattr(update, "run_knowledge_update", _raise_stop)
    monkeypatch.setattr(sys, "argv", ["update", "x.srt", "--llm-preset", "fast"])
    with pytest.raises(_Stop):
        update.main()
    assert model_routes._RUNTIME_PRESET == "fast"


def test_reference_ingest_takes_llm_preset(monkeypatch, runtime_preset) -> None:
    from finesub.llm.routing import model_routes
    from finesub.workflows import reference_ingest

    monkeypatch.setattr(reference_ingest, "gather_tasks", _raise_stop)
    monkeypatch.setattr(sys, "argv", ["ingest", "--task", "a.srt|a.wav", "--llm-preset", "fast"])
    with pytest.raises(_Stop):
        reference_ingest._main()
    assert model_routes._RUNTIME_PRESET == "fast"


@pytest.mark.parametrize("command", [["verify"], ["repair"], ["ingest", "--subject", "x", "--material", "m.txt"]])
def test_knowledge_maintain_takes_llm_preset(monkeypatch, runtime_preset, command) -> None:
    from finesub.llm.knowledge import maintain
    from finesub.llm.routing import model_routes

    for name in ("_cmd_verify", "_cmd_repair", "_cmd_ingest"):
        monkeypatch.setattr(maintain, name, _raise_stop)
    with pytest.raises(_Stop):
        maintain._main([*command, "--llm-preset", "fast"])
    assert model_routes._RUNTIME_PRESET == "fast"

