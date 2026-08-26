from __future__ import annotations

from dataclasses import fields
import hashlib
from pathlib import Path
import tomllib

import pytest

from finesub.llm.routing.config import LLMRole, default_role_configs, role_config_for
from finesub.llm.routing.model_routes import (
    DIFFICULTIES,
    TASK_GROUP_IDS,
    ModelRouteConfigError,
    default_model_routes,
    load_model_routes,
)
from finesub.llm.routing.model_catalog import load_model_catalog, merge_catalogs


# The declared expansion order (owner edit 2026-08-13): correction uses
# 3.7 → 3.6 → 3.5; research/knowledge use 3.6 → 3.5 → 3.7. Paid 3.7 is
# the capable tail for both.
EXPECTED_GROUPS = {
    "correction-capable": [
        "gemini-free-3_7-flash",
        "gemini-free-3_6-flash",
        "gemini-free-3_5-flash",
        "gemini-paid-3_7-flash",
    ],
    "correction-basic": [
        "gemini-free-3_5-flash-lite",
        "gemini-paid-3_5-flash-lite",
    ],
    "research-default": [
        "gemini-free-3_6-flash",
        "gemini-free-3_5-flash",
        "gemini-free-3_7-flash",
        "gemini-paid-3_7-flash",
    ],
    "knowledge-capable": [
        "gemini-free-3_6-flash",
        "gemini-free-3_5-flash",
        "gemini-free-3_7-flash",
        "gemini-paid-3_7-flash",
    ],
    "lightweight-default": [
        "gemini-free-3_5-flash-lite",
        "gemini-paid-3_5-flash-lite",
    ],
    "gemini-native-search": [
        "gemini-free-2_5-flash",
        "gemini-paid-3_7-flash",
    ],
    # The agy preset's two groups. Opus is text-only on purpose: it sits ahead
    # of the media target so text windows prefer it and multimodal ones skip
    # past it to the Gemini that agy fronts.
    "agy-capable": [
        "gemini-free-3_7-flash",
        "local-agy-opus-4_6",
        "local-agy-media-gemini-3_7-flash",
        "local-agy-native-gemini-3_7-flash",
    ],
    "agy-basic": [
        "gemini-free-3_5-flash",
        "local-agy-opus-4_6",
        "local-agy-media-gemini-3_7-flash",
        "local-agy-native-gemini-3_7-flash",
    ],
    # Tool-protocol only, so bound by nothing out of the box: `dsh --profile
    # headless` takes its task from its command line, which a whole window
    # does not fit into.
    "dsh-capable": [
        "local-dsh-deepseek-v4-pro",
        "local-dsh-deepseek-v4-flash",
        "local-dsh-native-deepseek-v4-pro",
    ],
    "dsh-basic": [
        "local-dsh-deepseek-v4-flash",
        "local-dsh-native-deepseek-v4-pro",
    ],
    # Reserved and bound by nothing: a conversational worker claims tasks
    # instead of being called, so it may only ever be alone here.
    "conversational-agent": ["conversational-agent"],
}


def test_packaged_groups_expand_to_the_declared_order() -> None:
    default_model_routes.cache_clear()
    routes = default_model_routes()

    assert {
        group_id: list(group.target_ids)
        for group_id, group in routes.model_groups.items()
    } == EXPECTED_GROUPS


# Every local-agent target, and what each one promises. No packaged group
# binds the Codex/Claude rows any more, so nothing else in the suite would
# notice a rename or a deletion -- and the manuals tell people to bind these
# ids by hand (quick model selection), which makes them public API.
EXPECTED_LOCAL_AGENT_TARGETS = {
    "local-codex-completion-gpt-5_6-luna": ("LOCAL_CODEX", ""),
    "local-codex-native-gpt-5_6-luna": ("LOCAL_CODEX", "web_search"),
    "local-codex-completion-gpt-5_6-sol": ("LOCAL_CODEX", ""),
    "local-codex-native-gpt-5_6-sol": ("LOCAL_CODEX", "web_search"),
    "local-claude-completion-opus-5": ("LOCAL_CLAUDE", ""),
    "local-claude-native-opus-5": ("LOCAL_CLAUDE", "web_search"),
    "local-claude-completion-sonnet-5": ("LOCAL_CLAUDE", ""),
    "local-claude-native-sonnet-5": ("LOCAL_CLAUDE", "web_search"),
    "local-claude-completion-haiku-4_5": ("LOCAL_CLAUDE", ""),
    "local-claude-native-haiku-4_5": ("LOCAL_CLAUDE", "web_search"),
    "local-agy-media-gemini-3_7-flash": ("LOCAL_AGY", ""),
    "local-agy-native-gemini-3_7-flash": ("LOCAL_AGY", "search_web"),
    "local-agy-opus-4_6": ("LOCAL_AGY", ""),
    "local-dsh-deepseek-v4-flash": ("LOCAL_DSH", ""),
    "local-dsh-deepseek-v4-pro": ("LOCAL_DSH", ""),
    "local-dsh-native-deepseek-v4-pro": ("LOCAL_DSH", "web_search"),
}


def test_every_local_agent_target_stays_declared_and_keeps_its_promise() -> None:
    """Unbound is not unused: these are ids people write in their own config.

    The `native` half of each pair exists because `retrieval=native` filters on
    the *target's* declared search tool before it ever consults the fact, so a
    model that can ground still needs two targets to express "this call may".
    """

    default_model_routes.cache_clear()
    routes = default_model_routes()
    declared = {
        target_id: (
            routes.target_fact(target_id).provider_tier,
            routes.target_profile(target_id).native_search_tool,
        )
        for target_id, target in routes.targets.items()
        if target.backend == "local_agent"
    }

    assert declared == EXPECTED_LOCAL_AGENT_TARGETS
    # The fact has to agree, or the two-layer filter drops the native target.
    for target_id, (_tier, tool) in EXPECTED_LOCAL_AGENT_TARGETS.items():
        if tool:
            assert routes.target_fact(target_id).supports_native_search, target_id
    # agy grounds through a *second* project (2026-08-15). Opus is text-only and
    # still declares nothing; the Gemini agy fronts backs both the media target
    # and the search-entitled one, so its fact says it can ground while only the
    # native target is allowed to.
    assert routes.target_fact("local-agy-opus-4_6").supports_native_search is False
    assert (
        routes.target_fact("local-agy-native-gemini-3_7-flash").supports_native_search
        is True
    )
    assert routes.target_profile("local-agy-media-gemini-3_7-flash").native_search_tool == ""


def test_role_configs_are_derived_from_the_default_preset() -> None:
    routes = default_model_routes()
    configs = default_role_configs()
    group_by_role = {
        LLMRole.AUDIO_MULTIMODAL: "correction-capable",
        LLMRole.GENERAL_CAPABLE: "research-default",
        LLMRole.LIGHTWEIGHT: "lightweight-default",
        LLMRole.LIGHTWEIGHT_MULTIMODAL: "lightweight-default",
    }

    for role, group_id in group_by_role.items():
        config = configs[role]
        assert [
            endpoint.target_id for endpoint in config.endpoint_chain
        ] == EXPECTED_GROUPS[group_id]
        assert (
            config.test_endpoint.target_id
            == routes.presets["default"].test_target_id
        )
        # The knob is per cell: research/knowledge think high at difficulty
        # quality (owner tuning 2026-08-12), the rest stay medium.
        assert config.thinking_level == (
            "high" if group_id in ("research-default", "knowledge-capable") else "medium"
        )

    # Cell knobs land on the config: correction intermediate binds the basic
    # group with the basicB variant; efficiency thinks low.
    intermediate = role_config_for("correction-mm", "intermediate")
    assert intermediate.model_group_id == "correction-basic"
    assert intermediate.variant == "basicB" and intermediate.thinking_level == "medium"
    efficiency = role_config_for("correction-text", "efficiency")
    assert efficiency.variant == "basicB" and efficiency.thinking_level == "low"


def test_runtime_facts_remain_independent_per_provider_tier() -> None:
    routes = default_model_routes()
    free = routes.target_fact("gemini-free-3_7-flash")
    paid = routes.target_fact("gemini-paid-3_7-flash")

    assert free.api_model_id == paid.api_model_id
    assert free.fact_id != paid.fact_id
    assert (free.rpm, free.tpm, free.rpd) != (paid.rpm, paid.tpm, paid.rpd)


def test_same_model_can_have_different_capabilities_per_tier(tmp_path: Path) -> None:
    path = tmp_path / "facts.psv"
    path.write_text(
        "|".join(
            (
                "fact_id",
                "provider_tier",
                "provider_kind",
                "display_name",
                "api_model_id",
                "max_input_tokens",
                "max_output_tokens",
                "supports_audio",
                "supports_video",
                "supports_native_search",
                "thinking",
                "rpm",
                "tpm",
                "rpd",
                "tpd",
                "is_free",
                "quality_score",
            )
        )
        + "\n"
        + "free|FREE|gemini|X|provider/x|100|20|false|false|false|false|1|10|2|-1|true|40\n"
        + "paid|PAID|gemini|X|provider/x|200|40|true|true|true|xhigh,medium,low|9|90|-1|-1|false|80\n",
        encoding="utf-8",
    )

    free, paid = load_model_catalog(path)
    assert free.api_model_id == paid.api_model_id
    assert free.supports_audio is False and paid.supports_audio is True
    assert free.supports_native_search is False and paid.supports_native_search is True
    # Per-model thinking mapping (owner design 2026-08-11): none vs an
    # explicit high,med,low -> provider-value triple.
    assert free.thinking_levels is None
    assert paid.thinking_levels == ("xhigh", "medium", "low")
    assert free.quality_score == 40
    assert paid.quality_score == 80


def test_catalog_rejects_out_of_range_quality_score(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_catalog.psv"
    data = source.read_text(encoding="utf-8").replace("|false|70", "|false|101", 1)
    path = tmp_path / "facts.psv"
    path.write_text(data, encoding="utf-8")

    with pytest.raises(ValueError, match="quality_score must be in"):
        load_model_catalog(path)


def test_quality_scores_are_advisory_data_on_every_fact() -> None:
    routes = default_model_routes()

    # Same model, different provider tier: the capability is the model's, so
    # the advisory score matches even though quotas differ.
    assert (
        routes.target_fact("gemini-free-3_7-flash").quality_score
        == routes.target_fact("gemini-paid-3_7-flash").quality_score
        == 75
    )
    assert all(
        fact.quality_score in range(0, 101) for fact in routes.facts.values()
    )


def test_quality_floor_warnings_name_and_quantify_offenders() -> None:
    from finesub.llm.routing.model_catalog import catalog_by_fact_id, quality_floor_warnings

    facts = catalog_by_fact_id()
    members = [facts["gemini-free-3_5-flash"], facts["gemini-free-3_5-flash-lite"]]

    warnings = quality_floor_warnings(
        members,
        floor_score=70,
        reference_model="3.5 Flash",
        owner="correction-mm/quality",
    )
    assert len(warnings) == 1
    assert "gemini-free-3_5-flash-lite" in warnings[0]
    assert "quality_score=60" in warnings[0]
    assert "下限 70" in warnings[0]
    assert "3.5 Flash" in warnings[0]

    # At-or-above floor stays silent: the score is advisory, not a filter.
    assert (
        quality_floor_warnings(
            members,
            floor_score=60,
            reference_model="3.5 Flash Lite",
            owner="research/quality",
        )
        == []
    )


def test_route_loader_rejects_tier_enablement_mismatch(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    data = source.read_text(encoding="utf-8").replace(
        'enabled_by = "GEMINI_FREE"', 'enabled_by = "GEMINI_PAID"', 1
    )
    path = tmp_path / "routes.toml"
    path.write_text(data, encoding="utf-8")

    with pytest.raises(ModelRouteConfigError, match="does not match fact tier"):
        load_model_routes(path)


@pytest.mark.parametrize(
    "old,new",
    [
        ('fact = "local-codex-gpt-5_6-luna"', 'fact = "gemini-free-3_5-flash"'),
        ('fact = "gemini-free-3_7-flash"', 'fact = "local-codex-gpt-5_6-luna"'),
    ],
)
def test_route_loader_rejects_backend_fact_provider_mismatch(
    tmp_path: Path, old: str, new: str
) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    data = source.read_text(encoding="utf-8").replace(old, new, 1)
    path = tmp_path / "routes.toml"
    path.write_text(data, encoding="utf-8")

    with pytest.raises(ModelRouteConfigError, match="cannot use fact provider tier"):
        load_model_routes(path)


def test_route_loader_rejects_native_tool_on_incapable_fact(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    data = source.read_text(encoding="utf-8").replace(
        'execution_profile = "gemini-default"',
        'execution_profile = "gemini-google-search"',
        1,
    )
    path = tmp_path / "routes.toml"
    path.write_text(data, encoding="utf-8")

    with pytest.raises(ModelRouteConfigError, match="runtime fact does not support"):
        load_model_routes(path)


@pytest.mark.parametrize(
    "old,new",
    [
        ('native_search_tool = "google_search"', 'native_search_tool = "web_search"'),
        ('native_search_tool = "web_search"', 'native_search_tool = "google_search"'),
    ],
)
def test_route_loader_rejects_tool_backend_mismatch(
    tmp_path: Path, old: str, new: str
) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    data = source.read_text(encoding="utf-8").replace(old, new, 1)
    path = tmp_path / "routes.toml"
    path.write_text(data, encoding="utf-8")

    with pytest.raises(ModelRouteConfigError, match="does not support"):
        load_model_routes(path)


def test_cached_route_catalog_cannot_be_mutated() -> None:
    routes = default_model_routes()

    with pytest.raises(TypeError):
        routes.targets["new"] = routes.targets["gemini-free-3_7-flash"]  # type: ignore[index]


def test_package_data_declares_both_routing_files() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    # They live in `finesub.llm.routing` with the loaders that read them, so the
    # package-data key is that subpackage, not `llm`.
    package_data = config["tool"]["setuptools"]["package-data"]["finesub.llm.routing"]

    assert "model_catalog.psv" in package_data
    assert "model_routes.toml" in package_data


@pytest.mark.parametrize("model", ("luna", "sol"))
def test_local_completion_and_search_targets_reuse_one_runtime_fact(model: str) -> None:
    routes = default_model_routes()
    completion = routes.targets[f"local-codex-completion-gpt-5_6-{model}"]
    native = routes.targets[f"local-codex-native-gpt-5_6-{model}"]

    assert completion.fact_id == native.fact_id == f"local-codex-gpt-5_6-{model}"
    assert routes.target_profile(completion.id).native_search_tool == ""
    assert routes.target_profile(native.id).native_search_tool == "web_search"
    assert routes.target_fact(native.id).supports_audio is False
    assert routes.target_fact(native.id).supports_native_search is True


def test_route_policies_are_only_a_backend_gate() -> None:
    """A policy says which backends may answer and nothing else.

    Which *models* answer is the bound group's job alone -- policies used to
    prepend groups of their own, which put that answer in two places at once.
    """

    routes = default_model_routes()

    # api-only means "no local agent"; user-declared API backends ride along.
    assert routes.policies["api-only"].allowed_backends == frozenset(
        {"gemini_rest", "openai_compat", "anthropic"}
    )
    # Both agent policies admit a person's own agent too (docs §12.1.4).
    assert routes.policies["agent-only"].allowed_backends == frozenset(
        {"local_agent", "conversational_agent"}
    )
    # The mixed default gates nothing away: the group ordering decides.
    assert routes.policies["agent-text-preferred"].allowed_backends == frozenset(
        {"gemini_rest", "local_agent", "conversational_agent", "openai_compat", "anthropic"}
    )
    assert [field.name for field in fields(routes.policies["api-only"])] == [
        "id",
        "allowed_backends",
    ]


def test_route_digest_includes_runtime_fact_snapshot() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"

    assert default_model_routes().routing_identity_digest != hashlib.sha256(
        source.read_bytes()
    ).hexdigest()


def test_advisory_fact_edits_leave_the_routing_identity_alone(tmp_path: Path) -> None:
    """Score/display-name edits must not invalidate checkpoints."""

    import finesub.llm.routing.model_routes as model_routes_module

    src_dir = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing"
    baseline = load_model_routes(src_dir / "model_routes.toml")

    catalog_text = (src_dir / "model_catalog.psv").read_text(encoding="utf-8")
    edited = catalog_text.replace("|capable|75", "|capable|33").replace(
        "|3.7 Flash|", "|3.7 Flash (renamed)|"
    )
    assert edited != catalog_text
    catalog_path = tmp_path / "model_catalog.psv"
    catalog_path.write_text(edited, encoding="utf-8")

    entries = tuple(load_model_catalog(catalog_path))
    original_default = model_routes_module.default_model_catalog
    model_routes_module.default_model_catalog = lambda: entries  # type: ignore[assignment]
    try:
        rescored = load_model_routes(src_dir / "model_routes.toml")
    finally:
        model_routes_module.default_model_catalog = original_default

    assert rescored.routing_identity_digest == baseline.routing_identity_digest
    assert rescored.advisory_digest != baseline.advisory_digest


def test_default_preset_binds_all_cells_and_matches_the_plan() -> None:
    """Plan v2 §7: the default preset covers 7 task groups x 3 difficulties;
    equivalent-migration groups keep today's expansion order, the deliberate
    changes (lites out of correction/high, knowledge's own narrow group) are
    asserted explicitly."""

    routes = default_model_routes()
    assert set(routes.task_groups) == {
        "correction-mm",
        "correction-text",
        "planning-mm",
        "planning-text",
        "research",
        "search_judge",
        "knowledge",
    }
    # 9 bindings since the difficulty-fallback redesign (2026-08-11): every
    # group binds high; only correction -- whose quality deliberately steps
    # down at intermediate -- binds a second cell.
    default = routes.presets["default"]
    assert len(default.bindings) == 9
    # Unbound cells walk up: research intermediate/efficiency reuse quality's
    # group, while correction efficiency reuses intermediate's; requested knobs stay.
    assert routes.resolve_binding("default", "research", "intermediate")[0].id == "research-default"
    efficiency_group, efficiency_cell = routes.resolve_binding(
        "default", "correction-mm", "efficiency"
    )
    assert efficiency_group.id == "correction-basic"
    assert efficiency_cell.variant == "basicB"
    # Thinking is a preset-level knob (2026-08-11) tuned 2026-08-12: every
    # Every group thinks low at efficiency; research/knowledge think high at
    # quality, with intermediate written explicitly so upward fallback cannot
    # leak high into it.
    assert routes.resolve_thinking("default", "correction-mm", "efficiency") == "low"
    assert routes.resolve_thinking("default", "correction-mm", "intermediate") == "medium"
    assert routes.resolve_thinking("default", "research", "quality") == "high"
    assert routes.resolve_thinking("default", "research", "intermediate") == "high"
    assert routes.resolve_thinking("default", "research", "efficiency") == "low"

    # Equivalent migration: same expansion order as the v1 chains had.
    assert routes.model_groups["research-default"].target_ids == tuple(
        EXPECTED_GROUPS["research-default"]
    )
    assert routes.model_groups["lightweight-default"].target_ids == tuple(
        EXPECTED_GROUPS["lightweight-default"]
    )

    # Deliberate changes (§7): correction/high loses the lites (no silent
    # quality downgrade -- they live in the intermediate cell now)...
    assert routes.model_groups["correction-capable"].target_ids == (
        "gemini-free-3_7-flash",
        "gemini-free-3_6-flash",
        "gemini-free-3_5-flash",
        "gemini-paid-3_7-flash",
    )
    group, cell = routes.resolve_binding("default", "correction-mm", "intermediate")
    assert group.id == "correction-basic" and cell.variant == "basicB"
    # ...and knowledge gets its own narrow group (no lite fallback: its
    # output is auto-applied into the knowledge base).
    assert routes.model_groups["knowledge-capable"].target_ids == tuple(
        EXPECTED_GROUPS["knowledge-capable"]
    )

    # Cell knobs: correction quality carries capableC, intermediate basicB;
    # single-template sessions carry "". The thinking knob itself is asserted
    # exhaustively in test_default_thinking_knobs_resolve_as_tuned.
    assert routes.task_groups["correction-mm"].cells["quality"].variant == "capableC"
    assert routes.task_groups["research"].cells["quality"].variant == ""


def test_no_shipped_preset_triggers_binding_warnings() -> None:
    """§5.4, for *every* preset we ship rather than only `default`.

    Checking one preset let `agy` ship printing six media warnings on every
    run -- one per `-mm` cell, for Opus sitting ahead of the target that can
    hear, which is the deliberate ordering rather than a mistake. "Empty" must
    come from the data being right, not from the warning being off; the next
    test proves the warning still fires.
    """

    routes = default_model_routes()
    assert set(routes.presets) >= {"default", "agy"}
    for preset_id in routes.presets:
        assert routes.preset_binding_warnings(preset_id) == [], preset_id


def test_a_media_cell_with_nothing_that_can_hear_still_warns() -> None:
    """The case the per-member warning was actually protecting against."""

    routes = load_model_routes(
        user_config={
            "model_groups": {"deaf": {"targets": ["local-agy-opus-4_6"]}},
            "presets": {
                "deaf-mm": {
                    "name": "全聋",
                    "test_target": "gemini-free-3_5-flash-lite",
                    "bindings": {"correction-mm/quality": "deaf"},
                }
            },
        }
    )

    warnings = routes.preset_binding_warnings("deaf-mm")
    assert any("没有成员支持音频" in message for message in warnings), warnings


def test_a_media_cell_with_one_hearing_member_reports_its_depth() -> None:
    """"Someone can hear" is not the same as "this chain has any depth".

    Collapsing the per-member warning to a boolean removed the noise but also
    the signal: a four-member cell whose media calls have exactly one real
    candidate reads as healthy while being one failure from having nowhere to
    go. So the warning reports the count instead.
    """

    routes = load_model_routes(
        user_config={
            "model_groups": {
                "thin": {
                    "targets": [
                        "local-agy-opus-4_6",
                        "local-agy-media-gemini-3_7-flash",
                    ]
                }
            },
            "presets": {
                "thin-mm": {
                    "name": "只剩一个能听",
                    "test_target": "gemini-free-3_5-flash-lite",
                    "bindings": {"correction-mm/quality": "thin"},
                }
            },
        }
    )

    warnings = routes.preset_binding_warnings("thin-mm")
    depth = [m for m in warnings if "链长为 1" in m]
    assert depth, warnings
    assert "local-agy-opus-4_6" in depth[0]


def test_binding_warnings_fire_on_floor_envelope_and_media(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    data = source.read_text(encoding="utf-8").replace(
        '[model_groups.research-default]\ntargets = ["gemini-free-3_6-flash",',
        '[model_groups.research-default]\ntargets = ["gemini-free-gemma-4-31b", "gemini-free-3_6-flash",',
        1,
    )
    path = tmp_path / "routes.toml"
    path.write_text(data, encoding="utf-8")

    warnings = load_model_routes(path).preset_binding_warnings("default")
    floor = [w for w in warnings if "quality_score=30" in w]
    envelope = [w for w in warnings if "规划包络" in w]
    assert floor and "gemini-free-gemma-4-31b" in floor[0] and "下限 50" in floor[0]
    assert len(envelope) == 1
    assert "gemini-free-gemma-4-31b" in envelope[0]
    assert "16000" in envelope[0]


def test_a_text_only_member_ahead_of_one_that_hears_is_not_a_warning(
    tmp_path: Path,
) -> None:
    """That ordering is the design, not a mistake.

    A text-only model in front of an audio-capable one is how "prefer this for
    text, hand media to the next" is expressed -- the capability filter drops
    it before any call, for free. Warning per member made the shipped `agy`
    preset print six lines a run; the warning now fires only when *nothing* in
    the group can hear, which is the case that actually breaks at dispatch.
    """

    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    data = source.read_text(encoding="utf-8").replace(
        '[model_groups.lightweight-default]\ntargets = ["gemini-free-3_5-flash-lite",',
        '[model_groups.lightweight-default]\ntargets = ["local-codex-completion-gpt-5_6-luna", "gemini-free-3_5-flash-lite",',
        1,
    )
    path = tmp_path / "routes.toml"
    path.write_text(data, encoding="utf-8")

    warnings = load_model_routes(path).preset_binding_warnings("default")
    assert not any("音频" in message for message in warnings), warnings


def test_user_preset_falls_back_to_default_cells(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    data = source.read_text(encoding="utf-8") + (
        '\n[presets.mine]\nname = "mine"\ntest_target = "gemini-free-3_5-flash-lite"\n'
        '[presets.mine.bindings]\n"correction-mm/quality" = "correction-basic"\n'
    )
    path = tmp_path / "routes.toml"
    path.write_text(data, encoding="utf-8")

    routes = load_model_routes(path)
    bound, _ = routes.resolve_binding("mine", "correction-mm", "quality")
    assert bound.id == "correction-basic"
    fallback, _cell = routes.resolve_binding("mine", "research", "quality")
    assert fallback.id == "research-default"
    # ...including the knob: the user preset fills neither, so research/high
    # inherits the default preset's high.
    assert routes.resolve_thinking("mine", "research", "quality") == "high"


def test_route_loader_rejects_incomplete_default_preset(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    data = source.read_text(encoding="utf-8").replace(
        '"knowledge/quality" = "knowledge-capable"\n', "", 1
    )
    path = tmp_path / "routes.toml"
    path.write_text(data, encoding="utf-8")

    with pytest.raises(
        ModelRouteConfigError, match="must bind every task group's high cell"
    ):
        load_model_routes(path)


def test_route_loader_rejects_preset_test_target_outside_bound_groups(
    tmp_path: Path,
) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    data = source.read_text(encoding="utf-8").replace(
        'test_target = "gemini-free-3_5-flash-lite"\n\n[presets.default.bindings]',
        'test_target = "gemini-free-2_5-flash"\n\n[presets.default.bindings]',
        1,
    )
    path = tmp_path / "routes.toml"
    path.write_text(data, encoding="utf-8")

    with pytest.raises(ModelRouteConfigError, match="not in any bound model group"):
        load_model_routes(path)


def test_an_unselected_presets_test_target_cannot_break_your_config() -> None:
    """Only the presets this run can use are checked.

    Checking every *declared* preset let a packaged one fail somebody's
    config: replace `default` in your own config.toml and the shipped `agy`
    preset -- which you never selected -- stopped loading, because its
    inherited test target lived in the group you replaced.
    """

    user = {
        "model_groups": {"mine": {"targets": ["gemini-free-3_5-flash"]}},
        "presets": {
            "default": {
                "name": "我的默认",
                "test_target": "gemini-free-3_5-flash",
                "bindings": {
                    f"{task_group}/quality": "mine" for task_group in TASK_GROUP_IDS
                },
            }
        },
    }

    routes = load_model_routes(user_config=user)
    assert routes.presets["agy"].test_target_id == "gemini-free-3_5-flash-lite"

    # Selecting it *does* check it, and now it really is unreachable.
    with pytest.raises(ModelRouteConfigError, match="not in any bound model group"):
        load_model_routes(user_config={**user, "preset": "agy"})


def test_the_agy_test_target_stays_on_the_lite_tier() -> None:
    """A `--test-profile` run pins this target for every single call.

    Full 3.5 Flash carries 20 RPD against the lite tier's 500, so quietly
    moving this off the lite model costs 25x the test headroom.
    """

    routes = default_model_routes()

    assert routes.presets["agy"].test_target_id == "gemini-free-3_5-flash-lite"
    assert routes.target_fact("gemini-free-3_5-flash-lite").rpd == 500


# A user override catalog + the config.toml half that composes it. Facts live
# in the catalog, composition in config.toml (owner decision 2026-08-12).
USER_CATALOG = """fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|max_output_tokens|thinking|quality_score
ds-flash|deepseek|openai_compat|https://api.deepseek.com/|deepseek-v4-flash|128000|32768|high,medium,low|60
"""


def _user_catalog(tmp_path: Path, text: str = USER_CATALOG):
    path = tmp_path / "model_catalog.psv"
    path.write_text(text, encoding="utf-8")
    return merge_catalogs(
        load_model_catalog(), load_model_catalog(path, self_reported=True)
    )


def _user_config() -> dict:
    return {
        "preset": "mine",
        "model_groups": {"my-corr": {"targets": ["ds-flash", "gemini-free-3_7-flash"]}},
        "presets": {
            "mine": {"name": "mine", "bindings": {"correction-text/quality": "my-corr"}}
        },
    }


def test_user_catalog_row_becomes_a_provider_a_fact_and_a_target(
    tmp_path: Path,
) -> None:
    """Facts in the catalog, composition in config.toml (2026-08-12).

    One override row is enough to add a model: it declares the provider (kind
    + URL are columns now), becomes a self-reported fact, and gets a target of
    its own, which config.toml then composes into a group and a preset.
    """

    routes = load_model_routes(
        user_config=_user_config(), catalog=_user_catalog(tmp_path)
    )

    assert routes.active_preset_id == "mine"
    assert routes.self_reported_fact_ids == {"ds-flash"}
    assert routes.targets["ds-flash"].backend == "openai_compat"
    provider = routes.providers["deepseek"]
    assert provider.kind == "openai_compat"
    assert provider.base_url == "https://api.deepseek.com"  # trailing / stripped
    assert provider.key_env == "FINESUB_KEY_DEEPSEEK"  # naming convention

    fact = routes.facts["ds-flash"]
    assert fact.provider_tier == "deepseek"
    # Blank cells take their defaults: paid-tier quotas, no media, identity
    # thinking is overridden here by an explicit triple.
    assert (fact.rpm, fact.tpm, fact.rpd) == (100, 4_000_000, -1)
    assert not fact.supports_audio and not fact.supports_native_search
    assert fact.token_scale == 1.0

    group, cell = routes.resolve_binding("mine", "correction-text", "quality")
    assert group.id == "my-corr" and cell.variant == "capableC"
    # Difficulty fallback first (2026-08-11): switching to intermediate keeps *my*
    # group (only the prompt changes), instead of silently swapping in the
    # default preset's lite group.
    med_group, med_cell = routes.resolve_binding("mine", "correction-text", "intermediate")
    assert med_group.id == "my-corr" and med_cell.variant == "basicB"
    fallback, _ = routes.resolve_binding("mine", "research", "quality")
    assert fallback.id == "research-default"
    # Inherited test target (user preset omitted it).
    assert routes.presets["mine"].test_target_id == "gemini-free-3_5-flash-lite"
    # D13 envelope: the smallest member decides.
    assert routes.group_planning_envelope("my-corr") == (128_000, 32_768)


def test_unreferenced_user_model_is_advisory_only(tmp_path: Path) -> None:
    """Adding a model the active run cannot reach must not
    invalidate checkpoints -- it changes only the advisory digest."""

    catalog = _user_catalog(tmp_path)
    base = load_model_routes()
    # The row exists but nothing composes it: no group, no preset.
    with_extra = load_model_routes(catalog=catalog)
    assert with_extra.routing_identity_digest == base.routing_identity_digest
    assert with_extra.advisory_digest != base.advisory_digest

    # ...while a *referenced* model (bound by the active preset) is routing.
    referenced = load_model_routes(user_config=_user_config(), catalog=catalog)
    assert referenced.routing_identity_digest != base.routing_identity_digest


def test_advisory_route_fields_stay_out_of_the_routing_identity(tmp_path: Path) -> None:
    """The packaged declaration side of digest reachability (fixed 2026-08-12):
    enters the routing identity as parsed structure, not as file bytes.

    Hashing the text put ``floor_score``, the preset display name -- and any
    comment -- into every resume key, so re-tuning an advisory floor silently
    discarded every completed window. The criterion is unchanged: changes what
    the run produces -> routing; changes how we judge it -> advisory.
    """

    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    original = source.read_text(encoding="utf-8")
    base = load_model_routes(source)

    for edit, expect_advisory_change in (
        (lambda text: text.replace("floor_score = 70", "floor_score = 65", 1), True),
        (lambda text: text.replace('name = "默认"', 'name = "出厂"', 1), True),
        (lambda text: text.replace("# floor_score guards", "# tweaked comment", 1), False),
    ):
        path = tmp_path / "routes.toml"
        path.write_text(edit(original), encoding="utf-8")
        edited = load_model_routes(path)
        assert edited.routing_identity_digest == base.routing_identity_digest
        assert (
            edited.advisory_digest != base.advisory_digest
        ) is expect_advisory_change

    # A routing edit still moves it: the bound group is what a call selects.
    path = tmp_path / "routes.toml"
    path.write_text(
        original.replace(
            '"research/quality" = "research-default"',
            '"research/quality" = "knowledge-capable"',
            1,
        ),
        encoding="utf-8",
    )
    assert (
        load_model_routes(path).routing_identity_digest
        != base.routing_identity_digest
    )


def test_task_group_reference_model_is_derived_from_the_default_binding() -> None:
    """The reference model is not declared any more (2026-08-12): it is
    whatever the default preset puts first in that *cell*, so a roster change
    cannot leave a stale name behind.

    Per cell, not per task group: correction binds a different group at intermediate
    (the declared downgrade tier), so that cell is calibrated to the lite.
    """

    routes = load_model_routes()

    assert routes.task_group_reference_model("correction-mm") == "3.7 Flash"
    assert (
        routes.task_group_reference_model("correction-mm", "intermediate")
        == "3.5 Flash Lite"
    )
    assert (
        routes.task_group_reference_model("correction-text", "efficiency")
        == "3.5 Flash Lite"
    )
    assert routes.task_group_reference_model("research") == "3.6 Flash"
    assert routes.task_group_reference_model("search_judge") == "3.5 Flash Lite"
    # Task groups that bind one group for every difficulty read the same.
    assert routes.task_group_reference_model("research", "efficiency") == "3.6 Flash"


def test_default_thinking_knobs_resolve_as_tuned() -> None:
    """Owner tuning 2026-08-12: research/knowledge think high at quality *and*
    intermediate (the model gets cheaper there, the reasoning does not);
    everything thinks low at efficiency; the rest stay medium.

    The intermediate row is inherited, not written: an unbound cell falls back
    *upward*, so research/intermediate picks up quality's knob on purpose.
    """

    routes = load_model_routes()
    resolved = {
        (task_group_id, difficulty): routes.resolve_thinking(
            "default", task_group_id, difficulty
        )
        for task_group_id in TASK_GROUP_IDS
        for difficulty in DIFFICULTIES
    }

    assert {key: value for key, value in resolved.items() if value == "high"} == {
        ("research", "quality"): "high",
        ("knowledge", "quality"): "high",
        ("research", "intermediate"): "high",
        ("knowledge", "intermediate"): "high",
    }
    assert all(
        resolved[(task_group_id, "efficiency")] == "low"
        for task_group_id in TASK_GROUP_IDS
    )
    assert all(
        resolved[(task_group_id, "intermediate")] == "medium"
        for task_group_id in TASK_GROUP_IDS
        if task_group_id not in ("research", "knowledge")
    )


def test_catalog_row_requires_a_context_window(tmp_path: Path) -> None:
    """The one field with no optimistic default (plan §5.2): windows are
    planned against it, so a wrong 194k explodes far from its cause."""

    path = tmp_path / "model_catalog.psv"
    path.write_text(
        "fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens\n"
        "ds|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing max_input_tokens"):
        load_model_catalog(path)


def test_catalog_row_requires_a_base_url_for_text_dialects(tmp_path: Path) -> None:
    path = tmp_path / "model_catalog.psv"
    path.write_text(
        "fact_id|provider_tier|provider_kind|api_model_id|max_input_tokens\n"
        "ds|deepseek|openai_compat|deepseek-v4-flash|128000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_url is required"):
        load_model_catalog(path)


def test_catalog_rejects_unknown_columns(tmp_path: Path) -> None:
    """A typo in a header is otherwise a silently missing fact."""

    path = tmp_path / "model_catalog.psv"
    path.write_text(
        "fact_id|provider_tier|api_model_id|max_input_tokens|max_output_token\n"
        "x|GEMINI_FREE|gemini/x|1000|20\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown="):
        load_model_catalog(path)


def test_catalog_names_the_replacement_for_a_retired_column(tmp_path: Path) -> None:
    """An override file written before the rename must be told what to type.

    Falling through to "unknown column" would be technically true and useless:
    the reader wrote a column that used to work.
    """

    path = tmp_path / "model_catalog.psv"
    path.write_text(
        "fact_id|provider_tier|model|litellm_model|max_input_tokens\n"
        "x|GEMINI_FREE|X|gemini/x|1000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        load_model_catalog(path)
    message = str(excinfo.value)
    assert "litellm_model -> api_model_id" in message
    assert "model -> display_name" in message


def test_override_catalog_replaces_a_packaged_row_in_place(tmp_path: Path) -> None:
    """Merging is by fact_id with the override winning, order preserved -- a
    re-tuned limit must not move the model inside its group."""

    path = tmp_path / "model_catalog.psv"
    path.write_text(
        "fact_id|provider_tier|api_model_id|max_input_tokens|quality_score\n"
        "gemini-free-3_6-flash|GEMINI_FREE|gemini/gemini-3.6-flash|100000|90\n",
        encoding="utf-8",
    )
    packaged = load_model_catalog()
    merged = merge_catalogs(packaged, load_model_catalog(path, self_reported=True))

    assert [entry.fact_id for entry in merged] == [
        entry.fact_id for entry in packaged
    ]
    overridden = {entry.fact_id: entry for entry in merged}["gemini-free-3_6-flash"]
    assert overridden.max_input_tokens == 100_000
    assert overridden.quality_score == 90
    assert overridden.self_reported is True
    # Blank cells took defaults rather than the packaged row's values: an
    # override row is a whole row, not a patch.
    assert overridden.supports_audio is False


def test_config_toml_no_longer_declares_models_or_providers() -> None:
    """They moved into the catalog; a stale config must say so, not be ignored."""

    for section in ("providers", "models"):
        with pytest.raises(ModelRouteConfigError, match="moved into the model catalog"):
            load_model_routes(user_config={section: {"x": {}}})


def test_user_preset_selection_must_exist() -> None:
    with pytest.raises(ModelRouteConfigError, match="names no known preset"):
        load_model_routes(user_config={"preset": "nope"})


def test_sample_config_llm_examples_stay_loadable(tmp_path: Path) -> None:
    """§11 drift guard: the commented [llm.*] example block in
    config.example.toml -- and the catalog row it documents -- must keep
    parsing and loading against the real loader. A sample that "looks right
    but isn't" is worse than none."""

    import re
    import tomllib

    sample_path = Path(__file__).resolve().parents[1] / "config.example.toml"
    text = sample_path.read_text(encoding="utf-8")
    # The whole sample must parse as-is (the examples are comments).
    parsed = tomllib.loads(text)
    assert "llm" in parsed

    block = text.split("自定义模型与预设", 1)[1].split("[pools]", 1)[0]

    # The documented catalog row must load through the real catalog parser.
    catalog_lines = [
        line[2:].strip()
        for line in block.splitlines()
        if line.startswith("#   ") and "|" in line
    ]
    assert len(catalog_lines) == 2, catalog_lines
    catalog_path = tmp_path / "model_catalog.psv"
    catalog_path.write_text("\n".join(catalog_lines) + "\n", encoding="utf-8")
    catalog = merge_catalogs(
        load_model_catalog(), load_model_catalog(catalog_path, self_reported=True)
    )

    # Uncomment the TOML-shaped example lines (table headers / assignments);
    # prose comment lines stay comments.
    toml_line = re.compile(r"^(\[llm\.|[A-Za-z_]+ = |\"[^\"]+\" = |preset = )")
    snippet_lines = []
    for line in block.splitlines():
        if line.startswith("# "):
            candidate = line[2:]
            if toml_line.match(candidate):
                snippet_lines.append(candidate)
    snippet = tomllib.loads("\n".join(snippet_lines))
    user = dict(snippet.get("llm", {}))
    if "preset" in snippet:
        user["preset"] = snippet["preset"]

    routes = load_model_routes(user_config=user, catalog=catalog)
    assert routes.active_preset_id == "my-deepseek"
    assert "ds-flash" in routes.self_reported_fact_ids
    assert routes.providers["deepseek"].base_url == "https://api.deepseek.com"
    group, _cell = routes.resolve_binding("my-deepseek", "correction-text", "quality")
    assert group.id == "my-corr"
    # The sample deliberately shows the knowledge caveat: binding a
    # self-reported model into knowledge trips the floor warning.
    warnings = routes.preset_binding_warnings("my-deepseek")
    assert any("knowledge" in w and "ds-flash" in w for w in warnings)


def test_a_binding_may_name_a_target_instead_of_a_group() -> None:
    """Quick model selection: "just use this one here" needs no group.

    The wrapper is a real group, so everything downstream -- plan, trace,
    resume digest, floor warning -- keeps one shape.
    """

    routes = load_model_routes(
        user_config={
            "presets": {
                "default": {
                    "name": "单模型",
                    "test_target": "gemini-free-3_5-flash-lite",
                    "bindings": {
                        f"{task_group}/quality": "gemini-free-3_5-flash-lite"
                        for task_group in TASK_GROUP_IDS
                    },
                }
            }
        }
    )

    group, _cell = routes.resolve_binding("default", "research", "quality")
    assert group.id == "target:gemini-free-3_5-flash-lite"
    assert group.target_ids == ("gemini-free-3_5-flash-lite",)
    assert group.variant_overrides == {}


def test_a_group_id_wins_over_a_target_of_the_same_name() -> None:
    """So adding a target can never quietly re-point somebody's binding."""

    routes = load_model_routes(
        user_config={
            "model_groups": {
                "gemini-free-3_5-flash-lite": {"targets": ["gemini-free-3_7-flash"]}
            },
            "presets": {
                "default": {
                    "name": "同名",
                    "test_target": "gemini-free-3_7-flash",
                    "bindings": {
                        f"{task_group}/quality": "gemini-free-3_5-flash-lite"
                        for task_group in TASK_GROUP_IDS
                    },
                }
            },
        }
    )

    group, _cell = routes.resolve_binding("default", "research", "quality")
    assert group.target_ids == ("gemini-free-3_7-flash",)


def test_the_single_target_namespace_cannot_be_declared() -> None:
    with pytest.raises(ModelRouteConfigError, match="reserved"):
        load_model_routes(
            user_config={
                "model_groups": {
                    "target:mine": {"targets": ["gemini-free-3_7-flash"]}
                }
            }
        )


def test_a_binding_that_names_nothing_says_so() -> None:
    with pytest.raises(ModelRouteConfigError, match="neither a model group nor a target"):
        load_model_routes(
            user_config={
                "presets": {
                    "default": {
                        "name": "错字",
                        "test_target": "gemini-free-3_5-flash-lite",
                        "bindings": {
                            f"{task_group}/quality": "gemini-free-3_5-flash-lit"
                            for task_group in TASK_GROUP_IDS
                        },
                    }
                }
            }
        )


def test_user_config_overrides_packaged_groups_and_presets() -> None:
    """Same-id means "mine wins", symmetric with the catalog's fact_id override
    (owner decision 2026-08-12). One mental model for both files instead of
    "your catalog row overrides, your group name is a collision error".

    The typo guard that costs is replaced by an announcement: every override
    is listed once so a misspelt id does not look like a working config.
    """

    base = load_model_routes()
    user = {
        "model_groups": {
            # A packaged id: replaces the shipped membership in place.
            "lightweight-default": {"targets": ["gemini-free-3_1-flash-lite"]},
            "mine": {"targets": ["gemini-free-3_5-flash"]},
        },
        "presets": {
            "default": {
                "name": "我的默认",
                # Must be reachable from a bound group -- the loader checks.
                "test_target": "gemini-free-3_5-flash",
                "bindings": {
                    f"{task_group}/quality": "mine"
                    for task_group in TASK_GROUP_IDS
                },
            }
        },
    }
    routes = load_model_routes(user_config=user)

    assert routes.model_groups["lightweight-default"].target_ids == (
        "gemini-free-3_1-flash-lite",
    )
    assert routes.presets["default"].name == "我的默认"
    group, _cell = routes.resolve_binding("default", "search_judge", "quality")
    assert group.id == "mine"
    # Both overrides are announced; the new id is not (it collides with nothing).
    assert set(routes.override_notices) == {"模型组 lightweight-default", "预设 default"}
    # Overriding is a routing change, so checkpoints invalidate as they should.
    assert routes.routing_identity_digest != base.routing_identity_digest


def test_overriding_the_default_preset_still_has_to_be_complete() -> None:
    """Overriding "default" is allowed because the completeness check runs on
    whatever ends up as the default -- an incomplete override is still caught."""

    with pytest.raises(ModelRouteConfigError, match="must bind every task group"):
        load_model_routes(
            user_config={
                "presets": {
                    "default": {
                        "name": "残缺",
                        "test_target": "gemini-free-3_5-flash-lite",
                        "bindings": {"research/quality": "research-default"},
                    }
                }
            }
        )


def _packaged_routes_text() -> str:
    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    return source.read_text(encoding="utf-8")


def test_a_conversational_target_may_not_share_a_model_group(tmp_path: Path) -> None:
    """A model group is an ordered fallback chain, and this cannot be a step.

    Falling back *to* it would mean calling something that only answers when a
    person happens to have an agent attached; falling back *from* it would mean
    taking over a task that agent may be halfway through.
    """

    text = _packaged_routes_text().replace(
        '[model_groups.conversational-agent]\ntargets = ["conversational-agent"]',
        '[model_groups.conversational-agent]\ntargets = ["conversational-agent", '
        '"gemini-free-3_7-flash"]',
        1,
    )
    path = tmp_path / "routes.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ModelRouteConfigError, match="group of its own"):
        load_model_routes(path)


def test_the_calling_path_never_selects_a_conversational_target() -> None:
    """Whether an agent is attached is not knowable here, and changes mid-run.

    So the route chain always declines it; the task queue is where it takes
    part. Anything else would be the harness dialling a worker that has no
    number.
    """

    from finesub.llm.routing.model_router import provider_enabled

    default_model_routes.cache_clear()
    routes = default_model_routes()
    target = routes.targets["conversational-agent"]
    fact = routes.target_fact("conversational-agent")

    class _Endpoint:
        backend = target.backend
        provider_tier = fact.provider_tier
        api_model_id = fact.api_model_id

    class _Candidate:
        endpoint = _Endpoint()

    assert provider_enabled(_Candidate(), agent_ready=lambda *_: True) is False


def test_no_driver_is_registered_for_the_conversational_tier() -> None:
    """Nothing may launch it: there is no CLI here, only somebody's session."""

    from finesub.llm.routing.execution_policy import ExecutionSettings, driver_for_provider_tier

    with pytest.raises(ValueError, match="No local-agent driver"):
        driver_for_provider_tier(
            ExecutionSettings(),
            provider_tier="LOCAL_CONVERSATIONAL",
            model="conversational-agent",
        )


def test_the_agent_session_tier_is_routing_identity(tmp_path: Path) -> None:
    """Identity hole 1 (owner decision 2026-08-22): the tier decides the call
    form, so two route tables differing only in `agent_session` must not
    share a checkpoint identity -- while the advisory digest stays put."""

    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    base = load_model_routes(source)
    path = tmp_path / "routes.toml"
    path.write_text(
        source.read_text(encoding="utf-8")
        + '\n[presets.default.agent_session]\n"research/quality" = "api"\n',
        encoding="utf-8",
    )
    tiered = load_model_routes(path)
    assert tiered.routing_identity_digest != base.routing_identity_digest
    assert tiered.advisory_digest == base.advisory_digest
