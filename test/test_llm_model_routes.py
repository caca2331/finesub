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
from finesub.llm.routing.model_catalog import Capability, load_effective_catalog, load_model_catalog


# The declared expansion order: correction uses 3.8 → 3.6,
# research/knowledge 3.6 → 3.8, paid 3.8 the capable tail for both.
EXPECTED_GROUPS = {
    "correction-capable": [
        "gemini-free-3_8-flash",
        "gemini-free-3_6-flash",
        "gemini-paid-3_8-flash",
    ],
    "correction-basic": [
        "gemini-free-3_5-flash-lite",
        "gemini-paid-3_5-flash-lite",
    ],
    "research-default": [
        "gemini-free-3_6-flash",
        "gemini-free-3_8-flash",
        "gemini-paid-3_8-flash",
    ],
    "knowledge-capable": [
        "gemini-free-3_6-flash",
        "gemini-free-3_8-flash",
        "gemini-paid-3_8-flash",
    ],
    "gemini-native-search": [
        "gemini-free-2_5-flash",
        "gemini-paid-3_8-flash",
    ],
    # The `agy-hybrid` preset's two groups. Opus is text-only on purpose: it sits
    # ahead of the media target so text windows prefer it and multimodal ones
    # skip past it to the Gemini that agy fronts.
    "agy-capable": [
        "gemini-free-3_8-flash",
        "local-agy-sonnet-5_5",
        "local-agy-gemini-3_8-flash",
        "local-agy-gemini-4-argon",
    ],
    "agy-basic": [
        "gemini-free-3_6-flash",
        "local-agy-sonnet-5_5",
        "local-agy-gemini-3_8-flash",
        "local-agy-gemini-4-argon",
    ],
    # The `agy` preset's pair: the same rosters with the API head removed.
    "agy-only-capable": [
        "local-agy-sonnet-5_5",
        "local-agy-gemini-3_8-flash",
        "local-agy-gemini-4-argon",
    ],
    "agy-only-fast": ["local-agy-gemini-3_8-flash", "local-agy-gemini-4-argon"],
    "agy-only-basic": [
        "local-agy-sonnet-5_5",
        "local-agy-gemini-3_8-flash",
        "local-agy-gemini-4-argon",
    ],
    # Tool-protocol only, so bound by nothing out of the box: `dsh --profile
    # headless` takes its task from its command line, which a whole window
    # does not fit into.
    "dsh-capable": [
        "local-dsh-deepseek-v4_1-flash",
        "local-dsh-deepseek-v4-pro",
    ],
    "dsh-basic": [
        "local-dsh-deepseek-v4_1-flash",
    ],
    # Bound by no packaged preset, like the dsh pair: a person names them, and
    # then only if their CodeBuddy plan actually reaches these models.
    "codebuddy-capable": [
        "local-codebuddy-glm-5_3-flash",
        "local-codebuddy-deepseek-v4_1-flash",
        "local-codebuddy-deepseek-v4-pro",
    ],
    "codebuddy-basic": [
        "local-codebuddy-hy3",
        "local-codebuddy-glm-5_3-flash",
        "local-codebuddy-deepseek-v4_1-flash",
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


def test_every_catalog_model_has_one_target_named_after_it_with_its_tier_tool() -> None:
    """One target per model (owner 2026-11-09): the name is the model's, and
    whether it may search is the model's `supports_native_search` with its
    tier's tool. Unbound is not unused: these are ids people write."""

    from finesub.llm.routing.model_catalog import NATIVE_SEARCH_TOOLS

    default_model_routes.cache_clear()
    routes = default_model_routes()
    local = {
        target_id: target
        for target_id, target in routes.targets.items()
        if target.backend == "local_agent"
    }
    packaged_local = {
        entry.fact_id: entry
        for entry in load_model_catalog()
        if entry.provider_kind == "local_agent"
    }
    assert set(local) == set(packaged_local)
    for target_id, target in local.items():
        entry = packaged_local[target_id]
        assert target.fact_id == target_id
        expected = NATIVE_SEARCH_TOOLS[entry.provider_tier] if entry.supports_native_search else ""
        assert target.native_search_tool == expected, target_id
    # Every packaged local agent can search (owner 2026-11-09), agy's Claude included.
    assert routes.targets["local-agy-sonnet-5_5"].native_search_tool == "search_web"
    assert routes.targets["local-codex-gpt-6_1-sol"].native_search_tool == "web_search"
    # Only the free Gemini 3.x API rows cannot.
    no_search = sorted(
        target_id for target_id, target in routes.targets.items()
        if target.backend == "gemini_rest" and not target.native_search_tool
    )
    assert no_search == [
        "gemini-free-3_1-flash-lite",
        "gemini-free-3_5-flash-lite",
        "gemini-free-3_6-flash",
        "gemini-free-3_8-flash",
    ]


@pytest.mark.parametrize(
    ("tier", "tool"),
    (
        ("LOCAL_CODEX", "web_search"),
        ("LOCAL_CLAUDE", "web_search"),
        ("LOCAL_AGY", "search_web"),
        ("LOCAL_DSH", "web_search"),
        ("LOCAL_CODEBUDDY", "web_search"),
    ),
)
def test_a_user_agent_model_gets_one_target_that_searches_unless_it_says_no(
    tmp_path: Path, tier: str, tool: str
) -> None:
    """A new Codex or agy release is one table, like a dsh gateway (D5)."""

    extra = tmp_path / "model_catalog.toml"
    extra.write_text(
        "[models.mine]\n"
        f'provider_tier = "{tier}"\n'
        'api_model_id = "vendor-model"\n'
        "max_input_tokens = 194000\n"
        f'quota_pool = "{"per-model"}"\n'
        "\n[models.mine-quiet]\n"
        f'provider_tier = "{tier}"\n'
        'api_model_id = "vendor-model-2"\n'
        "max_input_tokens = 194000\n"
        'quota_pool = "per-model"\n'
        "supports_native_search = false\n",
        encoding="utf-8",
    )
    routes = load_model_routes(catalog=load_effective_catalog(extra))
    assert [t for t, target in routes.targets.items() if target.fact_id == "mine"] == ["mine"]
    assert routes.targets["mine"].native_search_tool == tool
    assert routes.targets["mine-quiet"].native_search_tool == ""


def test_a_model_named_like_a_declared_target_is_refused(tmp_path: Path) -> None:
    extra = tmp_path / "model_catalog.toml"
    extra.write_text(
        "[models.conversational-agent]\n"
        'provider_tier = "LOCAL_CODEX"\n'
        'api_model_id = "x"\n'
        "max_input_tokens = 194000\n",
        encoding="utf-8",
    )
    with pytest.raises(ModelRouteConfigError, match="rename the model"):
        load_model_routes(catalog=load_effective_catalog(extra))


def test_patching_a_packaged_model_keeps_its_target(tmp_path: Path) -> None:
    extra = tmp_path / "model_catalog.toml"
    extra.write_text(
        "[models.local-codebuddy-hy3]\nmax_output_tokens = 32000\n",
        encoding="utf-8",
    )
    routes = load_model_routes(catalog=load_effective_catalog(extra))
    assert "local-codebuddy-hy3" in routes.self_reported_fact_ids
    assert routes.targets["local-codebuddy-hy3"].native_search_tool == "web_search"


def test_role_configs_are_derived_from_the_default_preset() -> None:
    routes = default_model_routes()
    configs = default_role_configs()
    group_by_role = {
        LLMRole.AUDIO_MULTIMODAL: "correction-capable",
        LLMRole.GENERAL_CAPABLE: "research-default",
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

    # Cell knobs land on the config: correction intermediate binds the capable
    # group with capableC; efficiency binds the basic group with basicA and
    # thinks low.
    intermediate = role_config_for("correction-mm", "intermediate")
    assert intermediate.model_group_id == "correction-capable"
    assert intermediate.variant == "capableC" and intermediate.thinking_level == "medium"
    efficiency = role_config_for("correction-text", "efficiency")
    assert efficiency.model_group_id == "correction-basic"
    assert efficiency.variant == "basicA" and efficiency.thinking_level == "low"


def test_runtime_facts_remain_independent_per_provider_tier() -> None:
    routes = default_model_routes()
    free = routes.target_fact("gemini-free-3_8-flash")
    paid = routes.target_fact("gemini-paid-3_8-flash")

    assert free.api_model_id == paid.api_model_id
    assert free.fact_id != paid.fact_id
    assert (free.rpm, free.tpm, free.rpd) != (paid.rpm, paid.tpm, paid.rpd)


def test_same_model_can_have_different_capabilities_per_tier(tmp_path: Path) -> None:
    path = tmp_path / "facts.toml"
    path.write_text(
        """
[models.free]
provider_tier = "FREE"
provider_kind = "gemini"
display_name = "X"
api_model_id = "provider/x"
max_input_tokens = 100
max_output_tokens = 20
supports_native_search = false
thinking = false
rpm = 1
tpm = 10
rpd = 2
is_free = true
capability = 40

[models.paid]
provider_tier = "PAID"
provider_kind = "gemini"
display_name = "X"
api_model_id = "provider/x"
max_input_tokens = 200
max_output_tokens = 40
supports_audio = true
supports_video = true
supports_native_search = true
thinking = ["xhigh", "medium", "low"]
rpm = 9
tpm = 90
capability = 80
""",
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
    # Scores still read as tiers (owner 2026-11-08): 40 is below low's 50.
    assert free.capability is Capability.VERY_LOW
    assert paid.capability is Capability.MEDIUM


@pytest.mark.parametrize(
    ("cell", "tier"),
    (
        ("very-low", Capability.VERY_LOW),
        ("HIGH", Capability.HIGH),
        (0, Capability.VERY_LOW),
        (49, Capability.VERY_LOW),
        (50, Capability.LOW),
        (82, Capability.MEDIUM),
        (90, Capability.HIGH),
        (100, Capability.HIGH),
    ),
)
def test_capability_takes_a_tier_name_or_a_score(cell: object, tier: Capability) -> None:
    from finesub.llm.routing.model_catalog import parse_capability

    assert parse_capability(cell, owner="x") is tier


@pytest.mark.parametrize("cell", ("101", "-1", '"mid"', "7.5", '"77"', "true"))
def test_catalog_rejects_an_capability_that_is_neither(tmp_path: Path, cell: str) -> None:
    path = tmp_path / "model_catalog.toml"
    path.write_text(
        f"[models.gemini-free-3_6-flash]\ncapability = {cell}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="capability must be"):
        load_effective_catalog(path)


def test_the_old_score_key_is_refused_with_its_new_name(tmp_path: Path) -> None:
    path = tmp_path / "model_catalog.toml"
    path.write_text(
        "[models.gemini-free-3_6-flash]\nquality_score = 82\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="quality_score is now called capability"):
        load_effective_catalog(path)


def test_capability_is_advisory_data_on_every_fact() -> None:
    routes = default_model_routes()

    # Same model, different provider tier: the capability is the model's, so
    # the advisory tier matches even though quotas differ.
    assert (
        routes.target_fact("gemini-free-3_8-flash").capability
        is routes.target_fact("gemini-paid-3_8-flash").capability
        is Capability.MEDIUM
    )
    # Every shipped row states one: blank is for rows nobody has judged.
    assert [f for f, fact in routes.facts.items() if fact.capability is None] == []
    # Owner 2026-11-08: Gemma 4 is the only very-low model.
    assert [
        fact_id
        for fact_id, fact in routes.facts.items()
        if fact.capability is Capability.VERY_LOW
    ] == ["gemini-free-gemma-4-31b"]


def test_capability_floor_warnings_name_both_tiers() -> None:
    from finesub.llm.routing.model_catalog import capability_floor_warnings, catalog_by_fact_id

    facts = catalog_by_fact_id()
    members = [facts["gemini-free-3_6-flash"], facts["gemini-free-3_5-flash-lite"]]

    warnings = capability_floor_warnings(
        members,
        min_capability=Capability.MEDIUM,
        reference_model="3.6 Flash",
        owner="correction-mm/quality",
    )
    assert len(warnings) == 1
    assert "gemini-free-3_5-flash-lite" in warnings[0]
    assert "能力档「低」" in warnings[0]
    assert "下限「中」" in warnings[0]
    assert "3.6 Flash" in warnings[0]

    # At-or-above floor stays silent: capability is advisory, not a filter.
    assert (
        capability_floor_warnings(
            members,
            min_capability=Capability.LOW,
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
        ('fact = "local-conversational-agent"', 'fact = "gemini-free-3_6-flash"'),
        ('fact = "gemini-free-3_8-flash"', 'fact = "local-codex-gpt-6-luna"'),
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


def test_cached_route_catalog_cannot_be_mutated() -> None:
    routes = default_model_routes()

    with pytest.raises(TypeError):
        routes.targets["new"] = routes.targets["gemini-free-3_8-flash"]  # type: ignore[index]


def test_package_data_declares_both_routing_files() -> None:
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    # They live in `finesub.llm.routing` with the loaders that read them, so the
    # package-data key is that subpackage, not `llm`.
    package_data = config["tool"]["setuptools"]["package-data"]["finesub.llm.routing"]

    assert "model_catalog.toml" in package_data
    assert "model_routes.toml" in package_data


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

    catalog_path = tmp_path / "model_catalog.toml"
    catalog_path.write_text(
        "[models.gemini-free-3_8-flash]\n"
        'capability = "low"\n'
        'display_name = "3.8 Flash (renamed)"\n'
        'note = "edited"\n'
        'avoid = { knowledge = "edited" }\n',
        encoding="utf-8",
    )

    entries = tuple(load_effective_catalog(catalog_path))
    original_default = model_routes_module.default_model_catalog
    model_routes_module.default_model_catalog = lambda: entries  # type: ignore[assignment]
    try:
        rescored = load_model_routes(src_dir / "model_routes.toml")
    finally:
        model_routes_module.default_model_catalog = original_default

    assert rescored.routing_identity_digest == baseline.routing_identity_digest
    assert rescored.advisory_digest != baseline.advisory_digest


def test_default_preset_binds_all_cells_and_matches_the_plan() -> None:
    """Plan v2 §7: the default preset covers 4 task groups x 3 difficulties;
    equivalent-migration groups keep today's expansion order, the deliberate
    changes (lites out of correction/high, knowledge's own narrow group) are
    asserted explicitly."""

    routes = default_model_routes()
    assert set(routes.task_groups) == {
        "correction-mm",
        "correction-text",
        "research",
        "knowledge",
    }
    # 8 bindings: every group binds quality; only correction -- which
    # deliberately steps down to the lites at efficiency -- binds more, its
    # intermediate written out on the capable group.
    default = routes.presets["default"]
    assert len(default.bindings) == 8
    # Unbound cells walk up: research intermediate/efficiency reuse quality's
    # group; requested knobs stay. Correction (owner 2026-10-30): the first two
    # tiers on the capable group, efficiency on the lites with low thinking.
    assert routes.resolve_binding("default", "research", "intermediate")[0].id == "research-default"
    for task_group in ("correction-mm", "correction-text"):
        group, cell = routes.resolve_binding("default", task_group, "intermediate")
        assert (group.id, cell.variant) == ("correction-capable", "capableC"), task_group
        group, cell = routes.resolve_binding("default", task_group, "efficiency")
        assert (group.id, cell.variant) == ("correction-basic", "basicA"), task_group
        assert routes.resolve_thinking("default", task_group, "efficiency") == "low"
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

    # Deliberate changes (§7): correction/high loses the lites (no silent
    # quality downgrade -- they live in the efficiency cell now)...
    assert routes.model_groups["correction-capable"].target_ids == tuple(
        EXPECTED_GROUPS["correction-capable"]
    )
    # ...and knowledge gets its own narrow group (no lite fallback: its
    # output is auto-applied into the knowledge base).
    assert routes.model_groups["knowledge-capable"].target_ids == tuple(
        EXPECTED_GROUPS["knowledge-capable"]
    )

    # Cell knobs: correction quality and intermediate carry capableC, efficiency basicA;
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
        assert routes.quality_cap_warnings(preset_id) == [], preset_id


def test_a_media_cell_with_nothing_that_can_hear_still_warns() -> None:
    """The case the per-member warning was actually protecting against."""

    routes = load_model_routes(
        user_config={
            "model_groups": {"deaf": {"targets": ["local-agy-sonnet-5_5"]}},
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
                        "local-agy-sonnet-5_5",
                        "local-agy-gemini-3_8-flash",
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
    assert "local-agy-sonnet-5_5" in depth[0]


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
    floor = [w for w in warnings if "能力档「极低」" in w]
    envelope = [w for w in warnings if "规划包络" in w]
    assert floor and "gemini-free-gemma-4-31b" in floor[0] and "下限「低」" in floor[0]
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
        '[model_groups.correction-basic]\ntargets = ["gemini-free-3_5-flash-lite",',
        '[model_groups.correction-basic]\ntargets = ["local-codex-gpt-6-luna", "gemini-free-3_5-flash-lite",',
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
    config: replace `default` in your own config.toml and the shipped
    `agy-hybrid` preset -- which you never selected -- stopped loading, because
    its inherited test target lived in the group you replaced.
    """

    user = {
        "model_groups": {"mine": {"targets": ["gemini-free-3_6-flash"]}},
        "presets": {
            "default": {
                "name": "我的默认",
                "test_target": "gemini-free-3_6-flash",
                "bindings": {
                    f"{task_group}/quality": "mine" for task_group in TASK_GROUP_IDS
                },
            }
        },
    }

    routes = load_model_routes(user_config=user)
    assert routes.presets["agy-hybrid"].test_target_id == "gemini-free-3_5-flash-lite"

    # Selecting it *does* check it, and now it really is unreachable.
    with pytest.raises(ModelRouteConfigError, match="not in any bound model group"):
        load_model_routes(user_config={**user, "preset": "agy-hybrid"})


def test_the_agy_hybrid_test_target_stays_on_the_lite_tier() -> None:
    """A `--test-profile` run pins this target for every single call.

    Full Flash carries 20 RPD against the lite tier's 500, so quietly
    moving this off the lite model costs 25x the test headroom.
    """

    routes = default_model_routes()

    assert routes.presets["agy-hybrid"].test_target_id == "gemini-free-3_5-flash-lite"
    assert routes.target_fact("gemini-free-3_5-flash-lite").rpd == 500


def test_the_pure_agy_preset_names_no_api_target_anywhere() -> None:
    """`agy` means "the subscription answers everything", test pass included.

    The test target is the trap: it is pinned for every call in a
    `--test-profile` run, so inheriting the default preset's lite Gemini would
    send that whole run to an API -- the one thing this preset exists to avoid.
    Its bound groups have to be clean too, or the preset is `agy-hybrid` with
    extra steps.
    """

    routes = default_model_routes()
    preset = routes.presets["agy"]

    bound = {
        target_id
        for group_id in set(preset.bindings.values())
        for target_id in routes.model_groups[group_id].target_ids
    }
    assert bound, preset.bindings
    assert all(
        routes.targets[target].backend == "local_agent" for target in bound
    ), bound
    assert preset.test_target_id in bound
    assert routes.targets[preset.test_target_id].backend == "local_agent"


# A user override catalog + the config.toml half that composes it. Facts live
# in the catalog, composition in config.toml (owner decision 2026-08-12).
USER_CATALOG = """[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com/"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000
max_output_tokens = 32768
thinking = ["high", "medium", "low"]
capability = "low"
"""


def _user_catalog(tmp_path: Path, text: str = USER_CATALOG):
    path = tmp_path / "model_catalog.toml"
    path.write_text(text, encoding="utf-8")
    return load_effective_catalog(path)


def _user_config() -> dict:
    return {
        "preset": "mine",
        "model_groups": {"my-corr": {"targets": ["ds-flash", "gemini-free-3_8-flash"]}},
        "presets": {
            "mine": {"name": "mine", "bindings": {"correction-text/quality": "my-corr"}}
        },
    }


def test_an_unstated_capability_is_no_claim_rather_than_a_bad_one(
    tmp_path: Path,
) -> None:
    """Blank used to mean a score of 50, i.e. below every shipped floor.

    So a person adding a model they had simply not judged got a warning
    phrased as if they had judged it badly, and the only way out was to invent
    a number. Since 2026-09-04 blank is `UNSTATED_CAPABILITY` -- it clears
    every floor -- and the fact that it did is a note, so the permissiveness is
    still visible to anyone reading the log.
    """

    from finesub.llm.routing.model_catalog import UNSTATED_CAPABILITY

    blank = USER_CATALOG.replace('capability = "low"\n', "")
    routes = load_model_routes(
        user_config=_user_config(), catalog=_user_catalog(tmp_path, blank)
    )
    fact = routes.facts["ds-flash"]

    assert fact.capability is None
    assert fact.effective_capability is UNSTATED_CAPABILITY
    assert not any(
        "低于下限" in message
        for message in routes.preset_binding_warnings("mine")
    )
    notes = routes.preset_binding_notes("mine")
    assert [note for note in notes if "ds-flash" in note and "未声明" in note]
    # Once per bound group, not once per cell: the same group answers several.
    assert len(notes) == len([note for note in notes if "my-corr" in note])


def test_a_stated_low_capability_still_trips_the_floor(tmp_path: Path) -> None:
    """The note is for silence, not for a low tier anyone actually wrote."""

    scored = USER_CATALOG.replace('capability = "low"', 'capability = "very-low"')
    routes = load_model_routes(
        user_config=_user_config(), catalog=_user_catalog(tmp_path, scored)
    )

    assert routes.facts["ds-flash"].capability is Capability.VERY_LOW
    assert any(
        "低于下限" in message
        for message in routes.preset_binding_warnings("mine")
    )
    assert routes.preset_binding_notes("mine") == []


# The context window binds the input ceiling, so the output reserve shows in it.
CAPPED_CATALOG = """[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com/"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000
max_output_tokens = 32768
context_window = 140000
thinking = ["high", "medium", "low"]
capability = "medium"
"""


def _capped_routes(tmp_path: Path, *, prepare: str = "", window: str = "", research: bool = False):
    config = _user_config()
    if research:
        config["presets"]["mine"]["bindings"]["research/quality"] = "my-corr"
    caps = "".join(f"{key} = {value}\n" for key, value in
                   (("prepare_chunk_tokens", prepare), ("window_subtitle_tokens", window)) if value)
    return load_model_routes(user_config=config, catalog=_user_catalog(tmp_path, CAPPED_CATALOG + caps))


@pytest.mark.parametrize("value", ["0", "-5", '"x"'])
def test_a_quality_cap_must_be_a_positive_integer(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match="window_subtitle_tokens"):
        _capped_routes(tmp_path, window=value)


def test_a_group_plans_with_its_smallest_quality_cap_a_blank_one_counting_as_the_default(tmp_path: Path) -> None:
    from finesub.llm.routing.config import DEFAULT_LIMITS, planning_limits_for

    blank = _capped_routes(tmp_path)
    assert blank.facts["ds-flash"].window_subtitle_tokens is None
    limits = planning_limits_for("correction-text", "quality", routes=blank)
    assert (limits.max_window_subtitle_tokens, limits.research_transcript_max_tokens, limits.knowledge_csv_tokens) == (
        10_000, 50_000, 100_000) == (DEFAULT_LIMITS.max_window_subtitle_tokens,
                                     DEFAULT_LIMITS.research_transcript_max_tokens, DEFAULT_LIMITS.knowledge_csv_tokens)
    smaller = _capped_routes(tmp_path, window="6000")
    assert planning_limits_for("correction-text", "quality", routes=smaller).max_window_subtitle_tokens == 6000
    # raising it takes every member: the blank Gemini member still counts as 10000
    larger = _capped_routes(tmp_path, window="20000")
    assert planning_limits_for("correction-text", "quality", routes=larger).max_window_subtitle_tokens == 10_000
    # a cap only shapes planning: the call's identity is unchanged
    assert smaller.routing_identity_digest == blank.routing_identity_digest != None  # noqa: E711
    assert smaller.advisory_digest != blank.advisory_digest


def test_the_quality_cap_warning_names_the_member_once_per_group_and_cap_its_cells_read(tmp_path: Path) -> None:
    warnings = _capped_routes(tmp_path, window="6000", prepare="30000").quality_cap_warnings("mine")
    # bound to the correction cell only: the chunk cap it states is read by no cell of this group
    assert warnings == ["模型组 my-corr 的成员 window_subtitle_tokens 不一致，按其中最小的 6000 切分（由 ds-flash 决定；"
                        "没写这项的成员按默认 10000 计；`config.toml` 的 `[chunking]` 写了时按它）"]
    both = _capped_routes(tmp_path, window="6000", prepare="30000", research=True)
    assert len(both.quality_cap_warnings("mine")) == 2
    native = both.quality_cap_warnings("mine", research_chunks=False)
    assert [warning for warning in native if "prepare_chunk_tokens" in warning] == [], "native research never chunks"
    assert _capped_routes(tmp_path).quality_cap_warnings("mine") == []
    assert not [note for note in both.preset_binding_notes("mine") if "切分" in note], "a warning, not a note"


def test_an_explicit_cap_outranks_the_groups_and_the_reserve_follows_the_groups(tmp_path, monkeypatch) -> None:
    from dataclasses import replace

    from finesub import config as app_config
    from finesub.llm.research import research_transcript_cap
    from finesub.llm.routing import model_routes
    from finesub.llm.routing.capabilities import correction_planning_limits
    from finesub.llm.routing.config import effective_window_subtitle_cap
    from finesub.llm.routing.profiles import DEFAULT_PROFILE

    profile = replace(DEFAULT_PROFILE, correction_media="text", difficulty="quality")
    monkeypatch.setattr(model_routes, "default_model_routes", lambda: _capped_routes(tmp_path))
    default = correction_planning_limits(profile)
    monkeypatch.setattr(model_routes, "default_model_routes",
                        lambda: _capped_routes(tmp_path, window="6000", prepare="30000", research=True))
    capped = correction_planning_limits(profile)
    assert capped.max_window_subtitle_tokens == 6000
    assert capped.prompt_input_limit > default.prompt_input_limit, "a smaller window reserves less output"
    assert effective_window_subtitle_cap(8000, capped) == 8000 and effective_window_subtitle_cap(None, capped) == 6000
    config = tmp_path / "config.toml"
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(config))
    for body, expected in (("", 30_000), ("research_transcript_max_tokens = 50000\n", 50_000)):
        config.write_text("[chunking]\n" + body, encoding="utf-8")
        app_config.clear_config_cache()
        assert research_transcript_cap(profile) == expected


@pytest.mark.parametrize("difficulty", ("quality", "efficiency"))
def test_binding_an_avoided_model_into_any_cell_of_that_task_warns_once(
    difficulty: str,
) -> None:
    """`avoid` speaks about the task itself, so unlike the capability floor it
    is checked in every difficulty cell -- and said once per model."""

    routes = load_model_routes(
        user_config={
            "preset": "mine",
            "presets": {
                "mine": {
                    "name": "mine",
                    "bindings": {
                        "correction-text/quality": "correction-capable",
                        f"knowledge/{difficulty}": "local-codex-gpt-6-luna",
                    },
                }
            },
        }
    )
    avoid = [m for m in routes.preset_binding_warnings("mine") if "不推荐" in m]
    assert len(avoid) == 1
    assert avoid[0].startswith("knowledge: 成员 local-codex-gpt-6-luna")
    assert "3.6 Flash" in avoid[0]
    # The packaged presets bind no avoided model.
    assert not [m for m in routes.preset_binding_warnings("default") if "不推荐" in m]


def test_avoid_must_name_a_task_group(tmp_path: Path) -> None:
    path = tmp_path / "model_catalog.toml"
    path.write_text(
        '[models.gemini-free-3_6-flash]\navoid = { knowlege = "typo" }\n', encoding="utf-8"
    )
    with pytest.raises(ModelRouteConfigError, match="avoid names unknown task groups"):
        load_model_routes(catalog=load_effective_catalog(path))


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
    assert med_group.id == "my-corr" and med_cell.variant == "capableC"
    low_group, low_cell = routes.resolve_binding("mine", "correction-text", "efficiency")
    assert low_group.id == "my-corr" and low_cell.variant == "basicA"
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

    Hashing the text put the task-group floor, the preset display name -- and any
    comment -- into every resume key, so re-tuning an advisory floor silently
    discarded every completed window. The criterion is unchanged: changes what
    the run produces -> routing; changes how we judge it -> advisory.
    """

    source = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm" / "routing" / "model_routes.toml"
    original = source.read_text(encoding="utf-8")
    base = load_model_routes(source)

    for edit, expect_advisory_change in (
        (lambda text: text.replace('min_capability = "medium"', 'min_capability = "low"', 1), True),
        (lambda text: text.replace('name = "默认"', 'name = "出厂"', 1), True),
        (lambda text: text.replace("# min_capability guards", "# tweaked comment", 1), False),
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

    Per cell, not per task group: correction binds a different group at efficiency
    (the declared downgrade tier), so that cell is calibrated to the lite.
    """

    routes = load_model_routes()

    assert routes.task_group_reference_model("correction-mm") == "3.8 Flash"
    assert (
        routes.task_group_reference_model("correction-mm", "intermediate")
        == "3.8 Flash"
    )
    assert (
        routes.task_group_reference_model("correction-text", "efficiency")
        == "3.5 Flash Lite"
    )
    assert routes.task_group_reference_model("research") == "3.6 Flash"
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

    path = tmp_path / "model_catalog.toml"
    for written in ("", 'max_input_tokens = ""\n'):
        path.write_text(
            "[models.ds]\n"
            'provider_tier = "deepseek"\n'
            'provider_kind = "openai_compat"\n'
            'base_url = "https://api.deepseek.com"\n'
            'api_model_id = "deepseek-v4-flash"\n' + written,
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing max_input_tokens"):
            load_model_catalog(path)


def test_catalog_row_requires_a_base_url_for_text_dialects(tmp_path: Path) -> None:
    path = tmp_path / "model_catalog.toml"
    path.write_text(
        "[models.ds]\n"
        'provider_tier = "deepseek"\n'
        'provider_kind = "openai_compat"\n'
        'api_model_id = "deepseek-v4-flash"\n'
        "max_input_tokens = 128000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_url is required"):
        load_model_catalog(path)


def test_catalog_rejects_unknown_keys(tmp_path: Path) -> None:
    """A typo in a key is otherwise a silently missing fact."""

    path = tmp_path / "model_catalog.toml"
    path.write_text(
        "[models.gemini-free-3_6-flash]\nmax_output_token = 20\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown key 'max_output_token'"):
        load_effective_catalog(path)


def test_override_catalog_patches_a_packaged_model_in_place(tmp_path: Path) -> None:
    """Merging is by model name, key by key, order preserved -- a re-tuned
    limit must not move the model inside its group."""

    path = tmp_path / "model_catalog.toml"
    path.write_text(
        "[models.gemini-free-3_6-flash]\n"
        "max_input_tokens = 100000\n"
        'capability = "high"\n',
        encoding="utf-8",
    )
    packaged = load_model_catalog()
    merged = load_effective_catalog(path)

    assert [entry.fact_id for entry in merged] == [
        entry.fact_id for entry in packaged
    ]
    overridden = {entry.fact_id: entry for entry in merged}["gemini-free-3_6-flash"]
    original = {entry.fact_id: entry for entry in packaged}["gemini-free-3_6-flash"]
    assert overridden.max_input_tokens == 100_000
    assert overridden.capability is Capability.HIGH
    assert overridden.self_reported is True
    # Keys the patch leaves out keep the packaged values.
    assert overridden.supports_audio is True
    assert overridden.rpm == original.rpm


def test_an_empty_string_takes_the_loader_default_not_the_packaged_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model_catalog.toml"
    path.write_text(
        "[models.local-codebuddy-hy3]\n"
        'capability = ""\n'
        'fallback_model = ""\n'
        'context_window = ""\n'
        "is_free = false\n",
        encoding="utf-8",
    )
    entry = {e.fact_id: e for e in load_effective_catalog(path)}["local-codebuddy-hy3"]
    assert entry.capability is None
    assert entry.fallback_model == ""
    assert entry.context_window == entry.max_input_tokens + entry.max_output_tokens
    # `false` is a value, not "unstated".
    assert entry.is_free is False


@pytest.mark.parametrize(
    "line",
    (
        'supports_audio = "true"',
        'rpm = "100"',
        'thinking = ["24576", "16384", "8192"]',
        'thinking = ["high", "low"]',
        "max_output_tokens = true",
        'avoid = ["knowledge"]',
    ),
)
def test_a_typed_key_refuses_another_type(tmp_path: Path, line: str) -> None:
    path = tmp_path / "model_catalog.toml"
    path.write_text(f"[models.gemini-free-3_6-flash]\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_effective_catalog(path)


def test_annotating_a_packaged_model_leaves_its_numbers_measured(tmp_path: Path) -> None:
    path = tmp_path / "model_catalog.toml"
    path.write_text(
        "[models.gemini-free-3_6-flash]\n"
        'note = "mine"\n'
        'avoid = { research = "slow" }\n',
        encoding="utf-8",
    )
    entry = {e.fact_id: e for e in load_effective_catalog(path)}["gemini-free-3_6-flash"]
    assert entry.self_reported is False
    assert entry.note == "mine"
    assert entry.avoid == (("research", "slow"),)


def test_the_packaged_catalog_writes_no_empty_strings() -> None:
    """`""` means "the loader's default" -- in the packaged file that is just
    leaving the key out, and writing it there would read as an inheritance
    that does not exist."""

    from finesub.llm.routing.model_catalog import _catalog_path, read_catalog_tables

    blanks = [
        f"{fact_id}.{key}"
        for fact_id, table in read_catalog_tables(_catalog_path()).items()
        for key, value in table.items()
        if value == ""
    ]
    assert blanks == []


def test_an_old_format_override_is_refused_with_the_conversion_command(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model_catalog.psv"
    path.write_text("fact_id|provider_tier|api_model_id|max_input_tokens\n", encoding="utf-8")
    with pytest.raises(ValueError, match="python -m finesub.llm.routing.migrate_catalog"):
        load_effective_catalog(path)


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

    # The documented catalog table must load through the real catalog parser.
    catalog_text = block.split("model_catalog.toml 的例子：", 1)[1].split("\n#\n")[1]
    catalog_lines = [line[4:] for line in catalog_text.splitlines() if line.startswith("#   ")]
    assert catalog_lines and catalog_lines[0].startswith("[models."), catalog_lines
    catalog_path = tmp_path / "model_catalog.toml"
    catalog_path.write_text("\n".join(catalog_lines) + "\n", encoding="utf-8")
    catalog = load_effective_catalog(catalog_path)

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
    # The sample's row states no `capability`, and since 2026-09-04 that is
    # a note rather than a floor warning -- an unstated tier is no claim, so
    # it clears every floor and is logged once instead.
    warnings = routes.preset_binding_warnings("my-deepseek")
    assert not any("低于下限" in message for message in warnings)
    notes = routes.preset_binding_notes("my-deepseek")
    assert any("ds-flash" in note and "未声明" in note for note in notes)


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
                "gemini-free-3_5-flash-lite": {"targets": ["gemini-free-3_8-flash"]}
            },
            "presets": {
                "default": {
                    "name": "同名",
                    "test_target": "gemini-free-3_8-flash",
                    "bindings": {
                        f"{task_group}/quality": "gemini-free-3_5-flash-lite"
                        for task_group in TASK_GROUP_IDS
                    },
                }
            },
        }
    )

    group, _cell = routes.resolve_binding("default", "research", "quality")
    assert group.target_ids == ("gemini-free-3_8-flash",)


def test_the_single_target_namespace_cannot_be_declared() -> None:
    with pytest.raises(ModelRouteConfigError, match="reserved"):
        load_model_routes(
            user_config={
                "model_groups": {
                    "target:mine": {"targets": ["gemini-free-3_8-flash"]}
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
            "correction-basic": {"targets": ["gemini-free-3_1-flash-lite"]},
            "mine": {"targets": ["gemini-free-3_6-flash"]},
        },
        "presets": {
            "default": {
                "name": "我的默认",
                # Must be reachable from a bound group -- the loader checks.
                "test_target": "gemini-free-3_6-flash",
                "bindings": {
                    f"{task_group}/quality": "mine"
                    for task_group in TASK_GROUP_IDS
                },
            }
        },
    }
    routes = load_model_routes(user_config=user)

    assert routes.model_groups["correction-basic"].target_ids == (
        "gemini-free-3_1-flash-lite",
    )
    assert routes.presets["default"].name == "我的默认"
    group, _cell = routes.resolve_binding("default", "research", "quality")
    assert group.id == "mine"
    # Both overrides are announced; the new id is not (it collides with nothing).
    assert set(routes.override_notices) == {"模型组 correction-basic", "预设 default"}
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
        '"gemini-free-3_8-flash"]',
        1,
    )
    path = tmp_path / "routes.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ModelRouteConfigError, match="group of its own"):
        load_model_routes(path)


def test_a_config_naming_a_deleted_variant_fails_and_lists_the_live_ones(tmp_path: Path) -> None:
    """capableB was deleted (2026-10): a config still naming it, as a cell's `variant` or in a model group's
    `variant_overrides`, fails on load with an error that names it and lists the variants that exist. The
    correct-only sets (`-polish`) are not named by a route: a task's mode derives them."""

    from finesub.llm.prompt_variants import CORRECT_ONLY_SUFFIX, VARIANTS
    from finesub.llm.routing.model_routes import ALLOWED_VARIANTS

    named = {name for name in VARIANTS if not name.endswith(CORRECT_ONLY_SUFFIX)}
    assert ALLOWED_VARIANTS == {"", *named}
    available = ", ".join(sorted(named))

    with pytest.raises(ModelRouteConfigError) as overrides:
        load_model_routes(user_config={"model_groups": {"correction-capable": {
            "targets": ["gemini-free-3_8-flash"], "variant_overrides": {"gemini-free-3_8-flash": "capableB"}}}})
    assert "'capableB'" in str(overrides.value) and available in str(overrides.value)

    marker = 'cells = { quality = { variant = "capableC" }'
    text = _packaged_routes_text()
    assert marker in text
    path = tmp_path / "routes.toml"
    path.write_text(text.replace(marker, marker.replace("capableC", "capableB"), 1), encoding="utf-8")
    with pytest.raises(ModelRouteConfigError) as cell:
        load_model_routes(path)
    assert "'capableB'" in str(cell.value) and available in str(cell.value)


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


def _bound_targets(routes, preset_id: str, cell: tuple[str, str]) -> tuple[str, ...]:
    return routes.model_groups[routes.presets[preset_id].bindings[cell]].target_ids


def test_a_preferred_target_rebinds_the_cell_exclusively() -> None:
    """"Use this one" now means EXACTLY this one (owner decision 2026-08-28,
    kb-followups plan B): the cell rebinds to a single-member group with no
    fallback chain — a call the pin cannot serve fails loudly instead of
    silently routing elsewhere, and a deafened media cell says so at load."""

    routes = load_model_routes(
        user_config={"preferred_targets": {"default": "local-agy-sonnet-5_5"}}
    )

    overlaid = _bound_targets(routes, "default", ("correction-mm", "quality"))
    assert overlaid == ("local-agy-sonnet-5_5",)
    # a text-only pin deafens the media cell — surfaced as a startup warning
    assert any(
        "没有成员支持音频" in message
        for message in routes.preset_binding_warnings("default")
    )


def test_a_per_task_group_preference_beats_the_blanket_one() -> None:
    routes = load_model_routes(
        user_config={
            "preferred_targets": {
                "default": "local-agy-sonnet-5_5",
                "research": "gemini-free-3_5-flash-lite",
            }
        }
    )

    assert _bound_targets(routes, "default", ("research", "quality")) == (
        "gemini-free-3_5-flash-lite",
    )
    assert _bound_targets(routes, "default", ("knowledge", "quality")) == (
        "local-agy-sonnet-5_5",
    )


def test_a_preferred_model_group_rebinds_wholesale() -> None:
    """A group value binds the cell to that group — its own chain, unchanged."""

    routes = load_model_routes(
        user_config={"preferred_targets": {"knowledge": "research-default"}}
    )
    assert routes.presets["default"].bindings[("knowledge", "quality")] == (
        "research-default"
    )
    assert _bound_targets(routes, "default", ("knowledge", "quality")) == (
        routes.model_groups["research-default"].target_ids
    )


def test_a_preference_takes_a_group_over_a_model_target_of_the_same_name() -> None:
    """As preset bindings do: a model added to the catalog never takes over a
    pin that already works. A declared target sharing the name is ambiguous."""

    from finesub.llm.routing.model_routes import install_runtime_preferred

    groups = {"local-agy-sonnet-5_5": {"targets": ["gemini-free-3_8-flash"]}}
    routes = load_model_routes(
        user_config={"model_groups": groups, "preferred_targets": {"default": "local-agy-sonnet-5_5"}}
    )
    assert _bound_targets(routes, "default", ("research", "quality")) == ("gemini-free-3_8-flash",)
    try:
        install_runtime_preferred({"default": "local-agy-sonnet-5_5"})
        routes = load_model_routes(user_config={"model_groups": groups})
        assert _bound_targets(routes, "default", ("research", "quality")) == ("gemini-free-3_8-flash",)
    finally:
        install_runtime_preferred(None)

    with pytest.raises(ModelRouteConfigError, match="ambiguous"):
        load_model_routes(
            user_config={
                "model_groups": {"gemini-free-3_8-flash": {"targets": ["gemini-free-3_5-flash-lite"]}},
                "preferred_targets": {"default": "gemini-free-3_8-flash"},
            }
        )


def test_a_preference_invalidates_a_resume_that_was_planned_without_it() -> None:
    """Which models answer is routing identity, so checkpoints must not survive."""

    assert (
        load_model_routes(
            user_config={"preferred_targets": {"default": "local-agy-sonnet-5_5"}}
        ).routing_identity_digest
        != default_model_routes().routing_identity_digest
    )


def test_a_preference_must_name_a_real_target_and_a_real_task_group() -> None:
    with pytest.raises(ModelRouteConfigError, match="names no known target"):
        load_model_routes(user_config={"preferred_targets": {"default": "nope"}})
    with pytest.raises(ModelRouteConfigError, match="names no task group"):
        load_model_routes(
            user_config={"preferred_targets": {"correction": "local-agy-sonnet-5_5"}}
        )


def test_a_declared_group_may_not_squat_the_overlay_namespace() -> None:
    with pytest.raises(ModelRouteConfigError, match="reserved for preferred-target"):
        load_model_routes(
            user_config={
                "model_groups": {
                    "preferred:x": {"targets": ["gemini-free-3_5-flash-lite"]}
                }
            }
        )


# ---- runtime overlay (--llm-model, kb-followups plan B) ----------------------


def test_parse_llm_model_args_forms() -> None:
    from finesub.llm.routing.model_routes import parse_llm_model_args

    assert parse_llm_model_args(None) == {}
    assert parse_llm_model_args(["X"]) == {"default": "X"}
    assert parse_llm_model_args(["X", "research=Y", "research=Z"]) == {
        "default": "X", "research": "Z",
    }
    with pytest.raises(ModelRouteConfigError, match="llm-model"):
        parse_llm_model_args(["research="])


def test_runtime_overlay_chain_and_lifecycle() -> None:
    """Resolution chain CLI[组] > CLI[default] > config[组] > config[default];
    installing empty clears — a second main() in one interpreter never
    inherits the previous run's override."""

    from finesub.llm.routing.model_routes import install_runtime_preferred

    try:
        # CLI bare default beats a config per-group pin
        install_runtime_preferred({"default": "local-agy-sonnet-5_5"})
        routes = load_model_routes(
            user_config={"preferred_targets": {"research": "gemini-free-3_5-flash-lite"}}
        )
        assert _bound_targets(routes, "default", ("research", "quality")) == (
            "local-agy-sonnet-5_5",
        )
        # CLI per-group beats CLI default
        install_runtime_preferred(
            {"default": "local-agy-sonnet-5_5", "research": "gemini-free-3_5-flash-lite"}
        )
        routes = load_model_routes(user_config={})
        assert _bound_targets(routes, "default", ("research", "quality")) == (
            "gemini-free-3_5-flash-lite",
        )
    finally:
        install_runtime_preferred(None)
    # cleared: back to the declared composition
    assert _bound_targets(load_model_routes(user_config={}), "default", ("research", "quality")) != (
        "local-agy-sonnet-5_5",
    )


def test_runtime_overlay_reaches_default_routes_and_planning() -> None:
    """贯通性: the memoized default loader — what planning envelopes, preflight
    and internally-constructed clients read — reflects the installed override,
    and the routing identity moves so stale resumes invalidate."""

    from finesub.llm.routing.model_routes import install_runtime_preferred

    baseline_identity = default_model_routes().routing_identity_digest
    try:
        install_runtime_preferred({"default": "local-agy-sonnet-5_5"})
        routes = default_model_routes()
        preset = routes.presets[routes.active_preset_id]
        for group_id in set(preset.bindings.values()):
            assert routes.model_groups[group_id].target_ids == ("local-agy-sonnet-5_5",)
        assert routes.routing_identity_digest != baseline_identity
    finally:
        install_runtime_preferred(None)
    assert default_model_routes().routing_identity_digest == baseline_identity


def test_the_output_cap_binds_a_window_before_the_input_ceiling_does() -> None:
    """What makes reserving the *estimate* safe while still requesting the ceiling.

    `correction_planning_limits` sets the input ceiling to
    `context_window - expected_output`, but a call still asks for the group's
    full `max_output` (`attempts.py`). Those two disagree by construction, and
    the disagreement is harmless only because no window can ever be planned
    large enough to enter the gap: `client._complete` drops a candidate on
    `estimated_input + max_tokens > context_window`, and a window that big is
    unreachable while the output cap binds first.

    The algebra, with `prompt ~= A + B*S` for `S` csv tokens (A ~ 10k, B ~ 2
    from a real window: 1925 csv -> 13893 prompt) and coefficient `c >= 2`:
    the input ceiling binds before the output cap only once
    `M * (0.9*B/c + 1) > W`, i.e. roughly `M > 0.6 * W`. Every member of every
    packaged model group sits far below that, so this is a tripwire for a
    future row (or a much larger quality cap), not a proof.

    Groups, not the catalog -- `gemini-free-gemma-4-31b` declares 32768 output
    against a 48768 window (0.67) and would trip it, but it is a grounded
    search target that belongs to no group and answers no correction window.
    Same reason `check_model_group_windows` scans groups.
    """

    routes = default_model_routes()
    offenders = []
    for group in routes.model_groups.values():
        for target_id in group.target_ids:
            fact = routes.target_fact(target_id)
            if fact.max_output_tokens > 0.6 * fact.context_window:
                offenders.append(
                    f"{fact.fact_id}: max_output {fact.max_output_tokens} > 0.6 x "
                    f"context_window {fact.context_window}"
                )
    assert not offenders, (
        "reserving the estimated output while still requesting the declared "
        "ceiling stops being safe for: " + "; ".join(sorted(offenders))
    )


def test_the_correction_envelope_reserves_the_estimate_not_the_ceiling() -> None:
    """The change itself: input ceiling follows `output_scale * c * cap`."""

    import math

    from finesub.llm.routing.capabilities import correction_planning_limits
    from finesub.llm.routing.config import effective_window_subtitle_cap
    from finesub.llm.routing.model_routes import (
        install_runtime_preferred,
        parse_llm_model_args,
    )
    from finesub.llm.routing.profiles import resolve_profile

    try:
        install_runtime_preferred(parse_llm_model_args(["local-codebuddy-hy3"]))
        profile = resolve_profile("text", "none", "quality")
        limits = correction_planning_limits(profile)
        routes = default_model_routes()
        fact = routes.facts["local-codebuddy-hy3"]
        cap = effective_window_subtitle_cap(None, limits)
        reserve = math.ceil(profile.output_scale * profile.output_coefficient * cap)

        assert limits.prompt_input_limit == fact.context_window - reserve
        # And the request is untouched: what a call asks for is still the
        # declared ceiling, which is the half that must not shrink.
        assert limits.output_limit == fact.max_output_tokens
    finally:
        install_runtime_preferred(None)
        default_model_routes.cache_clear()


def _catalog_row(**overrides):
    """A minimal packaged-shaped row, for guards that read whole tables."""

    from finesub.llm.routing.model_catalog import ModelCatalogEntry

    fields_ = dict(
        fact_id="row-1",
        provider_tier="LOCAL_CODEBUDDY",
        provider_kind="local_agent",
        base_url="",
        key_env="",
        display_name="row 1",
        api_model_id="m1",
        max_input_tokens=192000,
        max_output_tokens=64000,
        context_window=192000,
        supports_audio=False,
        supports_video=False,
        video_high_resolution_only=False,
        supports_native_search=True,
        thinking_levels=None,
        rpm=-1,
        tpm=-1,
        rpd=-1,
        tpd=-1,
        is_free=False,
        capability=Capability.LOW,
    )
    fields_.update(overrides)
    return ModelCatalogEntry(**fields_)


def test_per_model_quota_pool_derives_a_name_nobody_has_to_invent() -> None:
    """The sentinel is the point: a free line never needs a pool name.

    Naming them by hand (`CODEBUDDY_FREE_HY3`) means renaming one the day a
    line stops being free, which orphans whatever `.state` recorded under the
    old key -- and means two rows can share a pool by typo.
    """

    from finesub.llm.routing.model_catalog import PER_MODEL_QUOTA_POOL

    row = _catalog_row(quota_pool=PER_MODEL_QUOTA_POOL, api_model_id="hy3")

    assert row.effective_quota_pool == "LOCAL_CODEBUDDY:hy3"
    # Keyed on (tier, model) so an answering model name can be looked up by it.
    assert _catalog_row(quota_pool="").effective_quota_pool == "LOCAL_CODEBUDDY"
    assert (
        _catalog_row(quota_pool="CODEBUDDY_CREDITS").effective_quota_pool
        == "CODEBUDDY_CREDITS"
    )


def test_a_tier_with_several_allowances_refuses_to_guess_at_an_empty_pool() -> None:
    """"Empty" is a question nobody answered, not a default worth guessing.

    Either guess is wrong for some row on these tiers, and the expensive one
    (merging) takes a working line out for hours with no call left to thaw it.
    So the message says what to type instead.
    """

    from finesub.llm.routing.model_catalog import _validate_catalog_table

    with pytest.raises(ValueError) as excinfo:
        _validate_catalog_table([_catalog_row(quota_pool="")], "test table")

    message = str(excinfo.value)
    assert "per-model" in message and "CODEBUDDY_CREDITS" in message
    # The other per-line tier is covered by the same rule.
    with pytest.raises(ValueError):
        _validate_catalog_table(
            [_catalog_row(provider_tier="LOCAL_AGY", quota_pool="")], "test table"
        )
    # A tier backed by one subscription keeps the old meaning.
    _validate_catalog_table(
        [_catalog_row(provider_tier="LOCAL_CODEX", quota_pool="")], "test table"
    )


def test_a_free_row_sharing_a_paid_pool_warns_rather_than_failing_the_load(
    reported,
) -> None:
    """It warns on purpose, and the asymmetry is the finding.

    `is_free` drives nothing else and never has, so its values are unverified
    annotations. Promoting one to a load-blocking gate would stop a working
    configuration over a column nobody had reason to keep accurate -- which is
    exactly what happened the first time this ran (owner 2026-09-05).
    """

    from finesub.llm.routing.model_catalog import _validate_catalog_table

    _validate_catalog_table(
        [
            _catalog_row(fact_id="free", api_model_id="a", quota_pool="SHARED", is_free=True),
            _catalog_row(fact_id="paid", api_model_id="b", quota_pool="SHARED", is_free=False),
        ],
        "test table",
    )

    notices = [w for w in reported.warnings if "SHARED" in w.message]
    assert len(notices) == 1
    assert "free" in notices[0].message and "paid" in notices[0].message


def test_the_packaged_catalog_is_validated_even_with_no_override_file() -> None:
    """The effective table is reached two ways, and both must be checked.

    Both the single-file and the patched path go through one function that
    runs the whole-table checks, so neither can skip them.
    """

    import inspect

    from finesub.llm.routing import model_catalog

    assert "_entries_from_tables" in inspect.getsource(model_catalog.load_model_catalog)
    assert "_entries_from_tables" in inspect.getsource(model_catalog.load_effective_catalog)
    assert "_validate_catalog_table" in inspect.getsource(model_catalog._entries_from_tables)
