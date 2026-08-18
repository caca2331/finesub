"""Every switch combination still resolves to a coherent, routable plan.

The 0.3.2 restructure moved the LLM layer's pieces across module boundaries
(`profiles` -> `config` -> `model_routes` -> `model_router` -> `capabilities`).
Each of those has focused tests, but the thing a restructure actually breaks is
the *seam*: a combination that resolves in one module and is then rejected,
mis-typed or silently emptied by the next. Nothing walked the whole product of
the axes end to end, so a hole could only surface as a failed production run.

These are offline: planning, routing and capability checks only. No model is
called and no key is read.
"""

from __future__ import annotations

import itertools

import pytest

from finesub.llm.routing.capabilities import endpoint_supports
from finesub.llm.routing.config import planning_limits_for, role_config_for
from finesub.llm.routing.model_router import ModelRouter
from finesub.llm.routing.model_routes import default_model_routes, load_model_routes
from finesub.llm.routing.profiles import CONTINUITY, DIFFICULTY, MEDIA, RETRIEVAL, resolve_profile


ALL_VECTORS = list(
    itertools.product(MEDIA, MEDIA, RETRIEVAL, DIFFICULTY, CONTINUITY)
)


def _conflicts(vector) -> bool:
    """`difficulty=efficiency` pins the other axes (profiles.py constraint matrix).

    It is a hard error rather than a silent downgrade, so the matrix splits on
    it instead of skipping: the legal half must route, the illegal half must
    raise.
    """

    correction_media, planning_media, retrieval, difficulty, _continuity = vector
    return difficulty == "efficiency" and (
        correction_media != "text" or planning_media != "text" or retrieval != "none"
    )


LEGAL_VECTORS = [v for v in ALL_VECTORS if not _conflicts(v)]
CONFLICTING_VECTORS = [v for v in ALL_VECTORS if _conflicts(v)]


def _profile(correction_media, planning_media, retrieval, difficulty, continuity):
    return resolve_profile(
        retrieval=retrieval,
        difficulty=difficulty,
        continuity=continuity,
        correction_media=correction_media,
        planning_media=planning_media,
    )


def test_the_axes_are_the_ones_this_matrix_walks() -> None:
    """If an axis grows a value, this file must grow with it.

    Without this the matrix would keep passing while silently covering less.
    """

    assert MEDIA == ("text", "audio", "video")
    assert RETRIEVAL == ("none", "local", "native")
    assert DIFFICULTY == ("quality", "intermediate", "efficiency")
    assert CONTINUITY == ("serial", "parallel")
    assert len(ALL_VECTORS) == 3 * 3 * 3 * 3 * 2 == 162
    # efficiency pins media=text and retrieval=none, so it contributes only the
    # two continuity variants; everything else is free.
    assert len(LEGAL_VECTORS) == 3 * 3 * 3 * 2 * 2 + 2 == 110
    assert len(CONFLICTING_VECTORS) == 52


@pytest.mark.parametrize("vector", LEGAL_VECTORS, ids=lambda v: "-".join(v))
def test_every_switch_vector_resolves_and_routes(vector) -> None:
    """One combination, walked across every seam the restructure moved."""

    correction_media, planning_media, retrieval, difficulty, continuity = vector
    profile = _profile(*vector)

    # 1. The switch vector itself round-trips.
    assert profile.correction_media == correction_media
    assert profile.planning_media == planning_media
    assert profile.retrieval == retrieval
    assert profile.difficulty == difficulty
    assert profile.continuity == continuity

    routes = default_model_routes()
    correction_group = (
        "correction-mm" if correction_media != "text" else "correction-text"
    )
    planning_group = "planning-mm" if planning_media != "text" else "planning-text"

    for task_group in (correction_group, planning_group, "research", "search_judge"):
        # 2. The cell resolves to a config with a bound group and a variant.
        config = role_config_for(task_group, difficulty, routes=routes)
        assert config.model_group_id, (task_group, difficulty)
        assert config.task_group_id == task_group

        # 3. The group expands to a non-empty ordered chain under the shipped
        #    api-only policy -- an empty plan is exactly the failure mode a
        #    moved binding produces.
        plan = ModelRouter(routes, policy_id="api-only").plan(config)
        assert plan.candidates, (task_group, difficulty)
        assert plan.model_group_id == config.model_group_id

        # 4. Every candidate carries a usable endpoint identity.
        for candidate in plan.candidates:
            assert candidate.endpoint.api_model_id
            assert candidate.endpoint.provider_tier
            assert candidate.fact is not None

        # 5. Budgets are coherent, not merely present.
        limits = planning_limits_for(task_group, difficulty, routes=routes)
        assert limits.prompt_input_limit > 0
        assert limits.output_limit > 0
        assert limits.prompt_input_limit <= limits.context_limit


@pytest.mark.parametrize("vector", LEGAL_VECTORS, ids=lambda v: "-".join(v))
def test_media_and_retrieval_needs_are_actually_satisfiable(vector) -> None:
    """The capability filter must leave something behind for each cell.

    This is the half that bit before: routing produced a chain and the per-call
    filter then emptied it, which surfaces only at dispatch.
    """

    correction_media, planning_media, retrieval, difficulty, _continuity = vector
    routes = default_model_routes()

    for task_group, media in (
        ("correction-mm" if correction_media != "text" else "correction-text",
         correction_media),
        ("planning-mm" if planning_media != "text" else "planning-text",
         planning_media),
    ):
        config = role_config_for(task_group, difficulty, routes=routes)
        plan = ModelRouter(routes, policy_id="api-only").plan(config)
        survivors = [
            candidate
            for candidate in plan.candidates
            if endpoint_supports(
                candidate.endpoint,
                needs_audio=media in ("audio", "video"),
                needs_video=media == "video",
            )
        ]
        assert survivors, (task_group, media, difficulty)

    # retrieval=native is the one combination the shipped default cannot serve
    # from the free tier alone; it is served by the paid target's declared
    # google_search. Assert the reason rather than the outcome, so a catalog
    # edit that removes grounding fails here instead of in a run.
    if retrieval == "native":
        research = role_config_for("research", difficulty, routes=routes)
        plan = ModelRouter(routes, policy_id="api-only").plan(research)
        grounded = [
            candidate.target_id
            for candidate in plan.candidates
            if endpoint_supports(candidate.endpoint, needs_native_search=True)
        ]
        assert grounded, "no research candidate can ground under the default preset"


def test_forcing_a_lite_model_into_capable_cells_still_routes() -> None:
    """The cheap-coverage configuration, asserted rather than assumed.

    Pinning 3.5 Flash Lite into the capable cells is how these axes get
    exercised against a real API without spending the capable tier's budget --
    so the binding path it relies on (quick model selection) has to keep
    working, and it has to keep *warning*, since the lite model is below the
    correction and knowledge floors.
    """

    from finesub.llm.routing.model_routes import TASK_GROUP_IDS

    routes = load_model_routes(
        user_config={
            "preset": "lite-everywhere",
            "presets": {
                "lite-everywhere": {
                    "name": "3.5 Flash Lite everywhere",
                    "test_target": "gemini-free-3_5-flash-lite",
                    "bindings": {
                        f"{task_group}/{difficulty}": "gemini-free-3_5-flash-lite"
                        for task_group in TASK_GROUP_IDS
                        for difficulty in DIFFICULTY
                    },
                }
            },
        }
    )

    for task_group in TASK_GROUP_IDS:
        for difficulty in DIFFICULTY:
            config = role_config_for(task_group, difficulty, routes=routes)
            plan = ModelRouter(routes, policy_id="api-only").plan(config)
            assert [c.target_id for c in plan.candidates] == [
                "gemini-free-3_5-flash-lite"
            ], (task_group, difficulty)
            # Lite is multimodal, so even the -mm cells stay satisfiable.
            assert endpoint_supports(
                plan.candidates[0].endpoint, needs_audio=True, needs_video=True
            )

    # Below the floor for correction and knowledge -- and it says so.
    warnings = routes.preset_binding_warnings("lite-everywhere")
    warned_groups = {w.split("/", 1)[0] for w in warnings}
    assert "correction-mm" in warned_groups
    assert "knowledge" in warned_groups


@pytest.mark.parametrize("vector", CONFLICTING_VECTORS, ids=lambda v: "-".join(v))
def test_efficiency_refuses_the_axes_it_pins(vector) -> None:
    """A conflicting combination is an error, never a silent downgrade.

    The whole point of the constraint matrix is that asking for a cheap prompt
    *and* a video clip tells you so, rather than quietly dropping one.
    """

    from finesub.llm.routing.profiles import SwitchConflictError

    with pytest.raises(SwitchConflictError, match="efficiency"):
        _profile(*vector)
