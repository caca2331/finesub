"""GPU tiers: what a tier means, and how `auto` picks one.

A tier answers "what class of card is this", and its one VRAM number is what
has to be **free** -- not what the card holds. The card's own size exists for a
single moment inside `tier_for_vram`, which subtracts a flat reserve from it to
estimate the free figure and then matches. Most of what these tests pin is that
those two never get confused: "my card is 8GB so it fits" is the mistake.
"""

from __future__ import annotations

import pytest

from finesub.reporting import NullReporter, reporting_to
from finesub.speech.runtime.resources import (
    AUTO_CEILING_LIFT_CAPACITY_GB,
    AUTO_GPU_TIER,
    AUTO_TIER_CEILING,
    BYTES_PER_GIB,
    DEFAULT_GPU_TIER,
    GPU_TIERS,
    RAM_BUDGET_GB,
    RESOURCE_PROFILES,
    get_resource_profile,
    gpu_tier_cli_choices,
    gpu_tier_help,
    check_tier_device_agreement,
    detect_gpu_tier,
    gpu_backed_tiers,
    gpu_tier_names,
    resolve_gpu_tier,
    reserve_for_capacity,
    resource_limit_violations,
    tier_for_vram,
    warn_if_vram_is_short,
)


def test_a_tier_carries_one_vram_number_and_it_is_the_free_one() -> None:
    assert gpu_tier_names() == (
        "cpu",
        "entry",
        "standard",
        "standard_large_vram",
        "high",
    )
    for spec in GPU_TIERS:
        profile = RESOURCE_PROFILES[spec.name]
        assert profile.gpu_tier == spec.name
        assert profile.usable_gpu_gb == pytest.approx(spec.usable_gib)
        assert profile.gpu_limit_bytes == int(spec.usable_gib * BYTES_PER_GIB)
        assert profile.gpu == spec.gpu
        assert profile.ram_budget_gb == RAM_BUDGET_GB
        assert profile.ram_limit_bytes == RAM_BUDGET_GB * BYTES_PER_GIB
    # Nothing downstream is handed the card's nominal size to misread.
    assert not hasattr(RESOURCE_PROFILES["standard"], "vram_gb")


def test_tiers_are_ordered_and_scale_only_the_separator() -> None:
    """ASR runs one worker at every tier; only separation scales.

    The separator count is no longer monotone in the VRAM figure, and that is
    the point of `standard_large_vram`: the two answer different questions --
    workers are a throughput choice (two is the measured peak), VRAM is a
    budget the referee spends. Requiring them to move together is what made a
    12GB card choose between headroom and a slower third worker.
    """

    assert [spec.usable_gib for spec in GPU_TIERS] == [0.0, 3.0, 6.5, 10.0, 10.0]
    assert [spec.vocal_separator_instances for spec in GPU_TIERS] == [1, 1, 2, 2, 3]
    assert [spec.gpu for spec in GPU_TIERS] == [False, True, True, True, True]
    assert [
        RESOURCE_PROFILES[spec.name].vocal_separation_batch_size for spec in GPU_TIERS
    ] == [1, 1, 1, 1, 1]
    # Ordered by requirement, never decreasing -- `tier_for_vram` takes the
    # last one that fits, so an out-of-order row would make it pick wrong.
    figures = [spec.usable_gib for spec in GPU_TIERS]
    assert figures == sorted(figures)


def test_the_help_text_carries_each_tier_and_its_requirement() -> None:
    """The `--help` line has to answer "which one am I" on its own.

    A named tier is only an improvement over a number if the name comes with
    the requirement attached; `standard` alone says nothing about 6.5GB.
    """

    help_text = gpu_tier_help()
    for spec in GPU_TIERS:
        assert f"{spec.name}=" in help_text
    for spec in gpu_backed_tiers():
        # The FREE figure, never the card's own size: "my card is 8GB so it
        # fits" is exactly the mistake this text exists to prevent.
        assert f"{spec.usable_gib:g}GB free VRAM" in help_text
    # ...and the CPU tier must not quote one, because it has none to quote.
    # Asserted on its own summary, not on the whole help string: "0GB" is a
    # substring of "10GB", so the obvious check passes for the wrong reason.
    cpu_spec = next(spec for spec in GPU_TIERS if not spec.gpu)
    assert "VRAM" not in cpu_spec.summary
    assert f"{RAM_BUDGET_GB}GB RAM" in help_text


def _pin_vram(monkeypatch, total_gib: float | None, *, present: bool = True) -> None:
    """Answer for the two driver questions without touching a real driver.

    Both, because since the `cpu` tier they are separate: `cuda_device_present`
    decides whether a GPU tier is meaningful at all, and only then does
    `total_vram_gib` size it. Pinning the size alone left these tests passing
    on a machine with a card and failing on one without -- which is every CI
    runner.
    """

    import finesub.speech.runtime.device as device_module

    monkeypatch.setattr(device_module, "total_vram_gib", lambda: total_gib)
    monkeypatch.setattr(device_module, "cuda_device_present", lambda: present)


def test_no_usable_gpu_falls_back_to_the_smallest_tier(monkeypatch) -> None:
    """A card that is *there* but cannot be sized lands on `entry`.

    That is one case, not two, since the `cpu` tier split them: "no CUDA
    device" now lands on `cpu`, and what is left here is the card this torch
    build has no kernels for -- present, unsizable, and still a GPU tier
    because CTranslate2 may run on it.
    """

    _pin_vram(monkeypatch, None)
    assert resolve_gpu_tier(AUTO_GPU_TIER) == DEFAULT_GPU_TIER
    assert get_resource_profile().gpu_tier == DEFAULT_GPU_TIER
    assert get_resource_profile().vocal_separator_instances == 1


def test_a_card_too_small_for_any_tier_still_gets_entry() -> None:
    """The third case, and a different branch: a real card that fits nothing.

    There is nothing below `entry` to fall to, so it takes `entry` and finds
    out from the OOM -- exactly what happened when the tier was picked by hand.
    Falling back to *no* tier, or refusing to run, would be worse: plenty of
    small cards finish short clips fine.
    """

    for capacity in (1, 2, 3, 3.4):
        assert tier_for_vram(capacity) == "entry", capacity


def test_unset_and_auto_are_the_same_request(monkeypatch) -> None:
    """An omitted option must not drift away from an explicit `auto`."""

    _pin_vram(monkeypatch, 11.7)
    assert (
        resolve_gpu_tier(None)
        == resolve_gpu_tier(AUTO_GPU_TIER)
        == "standard_large_vram"
    )


def test_auto_rounds_to_the_nominal_card_size_before_flooring() -> None:
    """Round to the nominal size, then subtract the reserve, then match.

    Both steps guard a way `auto` could be quietly wrong. The driver never
    reports the number on the box -- a 16GB RTX 5070 Ti answers 15.92, a 12GB
    card just under 12 -- so carrying the shortfall in drops cards a rung: a
    separator instance short, with nothing in the output saying so. And a tier
    asks for *free* VRAM, so without the reserve a 10GB card would match
    `high`'s 10GiB requirement against headroom that was never there.
    """

    assert tier_for_vram(7.79) == "standard"  # rounds to 8; 8 - 1.5 == 6.5
    assert tier_for_vram(7.2) == "entry"      # rounds to 7; 7 - 1.375 < 6.5
    assert [
        tier_for_vram(value)
        for value in (3.62, 3.94, 5.9, 7.79, 9.8, 11.7, 15.92, 23.6, 24.0, 31.5)
    ] == [
        "entry",
        "entry",
        "entry",
        "standard",
        # 9.8 rounds to 10; 10 - 1.75 = 8.25, short of the 10 the next rung asks.
        "standard",
        # 11.7 rounds to 12; 12 - 2.0 == 10.0 exactly, so the headroom rung fits.
        "standard_large_vram",
        "standard_large_vram",
        # 23.6 rounds to 24, which is exactly where the ceiling lifts.
        "high",
        "high",
        # Nothing above `high` exists, so a bigger card simply gets it.
        "high",
    ]


def test_the_reserve_curve_passes_exactly_through_the_tier_figures() -> None:
    """`capacity / 8 + 0.5` is chosen, not fitted -- this is what it is chosen for.

    At each tier's nominal card size the curve leaves exactly that tier's
    requirement free: 4GB -> 3.0, 8GB -> 6.5, 12GB -> 10.0. That is the whole
    reason the reserve is a rule instead of a fourth column -- the same three
    numbers, extended to cards the table never named. It also means the match
    has to be `<=`; a strict comparison would drop every nominal card a rung.
    """

    for capacity, spec in zip((4, 8, 12), gpu_backed_tiers()):
        assert capacity - reserve_for_capacity(capacity) == pytest.approx(
            spec.usable_gib
        ), spec.name
    # The two smallest are also what `auto` picks there; `high` is not, because
    # the ceiling holds it back until 24GB (see the ceiling test).
    assert tier_for_vram(4) == "entry"
    assert tier_for_vram(8) == "standard"

    # It keeps scaling past the table rather than flattening out.
    assert reserve_for_capacity(24) == pytest.approx(3.5)
    assert reserve_for_capacity(6) == pytest.approx(1.25)


def test_auto_stops_at_standard_until_the_card_is_large_enough() -> None:
    """Room is not a reason to run three separator workers, and three is
    measurably slower than two (`separator-optimization.md` E7) -- so promoting
    a 16GB card to `high` would hand it a slower run for owning a bigger card.

    But room IS a reason to let the referee spend it, which is why the ceiling
    is `standard_large_vram` and not `standard`: same two workers, `high`'s
    VRAM budget. `high` itself stays reachable by hand; `auto` only reaches for
    it once the card is big enough that the question is headroom, not speed.
    """

    # Every card that could otherwise fit `high` is held at the ceiling.
    for capacity in (12, 15.92, 16, 20, 23):
        assert tier_for_vram(capacity) == AUTO_TIER_CEILING, capacity

    assert tier_for_vram(AUTO_CEILING_LIFT_CAPACITY_GB) == "high"
    assert tier_for_vram(32) == "high"

    # The ceiling only ever lowers a choice -- a small card is not promoted to it.
    assert tier_for_vram(4) == "entry"


def test_cli_offers_auto_first_then_the_tiers_smallest_up() -> None:
    assert gpu_tier_cli_choices() == (
        AUTO_GPU_TIER,
        "cpu",
        "entry",
        "standard",
        "standard_large_vram",
        "high",
    )


def test_a_tier_name_is_case_insensitive_but_a_wrong_one_is_rejected() -> None:
    """Loud on a typo: a silently ignored tier is the failure mode `auto`
    replaced, and falling back to a default here would recreate it."""

    assert resolve_gpu_tier("STANDARD") == "standard"
    assert resolve_gpu_tier("  high  ") == "high"
    with pytest.raises(ValueError, match="Unsupported GPU tier"):
        resolve_gpu_tier("gigantic")
    # The retired 4-separator tier is not silently accepted either.
    with pytest.raises(ValueError, match="Unsupported GPU tier"):
        resolve_gpu_tier("max")
    # The old numeric spelling is gone, and must not be quietly accepted.
    with pytest.raises(ValueError, match="Unsupported GPU tier"):
        resolve_gpu_tier("8")


# ---- the "not enough free VRAM" warning ------------------------------------


def _pin_free(monkeypatch, free_gib: float | None) -> None:
    import finesub.speech.runtime.device as device_module

    monkeypatch.setattr(device_module, "free_vram_gib", lambda: free_gib)


def _warnings(
    monkeypatch, tier: str, free_gib: float | None, device: str = "cuda"
) -> list:
    import io

    from finesub.reporting import TerminalReporter, reporting_to

    _pin_free(monkeypatch, free_gib)
    stream = io.StringIO()
    with reporting_to(TerminalReporter(stream, level="normal", isatty=False)):
        warn_if_vram_is_short(
            get_resource_profile(tier), stage="vocal separation", device=device
        )
    return [line for line in stream.getvalue().splitlines() if line.startswith("Warning:")]


def test_a_cpu_run_is_never_warned_about_vram(monkeypatch) -> None:
    """`--device cpu` on a machine that HAS a card is the easy one to get wrong.

    The card is perfectly visible, so a check that only asks the driver warns
    about VRAM for a run that will never touch it. The CPU-fallback case is the
    same sentence from the other direction, and `resolve_device` has already
    said its piece there.
    """

    assert _warnings(monkeypatch, "standard", free_gib=0.5, device="cpu") == []
    # ...and the same conditions on the GPU path do warn, so this is the device
    # deciding rather than the test pinning a silent function.
    assert _warnings(monkeypatch, "standard", free_gib=0.5, device="cuda")


def test_a_short_card_is_warned_about_but_never_downgraded(monkeypatch) -> None:
    """The 6GB-card-picks-standard case: say so, run anyway.

    Downgrading would change the separator's worker count, hence the block plan,
    hence where artifact boundaries fall -- the same file would come out
    differently depending on what else the user had open. And short is not the
    same as doomed: `entry` measures 2.26GiB against a 3GiB requirement, so a
    card a little under still finishes plenty of clips.
    """

    [warning] = _warnings(monkeypatch, "standard", free_gib=4.2)

    assert "standard tier" in warning
    assert "6.50 GiB" in warning and "4.20 GiB" in warning
    # The profile itself is untouched -- nothing downgraded, nothing refused.
    assert get_resource_profile("standard").vocal_separator_instances == 2


def test_the_same_rule_applies_however_the_tier_was_chosen(monkeypatch) -> None:
    """A hand-picked tier is not second-guessed, and not exempted either.

    `warn_if_vram_is_short` never learns where the tier came from, which is the
    point: the user asked for a tier, and what they are owed is the fact.
    """

    by_hand = _warnings(monkeypatch, "standard", free_gib=4.2)
    _pin_free(monkeypatch, 4.2)
    from_auto = _warnings(monkeypatch, resolve_gpu_tier("standard"), free_gib=4.2)

    assert by_hand == from_auto and by_hand


def test_enough_free_vram_says_nothing(monkeypatch) -> None:
    assert _warnings(monkeypatch, "standard", free_gib=6.5) == []
    assert _warnings(monkeypatch, "standard", free_gib=12.0) == []


def test_the_smallest_tier_is_not_told_to_downgrade_to_itself(monkeypatch) -> None:
    """"Use a smaller tier" is the least useful thing to say to `entry`.

    It is also the case where the reader most needs a way out, so the action
    keeps the two that still exist: free some VRAM, or take the CPU path.
    """

    import io

    from finesub.reporting import TerminalReporter, reporting_to

    _pin_free(monkeypatch, 1.3)
    stream = io.StringIO()
    with reporting_to(TerminalReporter(stream, level="normal", isatty=False)):
        warn_if_vram_is_short(get_resource_profile("entry"), stage="VAD-ASR")
    [warning] = [
        line for line in stream.getvalue().splitlines() if line.startswith("Warning:")
    ]

    assert "--gpu-tier" not in warning
    assert "--device cpu" in warning

    # ...while a larger tier does get pointed at the smaller one.
    assert "--gpu-tier entry" in _warnings(monkeypatch, "standard", free_gib=1.3)[0]


def test_a_machine_with_no_usable_card_is_not_warned(monkeypatch) -> None:
    """It is on the CPU path; `resolve_device` already said so, and a second
    line about VRAM would be noise about a resource it is not using."""

    assert _warnings(monkeypatch, "entry", free_gib=None) == []


def test_resource_limit_violations_cover_gpu_and_ram_caps() -> None:
    profile = get_resource_profile("standard")
    assert resource_limit_violations(
        peak_gpu_bytes=profile.gpu_limit_bytes,
        peak_ram_bytes=profile.ram_limit_bytes,
        profile=profile,
    ) == []
    violations = resource_limit_violations(
        peak_gpu_bytes=profile.gpu_limit_bytes + 1,
        peak_ram_bytes=profile.ram_limit_bytes + 1,
        profile=profile,
    )
    assert len(violations) == 2
    assert "peak_gmem exceeds" in violations[0]
    assert "peak_mem exceeds" in violations[1]


def test_a_violation_says_it_in_units_a_person_reads() -> None:
    """The warning quoted raw byte counts, which say nothing actionable.

    `(3639984783 > 2791728472)` is "over budget" and not one thing more -- by
    how much, and against which tier, both needed a calculator. The debug line
    two calls earlier was already formatted; only the warning a user actually
    sees was not.
    """

    profile = get_resource_profile("entry")
    [violation] = resource_limit_violations(
        peak_gpu_bytes=int(3.39 * BYTES_PER_GIB),
        peak_ram_bytes=None,
        profile=profile,
    )

    assert "entry tier" in violation
    assert "3.39 GiB" in violation and "3.00 GiB" in violation
    assert "3639984783" not in violation

class TestCpuTier:
    """`cpu` is a POLICY tier: "do not ask the GPU", not "asked and it said no".

    Split out of `entry` on 2026-09-02. Before that a machine with no CUDA
    device was handed `entry`, so its artifacts claimed a 3 GiB VRAM budget it
    never had and the VRAM warning needed a hardcoded "unless there is no
    CUDA" branch.
    """

    def test_it_carries_no_vram_claim(self) -> None:
        profile = get_resource_profile("cpu")
        assert profile.gpu is False
        assert profile.usable_gpu_gb == 0.0
        # Still one separator worker: a tier always answers "how much
        # parallelism", and one is the right answer on the CPU.
        assert profile.vocal_separator_instances == 1

    def test_auto_takes_it_only_when_no_card_answers(self, monkeypatch) -> None:
        """The policy branch is "no device", not "torch cannot use the device".

        A card too old for this PyTorch build is still a GPU tier: another
        backend (the patched CTranslate2, with its own arch list) may run on
        it, and folding torch's verdict into the tier would decide that for
        every stage at once.
        """

        from finesub.speech.runtime import resources as module

        monkeypatch.setattr(module, "detect_gpu_tier", detect_gpu_tier)

        def pin(present: bool, total):
            monkeypatch.setattr(
                "finesub.speech.runtime.device.cuda_device_present", lambda: present
            )
            monkeypatch.setattr(
                "finesub.speech.runtime.device.total_vram_gib", lambda: total
            )

        pin(False, None)
        assert detect_gpu_tier() == "cpu"

        # Device present, but torch cannot size it -> smallest GPU tier.
        pin(True, None)
        assert detect_gpu_tier() == "entry"

        pin(True, 12.0)
        assert detect_gpu_tier() == "standard_large_vram"

    def test_a_machine_with_no_visible_device_is_not_a_gpu_tier(
        self, monkeypatch
    ) -> None:
        """`is_available()` says True with `CUDA_VISIBLE_DEVICES=""`; measured.

        That spelling is one the `device` module names as a way to have no
        usable CUDA, and asking `is_available()` put a machine with zero
        visible devices on a GPU tier. The predicate has to be `device_count`.
        """

        import finesub.speech.runtime.device as device_module

        monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(device_module.torch.cuda, "device_count", lambda: 0)
        assert device_module.cuda_device_present() is False

        monkeypatch.setattr(device_module.torch.cuda, "device_count", lambda: 1)
        assert device_module.cuda_device_present() is True

    def test_the_vram_warning_is_silent_on_it(self, monkeypatch) -> None:
        """And silent structurally, not by a special case for "no CUDA"."""

        monkeypatch.setattr(
            "finesub.speech.runtime.device.free_vram_gib", lambda: 0.1
        )
        messages: list[str] = []
        with reporting_to(_CollectingReporter(messages)):
            warn_if_vram_is_short(
                get_resource_profile("cpu"), stage="vocal separation", device="cpu"
            )
            assert messages == []
            # A GPU tier in the same situation still warns -- otherwise this
            # test would pass with the check deleted.
            warn_if_vram_is_short(
                get_resource_profile("entry"), stage="vocal separation", device="cuda"
            )
        assert any("gpu-vram-short" in message for message in messages)

    def test_a_gpu_tier_never_suggests_it_as_a_downgrade(self, monkeypatch) -> None:
        """`--gpu-tier cpu` is a different answer, not a smaller rung."""

        monkeypatch.setattr(
            "finesub.speech.runtime.device.free_vram_gib", lambda: 0.1
        )
        messages: list[str] = []
        with reporting_to(_CollectingReporter(messages)):
            warn_if_vram_is_short(
                get_resource_profile("high"), stage="VAD-ASR", device="cuda"
            )
        assert any("--gpu-tier entry" in message for message in messages)
        assert not any("--gpu-tier cpu" in message for message in messages)

    def test_asking_for_both_is_refused(self) -> None:
        """`--gpu-tier cpu --device cuda` has no reading under which both hold."""

        with pytest.raises(ValueError, match="does not use the GPU"):
            check_tier_device_agreement("cpu", "cuda")

        # The ordinary combinations stay legal.
        check_tier_device_agreement("cpu", "cpu")
        check_tier_device_agreement("entry", "cuda")
        check_tier_device_agreement("entry", "cpu")
        check_tier_device_agreement("auto", "cuda")
        check_tier_device_agreement("cpu", None)


class _CollectingReporter(NullReporter):
    def __init__(self, sink: list[str]) -> None:
        self.sink = sink

    def warning(self, code, message, **kwargs) -> None:  # noqa: D102
        self.sink.append(f"{code}: {message} {kwargs}")
