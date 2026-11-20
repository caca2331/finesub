"""The ASR backend seam: declared options, per-backend acceptance, and the loop's use.

`asr_backend` is the one declaration; these tests are what make it binding.
The second-backend fakes exist because with CTranslate2 as the only real
backend every "all backends agree" check is green by construction -- a guard
has to be shown to go red before its green means anything.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.speech.recognition import asr_backend

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


def _load(target: str):
    module_name, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


@pytest.mark.parametrize("name", sorted(asr_backend.BACKENDS))
def test_every_registered_backend_holds_the_contract(name: str) -> None:
    pytest.importorskip("faster_whisper")
    backend = _load(asr_backend.BACKENDS[name])

    assert asr_backend.contract_violations(backend) == []


class _MissingRequired:
    capabilities = frozenset({asr_backend.BEAM})
    accepted_decode_options = asr_backend.accepted_for(capabilities) - {"force_teacher_force"}


class _ClaimsBeamWithoutCapability:
    capabilities = frozenset()
    accepted_decode_options = asr_backend.accepted_for(frozenset({asr_backend.BEAM}))


class _Undeclared:
    pass


def test_the_contract_check_goes_red_for_a_broken_second_backend() -> None:
    assert asr_backend.contract_violations(_MissingRequired) == [
        "does not accept ['force_teacher_force']"
    ]
    assert asr_backend.contract_violations(_ClaimsBeamWithoutCapability) == [
        "accepts ['beam_size', 'best_of'] without the capability for them"
    ]
    assert asr_backend.contract_violations(_Undeclared) == ["declares no capabilities"]


def test_a_backend_without_beam_accepts_only_required_options() -> None:
    accepted = asr_backend.accepted_for(frozenset())

    assert "beam_size" not in accepted and "best_of" not in accepted
    assert {"language", "temperature", "force_teacher_force"} <= accepted
    with pytest.raises(ValueError, match="unknown ASR backend capabilities"):
        asr_backend.accepted_for(frozenset({"streaming"}))


def test_undeclared_options_cannot_be_built_or_received() -> None:
    with pytest.raises(TypeError, match="DECODE_OPTIONS"):
        asr_backend.decode_options(condition_on_previous_text=False)
    with pytest.raises(TypeError, match="does not implement"):
        asr_backend.reject_unknown(
            {"beam_size": 5}, asr_backend.accepted_for(frozenset()), backend="fake"
        )


def test_an_ordinary_decode_needs_no_capability() -> None:
    """What every window sends must be acceptable to a backend with no capabilities.

    Otherwise a contract-abiding backend without beam search would refuse every
    window (greedy used to be spelled beam_size=None).
    """

    pytest.importorskip("numpy")
    from finesub.speech.recognition import transcribe

    ordinary = {
        **transcribe._build_transcribe_kwargs(language="en"),
        **transcribe._signal_options(),
    }
    retry = {**ordinary, **asr_backend.decode_options(force_teacher_force=True)}

    minimal = asr_backend.accepted_for(frozenset())
    assert set(ordinary) <= minimal
    assert set(retry) <= minimal
    assert set(transcribe._rescue_decode_options()) <= asr_backend.accepted_for(
        frozenset({asr_backend.BEAM})
    )


def test_capability_lookup() -> None:
    assert asr_backend.supports(SimpleNamespace(), asr_backend.BEAM) is True
    assert asr_backend.supports(SimpleNamespace(capabilities=frozenset()), asr_backend.BEAM) is False
    with pytest.raises(ValueError):
        asr_backend.supports(SimpleNamespace(), "streaming")


def test_the_loop_spells_decode_options_only_through_the_declaration() -> None:
    """No dict literal or subscript in the alignment loop names a decode option.

    That is the half a registry alone cannot enforce: an option spelled inline
    reaches the backend without ever being declared.
    """

    # "language" alone is also an ordinary metadata field; it counts only in the
    # company of a decode-only name.
    decode_only = asr_backend.OPTION_NAMES - {"language"}
    path = SOURCE_ROOT / "finesub/speech/recognition/transcribe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        keys: list[ast.expr] = []
        if isinstance(node, ast.Dict):
            keys = [key for key in node.keys if key is not None]
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            keys = [node.slice]
        names = {
            key.value for key in keys if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        if names & decode_only:
            offenders.extend(
                f"transcribe.py:{node.lineno} {name!r}"
                for name in sorted(names & asr_backend.OPTION_NAMES)
            )

    assert offenders == []


def test_fw_refine_refuses_an_option_it_does_not_implement() -> None:
    pytest.importorskip("faster_whisper")
    from finesub.speech.recognition import fw_refine_backend

    model = fw_refine_backend.RefinedWhisperModel.__new__(
        fw_refine_backend.RefinedWhisperModel
    )
    with pytest.raises(TypeError, match="condition_on_previous_text"):
        model.transcribe_wt(None, condition_on_previous_text=False)
    with pytest.raises(TypeError, match="force_teacher_force"):
        fw_refine_backend.transcribe_batch(model, [], force_teacher_force=True)


def test_a_backend_without_beam_skips_the_beam_rung(monkeypatch) -> None:
    pytest.importorskip("numpy")
    from finesub.speech.recognition import transcribe

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the beam rung must not run without the capability")

    monkeypatch.setattr(transcribe, "_beam_rescue", forbidden)
    monkeypatch.setattr(transcribe, "_coverage_shortfall", lambda *_a: (2.0, 0.5, 1.5))
    segments = [{"start": 0.0, "end": 0.5, "text": "x", "words": []}]
    model = SimpleNamespace(capabilities=frozenset())

    kept = transcribe._rescue_low_coverage(
        model,
        [{"start": 0.0, "end": 2.0}],
        segments,
        None,
        16000,
        0.3,
        language="en",
    )

    assert kept is segments


def test_a_new_contract_option_turns_an_unchanged_backend_red(monkeypatch) -> None:
    """The check has to notice an option nobody implemented yet.

    The mutation is the real scenario: someone adds a required option to
    `DECODE_OPTIONS` and stops there. While `accepted_decode_options` was
    computed from that same tuple, every backend silently grew the claim and
    this test family reported nothing -- which is exactly the "declared but
    ignored" failure the contract exists to prevent.
    """

    import importlib.util
    import sys

    pytest.importorskip("faster_whisper")  # the backend module imports it at load
    from finesub.speech.recognition import fw_refine_backend

    grown = asr_backend.DECODE_OPTIONS + (asr_backend.DecodeOption("nucleus_p"),)
    monkeypatch.setattr(asr_backend, "DECODE_OPTIONS", grown)
    # Import the backend afresh under the grown contract, because that is the
    # shape of the real mistake: the class body runs at import, so a derived set
    # is recomputed too and agrees with the contract again. Checking the class
    # already imported left the old value in place and passed either way -- a
    # test of nothing. A private copy under its own name, not a reload: a reload
    # replaces the module's classes for every later test in the worker.
    spec = importlib.util.spec_from_file_location(
        f"{fw_refine_backend.__package__}._contract_probe", fw_refine_backend.__file__
    )
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, fresh)  # dataclasses look their module up while the body runs
    spec.loader.exec_module(fresh)
    problems = asr_backend.contract_violations(fresh.RefinedWhisperModel)

    assert problems and "nucleus_p" in problems[0]
