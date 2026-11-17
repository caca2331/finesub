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
    if name == "fw-refine":
        pytest.importorskip("faster_whisper")
    else:
        pytest.importorskip("scipy")
    backend = _load(asr_backend.BACKENDS[name])

    assert asr_backend.contract_violations(backend) == []


def test_mlx_rejects_unknown_and_beam_before_runtime_load():
    pytest.importorskip("scipy")
    from finesub.speech.recognition.mlx_refine_backend import MlxRefineModel

    model = MlxRefineModel.__new__(MlxRefineModel)
    for option in ("beam_size", "best_of", "condition_on_previous_text", "typo"):
        with pytest.raises(TypeError, match="does not implement"):
            model.transcribe_wt(None, **{option: 1})
    assert not asr_backend.supports(model, asr_backend.BEAM)
    assert not asr_backend.supports(model, asr_backend.BATCH_DECODE)


def test_mlx_routing_respects_explicit_cpu_and_source_platform(monkeypatch):
    monkeypatch.setattr(asr_backend.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(asr_backend.platform, "machine", lambda: "arm64")
    assert asr_backend.resolve_backend(requested="auto") == "mlx-refine"
    assert asr_backend.resolve_backend("cpu", requested="mlx-refine") == "fw-refine"
    assert asr_backend.resolve_backend(gpu_tier="cpu", requested="auto") == "fw-refine"
    assert asr_backend.resolve_backend(requested="fw-refine") == "fw-refine"
    monkeypatch.setattr(asr_backend.platform, "system", lambda: "Windows")
    assert asr_backend.resolve_backend(requested="auto") == "fw-refine"
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        asr_backend.resolve_backend(requested="mlx-refine")


@pytest.mark.parametrize("tokens,unfinished", [
    ((200, 1, 205, 205, 2), True),
    ((200, 1, 205, 205, 2, 100), False),
    ((200, 1, 100), False),
])
def test_mlx_seek_and_trace_keep_tail_or_early_eot(tokens, unfinished):
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from finesub.speech.recognition.mlx_refine_backend import MlxDecodeTrace, _trace_spans
    from finesub.speech.recognition.mlx_seek import split_window

    tokenizer = SimpleNamespace(eot=100, timestamp_begin=200)
    trace = MlxDecodeTrace(tokens, tuple(-0.1 for _ in tokens),
                           np.zeros((len(tokens), 2, 1500)), np.zeros(1700))
    spans = _trace_spans(trace, tokenizer)
    decoded = np.array([t for t in tokens if t != tokenizer.eot])
    segments, seek, full_window = split_window(
        decoded, tokenizer, lambda **kw: kw, None, 30., 30., 3000, 3000, 2, .02)
    assert seek == 6000 and full_window
    assert len(segments) == len(spans)
    assert spans[-1].unfinished is unfinished
    assert segments[-1]["end"] == 60.
    assert sum(len(s["tokens"]) for s in segments) == len(decoded)
    for segment, span in zip(segments, spans):
        assert tuple(segment["tokens"]) == tuple(decoded[span.token_start:span.token_end + 1])
    if tokens[-1] == tokenizer.eot:
        assert spans[-1].tokens[-1] == tokenizer.eot
        assert trace.token_logprobs[spans[-1].token_end] == -0.1


def test_mlx_seek_source_rejects_unverified_shape():
    from finesub.speech.recognition.mlx_seek import adapt_seek_source

    with pytest.raises(RuntimeError, match="seek contract changed"):
        adapt_seek_source("def transcribe(): pass")


@pytest.mark.parametrize("early_eot", [False, True])
def test_mlx_adapter_uses_shared_confidence_and_retains_unfinished_word(monkeypatch, early_eot):
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from finesub.speech.recognition import mlx_refine_backend as backend

    class Tokenizer:
        eot, timestamp_begin, language = 100, 200, "en"

        def decode_with_timestamps(self, tokens):
            return "".join(f"<|{(t-200)*.02:.2f}|>" if t >= 200 else
                           {1: " Hello", 2: " world"}.get(t, "") for t in tokens)

    model = backend.MlxRefineModel.__new__(backend.MlxRefineModel)
    model.refine_frames = 0
    model._detect_disfluencies = model._collect_attention_signals = False
    model._collect_refine_signals = True
    tokens = (200, 1, 2, 100) if early_eot else (200, 1, 2)
    probs = (-9., -.1, -.3, -.2) if early_eot else (-9., -.1, -.3)
    model._latest_trace = backend.MlxDecodeTrace(tokens, probs,
                                                np.zeros((len(tokens), 2, 20)), np.zeros(240))
    monkeypatch.setattr(model, "_alignment", lambda *_a, **_kw:
                        ([(0, 0), (1, 1), (2, 2), (3 if early_eot else 2, 3)], np.ones((len(tokens), 20))))
    aligned = model._align_trace([{"tokens": [200, 1, 2]}], Tokenizer(), 40)
    assert [w["word"] for w in aligned[0].words] == ["Hello", "world"]
    assert aligned[0].words[-1]["confidence"] == .741
    assert any(event["type"] == "unfinished" for event in aligned[0].events) is not early_eot


def test_hardware_acceptance_counts_internal_teacher_force():
    from tools.hw_acceptance.score import rescue_comparison

    result = rescue_comparison({"alignment_retries": 2}, {"refine_teacher_force_windows": 3})
    assert result["refine_fallback"] == {"reference": 2, "candidate": 3}


def test_mlx_unknown_language_stays_null(monkeypatch):
    import sys
    import threading
    from collections import Counter

    pytest.importorskip("scipy")
    from finesub.speech.recognition import mlx_refine_backend as backend, mlx_session

    mx = SimpleNamespace(reset_peak_memory=lambda: None, get_peak_memory=lambda: 0)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    upstream = SimpleNamespace(add_word_timestamps=lambda **kw: None)
    monkeypatch.setattr(backend.importlib, "import_module", lambda _: upstream)
    monkeypatch.setattr(mlx_session, "bind_transcribe", lambda *a:
                        (lambda *a, **kw: {"segments": [], "language": None}, "test-source"))
    model = backend.MlxRefineModel.__new__(backend.MlxRefineModel)
    model._request_lock = threading.RLock()
    model._force_teacher_force = model._detect_disfluencies = False
    model._collect_refine_signals = model._collect_attention_signals = False
    model.request_model, model.model_path = object(), "model"
    model.runtime_versions, model.search_stats = {}, Counter()
    model.condition_on_previous_text, model.peak_memory_bytes = True, 0
    model.encoder = SimpleNamespace(hits=0, misses=0, evictions=0, bytes=0)
    assert model.transcribe_wt(None, language=None)["language"] is None


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

    import importlib

    pytest.importorskip("faster_whisper")  # the backend module imports it at load
    from finesub.speech.recognition import fw_refine_backend

    grown = asr_backend.DECODE_OPTIONS + (asr_backend.DecodeOption("nucleus_p"),)
    monkeypatch.setattr(asr_backend, "DECODE_OPTIONS", grown)
    # Reload under the grown contract, because that is the shape of the real
    # mistake: the class body runs at import, so a derived set is recomputed
    # too and agrees with the contract again. Patching without reloading left
    # the old value in place and passed either way -- a test of nothing.
    try:
        reloaded = importlib.reload(fw_refine_backend)
        problems = asr_backend.contract_violations(reloaded.RefinedWhisperModel)
    finally:
        monkeypatch.undo()
        importlib.reload(fw_refine_backend)

    assert problems and "nucleus_p" in problems[0]
