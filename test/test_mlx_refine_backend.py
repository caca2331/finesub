from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

# This adapter reuses the WT refine implementation, whose DTW filters require
# scipy from the [asr] extra.  The lean Linux CI intentionally installs only
# [harness,dev], so skip before importing the backend instead of failing test
# collection.  The full ASR environment runs this module normally.
pytest.importorskip("scipy", reason="[asr] extra not installed")

from finesub.speech.recognition import asr_backend, checkpoint, transcribe
from finesub.speech.recognition import mlx_refine_backend as mlx_backend


class _Tokenizer:
    eot = 100
    timestamp_begin = 200
    language = "en"
    _pieces: ClassVar = {1: " Hello", 2: ",", 3: " world"}

    def decode_with_timestamps(self, tokens):
        return "".join(
            f"<|{(token - 200) * 0.02:.2f}|>"
            if token >= 200
            else self._pieces.get(token, "")
            for token in tokens
        )

    decode = decode_with_timestamps


def test_auto_routes_only_apple_silicon_to_mlx() -> None:
    assert (
        asr_backend.resolve_backend("auto", system="Darwin", machine="arm64")
        == "mlx-refine"
    )
    assert (
        asr_backend.resolve_backend("auto", system="Darwin", machine="x86_64")
        == "fw-refine"
    )
    assert (
        asr_backend.resolve_backend("auto", system="Windows", machine="AMD64")
        == "fw-refine"
    )


def test_mlx_default_and_ct2_rejection() -> None:
    assert (
        asr_backend.resolve_model_name(
            "large-v3-turbo", "mlx-refine", default_model="large-v3-turbo"
        )
        == "mlx-community/whisper-large-v3-turbo"
    )
    with pytest.raises(ValueError, match="CTranslate2"):
        asr_backend.resolve_model_name(
            "TransWithAI/whisper-ja-1.5B-ct2",
            "mlx-refine",
            default_model="large-v3-turbo",
        )


def test_runtime_versions_are_an_exact_contract(monkeypatch) -> None:
    installed = {"mlx-whisper": "0.4.3", "mlx": "0.32.1"}
    monkeypatch.setattr(mlx_backend, "version", installed.__getitem__)
    with pytest.raises(RuntimeError, match=r"mlx==0\.32\.2 required"):
        mlx_backend._require_compatible_runtime()


@pytest.mark.parametrize(
    ("tokens", "unfinished", "expected"),
    [
        ((100, 7, 103, 99), False, (100, 7, 103)),
        ((100, 7, 8, 99), False, (100, 7, 8, 99)),
        ((100, 7, 8), True, (100, 7, 8)),
    ],
)
def test_trace_span_covers_normal_early_eot_and_decode_limit(
    tokens, unfinished, expected
) -> None:
    trace = mlx_backend.MlxDecodeTrace(
        tokens=tokens,
        token_logprobs=tuple(-0.1 for _ in tokens),
        attention=np.zeros((len(tokens), 2, 20), dtype=np.float32),
        tail_logprobs=np.zeros(120, dtype=np.float32),
    )
    spans = mlx_backend._trace_spans(
        trace, SimpleNamespace(eot=99, timestamp_begin=100)
    )
    assert len(spans) == 1
    assert spans[0].tokens == expected
    assert spans[0].unfinished is unfinished


def test_checkpoint_key_isolates_backend_revision_and_alignment_mode(tmp_path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"audio")
    common = {
        "model_name": "model",
        "language": "en",
        "gap_sec": 0.3,
        "audio_path": audio,
    }
    ct2 = checkpoint.build_key(**common, backend="fw-refine")
    mlx = checkpoint.build_key(
        **common,
        backend="mlx-refine",
        model_revision="abc",
        alignment_mode="one-pass-wt+teacher-force-fallback",
        trace_contract_version=1,
    )
    assert ct2 != mlx
    assert mlx["backend"] == "mlx-refine"
    assert mlx["model_revision"] == "abc"
    assert mlx["trace_contract_version"] == 1


def test_mlx_adapter_reuses_shared_word_and_confidence_contract(monkeypatch) -> None:
    model = mlx_backend.MlxRefineModel.__new__(mlx_backend.MlxRefineModel)
    model.refine_frames = 0
    model._detect_disfluencies = False
    model._collect_refine_signals = False
    model._collect_attention_signals = False
    model._latest_trace = mlx_backend.MlxDecodeTrace(
        tokens=(200, 1, 2, 3, 201, 100),
        token_logprobs=(-9.0, -0.1, -2.0, -0.3, -9.0, -0.5),
        attention=np.zeros((6, 2, 12), dtype=np.float32),
        tail_logprobs=np.zeros(300, dtype=np.float32),
    )
    monkeypatch.setattr(
        mlx_backend,
        "trace_alignment_path",
        lambda *_args, **_kwargs: [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
        ],
    )
    monkeypatch.setattr(
        mlx_backend,
        "prepare_alignment_weights",
        lambda *_args, **_kwargs: np.ones((5, 5), dtype=np.float32),
    )

    aligned = model._align_trace([{"tokens": [200, 1, 2, 3, 201]}], _Tokenizer(), 20)
    assert aligned is not None
    assert aligned[0].words == (
        {"word": "Hello,", "start": 0.02, "end": 0.04, "confidence": 0.905},
        {"word": "world", "start": 0.06, "end": 0.08, "confidence": 0.741},
    )


def test_mlx_has_no_ct2_batch_or_beam_contract() -> None:
    assert asr_backend.batch_decoder("mlx-refine") is None
    assert mlx_backend.MlxRefineModel.supports_beam is False


def test_mlx_low_coverage_never_calls_beam(monkeypatch) -> None:
    model = SimpleNamespace(supports_beam=False)
    group = [{"start": 0.0, "end": 1.0}]
    original = [{"start": 0.0, "end": 0.1, "words": []}]
    monkeypatch.setattr(
        transcribe, "_coverage_shortfall", lambda *_args: (1.0, 0.1, 0.8)
    )
    monkeypatch.setattr(
        transcribe,
        "_transcribe_group_candidate",
        lambda *_args, **_kwargs: pytest.fail("MLX must not enter beam rescue"),
    )
    with transcribe.collecting_stats() as stats:
        assert (
            transcribe._rescue_low_coverage(
                model, group, original, None, 16000, 0.3, language="en"
            )
            is original
        )
    assert stats["beam_rescue_skipped"] == 1
    assert "beam_rescue_attempted" not in stats


def test_macos_lock_pins_the_validated_pair_with_hashes() -> None:
    lock = tomllib.loads(
        Path("src/finesub_bootstrap/pylock.macos-arm64-py312.toml").read_text(
            encoding="utf-8"
        )
    )
    packages = {item["name"]: item for item in lock["packages"]}
    assert packages["mlx"]["version"] == "0.32.2"
    assert packages["mlx-whisper"]["version"] == "0.4.3"
    for name in ("mlx", "mlx-whisper"):
        wheels = packages[name]["wheels"]
        assert wheels and all(wheel["hashes"]["sha256"] for wheel in wheels)


def test_trace_desync_is_recorded_before_teacher_force(monkeypatch) -> None:
    model = mlx_backend.MlxRefineModel.__new__(mlx_backend.MlxRefineModel)
    model._force_teacher_force = False
    model._latest_trace = None
    model.one_pass_count = 0
    model.teacher_force_count = 0
    model.fallback_reasons = mlx_backend.Counter()
    model._last_fallback_reason = "trace-unavailable"
    monkeypatch.setattr(model, "_align_trace", lambda *_args: None)
    called = []

    def teacher_force(**kwargs):
        called.append(kwargs)

    model._add_word_timestamps(
        teacher_force,
        segments=[{"seek": 0}],
        tokenizer=object(),
        num_frames=10,
    )
    assert model.teacher_force_count == 1
    assert model.fallback_reasons == {"trace-unavailable": 1}
    assert len(called) == 1


def test_tiny_mlx_decode_records_one_alignment_row_per_selected_token() -> None:
    try:
        import mlx.core as mx
    except ImportError as exc:
        pytest.skip(f"MLX Metal device unavailable: {exc}")
    decoding = pytest.importorskip("mlx_whisper.decoding")
    whisper = pytest.importorskip("mlx_whisper.whisper")
    dims = whisper.ModelDimensions(
        n_mels=80,
        n_audio_ctx=10,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=51865,
        n_text_ctx=16,
        n_text_state=8,
        n_text_head=2,
        n_text_layer=2,
    )
    model = whisper.Whisper(dims, mx.float32)
    task = mlx_backend._TraceDecodingTask(
        model,
        decoding.DecodingOptions(language="en", sample_len=4, fp16=False),
    )
    task.run(mx.zeros((1, 20, 80)))
    trace = task.trace()
    assert trace is not None
    assert trace.attention.shape == (len(trace.tokens), 2, 10)
    assert len(trace.token_logprobs) == len(trace.tokens)
