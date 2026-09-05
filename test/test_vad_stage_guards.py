"""A guard that only fires on a branch the stage can take without a model.

`audio_coverage` was wired in the wrong place first, and the failure was
invisible: it was recorded after the "no speech at all" early return, so the
single most degenerate run -- the one the metric exists to make visible -- was
the one run that recorded no coverage.

It now sits where the fresh and reused paths meet, above that early return.
That location is reachable without loading Whisper, which is what lets this be
an ordinary test instead of a heavy-resource run.

(A second guard lived here for `vad_failover.report`, which had the mirror-image
placement bug -- it judged a freshly computed prefix and never a reused one.
The predicate was removed on 2026-08-30; see `docs/bench-baselines.md` 17.12.)
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from finesub.reporting import NullReporter, reporting_to
from finesub.run_metadata import load_run_metadata
from finesub.speech.preprocessing import energy as vad_energy
from finesub.speech.recognition import vad_asr_stage
from finesub.speech.runtime import hf_weights
from finesub.speech.runtime import device as device_module


@pytest.fixture()
def fw_refine_backend():
    """The refine backend, imported per test rather than at module scope.

    It pulls in `faster_whisper`, and only `TestWarmUpEncode` touches it -- the
    other twelve tests here resolve devices with no [asr] extra in sight. As a
    module-level import it took all of them down together on any environment
    without the extra, which is every CI runner.
    """

    pytest.importorskip("faster_whisper", reason="[asr] extra not installed")
    from finesub.speech.recognition import fw_refine_backend as backend

    return backend


class _Warnings(NullReporter):
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        self.entries.append((code, message))

    def codes(self) -> list[str]:
        return [code for code, _ in self.entries]


def _empty_prefix(audio_duration: float) -> vad_asr_stage.VadPrefix:
    """A prefix that found no speech in a long recording.

    The shape of a VAD that failed rather than a recording that is silent --
    which is precisely the pair the failover predicate cannot separate on its
    own, and why it reports rather than acts.
    """

    return vad_asr_stage.VadPrefix(
        raw_segments=[],
        segments=[],
        vad_meta={"vad": {"mode": "energy"}, "pause_hints": {"scorer": []}},
        audio_duration=audio_duration,
        timing={"vad_sec": 1.0},
        energy_track=vad_energy.VadEnergyTrack(
            energy_db=torch.tensor([-60.0, -59.0], dtype=torch.float32),
            hop_sec=0.01,
            frame_sec=0.025,
            energy_mode="weighted",
            frame_dbfs=None,
        ),
    )


def _stage_on_a_reused_empty_prefix(tmp_path, monkeypatch, reporter: _Warnings):
    """Run the stage far enough to hit the no-speech return, off a stored prefix.

    Decoding and weight prefetching are stubbed rather than satisfied: neither
    is what these guards are about, and requiring real media would push a check
    of *where two lines sit* behind the heavy-resource gate, where it would not
    run. The prefix stands in for the VAD pass, which is the point -- this is
    the reuse path.
    """

    monkeypatch.setattr(
        vad_asr_stage, "ensure_decodable_input", lambda path, _dir: (path, None)
    )
    monkeypatch.setattr(
        vad_asr_stage, "ensure_asr_weights", lambda name: hf_weights.UNMANAGED
    )

    source = tmp_path / "clip-vocal.ogg"
    source.write_bytes(b"stand-in bytes: identity is name + size + mtime")
    artifact = tmp_path / "clip-vad.json"
    vad_asr_stage.write_vad_prefix(
        artifact,
        _empty_prefix(600.0),
        source_path=source,
        vad_silero_assist=True,
    )

    output = tmp_path / "clip-aligned.json"
    metadata = tmp_path / "clip-metadata.json"
    with reporting_to(reporter):
        vad_asr_stage.run_vad_asr(
            source,
            output_path=output,
            device="cpu",
            vad_prefix_path=artifact,
            run_metadata_path=metadata,
            vad_silero_assist=True,
            qwen_verify="off",
            lang_redecode="off",
        )
    return output, metadata


def test_a_run_that_found_no_speech_still_records_coverage(tmp_path, monkeypatch) -> None:
    output, metadata = _stage_on_a_reused_empty_prefix(tmp_path, monkeypatch, _Warnings())

    recorded = load_run_metadata(metadata).get("audio_coverage")
    assert recorded == {
        "audio_sec": 600.0,
        "speech_sec": 0.0,
        "ratio": 0.0,
        "intervals": 0,
    }

    aligned = json.loads(output.read_text(encoding="utf-8"))
    assert aligned["metadata"]["asr_align"]["audio_coverage"]["ratio"] == 0.0

class TestAsrDeviceAsksCt2:
    """The ASR stage decodes with CTranslate2, so torch is the wrong oracle.

    `resolve_device` answers for torch; a machine with a working card and a
    CPU-only CT2 wheel used to hand `"cuda"` straight to CTranslate2 and fail
    at the first encode.
    """

    def test_a_ct2_without_cuda_falls_back_and_says_so(self, monkeypatch) -> None:
        messages: list[str] = []
        monkeypatch.setattr(
            "finesub.speech.runtime.device.ct2_cuda_unusable_reason",
            lambda: "this CTranslate2 build reports no CUDA device",
        )
        with reporting_to(_Collect(messages)):
            assert device_module.resolve_asr_device("cuda") == "cpu"
        assert any("cpu-fallback" in message for message in messages)
        assert any("CTranslate2" in message for message in messages)

    def test_a_working_ct2_keeps_the_gpu(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "finesub.speech.runtime.device.ct2_cuda_unusable_reason", lambda: None
        )
        assert device_module.resolve_asr_device("cuda") == "cuda"

    def test_a_cpu_run_never_pays_for_the_probe(self, monkeypatch) -> None:
        """`import ctranslate2` costs ~9 s cold; a CPU run must not buy it."""

        def explode() -> str:
            raise AssertionError("the probe ran on a CPU device")

        monkeypatch.setattr(
            "finesub.speech.runtime.device.ct2_cuda_unusable_reason", explode
        )
        assert device_module.resolve_asr_device("cpu") == "cpu"


class TestWarmUpEncode:
    """Building the model is not encoding, and only encoding surfaces the error.

    A CTranslate2 build with no matrix backend for the device constructs fine;
    `_missing_gemm_backend` catches the failure inside `encode`. So a warm-up
    that only constructs moves nothing -- which is exactly what `warm()` used
    to do.
    """

    def test_warm_encodes_every_pooled_model(self, monkeypatch, fw_refine_backend) -> None:
        encoded: list[object] = []

        class FakeModel:
            def __init__(self) -> None:
                self._encoder_cache = SimpleNamespace(clear=lambda: None)

            def encode(self, features):
                encoded.append(features)
                return features

        pool = fw_refine_backend.FwRefineModelPool.__new__(
            fw_refine_backend.FwRefineModelPool
        )
        pool._size = 2
        pool._idle = [FakeModel(), FakeModel()]
        pool._loaded = 2
        pool._condition = threading.Condition()
        monkeypatch.setattr(
            fw_refine_backend, "_encoder_window", lambda model, audio: audio
        )

        pool.warm()

        assert len(encoded) == 2, "warm() built the pool without encoding"

    def test_a_gemm_failure_reaches_the_caller_from_warm(
        self, monkeypatch, fw_refine_backend
    ) -> None:
        """The whole point: the translated error arrives at stage entry."""

        class BrokenModel:
            _encoder_cache = SimpleNamespace(clear=lambda: None)
            model = SimpleNamespace(device="cpu")

            def encode(self, features):
                raise fw_refine_backend._missing_gemm_backend(
                    RuntimeError("No SGEMM backend on CPU"), "cpu"
                )

        pool = fw_refine_backend.FwRefineModelPool.__new__(
            fw_refine_backend.FwRefineModelPool
        )
        pool._size = 1
        pool._idle = [BrokenModel()]
        pool._loaded = 1
        pool._condition = threading.Condition()
        monkeypatch.setattr(
            fw_refine_backend, "_encoder_window", lambda model, audio: audio
        )

        with pytest.raises(RuntimeError, match="ct2-patches"):
            pool.warm()


class TestTheRequestReachesEveryStage:
    """One request, three stages, two oracles -- and the same answer.

    Separation asks torch, the ASR stage asks CTranslate2, the referee asks
    torch again; what they share is the user's intent. These pin the two
    cases the pipeline promises: an explicit `cpu` keeps all of them off the
    card, and the desktop's "automatic" (`None`) is the default request, not
    whatever the ASR happened to resolve to.
    """

    def test_an_explicit_cpu_keeps_asr_and_referee_off_a_working_card(
        self, monkeypatch
    ) -> None:
        from finesub.speech.recognition import lang_redecode
        from finesub.speech.runtime.resources import get_resource_profile

        monkeypatch.setattr(
            "finesub.speech.runtime.device.cuda_usable", lambda: True
        )
        monkeypatch.setattr(
            "finesub.speech.runtime.device.ct2_cuda_unusable_reason", lambda: None
        )
        standard = get_resource_profile("standard")

        assert device_module.resolve_asr_device("cpu") == "cpu"
        assert (
            lang_redecode.referee_device(
                "cpu", standard, "large-v3-turbo", requested_device="cpu"
            )
            == "cpu"
        )
        assert (
            vad_asr_stage.referee_warm_device(
                qwen_verify="auto",
                device="cpu",
                resource_profile=standard,
                model_name="large-v3-turbo",
                requested_device="cpu",
            )
            is None
        )
        # (Separation is pinned in test_vocal_separation_pool.py, where the
        # stage can actually be run against a counting separator.)

    def test_the_desktop_automatic_puts_the_referee_on_the_idle_card(
        self, monkeypatch
    ) -> None:
        """CT2 cannot use the card, torch can, nobody chose a device.

        The ASR falls back to the CPU; the referee -- a torch model -- should
        take the idle card with the whole tier budget. It did not, because
        `None` was read as the resolved ASR device ("cpu") and treated as the
        user's wish (review 2026-09-02).
        """

        from finesub.speech.recognition import lang_redecode
        from finesub.speech.runtime.resources import get_resource_profile
        from finesub.speech.verification.qwen_referee import COMPILE_MIN_VRAM_GIB

        monkeypatch.setattr(
            "finesub.speech.runtime.device.cuda_usable", lambda: True
        )
        monkeypatch.setattr(
            "finesub.speech.runtime.device.ct2_cuda_unusable_reason",
            lambda: "this CTranslate2 build reports no CUDA device",
        )
        # "The idle card" is the premise, so it has to be stated rather than
        # borrowed from this host: referee placement's last question reads the
        # driver's live free VRAM, and the assertion below went red the day
        # another job held 14.9 of the 16.3 GiB.
        monkeypatch.setattr(
            "finesub.speech.runtime.device.free_vram_gib", lambda: 24.0
        )
        standard = get_resource_profile("standard")

        with reporting_to(_Collect([])):
            asr_device = device_module.resolve_asr_device(str(None or "cuda"))
        assert asr_device == "cpu"  # the CT2-only fallback

        assert (
            vad_asr_stage.referee_warm_device(
                qwen_verify="auto",
                device=asr_device,
                resource_profile=standard,
                model_name="large-v3-turbo",
                requested_device=None,
            )
            == "cuda"
        )
        # ...with the whole tier budget: nothing is resident beside it.
        assert (
            lang_redecode.referee_vram_budget(
                standard, "large-v3-turbo", beside_pool=False
            )
            >= COMPILE_MIN_VRAM_GIB
        )


class _Collect(NullReporter):
    def __init__(self, sink: list[str]) -> None:
        self.sink = sink

    def warning(self, code, message, **kwargs) -> None:  # noqa: D102
        self.sink.append(f"{code}: {message}")


class TestCt2ArchitectureFloor:
    """`get_cuda_device_count` counts devices, not kernels.

    It is `cudaGetDeviceCount`: on a GTX 10-series card the driver enumerates
    one device and the patched wheel -- built for `7.0` and up -- has no cubin
    for it. Without the floor the stage sent that card to CUDA and died inside
    the first encode, which is a hard failure where the documented behaviour is
    a CPU fallback.
    """

    @staticmethod
    def _ct2_sees_one_device(monkeypatch) -> None:
        """Stub the lazy `import ctranslate2` so the probe runs anywhere."""

        monkeypatch.setitem(
            sys.modules,
            "ctranslate2",
            SimpleNamespace(get_cuda_device_count=lambda: 1),
        )

    @staticmethod
    def _card_reports(monkeypatch, capability: tuple[int, int] | None) -> None:
        """What the driver says this card is -- `None` = torch sees no device."""

        monkeypatch.setattr(torch.cuda, "is_available", lambda: capability is not None)
        if capability is not None:
            monkeypatch.setattr(
                torch.cuda, "get_device_capability", lambda *_: capability
            )

    def test_a_card_below_the_floor_is_refused(self, monkeypatch) -> None:
        self._ct2_sees_one_device(monkeypatch)
        self._card_reports(monkeypatch, (6, 1))  # GTX 1080

        reason = device_module.ct2_cuda_unusable_reason()

        assert reason is not None
        assert "6.1" in reason

    def test_that_refusal_is_a_cpu_fallback_not_an_error(self, monkeypatch) -> None:
        """The whole point: the run continues on the CPU and says why."""

        self._ct2_sees_one_device(monkeypatch)
        self._card_reports(monkeypatch, (6, 1))
        messages: list[str] = []

        with reporting_to(_Collect(messages)):
            assert device_module.resolve_asr_device("cuda") == "cpu"

        assert any("cpu-fallback" in message for message in messages)

    def test_the_floor_itself_passes(self, monkeypatch) -> None:
        self._ct2_sees_one_device(monkeypatch)
        self._card_reports(monkeypatch, device_module.CT2_MIN_COMPUTE_CAPABILITY)

        assert device_module.ct2_cuda_unusable_reason() is None

    def test_a_capability_above_the_build_list_still_passes(self, monkeypatch) -> None:
        """`9.0+PTX` JITs forward -- this is how sm_120 decodes today.

        Requiring membership rather than a floor would push every card newer
        than the wheel onto the CPU, which is a far worse bug than the one the
        floor fixes.
        """

        self._ct2_sees_one_device(monkeypatch)
        self._card_reports(monkeypatch, (13, 0))

        assert device_module.ct2_cuda_unusable_reason() is None

    def test_a_card_torch_cannot_see_is_left_to_ct2(self, monkeypatch) -> None:
        """No reading, no verdict: the answer stays what it was before the floor.

        torch is asked for the capability as a hardware fact, not as an opinion
        about whether it could run there -- so when it sees no device at all
        there is nothing to compare, and CTranslate2 gets its chance.
        """

        self._ct2_sees_one_device(monkeypatch)
        self._card_reports(monkeypatch, None)

        assert device_module.ct2_cuda_unusable_reason() is None

    def test_no_device_at_all_still_reports_the_older_reason(self, monkeypatch) -> None:
        monkeypatch.setitem(
            sys.modules,
            "ctranslate2",
            SimpleNamespace(get_cuda_device_count=lambda: 0),
        )

        reason = device_module.ct2_cuda_unusable_reason()

        assert reason is not None
        assert "no CUDA device" in reason
