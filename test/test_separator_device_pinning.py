"""Where the separator's weights land, with no card and no audio-separator.

``audio_separator`` is a runtime dependency this suite does not install, so the
package is stood in for by a module that records the two things the stage can
get wrong: which device the wrapper was steered to, and in what order that
steering happened relative to ``load_model`` -- the weights are placed when the
model is loaded, so a device set afterwards would only affect the next session.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

from finesub.reporting import NullReporter, reporting_to
from finesub.speech.preprocessing.separator import separation as vocal_separation


class _Collect(NullReporter):
    def __init__(self) -> None:
        self.codes: list[str] = []

    def warning(self, code, message, **kwargs) -> None:  # noqa: D102
        self.codes.append(code)


def _install_fake_audio_separator(monkeypatch, *, hide: str | None = None):
    """Stand in for ``audio_separator.separator`` and collect its instances.

    The fake keeps the attribute names the stage steers (``torch_device``,
    ``torch_device_cpu``, ``onnx_execution_provider``) and records the device it
    carried at ``load_model`` time, which is the only moment that decides where
    the weights are put. ``hide`` drops one of the two steered names, standing in
    for a version bump that renames it.
    """

    instances: list[types.SimpleNamespace] = []
    module = types.ModuleType("audio_separator.separator")

    class FakeSeparator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.torch_device_cpu = torch.device("cpu")
            # What the real package chose inside __init__ off a bare
            # torch.cuda.is_available(): the CPU on an Intel-only machine.
            self.torch_device = self.torch_device_cpu
            self.onnx_execution_provider = ["CUDAExecutionProvider"]
            # Armed, as a wrapper built on a machine whose CUDA probe succeeds
            # would leave it. The builder must turn this off for the pinned
            # families, which is what makes the assertion below more than
            # "the initial value was already False".
            self.use_autocast = True
            self.device_at_load: object = "not loaded"
            if hide is not None:
                delattr(self, hide)
            instances.append(self)

        def load_model(self, **kwargs) -> None:
            self.load_model_kwargs = kwargs
            self.device_at_load = getattr(self, "torch_device", None)

    module.Separator = FakeSeparator
    package = types.ModuleType("audio_separator")
    package.separator = module
    monkeypatch.setitem(sys.modules, "audio_separator", package)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)

    monkeypatch.setattr(vocal_separation, "place_separator_files", lambda: None)
    monkeypatch.setattr(vocal_separation, "_mute_onnx_provider_notice", lambda: None)
    return instances


def _build(device: str | None):
    return vocal_separation._build_separator(
        str(Path("out")), "flac", 1, device=device
    )


def test_the_intel_gpu_is_pinned_before_the_weights_are_loaded(monkeypatch) -> None:
    instances = _install_fake_audio_separator(monkeypatch)

    separator = _build("xpu")

    assert separator is instances[0]
    # The library chose the CPU in __init__; the stage must have overridden it
    # before load_model, because that is when the weights are placed.
    assert separator.device_at_load == torch.device("xpu")
    assert separator.onnx_execution_provider == ["CPUExecutionProvider"]
    # Precision is forced off, not merely left off: fp32 is the answer certainly
    # right on this build, and the fake was armed with `use_autocast = True`.
    assert separator.use_autocast is False


def test_the_cpu_request_keeps_the_weights_on_the_cpu(monkeypatch) -> None:
    instances = _install_fake_audio_separator(monkeypatch)

    separator = _build("cpu")

    assert separator is instances[0]
    assert separator.device_at_load == separator.torch_device_cpu
    assert separator.onnx_execution_provider == ["CPUExecutionProvider"]
    assert separator.use_autocast is False


def test_a_renamed_xpu_pin_falls_back_to_the_cpu_and_says_so(monkeypatch) -> None:
    """A version bump that renames the attribute must not undo the pin quietly.

    The failure this guards against is not a message but a wrong placement: the
    stage would think it steered the wrapper while it ran wherever the library
    had put it -- here, the CPU its own probe chose -- and would then record the
    run on the card. So the builder falls back to the CPU explicitly, and the
    caller reads the landing back from the wrapper instead of trusting the plan.
    """

    _install_fake_audio_separator(monkeypatch, hide="torch_device")
    reporter = _Collect()

    with reporting_to(reporter):
        separator = _build("xpu")

    assert reporter.codes == ["separator-device"]
    assert separator.device_at_load == separator.torch_device_cpu
    assert separator.onnx_execution_provider == ["CPUExecutionProvider"]
    assert separator.use_autocast is False
    assert vocal_separation._pinned_separator_device(separator) == "cpu"


def test_a_renamed_cpu_pin_is_reported_instead_of_silently_ignored(monkeypatch) -> None:
    """The CPU branch cannot steer without its attribute either, and says so.

    There is no second way to keep the library off the card, so the honest end
    state is the warning; this test pins that the branch does not pretend.
    """

    instances = _install_fake_audio_separator(monkeypatch, hide="torch_device_cpu")
    reporter = _Collect()

    with reporting_to(reporter):
        _build("cpu")

    assert reporter.codes == ["separator-device"]
    assert instances[0].device_at_load == instances[0].torch_device


def test_the_landing_is_read_back_from_the_wrapper(monkeypatch) -> None:
    """The plan is not evidence; the attribute the wrapper loaded with is."""

    _install_fake_audio_separator(monkeypatch)

    assert vocal_separation._pinned_separator_device(_build("xpu")) == "xpu"
    assert vocal_separation._pinned_separator_device(_build("cpu")) == "cpu"
    assert vocal_separation._pinned_separator_device(_build("cuda")) == "cpu"


def test_a_cuda_token_leaves_the_library_choice_alone(monkeypatch) -> None:
    """Only the two families the library gets wrong are steered.

    Its ``__init__`` already asks for CUDA, so a CUDA token needs no second
    branch -- and adding one would be a place for the CUDA path to silently
    diverge from what the package does.
    """

    instances = _install_fake_audio_separator(monkeypatch)

    _build("cuda")

    # Left exactly as the library set it: this builder does not guess. The fake
    # starts armed, so an unforced CUDA path has to still be armed.
    assert instances[0].device_at_load == instances[0].torch_device_cpu
    assert instances[0].onnx_execution_provider == ["CUDAExecutionProvider"]
    assert instances[0].use_autocast is True
