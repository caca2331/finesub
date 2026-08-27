from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from finesub.speech.preprocessing.separator import accel


def _paths(tmp_path: Path) -> accel.AccelPaths:
    return accel.AccelPaths(root=tmp_path / "key")


def _install_package(paths: accel.AccelPaths) -> None:
    paths.aoti.mkdir(parents=True, exist_ok=True)
    (paths.aoti / "manifest.json").write_text("{}", encoding="utf-8")
    (paths.aoti / "time.pt2").write_bytes(b"")


def test_cache_root_prefers_the_explicit_model_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FINESUB_MODEL_DIR", str(tmp_path / "models"))

    assert accel._cache_root() == (
        (tmp_path / "models").resolve() / "audio-separator" / "accel"
    )


@pytest.mark.requires_main_checkout
def test_cache_root_uses_the_checkout_without_a_model_dir(monkeypatch) -> None:
    monkeypatch.delenv("FINESUB_MODEL_DIR", raising=False)
    checkout = Path(__file__).resolve().parents[1]

    assert accel._cache_root() == checkout / "cache" / "separator-accel"


def test_cache_key_binds_every_version_the_artefact_depends_on(monkeypatch) -> None:
    fake = type(
        "T",
        (),
        {
            "__version__": "2.11.0+cu128",
            "version": type("V", (), {"cuda": "12.8"})(),
            "cuda": type(
                "C",
                (),
                {
                    "is_available": staticmethod(lambda: True),
                    "get_device_capability": staticmethod(lambda: (12, 0)),
                },
            )(),
        },
    )()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    key = accel.cache_key("model.ckpt")

    assert key is not None
    # A mismatch on any of these makes the package unloadable, so each has to
    # land in a different directory rather than be silently reused.
    assert "2.11.0+cu128" in key
    assert "cu128" in key
    assert "sm120" in key
    assert key.startswith(f"v{accel.BUILD_FORMAT}-")
    assert accel.cache_key("other.ckpt") != key


def test_cache_key_is_none_without_cuda(monkeypatch) -> None:
    fake = type(
        "T",
        (),
        {"cuda": type("C", (), {"is_available": staticmethod(lambda: False)})()},
    )()
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    assert accel.cache_key("model.ckpt") is None


def test_probe_roundtrip_and_unreadable_probe_reads_as_absent(tmp_path) -> None:
    paths = _paths(tmp_path)
    assert accel.read_probe(paths) == {}

    accel.write_probe(paths, "aoti", "unavailable", "no compiler")
    recorded = accel.read_probe(paths)
    assert recorded["aoti"] == "unavailable"
    assert recorded["aoti_reason"] == "no compiler"

    accel.write_probe(paths, "jit", "unavailable", "triton")
    recorded = accel.read_probe(paths)
    assert recorded["aoti_reason"] == "no compiler"
    assert recorded["jit_reason"] == "triton"

    paths.probe.write_text("{ truncated", encoding="utf-8")
    assert accel.read_probe(paths) == {}


def test_existing_package_is_used_at_any_duration(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(accel, "triton_available", lambda: True)
    paths = _paths(tmp_path)
    _install_package(paths)

    # AOTI's per-process cost is small enough to win on short inputs too.
    assert accel.select_backend(paths, duration_sec=1.0) == "aoti"


def test_recorded_build_failure_falls_back_and_respects_the_jit_threshold(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(accel, "triton_available", lambda: True)
    paths = _paths(tmp_path)
    accel.write_probe(paths, "aoti", "unavailable", "cl.exe not found")

    below = accel.JIT_MIN_DURATION_SEC - 1
    assert accel.select_backend(paths, duration_sec=below) == "eager"
    assert accel.select_backend(paths, duration_sec=accel.JIT_MIN_DURATION_SEC) == "jit"


def test_first_run_tries_to_build_before_settling_for_jit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(accel, "triton_available", lambda: True)
    paths = _paths(tmp_path)

    monkeypatch.setattr(accel, "aoti_buildable", lambda: True)
    assert accel.select_backend(paths, duration_sec=10_000) == "aoti"
    assert accel.select_backend(paths, duration_sec=10_000, buildable=False) == "jit"


def test_a_machine_without_a_compiler_goes_straight_to_jit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(accel, "triton_available", lambda: True)
    monkeypatch.setattr(accel, "aoti_buildable", lambda: False)
    paths = _paths(tmp_path)

    # Not "attempt the build, fail, degrade to eager": the tier that this
    # machine can actually run has to be picked on the first run too.
    assert accel.select_backend(paths, duration_sec=10_000) == "jit"
    assert accel.select_backend(paths, duration_sec=1.0) == "eager"


def test_no_triton_means_no_compiled_tier(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(accel, "triton_available", lambda: False)
    paths = _paths(tmp_path)
    _install_package(paths)

    assert accel.select_backend(paths, duration_sec=10_000) == "eager"


def test_missing_cache_location_means_eager(monkeypatch) -> None:
    assert accel.select_backend(None, duration_sec=10_000) == "eager"


def test_opt_out_disables_acceleration(monkeypatch) -> None:
    monkeypatch.setenv("FINESUB_SEPARATOR_ACCEL", "off")
    assert accel.acceleration_disabled() is True
    assert accel.resolve_accel_paths("model.ckpt") is None

    monkeypatch.setenv("FINESUB_SEPARATOR_ACCEL", "on")
    assert accel.acceleration_disabled() is False


def test_half_written_package_is_not_treated_as_ready(tmp_path) -> None:
    paths = _paths(tmp_path)
    paths.aoti.mkdir(parents=True)
    (paths.aoti / "time.pt2").write_bytes(b"")

    # A build that died before writing its manifest must not be loaded.
    assert accel.aoti_package_ready(paths) is False


def _fake_builder(monkeypatch, *, build=None, load=None):
    """Stand in for the module apply_acceleration imports on use."""

    from finesub.speech.preprocessing import separator

    fake = type(
        "FakeAoti",
        (),
        {
            "build_packages": staticmethod(
                build or (lambda output_dir, **kwargs: {})
            ),
            "load_packages": staticmethod(load or (lambda instance, path: 1)),
        },
    )
    monkeypatch.setattr(separator, "separator_aoti", fake, raising=False)
    return fake


def test_leftovers_from_an_interrupted_build_do_not_block_the_next_one(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    paths.aoti.mkdir(parents=True)
    (paths.aoti / "time.pt2").write_bytes(b"half written")

    seen: dict[str, object] = {}

    def build(output_dir, **kwargs):
        # build_packages refuses a non-empty directory, so a crash that left
        # debris behind used to be recorded as "this machine cannot build".
        seen["empty"] = not any(output_dir.iterdir()) if output_dir.exists() else True
        _install_package(paths)
        return {}

    _fake_builder(monkeypatch, build=build)

    assert accel.apply_acceleration(object(), "aoti", paths).effective == "aoti"
    assert seen["empty"] is True
    assert accel.read_probe(paths)["aoti"] == "ok"


def test_a_package_that_builds_but_never_loads_is_not_rebuilt_forever(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    _install_package(paths)

    def load(instance, path):
        raise RuntimeError("sm mismatch")

    _fake_builder(monkeypatch, load=load)

    applied = accel.apply_acceleration(object(), "aoti", paths)
    assert (applied.requested, applied.effective) == ("aoti", "eager")
    assert applied.fallback_reason.startswith("RuntimeError: sm mismatch")
    # Without the probe the next run finds no package, rebuilds, and fails the
    # same way -- paying the full build on every run forever.
    assert accel.read_probe(paths)["aoti"] == "unavailable"
    monkeypatch.setattr(accel, "triton_available", lambda: True)
    assert accel.select_backend(paths, duration_sec=10_000) == "jit"


def test_probe_records_are_json_and_capped(tmp_path) -> None:
    paths = _paths(tmp_path)
    accel.write_probe(paths, "aoti", "unavailable", "x" * 5000)

    data = json.loads(paths.probe.read_text(encoding="utf-8"))
    assert len(data["aoti_reason"]) == 500
    assert isinstance(data["aoti_checked_at"], float)


# --- JIT: transactional install and first-forward revert -------------------


class Attend(torch.nn.Module):
    """Named like the Roformer's attention module, which is what accel patches."""

    def flash_attn(self, q, k, v):
        return "original"


class _Run(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.ModuleList([torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)])]
        )
        self.band_split = torch.nn.Linear(1, 1)
        self.mask_estimators = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
        self.attend = Attend()


class _Compiled(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self._orig_mod = inner


def _jit_fixture(monkeypatch, tmp_path, *, fail_on_call: int | None = None):
    run = _Run()
    originals = {
        "t0": run.layers[0][0],
        "t1": run.layers[0][1],
        "band": run.band_split,
        "mask": run.mask_estimators[0],
    }
    calls = {"n": 0}

    def fake_compile(module):
        calls["n"] += 1
        if fail_on_call is not None and calls["n"] == fail_on_call:
            raise RuntimeError("compile exploded")
        return _Compiled(module)

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "operator"))
    monkeypatch.setattr(accel, "_OPERATOR_INDUCTOR_CACHE_DIR", None)
    warnings: list[tuple] = []
    monkeypatch.setattr(
        accel,
        "current_reporter",
        lambda: SimpleNamespace(
            warning=lambda code, message, **kw: warnings.append((code, message, kw)),
            progress=lambda *a, **kw: None,
        ),
    )
    instance = SimpleNamespace(model_run=run)
    return instance, originals, warnings


def _is_original(run: _Run, originals: dict) -> bool:
    return (
        run.layers[0][0] is originals["t0"]
        and run.layers[0][1] is originals["t1"]
        and run.band_split is originals["band"]
        and run.mask_estimators[0] is originals["mask"]
        and not hasattr(run.attend, "_accel_original_flash_attn")
        and "flash_attn" not in run.attend.__dict__
    )


def test_jit_install_failure_midway_leaves_no_compiled_module_behind(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    instance, originals, warnings = _jit_fixture(monkeypatch, tmp_path, fail_on_call=3)

    applied = accel.apply_acceleration(instance, "jit", paths)

    # Two modules were compiled before the third call failed; eager must mean
    # eager, not "two compiled, two not".
    assert (applied.requested, applied.effective) == ("jit", "eager")
    assert applied.fallback_reason == "RuntimeError: compile exploded"
    assert applied.rollback is None
    assert _is_original(instance.model_run, originals)
    assert [code for code, *_ in warnings] == ["separator-jit-unavailable"]


def test_jit_first_forward_failure_reverts_in_place_and_records_the_verdict(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    accel.write_probe(paths, "aoti", "unavailable", "cl.exe not found")
    instance, originals, warnings = _jit_fixture(monkeypatch, tmp_path)

    applied = accel.apply_acceleration(instance, "jit", paths)
    assert applied.effective == "jit"
    assert isinstance(instance.model_run.band_split, _Compiled)
    assert "flash_attn" in instance.model_run.attend.__dict__
    assert paths.inductor.is_dir()
    (paths.inductor / "triton").mkdir()

    reverted = accel.revert_jit(
        applied, FileNotFoundError("triton_per_fused_0.json"), paths
    )

    assert (reverted.requested, reverted.effective) == ("jit", "eager")
    assert reverted.fallback_reason.startswith("FileNotFoundError")
    assert _is_original(instance.model_run, originals)
    assert instance.model_run.attend.flash_attn(1, 2, 3) == "original"
    # The managed cache is where the half-written kernel lives.
    assert not paths.inductor.exists()
    probe = accel.read_probe(paths)
    assert probe["jit"] == "unavailable"
    assert probe["jit_reason"].startswith("FileNotFoundError")
    # Recording the JIT verdict must not erase the AOTI one.
    assert probe["aoti"] == "unavailable"
    assert probe["aoti_reason"] == "cl.exe not found"
    assert warnings[-1][0] == "separator-jit-failed"
    assert str(paths.root) in warnings[-1][2]["action"]
    monkeypatch.setattr(accel, "triton_available", lambda: True)
    assert accel.select_backend(paths, duration_sec=10_000, buildable=False) == "eager"


def test_out_of_memory_at_first_forward_degrades_this_run_only(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    instance, originals, warnings = _jit_fixture(monkeypatch, tmp_path)
    applied = accel.apply_acceleration(instance, "jit", paths)

    reverted = accel.revert_jit(applied, torch.cuda.OutOfMemoryError("busy"), paths)

    assert reverted.effective == "eager"
    assert _is_original(instance.model_run, originals)
    assert "jit" not in accel.read_probe(paths)
    assert warnings[-1][2]["action"] == ""


def test_an_operator_managed_inductor_cache_is_never_cleared(
    tmp_path, monkeypatch
) -> None:
    paths = _paths(tmp_path)
    instance, originals, warnings = _jit_fixture(monkeypatch, tmp_path)
    operator_dir = tmp_path / "operator"
    operator_dir.mkdir()
    monkeypatch.setattr(accel, "_OPERATOR_INDUCTOR_CACHE_DIR", str(operator_dir))
    paths.inductor.mkdir(parents=True)

    applied = accel.apply_acceleration(instance, "jit", paths)
    accel.revert_jit(applied, RuntimeError("triton"), paths)

    assert operator_dir.is_dir()
    assert paths.inductor.is_dir()


# --- AOTI: a runner that fails partway leaves no installed forward behind ---


def test_aoti_install_failure_midway_restores_every_forward(tmp_path, monkeypatch) -> None:
    from finesub.speech.preprocessing.separator import separator_aoti

    run = _Run()
    originals = {id(m): m.forward for m in (run.layers[0][0], run.layers[0][1], run.band_split)}
    package_dir = tmp_path / "aoti"
    package_dir.mkdir()
    manifest = {
        "weights_serialized": False,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sm": "sm00",
        "packages": {
            "time": {"kind": "transformer", "axis": "time", "file": "time.pt2", "input_shape": [1, 8]},
            "frequency": {"kind": "transformer", "axis": "frequency", "file": "frequency.pt2", "input_shape": [1, 8]},
            "band_split": {"kind": "single", "module_path": "band_split", "file": "band_split.pt2"},
        },
    }
    (package_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("time", "frequency", "band_split"):
        (package_dir / f"{name}.pt2").write_bytes(b"")

    monkeypatch.setattr(separator_aoti, "_device_arch", lambda: "sm00")
    monkeypatch.setattr(separator_aoti, "_write_package_with_prefix", lambda data, dest: None)
    monkeypatch.setattr(separator_aoti, "_populate_rotary_cache", lambda module, length: None)
    monkeypatch.setattr(separator_aoti, "_constant_map", lambda compiled, target: {})
    loads = {"n": 0}

    class Runner:
        def load_constants(self, *args, **kwargs):
            return None

        def __call__(self, *args, **kwargs):
            return None

    def fake_load(path):
        loads["n"] += 1
        if loads["n"] == 3:
            raise RuntimeError("runner 3 would not load")
        return Runner()

    monkeypatch.setattr(torch._inductor, "aoti_load_package", fake_load)

    instance = SimpleNamespace(model_run=run)
    with pytest.raises(RuntimeError, match="runner 3"):
        separator_aoti.load_packages(instance, package_dir)

    # Two runners were installed before the third failed; both are gone.
    for module in (run.layers[0][0], run.layers[0][1], run.band_split):
        assert "forward" not in module.__dict__
        assert module.forward == originals[id(module)]
    assert not hasattr(instance, "_separator_aoti_scratch")


# --- AOTI: "there is a compiler" has to mean the compiler can compile ---


def test_cl_on_path_without_an_include_environment_is_not_a_toolchain(
    monkeypatch,
    tmp_path,
) -> None:
    """`cl.exe` reads its header search path from `INCLUDE`, which vcvars sets.

    Reporting a toolchain on the strength of the executable alone promises a
    90-second build and delivers a failed one, on a machine that would have
    been told "eager" honestly a moment earlier. With no vcvars to fall back
    to, the answer is no.
    """

    from finesub.speech.preprocessing.separator import separator_aoti

    monkeypatch.setattr(separator_aoti.shutil, "which", lambda _name: r"C:\msvc\cl.exe")
    monkeypatch.setattr(separator_aoti, "_find_vcvars", lambda: None)
    monkeypatch.setenv("INCLUDE", str(tmp_path / "empty"))

    assert not separator_aoti.cxx_toolchain_available()


def test_a_complete_msvc_environment_still_reads_as_a_toolchain(
    monkeypatch,
    tmp_path,
) -> None:
    from finesub.speech.preprocessing.separator import separator_aoti

    include = tmp_path / "include"
    include.mkdir()
    (include / "array").write_text("", encoding="utf-8")
    monkeypatch.setattr(separator_aoti.shutil, "which", lambda _name: r"C:\msvc\cl.exe")
    monkeypatch.setattr(separator_aoti, "_find_vcvars", lambda: None)
    monkeypatch.setenv("INCLUDE", str(include))

    assert separator_aoti.cxx_toolchain_available()


def test_the_probe_and_the_activation_cannot_disagree(monkeypatch, tmp_path) -> None:
    """The docstring's promise, as a test.

    A bare `cl.exe` with vcvars available is a toolchain -- activation will
    run vcvars and get a usable one -- and `_activate_msvc` must reach the
    same conclusion rather than short-circuiting on the executable.
    """

    from finesub.speech.preprocessing.separator import separator_aoti

    include = tmp_path / "include"
    include.mkdir()
    (include / "array").write_text("", encoding="utf-8")
    vcvars = tmp_path / "vcvars64.bat"
    vcvars.write_text("", encoding="utf-8")
    activated: list[str] = []

    def run(command, **_kwargs):
        activated.append(command)
        return SimpleNamespace(stdout=f"INCLUDE={include}", returncode=0)

    monkeypatch.setattr(separator_aoti, "_find_vcvars", lambda: vcvars)
    monkeypatch.setattr(separator_aoti.subprocess, "run", run)
    monkeypatch.setattr(separator_aoti.shutil, "which", lambda _name: r"C:\msvc\cl.exe")
    monkeypatch.setenv("INCLUDE", str(tmp_path / "empty"))

    assert separator_aoti.cxx_toolchain_available()
    assert separator_aoti._activate_msvc() == r"C:\msvc\cl.exe"
    assert activated, "an incomplete environment must go through vcvars"
