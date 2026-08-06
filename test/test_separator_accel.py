from __future__ import annotations

import json
from pathlib import Path

from asr_playground.speech.preprocessing import accel


def _paths(tmp_path: Path) -> accel.AccelPaths:
    return accel.AccelPaths(root=tmp_path / "key")


def _install_package(paths: accel.AccelPaths) -> None:
    paths.aoti.mkdir(parents=True, exist_ok=True)
    (paths.aoti / "manifest.json").write_text("{}", encoding="utf-8")
    (paths.aoti / "time.pt2").write_bytes(b"")


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

    accel.write_probe(paths, aoti="unavailable", reason="no compiler")
    recorded = accel.read_probe(paths)
    assert recorded["aoti"] == "unavailable"
    assert recorded["reason"] == "no compiler"

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
    accel.write_probe(paths, aoti="unavailable", reason="cl.exe not found")

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

    from asr_playground.speech import preprocessing

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
    monkeypatch.setattr(preprocessing, "separator_aoti", fake, raising=False)
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

    assert accel.apply_acceleration(object(), "aoti", paths) == "aoti"
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

    assert accel.apply_acceleration(object(), "aoti", paths) == "eager"
    # Without the probe the next run finds no package, rebuilds, and fails the
    # same way -- paying the full build on every run forever.
    assert accel.read_probe(paths)["aoti"] == "unavailable"
    monkeypatch.setattr(accel, "triton_available", lambda: True)
    assert accel.select_backend(paths, duration_sec=10_000) == "jit"


def test_probe_records_are_json_and_capped(tmp_path) -> None:
    paths = _paths(tmp_path)
    accel.write_probe(paths, aoti="unavailable", reason="x" * 5000)

    data = json.loads(paths.probe.read_text(encoding="utf-8"))
    assert len(data["reason"]) == 500
    assert isinstance(data["checked_at"], float)
