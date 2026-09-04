"""Build weightless regional AOTInductor packages for BS-Roformer.

The two artifacts contain generated code for the time and frequency
Transformer input signatures. Model parameters and rotary caches are injected
from the normal checkpoint-backed model at runtime and are not serialized.
Constant folding is deferred to load time so that every injected weight still
reaches the kernels, and each package is validated against a second block.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import types
import zipfile
from pathlib import Path
from typing import Any

import torch


AXES = {
    "time": (0, (62, 801, 512)),
    "frequency": (1, (801, 62, 512)),
}

# Modules that exist once per model. They are band-wise ModuleLists - 62 tiny
# Linear/RMSNorm pairs each - so eager spends most of their time launching
# kernels, which is what compiling them recovers.
SINGLE_MODULES = {
    "band_split": "band_split",
    "mask_estimator": "mask_estimators.0",
}

_FOLDED_CONST_PREFIX = "_FOLDED_CONST_"

#: Where `max_autotune` may be applied. Scoped rather than global because it is
#: only free on the fp16 Transformer axes; see `_inductor_configs`.
_MAX_AUTOTUNE_SCOPES = ("off", "transformers", "all")

# Per-axis SDPA choice measured on this GPU (see E3): cuDNN wins on the 801-long
# time axis, the memory-efficient kernel on the 62-long frequency axis.
_AXIS_ATTENTION_BACKENDS = {
    "time": "CUDNN_ATTENTION",
    "frequency": "EFFICIENT_ATTENTION",
}

# One package per axis is reused for all 12 transformer blocks, so nothing that
# differs between blocks may end up baked into the generated code. Compile-time
# folding does exactly that to small constants - it inlined the 8-element
# `to_gates.bias` and every runner then used block 0's gate bias, which cost
# 36dB of SI-SDR. Folding at load time keeps those tensors injectable, and the
# check below fails the build if anything block-specific gets baked again:
# with the bias inlined, block 1 was 12x further off than block 0.
_CROSS_BLOCK_ERROR_TOLERANCE = 3.0


def _separation() -> Any:
    """Imported on use: separation reaches this module through accel."""

    from . import separation

    return separation


_VSWHERE = (
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Microsoft Visual Studio"
    / "Installer"
    / "vswhere.exe"
)


def _find_vcvars() -> Path | None:
    """Locate the x64 build environment through vswhere.

    Guessing at install paths gets this wrong in the expensive direction: a
    machine with Professional, Enterprise or a non-default drive would be judged
    to have no compiler, and because the verdict is cached it would stay on the
    slower tier until someone cleared it. vswhere is what Microsoft ships to
    answer this, at a fixed location, and it covers every edition.
    """

    if not _VSWHERE.is_file():
        return None
    try:
        completed = subprocess.run(
            [
                str(_VSWHERE),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    root = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not root:
        return None
    vcvars = Path(root[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return vcvars if vcvars.is_file() else None


def _msvc_include_ready() -> bool:
    """Whether this environment's ``INCLUDE`` can actually reach the headers.

    ``cl.exe`` on PATH is not the same thing as a usable toolchain. A shell
    that picked up the compiler's directory without vcvars' environment has the
    binary and none of the standard headers, and the build then dies on
    ``#include <array>`` -- after the tier was chosen and the user was told a
    90-second build was starting, which is the one outcome
    :func:`cxx_toolchain_available` exists to prevent.
    """

    return any(
        (Path(item) / "array").is_file()
        for item in os.environ.get("INCLUDE", "").split(os.pathsep)
        if item
    )


def cxx_toolchain_available() -> bool:
    """Whether :func:`build_packages` would find a compiler, without side effects.

    Asked before a tier is chosen rather than after a build fails: a machine
    with no compiler would otherwise be told a 90-second build is starting and
    then land on eager, on exactly the run where the JIT tier was available.

    Deliberately mirrors what ``_activate_msvc`` checks, so the two cannot
    disagree -- including that it is Windows-shaped, which is the only platform
    the AOTI path has been run on. ``cl.exe`` alone does not count: only the
    pair of it and a reachable ``INCLUDE`` is a toolchain, and vcvars is
    trusted on its own because activating it is what sets ``INCLUDE`` up.
    """

    return (
        shutil.which("cl.exe") is not None and _msvc_include_ready()
    ) or _find_vcvars() is not None


def _activate_msvc() -> str:
    existing = shutil.which("cl.exe")
    if existing is not None and _msvc_include_ready():
        return existing

    vcvars = _find_vcvars()
    if vcvars is None:
        raise RuntimeError(
            "AOTInductor on Windows requires MSVC with its headers on INCLUDE "
            "(cl.exe alone is not enough)"
            if existing is not None
            else "AOTInductor on Windows requires MSVC (cl.exe)"
        )

    completed = subprocess.run(
        f'call "{vcvars}" >nul && set',
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            os.environ[key] = value
    compiler = shutil.which("cl.exe")
    if compiler is None:
        raise RuntimeError(f"Failed to activate MSVC through {vcvars}")
    return compiler


def _populate_rotary_cache(module: torch.nn.Module, seq_len: int) -> None:
    rotary = module.layers[0][0].rotary_embed
    positions = rotary.get_seq_pos(
        seq_len,
        device="cuda",
        dtype=torch.float16,
    )
    rotary(positions, seq_len=seq_len)


def _constant_map(
    compiled: Any,
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    available = dict(module.named_parameters())
    available.update(dict(module.named_buffers()))
    # Runtime constant folding adds derived entries (transposed weights, the
    # rotary cos/sin tables). They are recomputed from the injected checkpoint
    # tensors on the first run, so the caller must not supply them.
    required = [
        name
        for name in compiled.get_constant_fqns()
        if not name.startswith(_FOLDED_CONST_PREFIX)
    ]
    missing = sorted(set(required) - set(available))
    if missing:
        raise RuntimeError(f"Missing AOTI constants: {missing}")
    return {name: available[name] for name in required}


@contextlib.contextmanager
def _pinned_attention_backend(module: torch.nn.Module, backend: str | None) -> Any:
    """Build attention against one specific SDPA backend.

    The exported graph still holds the composite `scaled_dot_product_attention`;
    the backend is picked while Inductor lowers it, so the flags have to stay
    set across both export and compile. `Attend.flash_attn` opens its own
    `sdp_kernel` context during tracing, which would override them, so the
    method is replaced for the duration. What comes out is baked into the
    package - the unstable runtime selection E3 rejected no longer applies.
    """

    if backend is None:
        yield
        return

    from torch.nn.attention import SDPBackend, sdpa_kernel

    selected = getattr(SDPBackend, backend)

    def forced(self: Any, query: Any, key: Any, value: Any) -> Any:
        with sdpa_kernel(selected):
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
            )

    originals = [
        (attend, attend.flash_attn)
        for attend in module.modules()
        if attend.__class__.__name__ == "Attend"
    ]
    if not originals:
        raise RuntimeError("No Attend module found to pin")
    for attend, _ in originals:
        attend.flash_attn = types.MethodType(forced, attend)
    try:
        with sdpa_kernel(selected):
            yield
    finally:
        for attend, original in originals:
            attend.flash_attn = original


def _inductor_configs(
    emulate_precision_casts: bool,
    max_autotune: bool,
) -> dict[str, Any]:
    return {
        # Let Inductor benchmark Triton GEMM templates against cuBLAS instead of
        # always lowering to the latter. Two things come out of it (E14): the
        # templates win outright on this model's shapes -- the FFN down
        # projection measured triton_mm 1.1026ms against bias_addmm 1.1964ms --
        # and, because a template is Inductor's own code, the pointwise epilogue
        # fuses into it, which is where the compiled forward's second-largest
        # slice lives. Refused on the RTX 5060 Ti for insufficient SMs, which is
        # why E5 recorded it as unavailable; it is accepted on 70 SMs. A machine
        # that refuses it again gets a longer build and the cuBLAS lowering, not
        # a failure.
        #
        # The caller scopes this per target group rather than passing one bool
        # for the whole build: turning it on for band_split and mask_estimator
        # moved their validation error by 5x and 163x while the two Transformer
        # axes did not move at all, and end to end that cost 3.4dB of SI-SDR for
        # 1.6% of wall time. See E14's "scope" table.
        "max_autotune": max_autotune,
        # Weightless packages. 2.11 replaced the on-disk bool with
        # package_constants_on_disk_format, whose default (None) already means
        # "not on disk", so only the in-so switch is set here. Either way the
        # zip check in main() fails the build if weights slip in.
        "aot_inductor.package_constants_in_so": False,
        "emulate_precision_casts": emulate_precision_casts,
        # Keep injected constants reachable; see the note next to
        # _CROSS_BLOCK_ERROR_TOLERANCE.
        "aot_inductor.use_runtime_constant_folding": True,
        # Windows AOTI link list omits cudart although the generated wrapper
        # uses that API.
        "aot_inductor.custom_op_libs": ["cudart"],
    }


def _capture_module_inputs(
    model_instance: Any,
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Record what each single-instance module is actually called with.

    Their input shapes follow the checkpoint's band layout and chunk size, so
    reading them off one real forward beats hardcoding them here.
    """

    captured: dict[str, torch.Tensor] = {}

    def capture(name: str) -> Any:
        def hook(module: Any, inputs: tuple[Any, ...]) -> None:
            captured[name] = inputs[0].detach().clone()

        return hook

    handles = [
        model.get_submodule(path).register_forward_pre_hook(capture(name))
        for name, path in SINGLE_MODULES.items()
    ]
    try:
        _separation()._warm_up_shared_roformer(model_instance, use_amp=True)
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(SINGLE_MODULES) - set(captured))
    if missing:
        raise RuntimeError(f"Never reached during a forward: {missing}")
    return captured


def _inject_constants(compiled: Any, module: torch.nn.Module) -> None:
    compiled.load_constants(
        _constant_map(compiled, module),
        check_full_update=False,
        user_managed=True,
    )


def _relative_error(expected: torch.Tensor, actual: torch.Tensor) -> float:
    reference = expected.flatten().double()
    candidate = actual.flatten().double()
    return float((candidate - reference).norm() / reference.norm().clamp_min(1e-30))


def _archive_contains_weights(path: Path) -> bool:
    with zipfile.ZipFile(path) as archive:
        return any(name.startswith("data/weights/") for name in archive.namelist())

@contextlib.contextmanager
def _build_target(model_instance: Any) -> Any:
    """Yield the model to compile against, loading one only if none was given.

    Production passes its live model. Loading a second BS-Roformer instead
    would put two full copies of the weights on the GPU at once -- the master
    is already resident and warmed by the time a build starts -- which is a
    real risk on the 4GB profile, and an out-of-memory there gets recorded as
    "this machine cannot build AOTI".
    """

    if model_instance is not None:
        yield model_instance
        return
    with tempfile.TemporaryDirectory() as temp_output:
        yield _separation()._build_separator(
            temp_output, "flac", 1, use_cuda=True
        ).model_instance


def build_packages(
    output_dir: Path,
    *,
    model_instance: Any = None,
    emulate_precision_casts: bool = False,
    max_autotune: str = "transformers",
    attention_backend: str = "axis",
    targets: str = "all",
) -> dict[str, Any]:
    """Compile the weightless packages for this machine and return the manifest.

    Raises rather than degrading: the caller decides whether a machine without a
    C++ compiler should fall back, and records that verdict so the attempt is
    not repeated.
    """

    if max_autotune not in _MAX_AUTOTUNE_SCOPES:
        # Not a bool: an unrecognised string would silently mean "off", and the
        # only symptom would be a slower package that still validates.
        raise ValueError(
            f"max_autotune must be one of {_MAX_AUTOTUNE_SCOPES}, got "
            f"{max_autotune!r}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    compiler = _activate_msvc()

    attention_backends = {
        axis: (
            _AXIS_ATTENTION_BACKENDS[axis] if attention_backend == "axis" else None
        )
        for axis in AXES
    }

    packages: dict[str, Any] = {}
    with _build_target(model_instance) as target_instance:
        model = target_instance.model_run
        examples = (
            _capture_module_inputs(target_instance, model) if targets == "all" else {}
        )

        for axis, (module_index, shape) in AXES.items():
            module = model.layers[0][module_index].eval()
            example = torch.randn(shape, device="cuda", dtype=torch.float16)
            _populate_rotary_cache(module, shape[1])

            with torch.inference_mode(), torch.autocast(
                "cuda",
                dtype=torch.float16,
            ), _pinned_attention_backend(module, attention_backends[axis]):
                expected = module(example)
                export_started = time.perf_counter()
                exported = torch.export.export(module, (example,))
                export_sec = time.perf_counter() - export_started

                package_path = output_dir / f"{axis}.pt2"
                compile_started = time.perf_counter()
                torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=str(package_path),
                    inductor_configs=_inductor_configs(
                        emulate_precision_casts,
                        max_autotune in ("transformers", "all"),
                    ),
                )
                torch.cuda.synchronize()
                compile_sec = time.perf_counter() - compile_started

                load_started = time.perf_counter()
                compiled = torch._inductor.aoti_load_package(package_path)
                _inject_constants(compiled, module)
                torch.cuda.synchronize()
                load_and_inject_sec = time.perf_counter() - load_started
                actual = compiled(example)
                torch.cuda.synchronize()
                same_block_error = _relative_error(expected, actual)

                # A package built from block 0 is reused for every block, so
                # validate it against a different block's weights as well.
                # Re-injecting resets the runner's folding state, which is what
                # makes the second run pick up the new weights.
                other = model.layers[1][module_index].eval()
                other_expected = other(example)
                _inject_constants(compiled, other)
                other_actual = compiled(example)
                torch.cuda.synchronize()
                cross_block_error = _relative_error(other_expected, other_actual)

            if cross_block_error > _CROSS_BLOCK_ERROR_TOLERANCE * same_block_error:
                raise RuntimeError(
                    "AOTI package baked block-specific constants: relative error "
                    f"{cross_block_error:.3e} on block 1 against "
                    f"{same_block_error:.3e} on the block it was built from"
                )
            if _archive_contains_weights(package_path):
                raise RuntimeError(f"AOTI package unexpectedly contains weights: {package_path}")
            packages[axis] = {
                "file": package_path.name,
                "kind": "transformer",
                "axis": axis,
                "input_shape": list(shape),
                "input_dtype": "torch.float16",
                "attention_backend": attention_backends[axis] or "auto",
                "package_bytes": package_path.stat().st_size,
                "contains_weights": False,
                "constant_count": len(compiled.get_constant_fqns()),
                "injected_constant_fqns": sorted(
                    name
                    for name in compiled.get_constant_fqns()
                    if not name.startswith(_FOLDED_CONST_PREFIX)
                ),
                "export_sec": export_sec,
                "compile_sec": compile_sec,
                "validation_load_and_inject_sec": load_and_inject_sec,
                "validation_max_abs": float((expected - actual).abs().max()),
                "validation_cosine": float(
                    torch.nn.functional.cosine_similarity(
                        expected.flatten().float(),
                        actual.flatten().float(),
                        dim=0,
                    )
                ),
                "validation_relative_error": same_block_error,
                "cross_block_relative_error": cross_block_error,
            }

        if targets == "all":
            for name, module_path in SINGLE_MODULES.items():
                module = model.get_submodule(module_path).eval()
                example = torch.randn_like(examples[name])
                package_path = output_dir / f"{name}.pt2"

                with torch.inference_mode(), torch.autocast(
                    "cuda",
                    dtype=torch.float16,
                ):
                    expected = module(example)
                    export_started = time.perf_counter()
                    exported = torch.export.export(module, (example,))
                    export_sec = time.perf_counter() - export_started

                    compile_started = time.perf_counter()
                    torch._inductor.aoti_compile_and_package(
                        exported,
                        package_path=str(package_path),
                        inductor_configs=_inductor_configs(
                            emulate_precision_casts,
                            max_autotune == "all",
                        ),
                    )
                    torch.cuda.synchronize()
                    compile_sec = time.perf_counter() - compile_started

                    load_started = time.perf_counter()
                    compiled = torch._inductor.aoti_load_package(package_path)
                    _inject_constants(compiled, module)
                    torch.cuda.synchronize()
                    load_and_inject_sec = time.perf_counter() - load_started
                    actual = compiled(example)
                    torch.cuda.synchronize()

                injected = {
                    fqn
                    for fqn in compiled.get_constant_fqns()
                    if not fqn.startswith(_FOLDED_CONST_PREFIX)
                }
                # These modules are compiled once and used once, so there is no
                # second instance to validate against. Requiring every weight to
                # be injectable catches the same baking failure instead.
                baked = sorted(set(dict(module.named_parameters())) - injected)
                if baked:
                    raise RuntimeError(
                        f"AOTI package for {name} baked {len(baked)} weights, "
                        f"first: {baked[:3]}"
                    )
                if _archive_contains_weights(package_path):
                    raise RuntimeError(
                        f"AOTI package unexpectedly contains weights: {package_path}"
                    )
                packages[name] = {
                    "file": package_path.name,
                    "kind": "module",
                    "module_path": module_path,
                    "input_shape": list(example.shape),
                    "input_dtype": str(example.dtype),
                    "package_bytes": package_path.stat().st_size,
                    "contains_weights": False,
                    "constant_count": len(compiled.get_constant_fqns()),
                    "parameter_count": len(dict(module.named_parameters())),
                    "export_sec": export_sec,
                    "compile_sec": compile_sec,
                    "validation_load_and_inject_sec": load_and_inject_sec,
                    "validation_max_abs": float((expected - actual).abs().max()),
                    "validation_relative_error": _relative_error(expected, actual),
                }

    manifest = {
        "format": "separator-regional-aoti-v2",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "sm": _device_arch(),
        "compiler": compiler,
        "weights_serialized": False,
        "emulate_precision_casts": emulate_precision_casts,
        "max_autotune": max_autotune,
        "packages": packages,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_packages(model_instance: Any, package_dir: Path) -> int:
    """Install this machine's packages onto a live model; return runner count.

    Raises on any mismatch. The caller treats that as "this machine cannot use
    the package" and continues eagerly -- refusing to run because an accelerator
    is stale would be worse than running slowly. Transactional for that reason:
    a runner that fails to load partway through undoes the ones already
    installed, so "eager" is then true of the model and not just of the label.
    """

    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("weights_serialized") is not False:
        raise RuntimeError("AOTI manifest does not describe weightless packages")
    for field, actual in (
        ("torch", torch.__version__),
        ("cuda", torch.version.cuda),
        ("sm", _device_arch()),
    ):
        recorded = manifest.get(field)
        if recorded != actual:
            # An ABI or architecture mismatch is not a slow path, it is an
            # unloadable one: the packages carry SASS for one architecture and
            # link against one libtorch.
            raise RuntimeError(f"AOTI {field} mismatch: {recorded!r} != {actual!r}")

    packages = manifest["packages"]
    payloads = {
        name: (package_dir / entry["file"]).read_bytes()
        for name, entry in packages.items()
    }
    # 2.9's loader extracts to a directory keyed on the archive's top-level
    # prefix, so reusing one package across blocks needs a distinct prefix each.
    scratch = tempfile.TemporaryDirectory(prefix="separator_aoti_")
    model_run = model_instance.model_run
    installed = 0
    # Every module whose forward has been replaced, in order. The original is
    # the class method the instance attribute shadows, so the undo is a `del`.
    replaced: list[torch.nn.Module] = []

    package_log = logging.getLogger("torch._inductor.package.package")
    previous_level = package_log.level

    def install(name: str, target: torch.nn.Module, suffix: str) -> None:
        nonlocal installed
        runner = Path(scratch.name) / f"{name}_{suffix}.pt2"
        _write_package_with_prefix(payloads[name], runner)
        compiled = torch._inductor.aoti_load_package(runner)
        compiled.load_constants(
            _constant_map(compiled, target),
            check_full_update=False,
            user_managed=True,
        )
        replaced.append(target)
        target.forward = compiled
        installed += 1

    package_log.setLevel(logging.ERROR)
    try:
        for name, entry in packages.items():
            if entry.get("kind", "transformer") == "transformer":
                index = 0 if entry.get("axis", name) == "time" else 1
                for block_index, block in enumerate(model_run.layers):
                    transformer = block[index]
                    _populate_rotary_cache(transformer, int(entry["input_shape"][1]))
                    install(name, transformer, f"b{block_index}")
            else:
                install(name, model_run.get_submodule(entry["module_path"]), "single")
    except BaseException:
        for target in reversed(replaced):
            del target.forward
        scratch.cleanup()
        raise
    finally:
        package_log.setLevel(previous_level)

    # The extracted runners have to outlive this call.
    model_instance._separator_aoti_scratch = scratch
    return installed


def _write_package_with_prefix(package_bytes: bytes, destination: Path) -> None:
    import io

    with zipfile.ZipFile(io.BytesIO(package_bytes)) as source:
        names = source.namelist()
        if not names:
            raise RuntimeError("Empty AOTI package")
        original = names[0].split("/", 1)[0]
        with zipfile.ZipFile(destination, "w") as target:
            for info in source.infolist():
                prefix, separator, relative = info.filename.partition("/")
                if not separator or prefix != original:
                    raise RuntimeError(f"Unexpected AOTI archive member: {info.filename}")
                target.writestr(
                    f"{destination.stem}/{relative}",
                    source.read(info),
                    compress_type=info.compress_type,
                )


def _device_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"
