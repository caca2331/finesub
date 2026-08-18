"""Pick and locate the separator's compiled backend.

Three tiers, in order: an AOTInductor package built once on this machine, a
TorchInductor JIT cache, or plain eager. Everything here is best-effort -- a
missing toolchain, a stale cache or an unreadable probe all fall through to
eager, because losing an accelerator must never cost a separation run.

Layout, under the checkout so it is visible and deletable:

    cache/separator-accel/<key>/
        aoti/       weightless .pt2 packages plus their manifest
        inductor/   TORCHINDUCTOR_CACHE_DIR
        probe.json  whether an AOTI build has been attempted, and what happened

``<key>`` carries every version this artefact is bound to, so invalidation is
the key changing: a new torch, a new CUDA, a different GPU or a different
checkpoint lands in a fresh directory and the old one is simply never read
again. Deleting ``cache/separator-accel/`` resets all of it at once.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ....reporting import current_reporter

# Bumped when the build configuration changes in a way that invalidates
# artefacts the key alone would not distinguish -- a different target set, a
# different inductor flag. Part of the key, so old directories go unread.
BUILD_FORMAT = "1"

# JIT pays ~35s of graph reconstruction per process against ~2s for AOTI, so it
# only earns its keep on long inputs. Measured break-even is ~800s; 600s is used
# instead to match the block_seconds constant rather than introduce a second
# number, which costs a few seconds on inputs right at the threshold.
JIT_MIN_DURATION_SEC = 600.0

_ENV_DISABLE = "FINESUB_SEPARATOR_ACCEL"

# Snapshotted at import, before anything has touched Inductor: it is the only
# moment where an existing value is unambiguously the operator's, since
# Inductor writes its own default into the same variable the first time it is
# asked. A value set here is left alone.
_OPERATOR_INDUCTOR_CACHE_DIR = os.environ.get("TORCHINDUCTOR_CACHE_DIR")


@dataclass(frozen=True)
class AccelPaths:
    """Where this machine's artefacts for one exact stack version live."""

    root: Path

    @property
    def aoti(self) -> Path:
        return self.root / "aoti"

    @property
    def inductor(self) -> Path:
        return self.root / "inductor"

    @property
    def probe(self) -> Path:
        return self.root / "probe.json"


def acceleration_disabled() -> bool:
    """Honour an explicit opt-out, for bisecting a suspected accel problem."""

    return os.environ.get(_ENV_DISABLE, "").strip().lower() in {"0", "off", "no"}


def triton_available() -> bool:
    """Both compiled tiers need Triton; on Windows it comes from triton-windows."""

    try:
        return importlib.util.find_spec("triton") is not None
    except Exception:
        return False


def aoti_buildable() -> bool:
    """Whether a build could be attempted here at all, checked before choosing."""

    try:
        from . import separator_aoti

        return separator_aoti.cxx_toolchain_available()
    except Exception:
        return False


def cache_key(model_name: str) -> Optional[str]:
    """Identify the stack this artefact would be bound to, or None off CUDA.

    The torch comparison an AOTI package makes at load time is an exact string
    match, so the patch version belongs in the key; so does the GPU, because
    Triton emits SASS for the current architecture and no PTX to fall back on.
    """

    import torch

    if not torch.cuda.is_available():
        return None
    try:
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return None
    digest = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:8]
    # torch's local label usually already names the CUDA build, but not always
    # (a CPU wheel carries none), so the runtime version is kept separately --
    # spelled differently so the pair does not read as a duplicate.
    cuda = torch.version.cuda or "none"
    return (
        f"v{BUILD_FORMAT}-{torch.__version__}-cuda{cuda}-sm{major}{minor}-{digest}"
    )


def _cache_root() -> Optional[Path]:
    from ....paths import managed_separator_model_dir, resolve_checkout_root

    # An explicit model dir outranks the checkout: the desktop worker's app
    # source looks like a checkout but is a versioned directory that updates
    # orphan, and these artefacts are expensive enough to keep across updates.
    # Deliberately the *managed* directory, not wherever the weights were
    # found: a compiled package is specific to this install's torch and GPU.
    managed = managed_separator_model_dir()
    if managed is not None:
        return managed / "accel"
    try:
        root = resolve_checkout_root()
    except Exception:
        root = None
    if root is not None:
        return root / "cache" / "separator-accel"
    # Installed as a package rather than run from a checkout.
    return Path.home() / ".cache" / "audio-separator" / "accel"


def resolve_accel_paths(model_name: str) -> Optional[AccelPaths]:
    """Locate this machine's artefact directory, or None when there can be none."""

    if acceleration_disabled():
        return None
    key = cache_key(model_name)
    if key is None:
        return None
    root = _cache_root()
    if root is None:
        return None
    return AccelPaths(root=root / key)


def read_probe(paths: AccelPaths) -> dict:
    """Return the recorded AOTI build outcome; an unreadable probe reads as absent."""

    try:
        data = json.loads(paths.probe.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_probe(paths: AccelPaths, *, aoti: str, reason: str = "") -> None:
    """Record whether AOTI is usable here so a failed build is not retried.

    Best-effort: a probe that cannot be written costs a rebuild attempt next
    run, which is not worth failing a separation over.
    """

    payload = {"aoti": aoti, "checked_at": time.time()}
    if reason:
        payload["reason"] = reason[:500]
    try:
        paths.root.mkdir(parents=True, exist_ok=True)
        paths.probe.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def aoti_package_ready(paths: AccelPaths) -> bool:
    """A usable package directory is one with a manifest beside its .pt2 files."""

    try:
        return (paths.aoti / "manifest.json").is_file() and any(
            paths.aoti.glob("*.pt2")
        )
    except OSError:
        return False


def _axis_aware_attention(self, q, k, v):
    """Pin the SDPA backend measured per axis in E3.

    On 2.11 this is not only 6.4% faster than letting the dispatcher choose --
    `auto` also produced an extra VAD segment, so it is a correctness knob too.
    """

    import torch

    if q.is_cuda and q.dtype in {torch.float16, torch.bfloat16}:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            backend = (
                SDPBackend.CUDNN_ATTENTION
                if q.shape[-2] >= 256
                else SDPBackend.EFFICIENT_ATTENTION
            )
            with sdpa_kernel(backend):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0
                )
        except RuntimeError:
            pass
    return self._accel_original_flash_attn(q, k, v)


def _install_axis_attention(model_run) -> None:
    import types

    for module in model_run.modules():
        if module.__class__.__name__ != "Attend" or hasattr(
            module, "_accel_original_flash_attn"
        ):
            continue
        module._accel_original_flash_attn = module.flash_attn
        module.flash_attn = types.MethodType(_axis_aware_attention, module)


def _apply_jit(model_instance, paths: AccelPaths) -> None:
    """Compile the same modules the AOTI build targets, in-process.

    Two process-wide settings are changed here and deliberately not restored.
    Both have to outlive this call because ``torch.compile`` is lazy -- nothing
    is compiled until the first forward -- so a scope that put them back would
    put them back before they were ever read.
    """

    import torch

    if _OPERATOR_INDUCTOR_CACHE_DIR is None:
        paths.inductor.mkdir(parents=True, exist_ok=True)
        # Inductor's default cache lives in the system temp directory, so a
        # cleanup silently costs the 4-minute cold compile again. Keep it
        # beside the rest.
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(paths.inductor)
    # Without MSVC, Dynamo would try to build a C++ guard with cl.exe. Global,
    # and the only torch.compile user in the process is this one.
    torch._dynamo.config.enable_cpp_symbolic_shape_guards = False

    model_run = model_instance.model_run
    _install_axis_attention(model_run)
    for block in model_run.layers:
        for index, transformer in enumerate(block):
            block[index] = torch.compile(transformer)
    model_run.band_split = torch.compile(model_run.band_split)
    model_run.mask_estimators[0] = torch.compile(model_run.mask_estimators[0])


def apply_acceleration(
    model_instance,
    backend: str,
    paths: Optional[AccelPaths],
) -> str:
    """Install `backend` on a freshly warmed master model; return what stuck.

    Never raises. Anything that goes wrong here costs speed, not the run, so
    every failure degrades and says so on stderr rather than propagating.
    """

    if backend == "eager" or paths is None:
        return "eager"

    from . import separator_aoti

    if backend == "aoti":
        if not aoti_package_ready(paths):
            # A build that died partway leaves files behind, and build_packages
            # refuses a non-empty directory. Clearing first keeps that crash
            # from being recorded as "this machine cannot build", which would
            # turn one interrupted run into a permanent downgrade.
            # `shutil.rmtree` rather than `fsops.remove_tree`: this is our own
            # compile cache, holds nothing a user put there, and must not raise.
            shutil.rmtree(paths.aoti, ignore_errors=True)
            # Reported as progress, not as entering the stage: the stage is
            # already running, and announcing it again would draw a second
            # stage line and send the desktop a second "entered vocal" event.
            # Not a warning either -- nothing is wrong. But a silent
            # 90-second pause before the first block reads as a hang.
            current_reporter().progress(
                "vocal",
                completed=0,
                detail="正在为本机编译分离器（约 90 秒，仅一次）",
            )
            try:
                # Against the live model: the master is already resident and
                # warmed, and a second copy of the weights would not fit inside
                # the smaller GPU budgets.
                separator_aoti.build_packages(
                    paths.aoti,
                    model_instance=model_instance,
                )
            except Exception as exc:
                write_probe(paths, aoti="unavailable", reason=f"{type(exc).__name__}: {exc}")
                # Our own compile cache again -- see above for why it stays
                # `shutil.rmtree`.
                shutil.rmtree(paths.aoti, ignore_errors=True)
                # The verdict is cached so the next run does not pay the build
                # again, which also means a one-off failure (a busy GPU, a full
                # disk) sticks. Name the directory that clears it.
                current_reporter().warning(
                    "separator-compile-unavailable",
                    f"separator compilation unavailable ({exc}); continuing "
                    "without it.",
                    impact="分离会慢一些",
                    action=f"想再试一次就删掉 {paths.root}",
                )
                return "eager"
        try:
            separator_aoti.load_packages(model_instance, paths.aoti)
        except Exception as exc:
            # A package that will not load is stale, not merely slow: drop it.
            # Record the failure too -- without it a package that builds fine
            # and then never loads costs the full build again on every run.
            # Our own compile cache again -- see above for why it stays
            # `shutil.rmtree`.
            shutil.rmtree(paths.aoti, ignore_errors=True)
            write_probe(
                paths,
                aoti="unavailable",
                reason=f"load: {type(exc).__name__}: {exc}",
            )
            current_reporter().warning(
                "separator-package-unusable",
                f"separator package unusable ({exc}); discarded, continuing "
                "without it.",
                impact="分离会慢一些",
            )
            return "eager"
        write_probe(paths, aoti="ok")
        return "aoti"

    if backend == "jit":
        try:
            _apply_jit(model_instance, paths)
        except Exception as exc:
            current_reporter().warning(
                "separator-jit-unavailable",
                f"separator JIT unavailable ({exc}); continuing without it.",
                impact="分离会慢一些",
            )
            return "eager"
        return "jit"

    return "eager"


def select_backend(
    paths: Optional[AccelPaths],
    duration_sec: float,
    *,
    buildable: Optional[bool] = None,
) -> str:
    """Choose between ``aoti``, ``jit`` and ``eager`` for this run.

    ``buildable`` overrides the toolchain probe; a recorded failure in the probe
    outranks both, since a compiler that is present can still fail to produce a
    usable package.
    """

    if paths is None or not triton_available():
        return "eager"
    if aoti_package_ready(paths):
        return "aoti"
    if buildable is None:
        buildable = aoti_buildable()
    if buildable and read_probe(paths).get("aoti") != "unavailable":
        # No package yet and nothing says building one will fail: worth the
        # one-off build, which pays off at every input length.
        return "aoti"
    if duration_sec >= JIT_MIN_DURATION_SEC:
        return "jit"
    return "eager"
