from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
#: The harness's package path and its directory. Rules below are stated in the
#: full module path a reader would grep for; this is the one place it is
#: written down.
LLM = "finesub.llm"
LLM_ROOT = SOURCE_ROOT / "finesub" / "llm"
HEAVY_IMPORTS = {
    "audio_separator",
    "numba",
    "numpy",
    "torch",
    "torchaudio",
}


def _imports(path: Path, *, at_load: bool = False, root: Path = SOURCE_ROOT) -> set[str]:
    """Absolute module names a file under `src/` imports.

    Every import counts by default, function-level ones included: a deferred
    import is still a dependency, and it is the form a layering slip most
    often takes. `at_load` keeps only what runs when the module is imported
    (anything outside a function body), for rules about what module load needs
    or costs.

    Relative imports are resolved against the file's own package, so a rule is
    stated in the names a reader greps for; `from .. import llm` has no
    `module` and would otherwise vanish. `from x import y` yields both `x` and
    `x.y`, because `y` may be a submodule.
    """

    # The package a relative import starts from: the file's directory, for a
    # module and an `__init__.py` alike.
    package = list(path.relative_to(root).parent.parts)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    pending: list[ast.AST] = [tree]
    while pending:
        for node in ast.iter_child_nodes(pending.pop()):
            if at_load and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            pending.append(node)
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - (node.level - 1)]
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                imported.add(module)
                imported.update(f"{module}.{alias.name}" for alias in node.names if alias.name != "*")
    return imported


def test_the_import_scan_sees_deferred_and_package_relative_imports(tmp_path: Path) -> None:
    """The layering rules are only as wide as this scan.

    They used to read `tree.body` alone and skip a `from` with no module, so a
    function-level `from ...llm import x` or a module-less `from ... import
    llm` passed every one of them.
    """

    source = tmp_path / "finesub" / "speech" / "joint" / "probe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import json\n"
        "from ... import llm\n"
        "def later():\n"
        "    from ...llm.text import normalize\n",
        encoding="utf-8",
    )

    everything = _imports(source, root=tmp_path)
    assert {"json", "finesub.llm", "finesub.llm.text", "finesub.llm.text.normalize"} <= everything
    assert _imports(source, at_load=True, root=tmp_path) == {"json", "finesub", "finesub.llm"}


#: Packages that no longer exist: `asr_playground` became `finesub` and the
#: top-level `llm` moved under it (2026-08).
RENAMED_AWAY = ("asr_playground", "llm")

#: Everything tracked that Python reads. `cli/` too -- it is a separate suite,
#: so a stale import there is found by whoever runs it next, which may be a
#: release -- and `scripts/`, whose one module joined when the cn-lock
#: generator moved there (a tree outside the scan is a tree the guard does not
#: guard).
IMPORTING_TREES = ("src", "test", "tools", "cli", "scripts")


def _every_import(path: Path) -> set[str]:
    """Absolute module names a file imports *anywhere*, not just at load.

    `ast.walk`, not `tree.body`: the one import the rename missed sat inside a
    function, and the module-level scans this file had then would have skipped it.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        # `level` means relative, which cannot name a top-level package.
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module)
    return imported


def test_the_provisioning_layer_does_not_import_the_pipeline_package() -> None:
    """`finesub_bootstrap` stands on its own, at module load.

    The thin CLI vendors this package and runs `finesub` subcommands on its own
    interpreter before the managed runtime exists -- a module-level
    `import finesub.…` there is an ImportError at startup, not a missing
    feature. The one real dependency (`shell.py` resolving an output name) is a
    deliberate function-level import, which is why this checks module load only.

    docs/architecture.md marks this rule as test-enforced. It said so
    before this test existed, which is the failure this test also fixes.
    """

    offenders: list[str] = []
    for source in (SOURCE_ROOT / "finesub_bootstrap").rglob("*.py"):
        for imported in _imports(source, at_load=True):
            if imported == "finesub" or imported.startswith("finesub."):
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


#: `paths.py` is the one resolver, so it is the one file allowed to count its
#: way up to the repository root. See README_DEV「运行时路径解析契约」.
ROOT_WALK_EXEMPT = {"finesub/paths.py"}


def test_only_the_path_resolver_counts_its_way_to_the_repository_root() -> None:
    """`parents[N]` is an implicit contract with the directory depth.

    Move the module and it silently points somewhere else -- no import error,
    no failing call, just a wrong path. That is exactly what the 2026-08 move
    of `llm` into `finesub` did to one `parents[2]`, and the test holding it
    was `requires_main_checkout`, so a whole branch never ran it.
    """

    offenders: list[str] = []
    for source in SOURCE_ROOT.rglob("*.py"):
        relative = source.relative_to(SOURCE_ROOT).as_posix()
        if relative in ROOT_WALK_EXEMPT:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            # `Path(__file__)…parents[N]` -- the subscript is what makes it a
            # depth count rather than a `.parent` on something already known.
            if not isinstance(node, ast.Subscript):
                continue
            value = node.value
            if isinstance(value, ast.Attribute) and value.attr == "parents":
                offenders.append(f"{relative}:{node.lineno}")

    assert offenders == []


def test_nothing_imports_the_packages_the_rename_removed() -> None:
    """A stale import can read as green: it resolves to another checkout.

    `test_secrets.py` kept `from llm import llm_runtime` through the whole
    rename and passed, because the venv's editable install pointed at the main
    checkout, which still had `src/llm`. The suite was exercising a different
    tree's code and would only have failed once that tree changed too.

    Grep does not reliably find these either -- the rewrite that missed this
    line selected files by searching for `llm.` or `llm/`, and `from llm import
    x` contains neither.
    """

    repository_root = SOURCE_ROOT.parent
    offenders: list[str] = []
    for tree in IMPORTING_TREES:
        for source in (repository_root / tree).rglob("*.py"):
            if "node_modules" in source.parts or "_vendor" in source.parts:
                continue
            for imported in _every_import(source):
                head = imported.split(".", 1)[0]
                if head in RENAMED_AWAY:
                    relative = source.relative_to(repository_root).as_posix()
                    offenders.append(f"{relative} -> {imported}")

    assert offenders == []


def test_the_shared_key_pools_import_neither_llm_nor_speech() -> None:
    # `finesub.api_keys` serves both the LLM layer and the cloud ASR backends.
    forbidden = (LLM, "finesub.speech", "finesub.workflows")
    source = SOURCE_ROOT / "finesub" / "api_keys.py"
    assert [imported for imported in _imports(source) if imported.startswith(forbidden)] == []


def test_public_media_and_subtitles_have_no_upward_imports() -> None:
    forbidden = (LLM, "finesub.speech", "finesub.workflows")
    offenders: list[str] = []
    for domain in ("media", "subtitles"):
        for source in (SOURCE_ROOT / "finesub" / domain).rglob("*.py"):
            for imported in _imports(source):
                if imported.startswith(forbidden):
                    offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


def test_harness_public_layers_do_not_import_asr_dependencies_at_module_load() -> None:
    # Module load only: a function that needs torch imports it when called.
    offenders: list[str] = []
    roots = [
        SOURCE_ROOT / "finesub" / "media",
        SOURCE_ROOT / "finesub" / "subtitles",
        SOURCE_ROOT / "finesub" / "workflows",
        LLM_ROOT,
    ]
    sources = [source for root in roots for source in root.rglob("*.py")]
    # `finesub/__init__.py` runs before any of them. It did not use to matter --
    # the harness was a separate top-level package, so `import llm.x` never
    # touched the pipeline package at all. Since the 2026-08 move it is the
    # first thing every `import finesub.llm.…` executes, on installs that have
    # no torch: a `[harness]` one, and the thin CLI's own Python 3.10, which
    # runs `python -m finesub.llm.agent.agent_cleanup` with no managed runtime.
    # One convenience re-export added there would break both, and the root
    # suite would not notice -- its venv has torch.
    sources.append(SOURCE_ROOT / "finesub" / "__init__.py")
    for source in sources:
        for imported in _imports(source, at_load=True):
            if imported.split(".", 1)[0] in HEAVY_IMPORTS:
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


@pytest.mark.parametrize("argv", [["--help"], ["--gpu-tier", "no-such-tier"]], ids=["help", "bad-argument"])
def test_the_pipeline_cli_answers_help_and_argument_errors_without_torch(argv: list[str]) -> None:
    """`python -m finesub.pipeline --help` must not pay for torch.

    It used to: `pipeline` read the separator's sample rates from `separation`
    and imported `stages`, which imported the separator and the VAD-ASR stage
    at the top -- about 1.1 s of a 1.6 s `--help` was torch, and a mistyped
    option paid the same before argparse could say so. Run as the user runs it,
    in a fresh interpreter, because by now this process has torch loaded.
    """

    probe = (
        "import runpy, sys\n"
        f"sys.argv = ['pipeline', *{argv!r}]\n"
        "try:\n"
        "    runpy.run_module('finesub.pipeline', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "print(sorted(name for name in ('torch', 'torchaudio') if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": str(SOURCE_ROOT)},
        timeout=60,
    )

    assert result.stdout.strip().splitlines()[-1] == "[]", result.stdout + result.stderr


#: Standard-library modules that do not exist on Python 3.10. The published
#: CLI's shell runs on whatever interpreter the user installed it with
#: (`cli/pyproject.toml` declares `>=3.10`); only the *managed runtime* it
#: provisions is 3.12. A module-level import of one of these anywhere the shell
#: reaches turns a missing feature into `ModuleNotFoundError` for every command.
PY311_STDLIB = {"tomllib"}

#: Everything the thin CLI imports while dispatching. Deliberately a list of
#: modules rather than "all of finesub_bootstrap": the package also holds code
#: only the managed runtime reaches, and that is 3.12.
CLI_SHELL_MODULES = (
    "shell.py",
    "update_check.py",
    "secrets.py",
    "token_counter.py",
    "paths.py",
    "capabilities.py",
    "environment.py",
    "migrations/__init__.py",
    "task_index.py",
    "artifacts.py",
    "__init__.py",
)


def test_the_thin_cli_stays_importable_on_python_310() -> None:
    """`update_check.py` imported `tomllib` at module scope and broke 3.10.

    Not just the update notice: the module is imported before every non-help
    command, so `setup`, transcription and uninstall all died with
    `ModuleNotFoundError` on an interpreter the wheel says it supports. The
    import is lazy and guarded now; this keeps the next one from landing.

    Static, because a 3.10 interpreter is not available in this suite -- and a
    guard that needs one would never run.
    """

    bootstrap = SOURCE_ROOT / "finesub_bootstrap"
    offenders: list[str] = []
    for name in CLI_SHELL_MODULES:
        source = bootstrap / name
        if not source.exists():
            offenders.append(f"{name} -> listed but missing")
            continue
        for imported in _imports(source, at_load=True):
            if imported.split(".", 1)[0] in PY311_STDLIB:
                offenders.append(f"finesub_bootstrap/{name} -> {imported}")

    assert offenders == [], (
        "import these lazily inside the function that needs them, and degrade "
        f"when they are absent: {offenders}"
    )


#: Questions about *this machine's GPU* that only `device.py` may ask torch.
#: Using a card (empty_cache, synchronize, memory statistics, autocast) is
#: fine anywhere; asking whether there is one, which vendor, which architecture
#: or which build is not -- that is how a bare `is_available()` once put work
#: on a card with no kernels, and how an AMD card would be mistaken for NVIDIA.
DEVICE_PROBES = {
    "cuda": {
        "is_available",
        "device_count",
        "get_arch_list",
        "get_device_capability",
        "get_device_name",
        "get_device_properties",
        "mem_get_info",
        "is_bf16_supported",
        "set_device",
    },
    "version": {"cuda", "hip", "xpu"},
    "backends": {"mps"},
}
DEVICE_PROBE_HOME = "src/finesub/speech/runtime/device.py"


def _attribute_chain(node: ast.AST) -> list[str]:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return list(reversed(parts))


def test_device_probes_live_in_one_module() -> None:
    repository_root = SOURCE_ROOT.parent
    offenders: list[str] = []
    for source in SOURCE_ROOT.rglob("*.py"):
        relative = source.relative_to(repository_root).as_posix()
        if relative == DEVICE_PROBE_HOME:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            chain = _attribute_chain(node)
            if len(chain) < 2 or chain[0] != "torch":
                continue
            if chain[1] == "xpu" or (
                len(chain) >= 3 and chain[2] in DEVICE_PROBES.get(chain[1], set())
            ):
                offenders.append(f"{relative}:{node.lineno} {'.'.join(chain)}")

    assert offenders == [], (
        "ask finesub.speech.runtime.device instead of probing torch directly"
    )


#: The only production code allowed to call `torch.jit.load`. Handing it a path
#: string lets libtorch open the file with a narrow-character `fopen`, which
#: misreads a non-ASCII install path on a non-UTF-8 Windows code page -- the
#: 0.5.1 report of `silero_vad.jit` "not existing". The helper there passes a
#: file object instead.
TORCHSCRIPT_LOAD_HOME = "src/finesub/speech/runtime/native_paths.py"


def test_torchscript_loads_go_through_one_helper() -> None:
    repository_root = SOURCE_ROOT.parent
    offenders: list[str] = []
    for source in SOURCE_ROOT.rglob("*.py"):
        relative = source.relative_to(repository_root).as_posix()
        if relative == TORCHSCRIPT_LOAD_HOME:
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and _attribute_chain(node)[-2:] == ["jit", "load"]:
                offenders.append(f"{relative}:{node.lineno}")

    assert offenders == [], (
        "load TorchScript with finesub.speech.runtime.native_paths.load_torchscript"
    )


#: The only production code allowed to call `shutil.rmtree` directly. Every
#: other tree deletion goes through `fsops.remove_tree`, which refuses to follow
#: a directory link out of the tree it was asked to delete -- users do redirect
#: `models`/`cache`/`tasks` off the system drive with junctions, and on the
#: interpreters the CLI wheel supports `shutil.rmtree` walks straight through
#: one. The exemption is the separator's own compile cache: it holds nothing a
#: user put there, and clearing it must never raise.
RMTREE_EXEMPT = {"src/finesub/speech/preprocessing/separator/accel.py"}


def test_only_the_compile_cache_deletes_trees_with_shutil() -> None:
    repository_root = SOURCE_ROOT.parent
    offenders: list[str] = []
    for root in (SOURCE_ROOT,):
        for source in root.rglob("*.py"):
            relative = source.relative_to(repository_root).as_posix()
            if relative in RMTREE_EXEMPT:
                continue
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "rmtree"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "shutil"
                ):
                    offenders.append(f"{relative}:{node.lineno}")

    assert offenders == []


#: Where each `finesub.llm.routing` module sits in the one direction the package runs:
#: facts, then composition, then the execution identity, then the per-call
#: plan. A module may only import strictly below itself at load time. The
#: inversions that do exist -- `config` reading a resolved route back,
#: `model_router` classifying a `finesub.llm.client` failure -- are deferred imports
#: with a stated reason, which is why the two routing rules check module load only.
ROUTING_LAYERS = {
    "config": 0,
    "model_catalog": 0,
    # The old-format converter reads the catalog and nothing above it.
    "migrate_catalog": 1,
    "profiles": 1,
    # Above profiles since 2026-09-30: a preset's `defaults` table is checked
    # against the switch vocabulary profiles owns.
    "model_routes": 2,
    "execution_policy": 3,
    "model_router": 4,
    "capabilities": 5,
}


def test_the_routing_layers_only_import_downward() -> None:
    """At module load only; see `ROUTING_LAYERS` for the deferred inversions."""

    offenders: list[str] = []
    for source in (LLM_ROOT / "routing").glob("*.py"):
        if source.stem == "__init__":
            continue
        assert source.stem in ROUTING_LAYERS, f"unranked routing module: {source.stem}"
        rank = ROUTING_LAYERS[source.stem]
        for imported in _imports(source, at_load=True):
            if not imported.startswith(f"{LLM}.routing."):
                continue
            target = imported.split(".")[len(LLM.split(".")) + 1]
            assert target in ROUTING_LAYERS, f"unranked routing module: {target}"
            if ROUTING_LAYERS[target] >= rank:
                offenders.append(f"{source.stem} -> {target}")

    assert offenders == []


def test_routing_reaches_no_further_into_llm_than_the_agent_backends() -> None:
    """Routing decides *who* answers; it may not know how a call is made.

    `finesub.llm.agent` is the exception and not an accident: a local agent's driver
    config is part of the execution identity, and its spent-subscription
    freeze is a pre-filter on candidates.

    At module load only: `model_router` classifies a `finesub.llm.client`
    failure through a deferred import (see `ROUTING_LAYERS`).
    """

    allowed = (f"{LLM}.routing", f"{LLM}.agent")
    offenders: list[str] = []
    for source in (LLM_ROOT / "routing").glob("*.py"):
        for imported in _imports(source, at_load=True):
            if not imported.startswith(f"{LLM}."):
                continue
            if imported not in allowed and not imported.startswith(
                tuple(f"{name}." for name in allowed)
            ):
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


def test_the_knowledge_base_does_not_depend_on_the_stages() -> None:
    """The base is read and written by stages, never the other way around."""

    offenders: list[str] = []
    for source in (LLM_ROOT / "knowledge").rglob("*.py"):
        for imported in _imports(source):
            if imported.startswith(f"{LLM}.stages"):
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


def test_speech_has_no_llm_dependency() -> None:
    offenders: list[str] = []
    root = SOURCE_ROOT / "finesub" / "speech"
    for source in root.rglob("*.py"):
        for imported in _imports(source):
            if imported.startswith(LLM):
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []
