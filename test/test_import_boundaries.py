from __future__ import annotations

import ast
from pathlib import Path


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


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


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
    function, and every module-level scan in this file would have skipped it.
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
    deliberate function-level import, which is why this checks `tree.body` only.

    CLAUDE.md's architecture table marks this rule as test-enforced. It said so
    before this test existed, which is the failure this test also fixes.
    """

    offenders: list[str] = []
    for source in (SOURCE_ROOT / "finesub_bootstrap").rglob("*.py"):
        for imported in _top_level_imports(source):
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


def test_public_media_and_subtitles_have_no_upward_imports() -> None:
    # Bare `llm` stays alongside the full path: a relative `from ..llm import`
    # surfaces here as its unresolved module name, and that is exactly the
    # import this rule exists to refuse.
    forbidden = ("llm", LLM, "finesub.speech", "finesub.workflows")
    offenders: list[str] = []
    for domain in ("media", "subtitles"):
        for source in (SOURCE_ROOT / "finesub" / domain).glob("*.py"):
            for imported in _top_level_imports(source):
                if imported.startswith(forbidden):
                    offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


def test_harness_public_layers_do_not_import_asr_dependencies_at_module_load() -> None:
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
        for imported in _top_level_imports(source):
            if imported.split(".", 1)[0] in HEAVY_IMPORTS:
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


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
        for imported in _top_level_imports(source):
            if imported.split(".", 1)[0] in PY311_STDLIB:
                offenders.append(f"finesub_bootstrap/{name} -> {imported}")

    assert offenders == [], (
        "import these lazily inside the function that needs them, and degrade "
        f"when they are absent: {offenders}"
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


def _module_level_llm_imports(path: Path) -> set[str]:
    """Absolute `finesub.llm.*` modules a file imports at module load.

    Relative imports are resolved against the file's own package, so a rule can
    be stated in the names a reader uses rather than in dot counts.
    """

    parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    package = parts[:-1]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(
                alias.name
                for alias in node.names
                if alias.name.startswith(f"{LLM}.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
                if node.module is None:
                    # `from . import a, b` -- the names are the modules.
                    prefix = ".".join(base)
                    imported.update(
                        f"{prefix}.{alias.name}"
                        for alias in node.names
                        if prefix.startswith(LLM)
                    )
                    continue
                target = ".".join(base + [node.module])
            else:
                target = node.module or ""
            if target.startswith(f"{LLM}."):
                imported.add(target)
    return imported


#: Where each `finesub.llm.routing` module sits in the one direction the package runs:
#: facts, then composition, then the execution identity, then the per-call
#: plan. A module may only import strictly below itself at load time. The
#: inversions that do exist -- `config` reading a resolved route back,
#: `model_router` classifying a `finesub.llm.client` failure -- are deferred imports
#: with a stated reason, and deferred imports are not module-level.
ROUTING_LAYERS = {
    "api_keys": 0,
    "config": 0,
    "model_catalog": 0,
    "model_routes": 1,
    "profiles": 1,
    "execution_policy": 2,
    "model_router": 3,
    "capabilities": 4,
}


def test_the_routing_layers_only_import_downward() -> None:
    offenders: list[str] = []
    for source in (LLM_ROOT / "routing").glob("*.py"):
        if source.stem == "__init__":
            continue
        assert source.stem in ROUTING_LAYERS, f"unranked routing module: {source.stem}"
        rank = ROUTING_LAYERS[source.stem]
        for imported in _module_level_llm_imports(source):
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
    """

    allowed = (f"{LLM}.routing", f"{LLM}.agent")
    offenders: list[str] = []
    for source in (LLM_ROOT / "routing").glob("*.py"):
        for imported in _module_level_llm_imports(source):
            if imported not in allowed and not imported.startswith(
                tuple(f"{name}." for name in allowed)
            ):
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


def test_the_knowledge_base_does_not_depend_on_the_stages() -> None:
    """The base is read and written by stages, never the other way around."""

    offenders: list[str] = []
    for source in (LLM_ROOT / "knowledge").rglob("*.py"):
        for imported in _module_level_llm_imports(source):
            if imported.startswith(f"{LLM}.stages"):
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []


def test_speech_has_no_llm_dependency() -> None:
    offenders: list[str] = []
    root = SOURCE_ROOT / "finesub" / "speech"
    for source in root.rglob("*.py"):
        for imported in _top_level_imports(source):
            # Bare `llm` and `llm.` catch the relative form, which surfaces
            # here unresolved; the full path catches the absolute one.
            if imported == "llm" or imported.startswith(("llm.", LLM)):
                offenders.append(f"{source.relative_to(SOURCE_ROOT)} -> {imported}")

    assert offenders == []
