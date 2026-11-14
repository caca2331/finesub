"""Health of the managed Python runtime, in four tiers.

| tier     | asks                                                       | cost          |
| -------- | ---------------------------------------------------------- | ------------- |
| instant  | required package dirs, patched CTranslate2, base interpreter | milliseconds |
| quick    | every file each locked package's RECORD lists, and versions | about 0.5 s  |
| full     | quick + sha256 of every one of those files                  | about 20 s   |
| imports  | spawn the runtime Python and import the whole chain         | 15 s or more |

The instant tier answers `status()`, which is asked many times per command, so
it only looks at the filesystem. The quick tier is what catches a file an
antivirus quarantined or a cleaner deleted: importing `silero_vad` never reads
its `.jit`, so the import probe passes over exactly that damage.

The expected contents need no list of ours: the lock says which distributions
at which versions make up the runtime, and every installed wheel carries a
RECORD of the files it put down with their digests. Only this module and
`environment.py` decide what "healthy" means; the shell decides what to do
about it.

Stdlib only: the thin launcher imports it on Python 3.10.
"""

from __future__ import annotations

import base64
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from finesub_bootstrap.fsops import write_atomic
from finesub_bootstrap.processes import no_window
from finesub_bootstrap.user_config import read_section, toml_parser


# ---- instant tier ------------------------------------------------------------

#: One name per thing that can go missing independently: the worker's own IPC
#: models, the separator stack (whose deps it pulls in transitively and has
#: broken on before), the ASR decode chain, and the optional-by-CLI-default
#: extras the pipeline reaches for. An environment that imports all of these can
#: run a task end to end; one that cannot must be reported as needing repair
#: rather than failing halfway through a job.
REQUIRED_RUNTIME_IMPORTS = (
    "pydantic",
    "audio_separator.separator",
    "beartype",
    "ml_collections",
    "faster_whisper",
    "ctranslate2",
    "silero_vad",
    "transformers",
)

#: Stock CTranslate2 satisfies `import ctranslate2` and even the version pin, but
#: cannot run fw-refine -- only the patched build emits the decoder trace. The
#: lock installs the right one by hashed URL; this catches an environment that
#: drifted off it. See docs/ct2-distribution.md.
#:
#: Deliberately the product name and not the whole local label: the question here
#: is patched-versus-stock, and stock CTranslate2 never carries a `+finesub...`
#: label. Pinning the version too would mean editing this line on every wheel
#: rebuild, and getting it wrong reads as "the stock build is installed" against a
#: perfectly good runtime.
REQUIRED_CTRANSLATE2_LOCAL_LABEL = "finesub"

#: The same requirement expressed as directories under site-packages, for the
#: checks that must not cost 15 seconds. Import names differ from distribution
#: names, so these are the on-disk package directories, not the pip names.
REQUIRED_RUNTIME_PACKAGE_DIRS = (
    "pydantic",
    "audio_separator",
    "faster_whisper",
    "ctranslate2",
    "silero_vad",
    "transformers",
    "torch",
)


def site_packages_of(python_executable: Path) -> Path:
    return python_executable.parent.parent / "Lib" / "site-packages"


def instant_health(python_executable: Path, *, python_builds: Path) -> tuple[bool, str]:
    """Cheap, synchronous check that an environment is still intact.

    Deliberately *not* the import probe: that spawns a Python which loads
    torch and the whole decode chain, ~15s warm and far worse cold, and
    `status()` is asked several times per command. Here we only look at the
    filesystem, which catches packages removed after a good install.
    """

    site_packages = site_packages_of(python_executable)
    if not python_executable.is_file():
        return False, f"找不到运行环境的 Python 解释器（{python_executable}）"
    if not site_packages.is_dir():
        return False, f"运行环境里没有 site-packages 目录（{site_packages}）"
    healthy, detail = repair_base_interpreter(python_executable, python_builds=python_builds)
    if not healthy:
        return False, detail
    missing = [
        name
        for name in REQUIRED_RUNTIME_PACKAGE_DIRS
        if not (site_packages / name).is_dir()
    ]
    if missing:
        return False, f"Python 运行环境缺少必需依赖：{', '.join(missing)}"
    # Stock CTranslate2 satisfies every path check above; only the version
    # separates it from the build fw-refine needs, and dist-info carries it
    # without importing anything.
    labels = [item.name for item in site_packages.glob("ctranslate2-*.dist-info")]
    if labels and not any(REQUIRED_CTRANSLATE2_LOCAL_LABEL in label for label in labels):
        return False, "ctranslate2 是原版而非补丁版，ASR 无法运行。"
    return True, ""


def repair_base_interpreter(python_executable: Path, *, python_builds: Path) -> tuple[bool, str]:
    """Keep a moved environment usable by fixing `pyvenv.cfg`'s `home`.

    A virtual environment carries no standard library: `Scripts\\python.exe`
    finds it through the base interpreter recorded in `pyvenv.cfg`. Moving
    the environment is otherwise harmless -- `sys.prefix` follows the
    executable -- so the only thing a hand-moved installation breaks is that
    one absolute path, and only when the base lives inside the tree that
    moved (an interpreter uv installed under `python-builds`, rather than a
    system Python that stayed put).

    Rewriting one line rescues several GB, and it is the one repair made
    without asking: it only ever points `home` at the same interpreter under
    this installation, when the recorded one no longer exists. This runs inside
    the instant check, so it reads first and writes only when actually broken.
    """

    config = python_executable.parent.parent / "pyvenv.cfg"
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except OSError:
        return True, ""  # No pyvenv.cfg: not a venv we manage.
    recorded: str | None = None
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip() == "home":
            recorded = value.strip()
            break
    if recorded is None or Path(recorded).is_dir():
        return True, ""
    replacement = python_builds / Path(recorded).name
    if not replacement.is_dir():
        return False, "Python 运行环境的基础解释器已不存在，需要重新安装运行环境。"
    rewritten = [
        f"home = {replacement}" if line.partition("=")[0].strip() == "home" else line
        for line in lines
    ]
    write_atomic(config, "\n".join(rewritten) + "\n")
    return True, ""


# ---- import tier -------------------------------------------------------------


def runtime_probe_source(modules: tuple[str, ...], ctranslate2_label: str) -> str:
    """Build the `python -c` program that decides whether a runtime is usable.

    Kept separate from the subprocess call so it can be exercised against a
    real interpreter with a cheap module list, rather than only against an
    environment that already has the multi-gigabyte ASR stack installed.
    """

    return (
        "import importlib\n"
        "mods = {}\n"
        f"for name in {list(modules)!r}:\n"
        "    mods[name] = importlib.import_module(name)\n"
        "ct2 = mods.get('ctranslate2')\n"
        f"label = {ctranslate2_label!r}\n"
        "version = getattr(ct2, '__version__', '') if ct2 is not None else ''\n"
        "problem = (\n"
        "    f'ctranslate2 {version} is the stock build, not the patched one'\n"
        "    if ct2 is not None and label not in version\n"
        "    else None\n"
        ")\n"
        # None exits 0; any string -- including '' -- exits 1 and is printed.
        "raise SystemExit(problem)\n"
    )


def import_probe(python_executable: Path) -> tuple[bool, str]:
    """Import the whole chain in the runtime's own Python."""

    probe = runtime_probe_source(REQUIRED_RUNTIME_IMPORTS, REQUIRED_CTRANSLATE2_LOCAL_LABEL)
    command = [str(python_executable), "-I", "-c", probe]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            # Generous on purpose: the probe imports torch (via the separator
            # and silero) plus the whole decode chain, which on a cold cache is
            # seconds, not milliseconds. A timeout here is reported as a broken
            # runtime, so erring short is the costly direction.
            timeout=120,
            creationflags=no_window(),
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"Python 运行环境检测失败：{error}"
    if result.returncode == 0:
        return True, ""
    details = result.stderr.strip().splitlines()
    reason = details[-1] if details else "required module import failed"
    return False, f"Python 运行环境缺少必需依赖：{reason}"


# ---- quick and full tiers ----------------------------------------------------


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _same_version(installed: str, locked: str) -> bool:
    """PEP 440 equality for the forms that actually differ on disk.

    A false "version differs" is expensive: it rebuilds a runtime that is
    fine, and a freshly built one would fail its own check. The release
    segment is compared as integers with trailing zeros dropped (`1.0` is
    `1.0.0`, `2024.01` is `2024.1`); the rest (pre/post/dev/local) as text.
    """

    def canonical(version: str) -> tuple[tuple[int, ...], str]:
        match = re.match(r"v?(\d+(?:\.\d+)*)(.*)", version.strip().lower())
        if match is None:
            return (), version.strip().lower()
        release = [int(part) for part in match.group(1).split(".")]
        while len(release) > 1 and release[-1] == 0:
            release.pop()
        return tuple(release), match.group(2)

    return canonical(installed) == canonical(locked)


def locked_packages(lock: Path) -> dict[str, str] | None:
    """Normalized distribution name -> version, from a `pylock.toml`.

    Environment markers are not evaluated -- that needs `packaging`, and this
    module is stdlib-only. Every marker in the lock is true on the platform the
    lock targets, which `test_runtime_health.py` holds; a marker that were false
    would read here as a package gone missing.

    None when there is no TOML parser (3.10 without `tomli`): then the quick
    check cannot run at all.
    """

    parser = toml_parser()
    if parser is None:
        return None
    with lock.open("rb") as handle:
        data = parser(handle)
    return {
        _normalized(str(entry["name"])): str(entry["version"])
        for entry in data.get("packages", [])
        if "name" in entry and "version" in entry
    }


@dataclass
class RuntimeReport:
    """How an environment differs from its lock."""

    #: Locked distributions with no `.dist-info` at all.
    missing_packages: list[str] = field(default_factory=list)
    #: (name, installed version, locked version).
    version_mismatches: list[tuple[str, str, str]] = field(default_factory=list)
    #: Locked distribution -> the versions of its several `.dist-info`. An
    #: interrupted upgrade or a hand-run `pip install` leaves two; which one a
    #: check would read is glob order, so neither can vouch for the package.
    duplicate_distributions: dict[str, list[str]] = field(default_factory=dict)
    #: Distribution -> RECORD paths that are not on disk.
    missing_files: dict[str, list[str]] = field(default_factory=dict)
    #: Distribution -> RECORD paths whose digest differs (full check only).
    hash_mismatches: dict[str, list[str]] = field(default_factory=dict)
    #: (name, version) of distributions the lock does not know. Not a
    #: mismatch: they break nothing, but a rebuild from the lock drops them.
    extras: list[tuple[str, str]] = field(default_factory=list)

    @property
    def mismatched(self) -> bool:
        return bool(
            self.missing_packages
            or self.version_mismatches
            or self.duplicate_distributions
            or self.missing_files
            or self.hash_mismatches
        )

    def packages_to_reinstall(self) -> list[str]:
        names = {
            *self.missing_packages,
            *(name for name, _installed, _locked in self.version_mismatches),
            *self.duplicate_distributions,
            *self.missing_files,
            *self.hash_mismatches,
        }
        return sorted(names)

    def problems(self) -> list[str]:
        """One line per kind of difference, for a person to read."""

        lines: list[str] = []
        if self.missing_files:
            files = [f"{name}: {path}" for name, paths in self.missing_files.items() for path in paths]
            lines.append(_listed("缺少文件", files))
        if self.missing_packages:
            lines.append(_listed("缺少整个包", self.missing_packages))
        if self.version_mismatches:
            lines.append(
                _listed(
                    "版本与锁不同",
                    [f"{name} {installed}（锁里是 {locked}）" for name, installed, locked in self.version_mismatches],
                )
            )
        if self.duplicate_distributions:
            lines.append(
                _listed(
                    "同一个包装了多份",
                    [f"{name}（{'、'.join(versions)}）" for name, versions in self.duplicate_distributions.items()],
                )
            )
        if self.hash_mismatches:
            files = [f"{name}: {path}" for name, paths in self.hash_mismatches.items() for path in paths]
            lines.append(_listed("文件内容与安装时不同", files))
        return lines

    def losses(self, *, include_versions: bool) -> list[str]:
        """What a rebuild from the lock takes away from this environment.

        Version differences are left out when the rebuild is an upgrade: the
        new lock makes every upgraded dependency look "different", and that is
        the upgrade itself, not something the user did.
        """

        lost = [f"{name} {version}" for name, version in self.extras]
        if include_versions:
            lost += [f"{name} {installed}" for name, installed, _locked in self.version_mismatches]
        return lost


def _listed(label: str, items: list[str], shown: int = 2) -> str:
    head = "、".join(items[:shown])
    rest = f" 等 {len(items)} 项" if len(items) > shown else ""
    return f"{label}：{head}{rest}"


def _installed_distributions(site_packages: Path) -> dict[str, list[tuple[str, Path]]]:
    """Every `.dist-info` by normalised name; more than one is a damaged install."""

    installed: dict[str, list[tuple[str, Path]]] = {}
    for entry in sorted(site_packages.glob("*.dist-info")):
        name, separator, version = entry.name[: -len(".dist-info")].rpartition("-")
        if separator:
            installed.setdefault(_normalized(name), []).append((version, entry))
    return installed


def _record_rows(dist_info: Path) -> list[tuple[str, str]] | None:
    """(relative path, `algo=digest`) for every hashed file inside site-packages.

    Rows without a digest are the RECORD itself; rows reaching outside
    site-packages are console-script launchers under `Scripts/`, which the
    pipeline never runs. Bytecode is left out even when a wheel ships it with a
    digest -- the patched CTranslate2 does: its `.pyc` records the source's
    original mtime, the install gives the source a new one, and the first
    import rewrites the `.pyc`. A cache Python regenerates by itself is neither
    damage nor worth a rebuild.

    None when the RECORD cannot be read: then nothing about the package can
    be vouched for, and the caller counts it as damaged rather than clean.
    """

    try:
        with (dist_info / "RECORD").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    return [
        (row[0], row[1])
        for row in rows
        if len(row) >= 2
        and row[1]
        and not row[0].startswith(("../", "/"))
        and not row[0].endswith(".pyc")
    ]


def _present_files(site_packages: Path) -> set[str]:
    """Every file under site-packages, as normcased relative paths.

    One directory walk instead of a `stat` per RECORD row: 1.8 s warm against
    5.5 s for 66 000 rows on the maintainer's machine.
    """

    present: set[str] = set()
    for directory, _subdirectories, files in os.walk(site_packages):
        relative = os.path.relpath(directory, site_packages)
        for name in files:
            present.add(os.path.normcase(os.path.normpath(os.path.join(relative, name))))
    return present


def _digest_matches(path: Path, expected: str) -> bool:
    algorithm, _separator, value = expected.partition("=")
    try:
        digest = hashlib.new(algorithm)
    except ValueError:
        return True  # An algorithm we cannot compute proves nothing either way.
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError:
        return False
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii") == value


def check_runtime(site_packages: Path, lock: Path, *, verify_hashes: bool = False) -> RuntimeReport | None:
    """Compare an environment with its lock. None when the lock cannot be read."""

    try:
        locked = locked_packages(lock)
    except (OSError, ValueError):
        return None
    if locked is None:
        return None
    installed = _installed_distributions(site_packages)
    present = _present_files(site_packages)
    report = RuntimeReport()
    for name, version in sorted(locked.items()):
        found = installed.get(name)
        if not found:
            report.missing_packages.append(name)
            continue
        if len(found) > 1:
            report.duplicate_distributions[name] = [found_version for found_version, _entry in found]
            continue
        [(installed_version, dist_info)] = found
        if not _same_version(installed_version, version):
            report.version_mismatches.append((name, installed_version, version))
            continue
        rows = _record_rows(dist_info)
        if rows is None:
            report.missing_files.setdefault(name, []).append(f"{dist_info.name}/RECORD")
            continue
        for relative, expected in rows:
            key = os.path.normcase(os.path.normpath(relative))
            if key not in present:
                report.missing_files.setdefault(name, []).append(relative)
            elif verify_hashes and not _digest_matches(site_packages / relative, expected):
                report.hash_mismatches.setdefault(name, []).append(relative)
    report.extras = sorted(
        (name, version)
        for name, found in installed.items()
        if name not in locked
        for version, _dist_info in found
    )
    return report


# ---- when to check, and what the user asked for ------------------------------

INTEGRITY_CHECK_VALUES = ("auto", "on", "off")

#: Written into the runtime directory, so a rebuild -- which replaces that
#: directory -- starts from "this version has not run yet".
STATE_FILE = "finesub-health.json"


@dataclass(frozen=True)
class RuntimeSettings:
    integrity_check: str = "auto"
    auto_repair: bool = True
    #: What was wrong with the configured values, for the caller to warn about.
    problems: tuple[str, ...] = ()


def read_runtime_settings(user_data: Path | None) -> RuntimeSettings:
    """`[runtime] integrity_check` and `[runtime] auto_repair`.

    Without a TOML parser nothing the user wrote can be read. `auto_repair` then
    fails closed -- the same reason `update_check` does: rebuilding is the thing
    a user may have switched off, and a rebuild erases what they put into the
    runtime. Checking stays at its default.
    """

    section = read_section(user_data, "runtime")
    if section is None:
        return RuntimeSettings(
            auto_repair=False,
            problems=("config.toml 读不了（缺 TOML 解析器），自动重建按关闭处理。",),
        )
    problems: list[str] = []
    check = section.get("integrity_check", "auto")
    if check not in INTEGRITY_CHECK_VALUES:
        problems.append(f"[runtime] integrity_check = {check!r} 不认识，按 \"auto\" 处理。")
        check = "auto"
    repair = section.get("auto_repair", True)
    if not isinstance(repair, bool):
        problems.append(f"[runtime] auto_repair = {repair!r} 不是 true/false，按 true 处理。")
        repair = True
    return RuntimeSettings(integrity_check=check, auto_repair=repair, problems=tuple(problems))


def _last_successful_version(runtime_root: Path) -> str | None:
    try:
        data = json.loads((runtime_root / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("lastSuccessfulVersion") if isinstance(data, dict) else None
    return value if isinstance(value, str) else None


def should_check_before_run(policy: str, runtime_root: Path, version: str) -> bool:
    """Whether the quick check runs before this transcription.

    `auto` checks until this version has finished one run: a first run may be
    interrupted or fail for other reasons before it reaches a damaged file, and
    a completed run proves every file this version uses is there. Without a
    version (no installed distribution) there is nothing to compare, so check.
    """

    if policy == "on":
        return True
    if policy == "off":
        return False
    return not version or _last_successful_version(runtime_root) != version


def record_success(runtime_root: Path, version: str) -> None:
    if not version or not runtime_root.is_dir():
        return
    try:
        write_atomic(
            runtime_root / STATE_FILE,
            json.dumps({"lastSuccessfulVersion": version}, ensure_ascii=False) + "\n",
        )
    except OSError:
        pass  # Only costs one more check next time.


# ---- non-ASCII paths ---------------------------------------------------------


def ansi_code_page() -> int | None:
    """The system ANSI code page, or None off Windows.

    Not `locale.getpreferredencoding()`: it reports UTF-8 under `PYTHONUTF8=1`,
    which the runtime's worker sets. `locale.getencoding()` is 3.11+.
    """

    if sys.platform != "win32":
        return None
    import ctypes

    return int(ctypes.windll.kernel32.GetACP())


def non_ascii_path_problem(paths: list[Path]) -> str | None:
    """Explain why native loaders may misread these paths, or None."""

    code_page = ansi_code_page()
    if code_page in (None, 65001):
        return None
    offending = [str(path) for path in paths if not str(path).isascii()]
    if not offending:
        return None
    return (
        f"FineSub 装在含非英文字符的路径下（{offending[0]}），而系统代码页是 {code_page} 不是 UTF-8；"
        "部分原生库会因此读错路径、报「文件不存在」。把环境变量 FINESUB_HOME 设到纯英文路径"
        "（例如 D:\\FineSub）后重新运行可以避开。"
    )


def repair_command() -> str:
    """The command a user can paste, with FINESUB_HOME when it is not default."""

    home = os.environ.get("FINESUB_HOME", "").strip()
    command = "finesub setup --repair"
    if home:
        return f"$env:FINESUB_HOME='{home}'; {command}"
    return command
