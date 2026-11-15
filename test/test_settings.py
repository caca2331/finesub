"""`finesub.settings`: one declaration per key, one way to resolve it."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from finesub import config as app_config
from finesub import settings
from finesub.reporting import NullReporter, reporting_to


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    app_config.clear_config_cache()
    monkeypatch.setattr(settings, "_warned", set())
    yield
    app_config.clear_config_cache()


def _write(tmp_path, monkeypatch, body: str):
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    app_config.clear_config_cache()
    return path


# ------------------------------------------------------------------ the registry


def test_every_key_is_declared_once() -> None:
    names = [s.name for s in settings.REGISTRY]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("setting", settings.REGISTRY, ids=lambda s: s.name)
def test_every_declaration_is_complete_and_its_default_valid(setting) -> None:
    assert setting.kind in (bool, int, float, str)
    assert setting.doc
    if setting.default is not None:
        assert setting.check(setting.default) == setting.default


def test_importing_the_registry_pulls_in_no_consumer() -> None:
    """The documentation guard imports all of it; consumers bring torch."""

    probe = (
        "import sys, finesub.settings; "
        "print(sorted(m for m in sys.modules if m.startswith(('finesub.llm', "
        "'finesub.speech', 'torch'))))"
    )
    # This tree's `src`, explicitly: a bare interpreter imports whatever the
    # editable install points at, which from a worktree is the main checkout.
    source = str(Path(__file__).resolve().parents[1] / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [source, *filter(None, [os.environ.get("PYTHONPATH")])]
    )}
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
    )
    assert result.stdout.strip() == "[]"


# ------------------------------------------------------------------ guards

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src"

#: The only modules that read config.toml themselves: the registry (every
#: scalar key), and the two domains that own structured tables.
DIRECT_READERS = {
    "finesub/settings.py",
    "finesub/api_keys.py",  # [providers] / [pools]
    "finesub/llm/routing/model_routes.py",  # presets and groups under [llm]
}


def test_only_the_registry_and_the_structured_tables_read_the_file() -> None:
    """A second reader is how the four ways of reading a key came about:
    each one re-decided what "unset", `true` or `3.7` meant."""

    import ast

    offenders: list[str] = []
    for path in (SOURCE / "finesub").rglob("*.py"):
        relative = path.relative_to(SOURCE).as_posix()
        if relative in DIRECT_READERS or relative == "finesub/config.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # From any module: `api_keys` re-exports `read_config`, and that is the
        # door the `[chunking]` readers used.
        readers = {"read_config", "read_config_with_path"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and {a.name for a in node.names} & readers:
                offenders.append(f"{relative}:{node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr in readers:
                offenders.append(f"{relative}:{node.lineno}")

    assert offenders == [], "read scalar settings through finesub.settings"


@pytest.mark.parametrize("setting", settings.REGISTRY, ids=lambda s: s.name)
def test_every_key_is_documented_where_its_declaration_says(setting) -> None:
    """The rule "a user-facing option gets a line in the docs" (AGENTS.md),
    made into a test. Keys documented under `docs/manual/` are for users, so
    they also belong in the template and their table in the resources page."""

    import re

    # As a setting, not as a word: `enabled`, `remote` and `style` occur in
    # prose anyway, so a bare substring would stay green with the real
    # explanation deleted. Either TOML (`key =`) or a code span holding the key.
    token = rf"(?<![\w-]){re.escape(setting.key)}(?![\w-])"  # `[llm].key` counts
    as_setting = re.compile(rf"{token}\s*=|`[^`\n]*{token}[^`\n]*`")

    doc = (REPO / setting.doc).read_text(encoding="utf-8")
    assert as_setting.search(doc), f"{setting.name} is not documented in {setting.doc}"
    if setting.doc.startswith("docs/manual/"):
        example = (REPO / "config.example.toml").read_text(encoding="utf-8")
        assert re.search(rf"{token}\s*=", example), (
            f"{setting.name} is missing from config.example.toml"
        )
        resources = (REPO / "docs/manual/resources.md").read_text(encoding="utf-8")
        assert f"`[{setting.section}]`" in resources, (
            f"[{setting.section}] is missing from the table in docs/manual/resources.md"
        )


#: Flags whose setting is neither `--<key>` nor `--<section>-<key>`.
_FLAG_SETTINGS = {"--asr": "asr.backend", "--separate": "separator.enabled"}


def _tuning_table_defaults() -> list[tuple[settings.Setting, str]]:
    """Each setting a row of a `| 选项 | 默认 |` table in docs/manual/tuning.md names, with the default
    that row gives it. A row names a setting by its flag (the first one in the cell) or as
    `` `[section] key` / `key2` ``, with the defaults split by `` / `` the same way."""

    import re

    by_name = {s.name: s for s in settings.REGISTRY}
    by_flag = {**{f"--{s.key}".replace("_", "-"): s for s in settings.REGISTRY},
               **{f"--{s.section}-{s.key}".replace("_", "-"): s for s in settings.REGISTRY},
               **{flag: by_name[name] for flag, name in _FLAG_SETTINGS.items()}}
    found = []
    for line in (REPO / "docs/manual/tuning.md").read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("| `") or len(cells) < 3:
            continue
        names = re.findall(r"`([^`]+)`", cells[0].split("（")[0])  # not the `（`config.toml`）` note
        table = re.match(r"\[(\w+)\] (\w+)$", names[0])
        if table:
            section = table[1]
            keys = [table[2], *names[1:]]
            defaults = cells[1].split(" / ")
            found += [(by_name[f"{section}.{key}"], default) for key, default in zip(keys, defaults, strict=True)]
        elif names[0] in by_flag:
            found.append((by_flag[names[0]], cells[1]))
    return found


def test_the_tuning_tables_give_each_setting_its_declared_default() -> None:
    """The manual's 「默认」 column, where the default is a setting's: a single enumerable value
    (`test_doc_facts.py` keeps prose defaults out of scope; these are not prose). A setting the
    manual only explains in a paragraph -- the length scale, the two linger times -- is not read."""

    import re

    words = {True: {"开", "`true`", "`on`"}, False: {"关", "`false`", "`off`"}}
    rows = _tuning_table_defaults()
    wrong = []
    for setting, cell in rows:
        if setting.kind is bool:
            agrees = cell in words[setting.default]
        elif setting.kind is str:
            value = re.match(r"`([^`]*)`", cell)
            agrees = value is not None and value[1] == setting.default
        else:
            number = re.match(r"`?(-?\d+(?:\.\d+)?)", cell)
            agrees = number is not None and float(number[1]) == setting.default
        if not agrees:
            wrong.append(f"{setting.name}: manual says {cell!r}, declared {setting.default!r}")
    assert wrong == []
    # The scan surface is part of the guard: a table whose shape the reader stops recognising goes quiet.
    assert {"asr.backend", "separator.enabled", "postprocess.empty_cues", "llm.max_retries_per_window",
            "llm.api_soft_timeout_fraction", "llm.local_agent_exit_on_accept"} <= {s.name for s, _ in rows}


# ------------------------------------------------------------------ resolve


@pytest.mark.parametrize("setting", settings.REGISTRY, ids=lambda s: s.name)
def test_every_key_absent_valid_and_wrongly_typed(setting, tmp_path, monkeypatch) -> None:
    """New keys get this for free: unset, a valid value, a value of the wrong type."""

    _write(tmp_path, monkeypatch, "[providers]\ntavily = false\n")
    assert setting.resolve() == setting.default

    valid = {bool: "true", int: "7", float: "7.5", str: '"x"'}[setting.kind]
    if setting.minimum is not None and setting.kind is not bool:
        valid = str(setting.minimum if setting.maximum is None else setting.maximum)
    _write(tmp_path, monkeypatch, f"[{setting.section}]\n{setting.key} = {valid}\n")
    assert setting.resolve() == setting.check(setting.configured())

    wrong = '"text"' if setting.kind is not str else "3"
    _write(tmp_path, monkeypatch, f"[{setting.section}]\n{setting.key} = {wrong}\n")
    with pytest.raises(ValueError, match=f"{setting.section}\\.{setting.key} must be"):
        setting.resolve()


def test_false_zero_and_empty_are_answers_not_absence(tmp_path, monkeypatch) -> None:
    _write(
        tmp_path, monkeypatch,
        '[vad]\nsilero_assist = false\n'
        '[chunking]\nmax_window_subtitle_tokens = 0\n'
        '[llm]\nstyle = ""\n',
    )

    assert settings.VAD_SILERO_ASSIST.resolve() is False
    assert settings.MAX_WINDOW_SUBTITLE_TOKENS.resolve() == 0
    assert settings.STYLE.resolve() == ""
    assert settings.STYLE.configured() == ""


def test_an_explicit_value_wins_without_reading_the_file_for_that_key(
    tmp_path, monkeypatch
) -> None:
    """Today's resolvers short-circuit; a broken value the flag overrides
    must not stop the run."""

    _write(tmp_path, monkeypatch, '[separator]\nenabled = "no"\n')

    assert settings.SEPARATE.resolve(False) is False
    with pytest.raises(ValueError):
        settings.SEPARATE.resolve()


def test_an_explicit_value_is_checked_and_blamed_as_explicit(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, monkeypatch, "[segmentation]\nlength_scale = 0.8\n")

    with pytest.raises(ValueError, match="explicit value") as raised:
        settings.SPLIT_LENGTH_SCALE.resolve(1.7)
    assert str(path) not in str(raised.value)
    # The manifest case that used to be read as True.
    with pytest.raises(ValueError, match="separator.enabled must be true or false"):
        settings.SEPARATE.resolve("false")


def test_a_bad_file_value_names_the_file(tmp_path, monkeypatch) -> None:
    path = _write(tmp_path, monkeypatch, "[segmentation]\nlength_scale = 2.0\n")

    with pytest.raises(ValueError, match=r"within \[0\.6, 1\.6\]") as raised:
        settings.SPLIT_LENGTH_SCALE.resolve()
    assert str(path) in str(raised.value)


@pytest.mark.parametrize(
    ("setting", "value", "wording"),
    [
        (settings.SEPARATE, 1, "true or false"),
        (settings.SEPARATE, "true", "true or false"),
        (settings.MAX_WINDOW_SUBTITLE_TOKENS, True, "an integer"),
        (settings.MAX_WINDOW_SUBTITLE_TOKENS, 3.7, "an integer"),
        (settings.MAX_WINDOW_SUBTITLE_TOKENS, "12", "an integer"),
        (settings.MAX_WINDOW_SUBTITLE_TOKENS, -1, "at least 0"),
        (settings.END_EXTEND, True, "a number"),
        (settings.END_EXTEND, "0.3", "a number"),
        (settings.END_EXTEND, float("nan"), "a finite number"),
        (settings.LOCAL_AGENT_TIMEOUT_SECONDS, 5, "at least 10"),
        (settings.PROXY, 3, "a string"),
    ],
)
def test_values_are_never_coerced(setting, value, wording) -> None:
    with pytest.raises(ValueError, match=wording):
        setting.check(value)


def test_a_float_setting_takes_an_integer() -> None:
    assert settings.END_EXTEND.check(1) == 1.0


def test_normalisation_is_left_to_the_domain() -> None:
    """`style_mode = ""` means "read" today; that is the domain's call."""

    assert settings.STYLE_MODE.check("") == ""
    assert settings.STYLE_MODE.check(" Update ") == " Update "


def test_a_shown_value_never_carries_a_credential() -> None:
    with pytest.raises(ValueError) as raised:
        settings.PROXY.check(["https://carl:hunter2@proxy.example"])
    assert "hunter2" not in str(raised.value)
    assert "proxy.example" in str(raised.value)


def test_a_section_that_is_not_a_table_is_an_error(tmp_path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, 'segmentation = "nope"\n')

    with pytest.raises(ValueError, match=r"\[segmentation\] must be a TOML table"):
        settings.SPLIT_LENGTH_SCALE.resolve()


def test_one_read_serves_several_keys(tmp_path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "[postprocess]\nend_extend_seconds = 0.5\n")
    loaded = settings.read()
    _write(tmp_path, monkeypatch, "[postprocess]\nend_extend_seconds = 0.9\n")

    assert settings.END_EXTEND.resolve(config=loaded) == 0.5
    assert settings.END_EXTEND.resolve() == 0.9


# ------------------------------------------------------------------ unknown keys


class _Warnings(NullReporter):
    def __init__(self) -> None:
        self.codes: list[str] = []
        self.messages: list[str] = []

    def warning(self, code, message, *, impact="", action="") -> None:
        self.codes.append(code)
        self.messages.append(message)


def test_a_misspelt_key_is_warned_about_once(tmp_path, monkeypatch) -> None:
    _write(tmp_path, monkeypatch, "[postprocess]\nend_extend_second = 0.5\n")
    reporter = _Warnings()

    with reporting_to(reporter):
        assert settings.END_EXTEND.resolve() == settings.END_EXTEND.default
        settings.END_EXTEND.resolve()

    assert reporter.codes == ["config-unknown-key"]
    assert "end_extend_second" in reporter.messages[0]
    assert "end_extend_seconds" in reporter.messages[0], "names what it probably meant"


def test_model_route_tables_and_foreign_tables_are_not_warned_about(tmp_path, monkeypatch) -> None:
    """What model_routes reads under `[llm]` is known; `[cli]` and `[providers]` are not ours."""

    _write(
        tmp_path, monkeypatch,
        '[llm]\npreset = "agy"\n[llm.presets.my]\nname = "mine"\n'
        '[llm.model_groups.g]\ntargets = []\n[llm.preferred_targets]\n'
        '[cli]\nupdate_check = false\n[providers]\ntavily = false\n',
    )
    reporter = _Warnings()

    with reporting_to(reporter):
        settings.read()

    assert reporter.codes == []


def test_a_stray_table_under_llm_is_warned_about(tmp_path, monkeypatch) -> None:
    """`[llm.bindings]` (bindings belong under a preset) was copied from an old
    manual example; nothing reads it, and nothing used to say so."""

    _write(tmp_path, monkeypatch, '[llm]\npreset = "agy"\n[llm.bindings]\ncorrection = "x"\n')
    reporter = _Warnings()

    with reporting_to(reporter):
        settings.read()

    assert reporter.codes == ["config-unknown-key"]
    assert "'bindings'" in reporter.messages[0]
    known = set(reporter.messages[0].split("Known: ", 1)[1].split(", "))
    assert "preset" in known
    assert not known & {"providers", "models"}, (
        "the tables that moved out are not offered as settings"
    )


def test_the_llm_keys_read_elsewhere_are_what_model_routes_reads() -> None:
    """A source guard: the unknown-key warning names model_routes' keys rather
    than importing it, so a key it starts reading has to be added here too, or
    every user of it gets a warning that their working setting is ignored."""

    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src/finesub/llm/routing/model_routes.py"
    ).read_text(encoding="utf-8")
    read = set(re.findall(r'_table\(user, "([a-z_]+)"\)', source))
    read |= set(re.findall(r'user\.get\("([a-z_]+)"', source))
    loop = re.search(r'for section in \(([^)]*)\):\n\s+if _table\(user, section\)', source)
    assert loop, "the moved-table check in model_routes changed shape"
    moved = set(re.findall(r'"([a-z_]+)"', loop.group(1)))
    read |= moved

    assert read == set(settings._READ_ELSEWHERE["llm"])
    assert moved == set(settings._MOVED["llm"]), "the tables model_routes raises on"


def test_only_model_routes_reads_the_llm_table_directly() -> None:
    """The guard above scans model_routes only; this one keeps that scan
    sufficient. A new reader of `config.toml`'s `[llm]` elsewhere would bring
    keys the unknown-key warning does not know. It looks for `.get("llm")`,
    the way the table is read; `["llm"]` also indexes the scheduler's worker
    counts, so it is not a usable signal."""

    import re
    from pathlib import Path

    source_root = Path(__file__).resolve().parents[1] / "src"
    readers = sorted(
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
        if re.search(r"""\.get\(\s*["']llm["']""", path.read_text(encoding="utf-8"))
    )

    assert readers == ["finesub/llm/routing/model_routes.py"], (
        "a new module reads config.toml's [llm] directly: add its keys to "
        "settings._READ_ELSEWHERE (and to the guard above), or read them "
        "through finesub.settings"
    )


def test_an_undelivered_warning_is_tried_again(tmp_path, monkeypatch) -> None:
    """A worker thread with no reporter must not use up the one warning."""

    _write(tmp_path, monkeypatch, "[vad]\nsilero_asist = false\n")
    settings.read()  # no reporter bound: dropped

    reporter = _Warnings()
    with reporting_to(reporter):
        settings.read()

    assert reporter.codes == ["config-unknown-key"]
