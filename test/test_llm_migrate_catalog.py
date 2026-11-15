"""The old-format override converter keeps the old meaning, not the old cells."""

from __future__ import annotations

from pathlib import Path

import pytest

from finesub.llm.routing.migrate_catalog import MigrationError, convert, main
from finesub.llm.routing.model_catalog import load_effective_catalog

HEADER = (
    "fact_id|provider_tier|provider_kind|base_url|display_name|api_model_id"
    "|max_input_tokens|max_output_tokens|context_window|supports_native_search"
    "|thinking|rpm|tpm|capability|quota_pool"
)


def _convert(tmp_path: Path, rows: str) -> tuple[dict, list[str], Path]:
    old = tmp_path / "model_catalog.psv"
    old.write_text(rows, encoding="utf-8")
    text, notices = convert(old)
    new = tmp_path / "model_catalog.toml"
    new.write_text(text, encoding="utf-8")
    return {e.fact_id: e for e in load_effective_catalog(new)}, notices, new


def test_an_old_whole_row_override_keeps_its_defaults_instead_of_inheriting(
    tmp_path: Path,
) -> None:
    # The old row replaced the packaged one, so its blank cells meant
    # "default", not "the shipped value".
    facts, _, new = _convert(
        tmp_path,
        "# my endpoints\n"
        + HEADER
        + "\nlocal-codebuddy-glm-5_3-flash|LOCAL_CODEBUDDY||||glm-5.3-flash|300000"
        "|32000||true|||||CODEBUDDY_CREDITS\n",
    )
    entry = facts["local-codebuddy-glm-5_3-flash"]
    assert entry.capability is None
    assert entry.thinking_levels == ("high", "medium", "low")
    assert entry.context_window == 332000
    assert entry.display_name == "glm-5.3-flash"
    # Old default for a local agent's rates, written out because the new one differs.
    assert (entry.rpm, entry.tpm) == (100, 4_000_000)
    text = new.read_text(encoding="utf-8")
    assert text.startswith("# my endpoints\n")
    assert 'capability = ""' in text and 'display_name = ""' in text
    assert "note" not in text and "avoid" not in text


def test_a_new_agent_model_gets_the_old_rates_and_the_new_search_default(
    tmp_path: Path,
) -> None:
    facts, notices, _ = _convert(
        tmp_path,
        HEADER + "\nmy-dsh|LOCAL_DSH||||prov/model|194000|||||||medium|\n",
    )
    entry = facts["my-dsh"]
    assert (entry.rpm, entry.tpm) == (100, 4_000_000)
    assert entry.supports_native_search is True
    assert entry.self_reported is True
    assert any("my-dsh" in notice and "supports_native_search" in notice for notice in notices)


def test_an_old_true_on_an_openai_compat_row_is_dropped(tmp_path: Path) -> None:
    """It never searched (an old custom endpoint had no tool), and the new loader
    refuses true without `search_request` -- so it goes, with a notice."""

    facts, notices, new = _convert(
        tmp_path,
        HEADER + "\nds|deepseek|openai_compat|https://api.deepseek.com||deepseek-x|128000|||true|||||\n",
    )

    assert facts["ds"].supports_native_search is False
    assert "supports_native_search" not in new.read_text(encoding="utf-8")
    assert notices == [
        "ds: supports_native_search = true is dropped -- it never searched; "
        "add search_request to make this endpoint search"
    ]


def test_a_custom_endpoint_converts_with_typed_values(tmp_path: Path) -> None:
    facts, notices, new = _convert(
        tmp_path,
        "fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens"
        "|thinking|quality_score|is_free\n"
        "anth|ANTHROPIC|anthropic|https://api.anthropic.com|claude-x|1000000"
        "|high,medium,low|77|false\n"
        "oai|DEEPSEEK|openai_compat|https://api.deepseek.com|deepseek-x|128000||50|false\n",
    )
    entry = facts["anth"]
    assert entry.capability is not None and int(entry.capability) == 70
    # An Anthropic endpoint now searches by default and is named;
    # an OpenAI-compatible one has no `search_request`, so stays off, silently.
    assert entry.supports_native_search is True
    assert facts["oai"].supports_native_search is False
    assert len(notices) == 1 and notices[0].startswith("anth: supports_native_search is now true")
    text = new.read_text(encoding="utf-8")
    assert "capability = 77" in text
    assert 'thinking = ["high", "medium", "low"]' in text
    assert "is_free = false" in text


def test_a_token_budget_thinking_cell_becomes_integers(tmp_path: Path) -> None:
    facts, _, new = _convert(
        tmp_path,
        "fact_id|provider_tier|api_model_id|max_input_tokens|thinking\n"
        "g|GEMINI_FREE|gemini/x|1000|24576,16384,8192\n",
    )
    assert facts["g"].thinking_levels == ("24576", "16384", "8192")
    assert "thinking = [24576, 16384, 8192]" in new.read_text(encoding="utf-8")


def test_a_broken_old_file_is_refused(tmp_path: Path) -> None:
    old = tmp_path / "model_catalog.psv"
    old.write_text("fact_id|provider_tier|api_model_id|max_input_tokens\nx|T|m|abc\n", encoding="utf-8")
    with pytest.raises(MigrationError, match="max_input_tokens must be an integer"):
        convert(old)


def test_the_command_never_overwrites(tmp_path: Path, capsys) -> None:
    old = tmp_path / "model_catalog.psv"
    old.write_text("fact_id|provider_tier|api_model_id|max_input_tokens\nx|GEMINI_FREE|gemini/x|1000\n", encoding="utf-8")
    existing = tmp_path / "model_catalog.toml"
    existing.write_text("# mine\n", encoding="utf-8")

    assert main([str(old)]) == 1
    assert existing.read_text(encoding="utf-8") == "# mine\n"
    assert "not overwriting" in capsys.readouterr().err

    assert main([str(old), "-o", str(tmp_path / "out.toml")]) == 0
    assert "[models.x]" in (tmp_path / "out.toml").read_text(encoding="utf-8")
    assert not old.exists()
    assert (tmp_path / "model_catalog.psv.bak").read_text(encoding="utf-8").startswith("fact_id|")


def test_the_old_file_is_kept_under_the_first_free_backup_name(tmp_path: Path) -> None:
    old = tmp_path / "model_catalog.psv"
    old.write_text("fact_id|provider_tier|api_model_id|max_input_tokens\nx|GEMINI_FREE|gemini/x|1000\n", encoding="utf-8")
    (tmp_path / "model_catalog.psv.bak").write_text("earlier\n", encoding="utf-8")

    assert main([str(old)]) == 0
    assert (tmp_path / "model_catalog.psv.bak").read_text(encoding="utf-8") == "earlier\n"
    assert (tmp_path / "model_catalog.psv.2.bak").read_text(encoding="utf-8").startswith("fact_id|")
    assert not old.exists()


def test_a_row_for_a_renamed_packaged_model_moves_to_the_new_name(tmp_path: Path) -> None:
    facts, notices, new = _convert(
        tmp_path,
        "fact_id|provider_tier|api_model_id|max_input_tokens|max_output_tokens|context_window|capability\n"
        "local-claude-haiku-4_5|LOCAL_CLAUDE|haiku|1000000|128000|1000000|low\n",
    )
    assert "local-claude-haiku-4_5" not in facts
    assert int(facts["local-claude-haiku"].capability) == 50
    assert "[models.local-claude-haiku]" in new.read_text(encoding="utf-8")
    assert any("now called local-claude-haiku" in notice for notice in notices)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_a_rename_onto_a_model_of_your_own_is_refused(tmp_path: Path, order) -> None:
    rows = [
        "local-claude-haiku-4_5|LOCAL_CLAUDE|haiku|1000000|128000|1000000|low",
        "local-claude-haiku|LOCAL_CLAUDE|claude-haiku-own|1000000|128000|1000000|low",
    ]
    old = tmp_path / "model_catalog.psv"
    old.write_text(
        "fact_id|provider_tier|api_model_id|max_input_tokens|max_output_tokens|context_window|capability\n"
        + "".join(rows[i] + "\n" for i in order),
        encoding="utf-8",
    )
    with pytest.raises(MigrationError, match="would both be written as local-claude-haiku"):
        convert(old)


def test_the_config_scan_names_each_dropped_or_renamed_target(tmp_path: Path) -> None:
    from finesub.llm.routing.migrate_catalog import scan_config

    config = tmp_path / "config.toml"
    config.write_text(
        '[llm.presets.mine.bindings]\n'
        '"knowledge/quality" = "local-codex-completion-gpt-6_1-sol"\n'
        '"research/quality" = "local-dsh-native-deepseek-v4-pro"\n'
        '"correction-text/quality" = "local-codex-gpt-6_1-sol"\n',
        encoding="utf-8",
    )
    found = scan_config(config)
    assert found == [
        f"{config}:2: local-codex-completion-gpt-6_1-sol -> local-codex-gpt-6_1-sol",
        f"{config}:3: local-dsh-native-deepseek-v4-pro -> local-dsh-deepseek-v4-pro",
    ]


def test_an_old_target_name_in_a_binding_names_its_replacement() -> None:
    from finesub.llm.routing.model_routes import ModelRouteConfigError, load_model_routes

    with pytest.raises(ModelRouteConfigError, match="local-codex-native-gpt-6_1-sol -> local-codex-gpt-6_1-sol"):
        load_model_routes(
            user_config={
                "preset": "mine",
                "presets": {
                    "mine": {
                        "name": "mine",
                        "bindings": {"correction-text/quality": "local-codex-native-gpt-6_1-sol"},
                    }
                },
            }
        )


def test_a_patched_free_gemini_row_keeps_its_no_search(tmp_path: Path) -> None:
    # The packaged row says false; writing "" would hand it the new default (true).
    facts, notices, new = _convert(
        tmp_path,
        "fact_id|provider_tier|api_model_id|max_input_tokens|rpm|supports_native_search\n"
        "gemini-free-3_8-flash|GEMINI_FREE|gemini/gemini-3.8-flash|194000|10|\n",
    )
    assert facts["gemini-free-3_8-flash"].supports_native_search is False
    assert "supports_native_search" not in new.read_text(encoding="utf-8")
    assert not [n for n in notices if "supports_native_search" in n]


def test_the_manual_catalog_examples_load(tmp_path: Path) -> None:
    """Every TOML block in the routing manual that declares models loads."""

    manual = (Path(__file__).resolve().parents[1] / "docs" / "manual" / "model-routing.md").read_text(
        encoding="utf-8"
    )
    blocks = [
        block.split("\n", 1)[1]
        for block in manual.split("```")
        if block.startswith("toml") and "[models." in block
    ]
    assert len(blocks) >= 3
    for index, block in enumerate(blocks):
        path = tmp_path / f"example-{index}.toml"
        path.write_text(block, encoding="utf-8")
        load_effective_catalog(path)
