"""The preset directory and `CATEGORIES` must name the same set.

Nothing checked this before. Both directions failed silently, in different
ways, and neither produced a test failure:

* a `presets/*.toml` not listed in `CATEGORIES` is never loaded by anything --
  so it is not merely inert, its *syntax is never checked either*; it can sit
  in the tree malformed for as long as nobody notices;
* a name in `CATEGORIES` with no file behind it passes `preset_for_category`'s
  membership check and then dies inside `load_preset` with a bare
  `FileNotFoundError` -- an error that reads like a packaging problem rather
  than like the missing declaration it is.

The second half of this file is the reason the first half can be trusted: the
guard scans a directory, and a guard whose scan comes back empty is green
forever. `test_the_preset_directory_is_not_empty` is what makes a renamed or
moved directory red instead.
"""

from __future__ import annotations

from finesub.llm.knowledge.node import presets
from finesub.llm.knowledge.node.presets import (
    CATEGORIES,
    MATCHABLE_CATEGORIES,
    STANDALONE_CATEGORIES,
    load_preset,
)


def _shipped(*, legacy: bool = False) -> set[str]:
    """Preset names on disk. Reads through `_preset_folder` on purpose -- the
    seam a test would redirect to inject a category (see that function)."""

    folder = presets._preset_folder()
    if legacy:
        folder = folder / "legacy"
    return {
        entry.name[: -len(".toml")]
        for entry in folder.iterdir()
        if entry.is_file() and entry.name.endswith(".toml")
    }


def test_the_preset_directory_is_not_empty() -> None:
    assert _shipped(), "no preset files found -- the guard below would be vacuous"
    assert _shipped(legacy=True), "no legacy presets found"


def test_every_category_has_a_preset_and_every_preset_has_a_category() -> None:
    assert _shipped() == set(CATEGORIES)


def test_every_shipped_preset_parses_at_the_expected_version() -> None:
    """`parse_preset` is strict by design, so "it loads" is a real assertion.

    Only the two categories named in a link or a table were ever loaded by a
    test before this; a third could ship malformed.
    """

    for name in CATEGORIES:
        assert load_preset(name).version == 2

    # The frozen version-1 archive presets travel a separate path with its own
    # expected version, and predate `style` entirely.
    for name in _shipped(legacy=True):
        assert load_preset(name, legacy=True).version == 1


def test_the_matching_split_partitions_the_categories_in_order() -> None:
    """`MATCHABLE_CATEGORIES` is derived from each preset's own `matchable`.

    Order is asserted, not just membership: this tuple is what `repo.resolve()`
    sweeps and what `repo` writes the index files in, so a reordering is a
    behaviour change that would otherwise pass silently. It is also why
    `CATEGORIES` is written out rather than globbed -- a glob is alphabetical.
    """

    assert MATCHABLE_CATEGORIES + STANDALONE_CATEGORIES == CATEGORIES
    assert not set(MATCHABLE_CATEGORIES) & set(STANDALONE_CATEGORIES)
    for name in CATEGORIES:
        expected = MATCHABLE_CATEGORIES if load_preset(name).matchable else STANDALONE_CATEGORIES
        assert name in expected


def test_index_headers_follow_the_matching_split() -> None:
    """`index_header` is required exactly when the category has an index.

    A standalone category carrying one would be a value nothing reads --
    `repo.index_text` raises for those rather than building an index -- and
    the parser rejects it for that reason.
    """

    for name in CATEGORIES:
        preset = load_preset(name)
        assert bool(preset.index_header.strip()) is preset.matchable


def test_every_category_describes_itself() -> None:
    """`blurb` feeds the structure table the prompts carry. It was a hardcoded
    dict in `render.py` until 2026-09-07, and that dict's own comment records
    the incident where a hardcoded pair of categories made the table stop
    covering the store."""

    for name in CATEGORIES:
        assert load_preset(name).blurb.strip()


def test_the_index_face_and_the_template_face_are_different_category_sets() -> None:
    """Two sets meet in the prompts and they are not interchangeable.

    The INDEX face is matchable-only: a standalone category has no index at
    all, and `repo.index_text` raises rather than building one. The TEMPLATE
    face is every category, `style` included -- the maintenance session is the
    one that may write a style entry, so it has to see that template.

    Collapsing the two into "loop over the categories" is the mistake this
    pins: `render_structure_spec()` defaults to the matchable set, so calling
    it without an argument silently dropped `style` from the table for as long
    as the category existed.
    """

    from finesub.llm.knowledge.node.render import render_structure_spec

    index_face = set(MATCHABLE_CATEGORIES)
    template_face = set(CATEGORIES)
    assert index_face < template_face, "style would make these differ; it does not"

    everything = render_structure_spec(CATEGORIES)
    for name in CATEGORIES:
        assert f"**{name}（" in everything, f"{name} missing from the structure table"

    matchable_only = render_structure_spec(MATCHABLE_CATEGORIES)
    for name in template_face - index_face:
        assert f"**{name}（" not in matchable_only


def test_the_maintenance_prompt_carries_every_category_template() -> None:
    """The end-to-end half of the test above: §0 requirement 3 is about what
    the maintenance session actually receives, not about what a helper can
    render when asked nicely. The refined session -- the one with style
    evidence -- gets every template; without refined subtitles, and in the
    proposal repair that speaks the same contract, the templates are the
    writable categories only (owner 2026-10-03, cold read Q6)."""

    from finesub.llm.prompts import (
        build_knowledge_proposal_repair_messages,
        build_knowledge_update_messages,
    )

    def system(**kwargs):
        messages = build_knowledge_update_messages(
            task_summary="（测试）", window_packs="（无）", knowledge_indexes={}, **kwargs
        )
        return next(m["content"] for m in messages if m["role"] == "system")

    refined = system(refined=True)
    for name in CATEGORIES:
        assert f"**{name}（" in refined, f"maintenance prompt lost {name}'s template"
    repair = build_knowledge_proposal_repair_messages(
        task_summary="（测试）", read_rev=1, current_rev=2, conflict_rows="", rejected_rows="", kb_entries=""
    )[0]["content"]
    for prompt in (system(refined=False), repair):
        for name in CATEGORIES:
            assert (f"**{name}（" in prompt) is (name in MATCHABLE_CATEGORIES), name


def test_the_injected_indexes_cover_every_matchable_category() -> None:
    """The other half: one rendered block, one tagged section per matchable
    category, and nothing for a standalone one."""

    from finesub.llm.prompt_compose import render_knowledge_indexes

    block = render_knowledge_indexes({name: f"- {name} 条目" for name in CATEGORIES})
    for name in MATCHABLE_CATEGORIES:
        assert f"<{name}_index>" in block and f"</{name}_index>" in block
    for name in STANDALONE_CATEGORIES:
        assert f"<{name}_index>" not in block

    # Empty means （空）, one behaviour for every caller.
    empty = render_knowledge_indexes({})
    assert empty.count("（空）") == len(MATCHABLE_CATEGORIES)
    assert render_knowledge_indexes(None) == empty


def test_a_changed_preset_must_say_what_happens_to_existing_entries() -> None:
    """Changing a template is a migration, and the ledger is where it is owed.

    Plan §0 requirement 12: a preset change comes with a script for the entries
    written against the old shape. ⚠ The trigger is the CONTENT, not the
    `version` field — dropping a section, renaming a label or tightening
    `max_lines` all leave `version = 2` while existing entries stop conforming.
    `preset_digest` therefore hashes the parsed shape; a reworded comment is not
    a change and will not wake this test.

    What to do when it goes red:

    1. work out what existing entries need — often nothing (a purely additive
       section), sometimes a `move_line` sweep, sometimes a `recategorize`;
    2. write that into the `migration` field in plain language, or name the
       script that does it;
    3. update the `digest`.

    Updating the digest alone passes this test, and that is deliberate: no
    guard can tell whether a migration was thought about. What it can do is
    make the change impossible to land without someone typing the sentence.
    """

    import json
    from pathlib import Path

    from finesub.llm.knowledge.node.presets import preset_digest

    ledger_path = Path(__file__).parent / "data" / "preset_migrations.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert set(ledger) == set(CATEGORIES), (
        "every category needs a ledger entry: " + str(sorted(set(CATEGORIES) ^ set(ledger)))
    )
    drifted = []
    for name in CATEGORIES:
        recorded = str(ledger[name].get("digest", ""))
        actual = preset_digest(load_preset(name))
        if recorded != actual:
            drifted.append(f"{name}: recorded {recorded[:12]}…, now {actual[:12]}…")
        assert str(ledger[name].get("migration", "")).strip(), (
            f"{name}: the ledger entry needs a `migration` note (write 'none — <why>'"
            " when a change needs nothing from existing entries)"
        )
    assert drifted == [], (
        "a preset changed shape; existing entries were written against the old one.\n"
        "Say what happens to them in test/data/preset_migrations.json, then update\n"
        "the digest:\n  " + "\n  ".join(drifted)
    )


def test_a_new_category_reaches_every_face_without_editing_them(
    tmp_path, monkeypatch
) -> None:
    """The stopping-point acceptance: add a category, change nothing else.

    Injected through the `_preset_folder` seam rather than shipped, because the
    acceptance category is not a product decision — it must be testable without
    becoming real (plan §6). The three faces it has to reach are the three that
    used to be spelled out by hand:

    * the INDEX face (matchable categories only) — one tagged block per category
      in the rendered injection, and a token-count key per category in the
      exchange metadata;
    * the TEMPLATE face (every category) — a section table the maintenance
      session receives;
    * the STORE face — a preset that loads, with its own sections and caps.

    ⚠ Three module-level values are computed once and have to be re-derived
    after the redirect: `load_preset`'s cache, and the matchable/standalone
    split. `_preset_folder`'s docstring says so; this test is what makes that
    sentence true rather than hopeful.
    """

    import shutil

    from finesub.llm.exchange_metadata import index_block_tokens
    from finesub.llm.knowledge.node import presets as presets_module
    from finesub.llm.knowledge.node.render import render_structure_spec
    from finesub.llm.prompt_compose import render_knowledge_indexes

    folder = tmp_path / "presets"
    folder.mkdir()
    shipped = presets_module._preset_folder()
    for name in CATEGORIES:
        shutil.copy(str(shipped / f"{name}.toml"), folder / f"{name}.toml")
    # A heterogeneous one: it borrows streamer's shape but is its own category,
    # which is exactly the case the old `category == "streamer"` rules missed.
    seiyuu = (folder / "streamer.toml").read_text(encoding="utf-8")
    seiyuu = seiyuu.replace('name = "streamer"', 'name = "seiyuu"', 1)
    seiyuu = seiyuu.replace('blurb = "主播"', 'blurb = "声优"', 1)
    seiyuu = seiyuu.replace("# 主播索引", "# 声优索引", 1)
    (folder / "seiyuu.toml").write_text(seiyuu, encoding="utf-8")

    extended = CATEGORIES + ("seiyuu",)
    monkeypatch.setattr(presets_module, "_preset_folder", lambda: folder)
    monkeypatch.setattr(presets_module, "CATEGORIES", extended)
    presets_module.load_preset.cache_clear()
    matchable, standalone = presets_module._split_by_matching()
    monkeypatch.setattr(presets_module, "MATCHABLE_CATEGORIES", matchable)
    monkeypatch.setattr(presets_module, "STANDALONE_CATEGORIES", standalone)
    try:
        assert "seiyuu" in matchable

        # STORE face: it loads, with its own identity.
        preset = presets_module.preset_for_category("seiyuu")
        assert preset.blurb == "声优" and preset.matchable
        assert "声优索引" in preset.index_header

        # INDEX face: a tagged block, and a metadata key, both derived.
        block = render_knowledge_indexes({"seiyuu": "- 某声优 | 别名 | 简介"})
        assert "<seiyuu_index>" in block and "某声优" in block
        assert "seiyuu_index_tokens" in index_block_tokens(None, block)

        # TEMPLATE face: named in the structure table the maintenance session gets.
        table = render_structure_spec(extended)
        assert "**seiyuu（声优）**" in table

        # CONTRACT face: and the whole prompt has to agree with the table.
        # Rendering the pieces was not enough -- the assembled system message
        # went on declaring `category` a two-value enum three paragraphs below
        # the table that had just introduced a third (review 2026-09-07).
        from finesub.llm.prompts import _knowledge_update_system

        system = _knowledge_update_system(True, inputs_block="")
        assert "**seiyuu（声优）**" in system
        assert "`seiyuu`（声优）" in system, "the roster sentence must name it too"
        assert "|".join(matchable) in system, "the writable enum is the grant"

        from finesub.llm.prompt_compose import PROMPT_TEMPLATE_DIR

        contract = (PROMPT_TEMPLATE_DIR / "fragment_knowledge_output_v1.md").read_text(
            encoding="utf-8"
        )
        assert "streamer" not in contract, (
            "the ops contract must not name a category by hand — that is the "
            "copy that goes stale when one is added"
        )
    finally:
        presets_module.load_preset.cache_clear()
