"""Documents that enumerate a set from the code must enumerate all of it.

`test_doc_links.py` guards that a citation *lands*; nothing guarded that a
sentence is still *true*. The failure this exists for is on record twice: the
2026-08-15 structure review found four factual errors in `README_DEV.md`, they
were fixed, and the 2026-09-03 review found four more in the same section --
the GPU tier table, still listing three tiers after the code had grown to five,
with `auto` capped at the wrong one. A commit had even read that section on its
way past without noticing.

**Scope, deliberately narrow.** Only *enumerable* facts: a set of names that
exists in the code and is spelled out in prose. Not numbers, not defaults, not
descriptions -- those need a human to read the sentence, and a guard that
approximates reading is one that cries wolf until people learn to ignore it
(the same reasoning `test_doc_links.py` gives for not guessing at unqualified
citations). One family today; add another when a second one burns us.
The one default that is checked, a setting's in the 「默认」 column of
`docs/manual/tuning.md`, is a single declared value read off a table cell, not
a sentence; it lives with the registry in `test_settings.py`.

⚠ What this does **not** catch: a doc that lists every name but miscounts them
in prose ("三个 GPU 档位"), or one that describes a tier wrongly. It catches the
shape the errors actually took -- a table that stopped growing with the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finesub.speech.runtime.resources import GPU_TIERS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: Documents that spell out the tier set, and therefore have to spell out all of
#: it. A document that merely *mentions* a tier in passing is not listed here --
#: `docs/manual/tuning.md` names `--gpu-tier` without enumerating, and demanding
#: five names of it would push the list toward "every doc that says the word".
#: `README.md` left the list on 2026-09-03: it no longer names any tier, the
#: user-facing table moved to `docs/manual/resources.md`.
TIER_ENUMERATING_DOCS = (
    "README_DEV.md",                # dev 向：档位表（历史上出错的正是这份）
    "docs/gpu-profiles.md",         # owner 文档：映射与实测依据
    "docs/manual/resources.md",     # 用户向：显卡档位与 CPU 回退
)


@pytest.mark.parametrize("relative", TIER_ENUMERATING_DOCS)
def test_every_gpu_tier_appears_in_the_docs_that_enumerate_them(relative: str) -> None:
    text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
    # Backticked, because that is how every one of these documents writes a
    # tier name, and because bare `high`/`standard` are ordinary English words
    # that would match anywhere.
    missing = [spec.name for spec in GPU_TIERS if f"`{spec.name}`" not in text]
    assert missing == [], (
        f"{relative} enumerates GPU tiers but is missing "
        + ", ".join(missing)
        + " -- `speech/runtime/resources.py` GPU_TIERS is the source of truth"
    )


def test_the_enumerating_list_still_points_at_real_documents() -> None:
    """A moved or renamed doc must not drop out of the guard silently."""

    missing = [name for name in TIER_ENUMERATING_DOCS if not (REPOSITORY_ROOT / name).exists()]
    assert missing == [], (
        "these are in TIER_ENUMERATING_DOCS but do not exist: " + ", ".join(missing)
    )


#: The user-facing document that spells out how one CodeBuddy login's
#: allowances are divided. Narrow on purpose, same reasoning as the tier list:
#: a document that merely mentions `hy3` is not enumerating anything.
CODEBUDDY_POOL_DOC = "docs/manual/agent.md"


def _codebuddy_pool_members() -> dict[str, list[str]]:
    from finesub.llm.routing.model_catalog import default_model_catalog

    members: dict[str, list[str]] = {}
    for entry in default_model_catalog():
        if entry.provider_tier != "LOCAL_CODEBUDDY":
            continue
        members.setdefault(entry.effective_quota_pool, []).append(entry.api_model_id)
    return members


def test_the_manual_names_every_model_line_on_each_codebuddy_allowance() -> None:
    """A row that changes allowance has to change this paragraph too.

    The failure this exists for is on record: the manual said "每条模型线各有
    各的额度" and "每一行都记了独立的额度池" for a month after that stopped being
    true -- the paid rows share one credit balance, and only the two free lines
    are metered apart. Prose that describes a *structure* rots silently; the
    part of it that a guard can hold is the membership, which is enumerable
    from the catalog and has to grow with it.
    """

    text = (REPOSITORY_ROOT / CODEBUDDY_POOL_DOC).read_text(encoding="utf-8")
    members = _codebuddy_pool_members()

    assert len(members) >= 2, (
        "this guard assumes the tier has several allowances; if that changed, "
        "the paragraph it guards needs rewriting rather than this assertion "
        "relaxing"
    )
    missing = sorted(
        model
        for models in members.values()
        for model in models
        if f"`{model}`" not in text
    )
    assert missing == [], (
        f"{CODEBUDDY_POOL_DOC} explains how the CodeBuddy allowances are "
        "divided but never names " + ", ".join(missing)
        + " -- model_catalog.toml's quota_pool key is the source of truth"
    )


def test_the_manual_does_not_claim_one_allowance_per_codebuddy_row() -> None:
    """The specific sentence that was wrong, kept out by name.

    Membership alone cannot catch it: a paragraph can name all five models and
    still describe the wrong structure around them. This is the one claim that
    was actually made and actually false, so it is worth a line -- not a
    general attempt to read the prose.
    """

    text = (REPOSITORY_ROOT / CODEBUDDY_POOL_DOC).read_text(encoding="utf-8")
    members = _codebuddy_pool_members()
    rows = sum(len(models) for models in members.values())

    if len(members) == rows:
        return  # one pool per row really is the structure; the claim is fine
    for claim in ("每条模型线各有各的额度", "每一行都记了独立的额度池"):
        assert claim not in text, (
            f"{CODEBUDDY_POOL_DOC} still says {claim!r}, but the catalog gives "
            f"{rows} rows only {len(members)} allowances"
        )
