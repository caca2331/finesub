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
