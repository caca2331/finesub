"""Sub-entry (term) pre-injection: matching below the entry level.

The engine already existed and already ran -- `node/matching.py` indexes
subject surfaces AND term surfaces AND items (aliases, misheard variants), and
the correction stage scans every window with it -- but it was wired in shadow:
matches were booked as events and changed nothing about injection.

The measurement that justified promoting it, from that ledger: **347 of 351
matches were terms and only 4 were subjects**, and **87% of the term hits sat
under a subject that was never itself mentioned**. Entry names are game titles
and streamer names, which people rarely say out loud; what they say are the
characters and proper nouns inside them. So the entry-level matcher is not
broken -- it is answering a question the transcript rarely poses.
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.llm.knowledge import base


def _match(subject="原神", section="角色", line="- ルミ|露米|Lumi|霜精", via=(), hits=1):
    return base.TermMatch(
        subject=subject, section=section, line=line, via=tuple(via), hits=hits
    )


class TestRendering:
    def test_a_hit_carries_its_address(self) -> None:
        """A bare term line says nothing about which work it belongs to; the
        parent and section are what make it an answerable fact."""

        rendered = base.render_term_matches([_match()])

        assert rendered == "- [原神 / 角色] ルミ|露米|Lumi|霜精"

    def test_a_hit_with_no_address_still_renders(self) -> None:
        rendered = base.render_term_matches([_match(subject="", section="")])

        assert rendered == "- ルミ|露米|Lumi|霜精"

    def test_only_the_line_is_injected_not_the_parent_entry(self) -> None:
        """The point of matching at this level is that one spoken name costs
        one line rather than a whole entry."""

        rendered = base.render_term_matches([_match(), _match(line="- A|B|C|D")])

        assert rendered.count("\n") == 1
        assert len(rendered.splitlines()) == 2

    def test_nothing_matched_renders_nothing(self) -> None:
        assert base.render_term_matches([]) == ""


class TestSelection:
    def test_empty_text_never_touches_the_store(self, tmp_path) -> None:
        """Cheap guard with a real consequence: building the exact index opens
        the store, and the injection path runs on every window."""

        assert base.match_terms(tmp_path / "missing", "") == []
        assert base.match_terms(tmp_path / "missing", "   ") == []

    def test_the_cap_is_sized_from_the_ledger(self) -> None:
        """24 is not a round number pulled from the air: the shadow ledger's
        median is 8 hits per task and its maximum is 59, over 42 distinct
        nodes. A cap far below the median would drop the ordinary case."""

        assert base.KB_TERM_PREINJECT_MAX_HITS == 24

    def test_a_match_serialises_for_the_run_report(self) -> None:
        """The report is how an A/B on this reads what was injected."""

        payload = _match(via=("ルミィ",), hits=3).to_dict()

        assert payload == {
            "subject": "原神",
            "section": "角色",
            "line": "- ルミ|露米|Lumi|霜精",
            "via": ["ルミィ"],
            "hits": 3,
        }


def _seed(root: Path) -> Path:
    """One game entry, aliases 崩铁 / 星铁, with one character line under it."""

    (root / "common").mkdir(parents=True)
    (root / "common" / "index.md").write_text(
        "- 崩坏星穹铁道 [游戏] | Honkai: Star Rail | 崩铁、星铁 | HoYoverse RPG\n", encoding="utf-8"
    )
    (root / "common" / "崩坏星穹铁道.md").write_text(
        "# 崩坏星穹铁道\nHoYoverse RPG\n\n## 档案\n本名: 崩坏星穹铁道 / Honkai: Star Rail\n别名: 崩铁、星铁\n\n"
        "## 角色\n\n三月七|三月七|March 7th||开朗少女。\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )
    return root


class TestItIsActuallyWired:
    def test_the_render_path_injects_a_term_and_dedupes_it_under_a_whole_entry(self, tmp_path) -> None:
        """Shadow-only was the previous state: the events got booked and the
        prompt carried nothing. A named character alone now injects its line;
        an entry injected whole already holds the line, so it is not repeated."""

        from finesub.llm.research import render_preinjected_entries

        root = _seed(tmp_path)
        text, report = render_preinjected_entries(root, "今天三月七出场", count_tokens=len)
        assert report["matches"] == []
        assert [m["line"] for m in report["term_matches"]] == ["三月七|三月七|March 7th|开朗少女。"]
        assert "[崩坏星穹铁道 / 角色] 三月七|三月七|March 7th" in text

        text, report = render_preinjected_entries(root, "崩坏星穹铁道的三月七", count_tokens=len)
        assert len(report["matches"]) == 1 and "term_matches" not in report
        assert text.count("March 7th") == 1


class TestTermOnlyHitsAreObservable:
    """A window can be reached by a sub-entry match with no entry-level match
    at all -- measured, 87% of term hits are exactly that. Gating the record on
    the entry-level matches means the prompt carried knowledge the report says
    nothing about, which is the opposite of the A/B observability the feature
    was justified with."""

    def test_a_term_only_hit_is_recorded(self, tmp_path) -> None:
        import json

        from finesub.llm.knowledge.base import TASK_ARTIFACT_FILENAME
        from finesub.llm.stages.prepare_task import _opening_inputs
        from finesub.llm.token_budget import default_token_counter

        artifacts = tmp_path / "task"
        _, text, report, _ = _opening_inputs(
            "今天三月七出场", local_search=False, knowledge_enabled=True, knowledge_root=_seed(tmp_path / "kb"),
            search_client=None, execution_settings=None, token_counter=default_token_counter(),
            task_artifact_dir=artifacts, task_id="t",
        )
        assert report["matches"] == [] and report["term_matches"]
        records = [json.loads(line) for line in (artifacts / TASK_ARTIFACT_FILENAME).read_text(
            encoding="utf-8").splitlines()]
        assert [r["kind"] for r in records] == ["knowledge_preinjection"]
        assert records[0]["payload"]["term_matches"] == report["term_matches"]

    def test_the_audit_hash_covers_the_terms(self, tmp_path) -> None:
        """A hash over entries alone reports "the knowledge input did not
        change" while the injected text did -- the one thing it exists to
        prevent. Neither text names an entry; only the term hit differs."""

        from finesub.llm.research import research_knowledge_inputs_hash

        root = _seed(tmp_path)
        assert research_knowledge_inputs_hash(root, "今天三月七出场") != research_knowledge_inputs_hash(
            root, "今天天气很好")
