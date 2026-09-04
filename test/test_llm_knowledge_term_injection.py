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

import pytest

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


class TestItIsActuallyWired:
    def test_the_render_path_calls_it_and_dedupes(self) -> None:
        """Shadow-only was the previous state and is indistinguishable from
        this one by any other test: the events get booked either way, so the
        only thing that says "it now drives injection" is the call site."""

        import inspect

        from finesub.llm import research

        source = inspect.getsource(research.render_preinjected_entries)
        assert "match_terms(" in source
        assert "exclude_subjects" in source, (
            "an entry injected whole already contains its own term lines"
        )
        assert "render_term_matches(" in source


class TestTermOnlyHitsAreObservable:
    """A window can be reached by a sub-entry match with no entry-level match
    at all -- measured, 87% of term hits are exactly that. Gating the record on
    the entry-level matches means the prompt carried knowledge the report says
    nothing about, which is the opposite of the A/B observability the feature
    was justified with."""

    def test_the_report_gate_accepts_term_only(self) -> None:
        import inspect

        from finesub.llm import research
        from finesub.llm.stages import fast_session

        for module, function in (
            (research, "run_research"),
            (fast_session, "run_fast_session"),
        ):
            source = inspect.getsource(module)
            assert 'preinjection_report.get("term_matches")' in source, (
                f"{module.__name__} still gates the record on entry hits alone"
            )

    def test_the_audit_hash_covers_the_terms(self) -> None:
        """A hash over entries alone reports "the knowledge input did not
        change" while the injected text did -- the one thing it exists to
        prevent."""

        import inspect

        from finesub.llm import research

        source = inspect.getsource(research.research_knowledge_inputs_hash)
        assert "match_terms(" in source
        assert "preinjected_terms" in source
