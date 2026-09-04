"""P9: what the ASR referee is told about the knowledge base, in three depths.

The layering is the point. `speech` must not import `llm`, and the referee
lives in `speech` -- so the selection happens in `pipeline.py`, the one module
that already knows both sides, and what crosses the boundary is a plain string.
No new dependency in either direction, and the recogniser never learns where
names come from.

The three depths answer different questions:

* ``off``   nothing. The default, because injecting into a second model's
            prompt changes what it hears and no measurement justifies more yet.
* ``terms`` every NAME and nothing else -- the strings a recogniser has to get
            right, without the prose a corrector reads.
* ``full``  what the correction layer would see.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub import pipeline, stages
from finesub.llm.knowledge import base
from finesub.speech.recognition import vad_asr_stage
from finesub.speech.verification import qwen_referee


class _FakeReferee:
    """Records what the stage does to it -- no model, no GPU."""

    def __init__(self, device: str) -> None:
        self.requested_device = device
        self.calls: list[tuple[str, object]] = []

    def set_context(self, context: str) -> None:
        self.calls.append(("set_context", context))

    def set_vram_budget(self, budget) -> None:
        self.calls.append(("set_vram_budget", budget))

    def close(self) -> None:
        self.calls.append(("close", None))


def _unbuildable(**kwargs):  # pragma: no cover
    raise AssertionError("reused a usable referee AND built a second one")


class TestTermLineTrimming:
    def test_the_description_column_is_dropped_by_position(self) -> None:
        """A term line is `源语言|中文定名|别名|一句话描述` and the description
        may itself contain pipes, so counting fields would truncate it in the
        middle instead of removing it."""

        line = "ルミ|露米|Lumi|霜精，会喊 ヤッホー|所以描述里也有竖线"

        assert base.strip_term_description(line) == "ルミ|露米|Lumi"

    def test_a_line_that_is_not_a_term_line_is_untouched(self) -> None:
        """This trims; it does not filter."""

        assert base.strip_term_description("- 普通一行") == "- 普通一行"
        assert base.strip_term_description("") == ""

    def test_names_survive_and_prose_does_not(self) -> None:
        body = "## 角色\nルミ|露米|Lumi|一句话描述\n这是一段散文\n## 空节\n"

        assert base.term_lines_only(body) == "## 角色\nルミ|露米|Lumi"

    def test_a_section_left_empty_is_dropped(self) -> None:
        """A heading with nothing under it is noise once the prose is gone."""

        assert base.term_lines_only("## 只有散文\n随便写点什么\n") == ""


class TestLevels:
    def test_off_is_the_default_and_reads_nothing(self, tmp_path) -> None:
        text, report = stages.build_asr_context(
            None, knowledge_root=tmp_path, text="ルミ"
        )

        assert text == ""
        assert report == {"level": "off"}

    def test_an_unknown_level_degrades_to_off(self, tmp_path) -> None:
        """A typo in a switch must not silently inject a different depth."""

        text, report = stages.build_asr_context(
            "verbose", knowledge_root=tmp_path, text="ルミ"
        )

        assert (text, report["level"]) == ("", "off")

    def test_empty_text_buys_nothing(self, tmp_path) -> None:
        assert stages.build_asr_context(
            "full", knowledge_root=tmp_path, text="   "
        )[0] == ""

    def test_an_unreadable_store_is_not_fatal(self, tmp_path) -> None:
        """Context is a nicety; a missing knowledge base must not take a
        transcription down."""

        text, _report = stages.build_asr_context(
            "terms", knowledge_root=tmp_path / "nope", text="ルミ"
        )

        assert text == ""

    def test_the_levels_are_exactly_three(self) -> None:
        assert pipeline.ASR_CONTEXT_LEVELS == ("off", "terms", "full")


class TestTheRefereeCarriesIt:
    def test_no_context_keeps_the_documented_request_shape(self) -> None:
        """A run that injects nothing must not start decoding through a
        hand-built conversation."""

        seen = {}

        class Processor:
            def apply_transcription_request(self, audio):
                seen["helper"] = True
                return {"input_ids": audio}

            def apply_chat_template(self, *a, **k):  # pragma: no cover
                raise AssertionError("built a conversation with no context")

        referee = qwen_referee.QwenReferee(context="")
        referee._transcription_inputs([[0.0, 1.0]], Processor())

        assert seen == {"helper": True}

    def test_context_goes_into_the_system_turn(self) -> None:
        seen = {}

        class Processor:
            def apply_transcription_request(self, audio):  # pragma: no cover
                raise AssertionError("dropped the context on the floor")

            def apply_chat_template(self, conversations, **kwargs):
                seen["conversations"] = conversations
                seen["kwargs"] = kwargs
                return {"input_ids": []}

        referee = qwen_referee.QwenReferee(context="- [原神] ルミ|露米|Lumi")
        referee._transcription_inputs([[0.0, 1.0]], Processor())

        system, user = seen["conversations"][0]
        assert system["role"] == "system"
        assert system["content"][0]["text"] == "- [原神] ルミ|露米|Lumi"
        assert user["role"] == "user"
        assert user["content"][0]["type"] == "audio"
        assert seen["kwargs"]["add_generation_prompt"] is True

    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_blank_context_is_the_same_as_none(self, value) -> None:
        assert qwen_referee.QwenReferee(context=value)._context == ""


class TestLayering:
    def test_the_recogniser_never_imports_the_llm_layer(self) -> None:
        """The rule this whole design exists to keep. If the stage ever grows
        an `llm` import, the string-passing above stops being necessary and
        someone will quietly delete it."""

        import inspect

        from finesub.speech.recognition import vad_asr_stage

        for module in (vad_asr_stage, qwen_referee):
            source = inspect.getsource(module)
            assert "from ...llm" not in source
            assert "import finesub.llm" not in source

    def test_the_stage_takes_the_text_not_a_knowledge_root(self) -> None:
        import inspect

        from finesub.speech.recognition import vad_asr_stage

        signature = inspect.signature(vad_asr_stage.run_vad_asr)
        assert "asr_context" in signature.parameters
        assert "knowledge_root" not in signature.parameters


class TestTheReuseDoesNotDropIt:
    """The stage reuses the inline language referee for the verification pass
    to avoid a second model load. That instance is built context-free ON
    PURPOSE -- `lang_redecode` and `lang_audit` read its language field, and a
    list of Japanese names would bias exactly the quantity the audit exists to
    cross-check independently. So the context has to be attached at reuse, and
    the default `--lang-redecode auto` makes that the COMMON path: without it,
    `--asr-context terms` silently verifies with nothing injected."""

    def test_setting_it_afterwards_works(self) -> None:
        referee = qwen_referee.QwenReferee(context="")

        referee.set_context("- [原神] ルミ|露米|Lumi")

        assert referee._context == "- [原神] ルミ|露米|Lumi"

    def test_it_can_be_cleared_again(self) -> None:
        referee = qwen_referee.QwenReferee(context="something")

        referee.set_context("")

        assert referee._context == ""

    def test_reuse_attaches_the_context(self) -> None:
        inline = _FakeReferee("cuda")

        chosen = vad_asr_stage.resolve_verification_referee(
            inline,
            device="cuda",
            asr_context="- [原神] ルミ",
            build=_unbuildable,
            vram_budget_gib=6.5,
        )

        assert chosen is inline
        # The context lands, and so does the whole-tier VRAM budget: the
        # inline referee was told the spare-beside-Whisper figure.
        assert inline.calls == [
            ("set_context", "- [原神] ルミ"),
            ("set_vram_budget", 6.5),
        ]

    def test_a_device_mismatch_closes_the_inline_one_and_builds_with_context(
        self,
    ) -> None:
        """The 4GB path: a CPU inline referee is dropped for the CUDA one once
        Whisper is gone. The replacement has to carry the context too, and the
        one being replaced has to be released -- it may hold GiBs."""

        inline = _FakeReferee("cpu")
        built = {}

        def build(**kwargs):
            built.update(kwargs)
            return _FakeReferee(kwargs["device"])

        chosen = vad_asr_stage.resolve_verification_referee(
            inline, device="cuda", asr_context="ルミ", build=build, vram_budget_gib=3.0
        )

        assert chosen is not inline
        assert inline.calls == [("close", None)]
        assert built == {"device": "cuda", "context": "ルミ", "vram_budget_gib": 3.0}

    def test_with_no_inline_referee_it_just_builds_one(self) -> None:
        built = {}

        vad_asr_stage.resolve_verification_referee(
            None,
            device="cpu",
            asr_context="",
            build=lambda **kw: built.update(kw) or _FakeReferee("cpu"),
        )

        assert built == {"device": "cpu", "context": "", "vram_budget_gib": None}

    def test_the_stage_routes_through_the_helper(self) -> None:
        """The behaviour above is only the production behaviour while the
        stage still calls it -- and an inlined copy would pass every test
        above while dropping the context again."""

        import inspect

        source = inspect.getsource(vad_asr_stage.run_vad_asr)
        assert "resolve_verification_referee(" in source

    def test_the_language_referee_itself_stays_context_free(self) -> None:
        """The other half of the same rule: biasing the language vote with a
        term list would undo the audit's whole premise."""

        import inspect

        from finesub.speech.recognition import vad_asr_stage

        source = inspect.getsource(vad_asr_stage.run_vad_asr)
        build = source[source.index("redecode_referee = redecode_qwen.QwenReferee") :][:200]
        assert "context=" not in build


class TestTheTitleIsFetchedWhenTheContextNeedsIt:
    """`--asr-context` reads extra-info at `aligned`, four stages before
    `run_full_correction` does. A gate that only knows about the correction
    layer leaves the default `finesub URL --asr-context terms` looking at a URL
    and a file path, with the title it is meant to match subjects from never
    fetched."""

    def test_the_context_switch_opens_the_gate_from_aligned_on(self) -> None:
        assert not stages.stage_consumes_extra_info("raw-srt")
        assert stages.stage_consumes_extra_info("raw-srt", asr_context="terms")
        assert stages.stage_consumes_extra_info("aligned", asr_context="full")

    def test_a_vocal_only_run_still_stays_offline(self) -> None:
        """`--stage vocal` stops before the referee exists, so the switch buys
        nothing and the title probe would be paid for a consumer that never
        runs -- exactly the waste the stage half of the gate exists to avoid."""

        assert not stages.stage_consumes_extra_info("vocal", asr_context="terms")
        assert not stages.stage_consumes_extra_info("vocal", asr_context="full")

    def test_off_and_absent_leave_the_gate_where_it_was(self) -> None:
        for value in (None, "", "off", "OFF"):
            assert not stages.stage_consumes_extra_info("raw-srt", asr_context=value)

    def test_the_composed_block_carries_the_title(self, monkeypatch) -> None:
        monkeypatch.setattr(stages, "source_title_line", lambda url: "视频标题: X")

        block = stages.compose_url_extra_info(
            "https://x/1", "媒体文件: a.mp4", "", stage="raw-srt", asr_context="terms"
        )

        assert "视频标题: X" in block


class TestEveryFrontEndCarriesIt:
    """A front end that accepts the flag but never forwards it is worse than
    one that rejects it: the run looks configured and decodes without context.

    Since the multi-input refactor there is one entry point and the row
    whitelist is derived from `run_pipeline`'s signature, so these are
    regression tests for the derivation rather than for four hand-copied
    lists -- which is exactly why the hand-copied version is gone.
    """

    def test_the_manifest_accepts_it(self) -> None:
        merged = pipeline.merge_item_options(
            {"source": "https://x/1", "asr_context": "terms"}, {"stage": "raw-srt"}
        )

        assert merged["asr_context"] == "terms"

    def test_the_cli_carries_it_and_defers_the_default(self, monkeypatch) -> None:
        """`None`, not `"off"`: the resolved default lives once, in
        `build_asr_context`, and every front end asks it the same question."""

        import sys

        monkeypatch.setattr(sys, "argv", ["finesub", "a.wav"])
        args = pipeline.parse_args()

        assert args.asr_context is None
        # Absence, not `None`: the runner passes through only what was
        # given, so an unset switch never becomes a second default.
        assert "asr_context" not in pipeline._defaults_from_args(args, single=True)

    def test_the_run_actually_receives_it(self, tmp_path, monkeypatch) -> None:
        source = tmp_path / "input.wav"
        source.write_bytes(b"fake")
        calls: list[dict] = []
        monkeypatch.setattr(
            pipeline, "run_pipeline", lambda *a, **kw: calls.append(kw)
        )

        item = pipeline.build_item(
            {"source": str(source), "stage": "raw-srt", "asr_context": "full"}
        )
        item.stages["asr"](item.payload)

        assert calls[0]["asr_context"] == "full"
