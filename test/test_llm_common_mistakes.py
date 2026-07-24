from __future__ import annotations

import json
import subprocess

import pytest

from llm.knowledge.mistakes import (
    MistakeEntry,
    apply_mistake_proposals,
    common_mistakes_path,
    load_common_mistakes,
    parse_common_mistakes,
    render_common_mistakes,
    render_featured_mistakes_block,
)


def _add(source: str, wrong: str, correct: str, note: str = "备注") -> str:
    return json.dumps(
        {
            "op": "add_mistake",
            "source": source,
            "wrong": wrong,
            "correct": correct,
            "note": note,
            "prompt_version": "zh-subtitle-correction-csv-v5",
            "reason": "精修 SRT 对齐差异",
        },
        ensure_ascii=False,
    )


def _wrap(*lines: str) -> str:
    return "<mistake_proposals>\n" + "\n".join(lines) + "\n</mistake_proposals>"


def test_render_parse_round_trip() -> None:
    entries = [
        MistakeEntry(
            id="M0001",
            source="so the run is basically dead",
            wrong="所以这次跑步基本上完了",
            correct="所以这一把基本上寄了",
            note="游戏直播语境下 run 指单局",
            prompt_version="zh-subtitle-correction-csv-v4",
            month="2026-07",
        ),
        MistakeEntry(
            id="M0002",
            source="nice save",
            wrong="很好的存档",
            correct="救得漂亮",
            note="口语",
            prompt_version="zh-subtitle-correction-csv-v5",
            month="2026-07",
        ),
    ]

    text = render_common_mistakes(entries, ["M0002"])
    parsed_entries, featured = parse_common_mistakes(text)

    assert parsed_entries == entries
    assert featured == ["M0002"]


def test_apply_assigns_incrementing_ids_and_dedupes(tmp_path) -> None:
    report = apply_mistake_proposals(
        _wrap(_add("a", "错A", "对A"), _add("b", "错B", "对B")),
        knowledge_root=tmp_path,
        commit=False,
    )
    assert [record.status for record in report.applied] == ["applied", "applied"]

    # Duplicate (source, wrong) is skipped; a new pair still gets the next id.
    second = apply_mistake_proposals(
        _wrap(_add("a", "错A", "对A2"), _add("c", "错C", "对C")),
        knowledge_root=tmp_path,
        commit=False,
    )
    assert [record.status for record in second.skipped] == ["skipped"]
    assert "duplicate" in second.skipped[0].reason

    entries, featured = load_common_mistakes(tmp_path)
    assert [entry.id for entry in entries] == ["M0001", "M0002", "M0003"]
    assert entries[2].source == "c"
    assert featured == []


def test_set_featured_validates_ids_and_limit(tmp_path) -> None:
    from llm.knowledge.mistakes import MAX_FEATURED_MISTAKES

    seeded = MAX_FEATURED_MISTAKES + 2
    apply_mistake_proposals(
        _wrap(*(_add(f"s{i}", f"w{i}", f"c{i}") for i in range(seeded))),
        knowledge_root=tmp_path,
        commit=False,
    )

    unknown = apply_mistake_proposals(
        _wrap(json.dumps({"op": "set_featured", "ids": ["M9999"]})),
        knowledge_root=tmp_path,
        commit=False,
    )
    assert unknown.skipped and "unknown mistake id" in unknown.skipped[0].reason

    too_many = apply_mistake_proposals(
        _wrap(
            json.dumps(
                {
                    "op": "set_featured",
                    "ids": [f"M{i + 1:04d}" for i in range(MAX_FEATURED_MISTAKES + 1)],
                }
            )
        ),
        knowledge_root=tmp_path,
        commit=False,
    )
    assert too_many.skipped and "limit" in too_many.skipped[0].reason

    ok = apply_mistake_proposals(
        _wrap(json.dumps({"op": "set_featured", "ids": ["M0002", "M0005"]})),
        knowledge_root=tmp_path,
        commit=False,
    )
    assert ok.applied
    _entries, featured = load_common_mistakes(tmp_path)
    assert featured == ["M0002", "M0005"]


def test_add_example_writes_good_example_ledger_and_dedupes(tmp_path) -> None:
    from llm.knowledge.mistakes import load_good_examples

    example = json.dumps(
        {
            "op": "add_example",
            "source": "命に嫌われている",
            "translation": "被生命所厌恶。",
            "note": "歌名保留既有通行译法而非直译",
            "reason": "精修对照",
        },
        ensure_ascii=False,
    )
    report = apply_mistake_proposals(
        _wrap(example, _add("a", "错A", "对A")),
        knowledge_root=tmp_path,
        commit=False,
    )

    assert [record.op for record in report.applied] == ["add_example", "add_mistake"]
    examples = load_good_examples(tmp_path)
    assert [entry.id for entry in examples] == ["G0001"]
    assert examples[0].translation == "被生命所厌恶。"
    # Mistake ledger and example ledger are separate files.
    assert (tmp_path / "translation" / "good-example.md").exists()
    assert (tmp_path / "translation" / "common-mistake.md").exists()

    # Duplicate (source, translation) is skipped; missing translation is invalid.
    second = apply_mistake_proposals(
        _wrap(
            example,
            json.dumps({"op": "add_example", "source": "x", "note": "n", "reason": "r"}),
        ),
        knowledge_root=tmp_path,
        commit=False,
    )
    assert [record.status for record in second.skipped] == ["skipped", "skipped"]
    assert "duplicate" in second.skipped[0].reason
    assert "translation is empty" in second.skipped[1].reason
    assert [entry.id for entry in load_good_examples(tmp_path)] == ["G0001"]


def test_evidence_text_blocks_fabricated_wrong(tmp_path) -> None:
    # Every audited run contained add_mistake proposals whose `wrong` never
    # occurred in any output; with evidence_text the apply layer drops them.
    report = apply_mistake_proposals(
        _wrap(_add("so the run is dead", "这次跑步完了", "这把寄了")),
        knowledge_root=tmp_path,
        commit=False,
        evidence_text="sub|1|2.0|so the run is dead|这把基本上寄了|8|",
    )
    assert report.applied == []
    assert "not found in task outputs" in report.skipped[0].reason

    # The same proposal applies when wrong actually occurs (whitespace-insensitive).
    report = apply_mistake_proposals(
        _wrap(_add("so the run is dead", "这次跑步完了", "这把寄了")),
        knowledge_root=tmp_path,
        commit=False,
        evidence_text="sub|1|2.0|so the run is dead|这次 跑步 完了|8|",
    )
    assert [record.status for record in report.applied] == ["applied"]


def test_allow_featured_false_skips_set_featured_but_applies_adds(tmp_path) -> None:
    # The post-task knowledge update curates 精选 manually: a stray
    # set_featured is refused at the harness while add_mistake still lands.
    apply_mistake_proposals(_wrap(_add("a", "错A", "对A")), knowledge_root=tmp_path, commit=False)

    report = apply_mistake_proposals(
        _wrap(
            _add("b", "错B", "对B"),
            json.dumps({"op": "set_featured", "ids": ["M0001"]}),
        ),
        knowledge_root=tmp_path,
        commit=False,
        allow_featured=False,
    )

    assert [record.op for record in report.applied] == ["add_mistake"]
    assert report.skipped and "not allowed" in report.skipped[0].reason
    _entries, featured = load_common_mistakes(tmp_path)
    assert featured == []


def test_invalid_ops_and_empty_fields_are_skipped(tmp_path) -> None:
    report = apply_mistake_proposals(
        _wrap(
            json.dumps({"op": "delete_mistake", "ids": ["M0001"]}),
            json.dumps({"op": "add_mistake", "source": "", "wrong": "w", "correct": "c", "note": "n"}),
        ),
        knowledge_root=tmp_path,
        commit=False,
    )
    assert not report.applied
    assert len(report.skipped) == 2
    assert not common_mistakes_path(tmp_path).exists()


@pytest.mark.slow
def test_apply_commits_to_embedded_git_repo(tmp_path) -> None:
    report = apply_mistake_proposals(
        _wrap(_add("a", "错A", "对A")),
        knowledge_root=tmp_path,
        task_id="task-9",
    )

    assert report.committed
    assert (tmp_path / ".git").exists()
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True
    )
    assert "translation-ledger" in log.stdout
    assert "task-9" in log.stdout


def test_featured_block_renders_only_curated_entries(tmp_path) -> None:
    assert render_featured_mistakes_block(tmp_path) == ""

    apply_mistake_proposals(
        _wrap(
            _add("so the run is basically dead", "所以这次跑步基本上完了", "所以这一把基本上寄了"),
            _add("nice save", "很好的存档", "救得漂亮"),
            json.dumps({"op": "set_featured", "ids": ["M0001"]}),
        ),
        knowledge_root=tmp_path,
        commit=False,
    )

    block = render_featured_mistakes_block(tmp_path)
    assert "常见翻译错误对照" in block
    assert "所以这一把基本上寄了" in block
    assert "救得漂亮" not in block
