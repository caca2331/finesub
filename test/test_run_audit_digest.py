"""`agent-tasks/run-audit/scripts/extract_digest.py` against production.

The script is stdlib-only on purpose (it must not depend on the code it
audits), so its schema constants are copies. They drifted unseen for weeks: an
injection cap of 4000 against the presets' 6000, row types that kept the retired
`insert` and missed `discard`, knowledge ops that no longer exist. Every FLAG
built on a stale copy is noise the auditing agent has to argue its way past.
This compares each copy with its source.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from finesub.llm import output_protocol
from finesub.llm.knowledge.base import KNOWLEDGE_CATEGORIES
from finesub.llm.knowledge.feedback import KNOWLEDGE_HINT_DIRECTIONS
from finesub.llm.knowledge.node import presets
from finesub.llm.knowledge.node.proposals import MODEL_OPS


SCRIPT = Path(__file__).resolve().parents[1] / "agent-tasks" / "run-audit" / "scripts" / "extract_digest.py"


@pytest.fixture(scope="module")
def digest():
    spec = importlib.util.spec_from_file_location("run_audit_digest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_audit_digest"] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop("run_audit_digest", None)


def test_the_script_imports_only_the_standard_library() -> None:
    # `zhconv` is an optional, guarded import for one check.
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        name.split(".", 1)[0]
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names] if isinstance(node, ast.Import)
            else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        )
    }
    assert imported - set(sys.stdlib_module_names) == {"zhconv"}


def test_the_schema_copies_match_production(digest) -> None:
    assert digest.LEGAL_CATEGORIES == set(KNOWLEDGE_CATEGORIES)
    assert digest.PROPOSAL_CATEGORIES == set(presets.CATEGORIES)
    assert digest.LEGAL_DIRECTIONS == set(KNOWLEDGE_HINT_DIRECTIONS)
    assert digest.LEGAL_KNOWLEDGE_OPS == set(MODEL_OPS)
    assert set(digest.KNOWLEDGE_REQUIRED_FIELDS) == set(MODEL_OPS)
    assert digest.LEGAL_ROW_TYPES == {"", output_protocol.KIND_SUB, output_protocol.KIND_DISCARD}
    assert {
        presets.preset_for_category(category).max_entry_tokens for category in presets.CATEGORIES
    } == {digest.KB_ENTRY_TOKEN_LIMIT}
    streamer = presets.preset_for_category("streamer")
    assert digest.STREAMER_SECTIONS == tuple(section.name for section in streamer.sections)


def test_discard_rows_are_counted_and_insert_rows_flagged(digest, tmp_path: Path) -> None:
    content = (
        "<translated>\n"
        "type|position|corrected_text|translated_text|conf|note\n"
        "|1|こんにちは|你好|high|\n"
        "discard|2|复读幻觉\n"
        "insert|3|あ|啊|low|\n"
        "</translated>"
    )
    windows = tmp_path / "windows.jsonl"
    windows.write_text(json.dumps({"chunk_id": "w1", "content": content}) + "\n", encoding="utf-8")
    digest.FLAGS.clear()
    out: list[str] = []

    digest.audit_windows(windows, out)

    assert "  - w1: sub=1 discard=1" in out
    assert digest.FLAGS == ["窗口 w1 第 5 行 type 非法 'insert'"]
