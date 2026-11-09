"""Prompt prose lives in `prompt_templates/`, not in the Python that renders it.

`docs/llm_prompts.md` and `AGENTS.md` both say prompt text is never hardcoded
in Python -- templates, `mode_texts_v1.toml` and `phrases_v1.toml` hold it, so
iterating a prompt is a template edit and a cold reader who follows the docs to
the templates sees everything the model sees. `docs/architecture.md` marked the
rule ✱ ("a test pins it") for two years before any test did; the few dozen
sentences the renderers held by then moved into the templates on 2026-11-08, and
the guard now allows none outside the parameter layer.

Scan surface, stated because a guard does not see what it does not scan:

- **Files**: every module under `src/finesub/llm/` whose name contains
  `prompt` (`prompts.py`, `prompt_compose.py`, `prompt_variants.py`,
  `prepare_prompts.py`, `request_prompts.py`, `stages/*/prompt.py`, ...), plus
  `shape_criteria.py` -- the modules whose job is rendering prompts. Model-facing
  text elsewhere (validator feedback, the agent drivers' English task framing,
  the search agent's instructions) is NOT scanned: there the same heuristic
  cannot tell a prompt sentence from a user-facing warning.
- **Strings**: string literals and the literal parts of f-strings with at least
  `MIN_CJK_CHARS` CJK characters -- a clause, not a short label. Docstrings are
  skipped, and so is a literal that goes into a `raise`, an exception, a
  reporter call (`warning`, `info`, ...) or an argparse `help=`: those are for
  the user, not the model.

- **Parameter layer** (`_PARAMETER_LAYER`): the module-level dicts whose values
  fill one template `$slot` each and are picked by key (media mode, line kind)
  are a sanctioned exception, not debt (owner 2026-11-07; the boundary is in
  `docs/llm_prompts.md`'s rule on hardcoded text). They are named one by one,
  so a new dict is scanned until it is added here deliberately.

A failure names each literal by `path:scope` (the enclosing function's qualname,
or the name a module-level literal is assigned to) and line.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

LLM_ROOT = Path(__file__).resolve().parents[1] / "src" / "finesub" / "llm"

#: A clause rather than a label: the parameter layer's short slot values (a
#: media name, a column heading) stay under it.
MIN_CJK_CHARS = 12
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")
#: Calls whose string arguments are for the user (reporter methods, argparse).
_USER_FACING_CALLS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "print", "add_argument", "add_parser"}
)

#: The parameter layer: one slot value per key, chosen by Python, framed by a template. Not scanned.
_PARAMETER_LAYER = frozenset(
    {
        "prompt_compose.py:_MEDIA_PARAMS",
        "prompt_compose.py:_TEXT_MEDIA_PARAMS",
        "prompt_compose.py:_VIDEO_PARAM_OVERRIDES",
        "prompt_compose.py:_INPUT_INVENTORY_MEDIA",
        "stages/speaker_judge/prompt.py:PHRASES",
    }
)


def _scanned_modules() -> list[Path]:
    return sorted(
        path
        for path in LLM_ROOT.rglob("*.py")
        if "prompt" in path.name or path.name == "shape_criteria.py"
    )


def _callee(call: ast.Call) -> str:
    function = call.func
    return function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")


def _is_docstring(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    statement = parents.get(node)
    owner = parents.get(statement)
    return (
        isinstance(statement, ast.Expr)
        and isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and owner.body[0] is statement
    )


def _is_for_the_user(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents:
        parent = parents[current]
        if isinstance(parent, ast.Raise):
            return True
        if isinstance(parent, ast.keyword) and parent.arg == "help":
            return True
        if isinstance(parent, ast.Call) and current is not parent.func:
            name = _callee(parent)
            if name in _USER_FACING_CALLS or name.endswith(("Error", "Exception", "Warning")):
                return True
        current = parent
    return False


def _scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    names: list[str] = []
    assigned: str | None = None
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(current.name)
        elif not names and assigned is None and isinstance(current, (ast.Assign, ast.AnnAssign)):
            target = current.targets[0] if isinstance(current, ast.Assign) else current.target
            assigned = getattr(target, "id", None)
    return ".".join(reversed(names)) or assigned or "<module>"


def prompt_literals(source: str) -> list[tuple[str, int]]:
    """`(scope, line)` of every literal in `source` that reads as prompt prose."""

    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            text = "".join(part.value for part in node.values if isinstance(part, ast.Constant))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if isinstance(parents.get(node), ast.JoinedStr):
                continue  # counted as part of its f-string
            text = node.value
        else:
            continue
        if len(_CJK.findall(text)) < MIN_CJK_CHARS:
            continue
        if _is_docstring(node, parents) or _is_for_the_user(node, parents):
            continue
        found.append((_scope(node, parents), node.lineno))
    return found


def _counts() -> tuple[dict[str, int], dict[str, list[int]]]:
    counts: dict[str, int] = {}
    lines: dict[str, list[int]] = {}
    for path in _scanned_modules():
        relative = path.relative_to(LLM_ROOT).as_posix()
        for scope, line in prompt_literals(path.read_text(encoding="utf-8")):
            key = f"{relative}:{scope}"
            if key in _PARAMETER_LAYER:
                continue
            counts[key] = counts.get(key, 0) + 1
            lines.setdefault(key, []).append(line)
    return counts, lines


def test_no_prompt_prose_is_hardcoded() -> None:
    counts, lines = _counts()
    assert counts == {}, (
        "prompt prose in Python -- put it in src/finesub/llm/prompt_templates/ "
        "(a fragment, mode_texts_v1.toml or phrases_v1.toml) and fill it in: "
        + ", ".join(f"{key} (lines {lines[key]})" for key in sorted(counts))
    )


def test_the_scan_surface_still_holds_the_prompt_renderers() -> None:
    """A rename that took a renderer out of the glob would make it unguarded."""

    names = {path.relative_to(LLM_ROOT).as_posix() for path in _scanned_modules()}
    assert {"prompts.py", "prompt_compose.py", "prompt_variants.py", "shape_criteria.py"} <= names


def test_the_parameter_layer_names_live_dicts() -> None:
    """An entry whose dict was renamed or removed would exempt nothing and read as coverage."""

    scopes = set()
    for path in _scanned_modules():
        relative = path.relative_to(LLM_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
            if isinstance(getattr(node, "value", None), ast.Dict) and isinstance(target, ast.Name):
                scopes.add(f"{relative}:{target.id}")
    assert sorted(_PARAMETER_LAYER - scopes) == []


def test_the_check_tells_prompt_prose_from_user_messages() -> None:
    """A guard nobody has seen fail is a guard nobody should trust."""

    source = '''
"""模块说明也是中文，而且足够长，但它是文档字符串。"""
INSTRUCTION = "请逐行核对字幕与音频，发现听错的专有名词时改成知识库里的定名。"
SHORT = "音频与画面"

def render(n):
    note = f"本次调用是第 {n} 块，只含这一段窗口；其余块会在独立调用中处理。"
    if n < 0:
        raise ValueError("块号不能是负数，这说明调用方把切块的结果算错了。")
    reporter.warning("topic", "这一块没有可用的模型，按纯文本继续处理这一段。")
    return note
'''
    assert prompt_literals(source) == [("INSTRUCTION", 3), ("render", 7)]
