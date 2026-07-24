"""Deterministic digest extractor for run-audit.

Scans one finished run directory (pipeline / correction / reference_ingest)
and prints a markdown digest: layout, stats, and machine-detected FLAGS that
the auditing agent should investigate by opening the referenced files.

Stdlib only, on purpose: the audit tool must not depend on the project code
it is auditing.

Usage:
  python extract_digest.py <run-dir> [--refined <refined.srt>] [--json]
  python extract_digest.py --kb <knowledge-root>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# --- schema constants (mirror docs/knowledge.md; update together) -----------
LEGAL_CATEGORIES = {"streamer", "common"}
LEGAL_DIRECTIONS = {"new_entry", "replace_section", "append_lines"}
LEGAL_KNOWLEDGE_OPS = {"append_lines", "edit_lines", "replace_section", "create_entry"}
# set_featured is harness/manual only; add_example has its own field schema.
LEGAL_MISTAKE_OPS = {"add_mistake", "add_example"}
MISTAKE_REQUIRED_FIELDS = {
    "add_mistake": ("source", "wrong", "correct", "note"),
    "add_example": ("source", "translation", "note"),
}
# Top-level ``content`` is required for append/replace/create; edit_lines uses
# ``edits[].content`` instead (see knowledge/base.py KnowledgeProposal).
KNOWLEDGE_REQUIRED_FIELDS = {
    "append_lines": ("category", "entry", "op", "section", "content", "reason"),
    "replace_section": ("category", "entry", "op", "section", "content", "reason"),
    "create_entry": ("category", "entry", "op", "intro", "reason"),
    "edit_lines": ("category", "entry", "op", "edits", "reason"),
}
LEGAL_ROW_TYPES = {"", "sub", "insert"}
CSV_HEADER_TYPE = "type"

# Signals of systematic failure modes (references/failure-modes.md)
FREQUENCY_ADVERBS = re.compile(r"经常|常在|常常|倾向于|总是|一贯|习惯于|通常")
META_DISCOURSE = re.compile(r"（注|注：|谐音|意为|疑似|应为|直译|误听")
PLACEHOLDER_VALUE = re.compile(r"^[（(].*[)）]$")
KANA = re.compile(r"[぀-ヿ]")

LONG_SUB_SECONDS = 10.0
LONG_GAP_SECONDS = 8.0
KB_ENTRY_TOKEN_LIMIT = 4000  # kb_entries per-entry injection cap

TAG_RE = {
    name: re.compile(rf"<{name}\b[^>]*>(.*?)</{name}>", re.IGNORECASE | re.DOTALL)
    for name in (
        "reasoning",
        "translated",
        "knowledge_proposals",
        "mistake_proposals",
        "task_update_feedback",
    )
}

FLAGS: list[str] = []


def flag(text: str) -> None:
    FLAGS.append(text)


# --- SRT parsing -------------------------------------------------------------
SRT_TIME_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)


def parse_srt(path: Path) -> list[dict]:
    entries = []
    block: list[str] = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines() + [""]:
        if line.strip():
            block.append(line)
            continue
        if block:
            for i, bline in enumerate(block):
                match = SRT_TIME_RE.search(bline)
                if match:
                    g = [int(x) for x in match.groups()]
                    start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                    end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                    entries.append(
                        {
                            "start": start,
                            "end": end,
                            "text": "\n".join(block[i + 1 :]).strip(),
                        }
                    )
                    break
            block = []
    return entries


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# --- run layout --------------------------------------------------------------
def resolve_layout(run_dir: Path) -> dict:
    layout: dict = {"run_dir": run_dir}
    dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.endswith("llm-artifacts")]
    layout["artifact_dir"] = dirs[0] if dirs else None
    srts = sorted(run_dir.glob("*.srt"))
    layout["raw_srt"] = next((p for p in srts if p.stem.endswith("-raw")), None)
    aux_suffixes = ("-raw", "-translated", "-corrected")
    finals = [p for p in srts if not p.stem.endswith(aux_suffixes)]
    layout["final_srt"] = finals[0] if finals else None
    layout["annotated_csv"] = next(iter(run_dir.glob("*-annotated.csv")), None)
    layout["stable_json"] = next(iter(run_dir.glob("*-stable.json")), None)
    return layout


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            flag(f"{path.name} 第 {no} 行不是合法 JSON: {exc}")
    return records


# --- feedback audit ----------------------------------------------------------
def audit_feedback(records: list[dict], out: list[str]) -> None:
    blocks = [
        (r["kind"], r.get("payload") or {})
        for r in records
        if r.get("kind") in ("correction_window_task_feedback", "research_task_feedback")
    ]
    out.append(f"- feedback 块: {len(blocks)} 个")
    for kind, payload in blocks:
        where = f"{kind}(chunk {payload.get('chunk_id', '-')})"
        body = str(payload.get("feedback") or "").strip()
        if not body:
            out.append(f"  - {where}: 空")
            continue
        try:
            data = json.loads(re.sub(r"^```\w*\n|\n```$", "", body))
        except json.JSONDecodeError:
            flag(f"feedback JSON 解析失败: {where}")
            continue
        hints = data.get("knowledge_hints") or []
        corrections = [str(c) for c in (data.get("asr_corrections") or [])]
        for key in ("knowledge_hints", "asr_corrections", "uncertainties"):
            if key not in data:
                flag(f"feedback 缺字段 {key}: {where}")
        for hint in hints:
            if not isinstance(hint, dict):
                flag(f"hint 不是对象: {where}")
                continue
            cat, entry = str(hint.get("category", "")), str(hint.get("entry", ""))
            if cat not in LEGAL_CATEGORIES:
                flag(f"hint category 非法 {cat!r} (entry {entry!r}): {where}")
            direction = str(hint.get("direction", ""))
            if direction and direction not in LEGAL_DIRECTIONS:
                flag(f"hint direction 非法 {direction!r}: {where}")
            conf = hint.get("confidence")
            if conf is not None and not (isinstance(conf, int) and 1 <= conf <= 9):
                flag(f"hint confidence 越界 {conf!r}: {where}")
            if entry and any(entry in c for c in corrections):
                flag(f"hint 与 asr_corrections 重复填报 {entry!r}: {where}")
        out.append(
            f"  - {where}: hints={len(hints)} asr_corrections={len(corrections)} "
            f"uncertainties={len(data.get('uncertainties') or [])}"
        )


# --- knowledge update audit ---------------------------------------------------
def audit_knowledge_updates(records: list[dict], out: list[str]) -> None:
    responses = [r.get("payload") or {} for r in records if r.get("kind") == "knowledge_update_response"]
    out.append(f"- knowledge_update_response: {len(responses)} 个 chunk")
    for payload in responses:
        where = f"knowledge-update chunk {payload.get('chunk', '?')}"
        content = str(payload.get("response_content") or "")
        if TAG_RE["reasoning"].search(content):
            out.append(f"  - {where}: 含 <reasoning> 块（v17 起必须，注意长度）")
        for name in ("knowledge_proposals", "mistake_proposals"):
            match = TAG_RE[name].search(content)
            if not match:
                continue
            lines = [l.strip() for l in match.group(1).splitlines() if l.strip().startswith("{")]
            out.append(f"  - {where}: {name} {len(lines)} 条")
            for no, line in enumerate(lines, 1):
                ref = f"{where} {name}#{no}"
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    flag(f"proposal 不是合法 JSON: {ref}")
                    continue
                op = str(p.get("op", ""))
                if name == "knowledge_proposals":
                    if str(p.get("category", "")) not in LEGAL_CATEGORIES:
                        flag(f"proposal category 非法 {p.get('category')!r}: {ref}")
                    if op not in LEGAL_KNOWLEDGE_OPS:
                        flag(f"proposal op 非法 {op!r}: {ref}")
                        continue
                    content_blob = str(p.get("content", ""))
                    if op == "edit_lines":
                        edits = p.get("edits") or []
                        if isinstance(edits, list):
                            content_blob = "\n".join(
                                str(e.get("content", "")) for e in edits if isinstance(e, dict)
                            )
                    hit = FREQUENCY_ADVERBS.search(content_blob)
                    if hit:
                        flag(f"proposal content 含频率词「{hit.group(0)}」——单次证据常态化嫌疑: {ref}")
                    required = KNOWLEDGE_REQUIRED_FIELDS.get(
                        op, ("category", "entry", "op", "reason")
                    )
                else:
                    if op == "set_featured":
                        flag(f"模型输出了 set_featured（精选归人工管）: {ref}")
                        continue
                    if op not in LEGAL_MISTAKE_OPS:
                        flag(f"mistake op 非法 {op!r}: {ref}")
                        continue
                    required = MISTAKE_REQUIRED_FIELDS[op]
                    if op == "add_mistake":
                        if p.get("source") and p.get("source") == p.get("wrong"):
                            flag(f"mistake wrong==source（疑为 ASR 问题而非翻译错误）: {ref}")
                        if KANA.search(str(p.get("correct", ""))):
                            flag(f"mistake correct 含假名（应为简体中文译文）: {ref}")
                    text_fields = ("wrong", "correct") if op == "add_mistake" else ("translation",)
                    for field in text_fields:
                        value = str(p.get(field, ""))
                        if PLACEHOLDER_VALUE.match(value):
                            flag(f"mistake {field} 是占位符/操作说明 {value!r}: {ref}")
                for field in required:
                    value = p.get(field, "")
                    if field == "edits":
                        if not (isinstance(value, list) and value):
                            flag(f"proposal 缺字段 {field}: {ref}")
                        continue
                    if not str(value).strip():
                        flag(f"proposal 缺字段 {field}: {ref}")


# --- correction window audit ---------------------------------------------------
def audit_windows(path: Path, out: list[str]) -> None:
    windows = load_jsonl(path)
    out.append(f"- 纠错窗口（已提交）: {len(windows)} 个")
    for w in windows:
        cid = w.get("chunk_id", "?")
        content = str(w.get("content") or "")
        has_reasoning = bool(TAG_RE["reasoning"].search(content))
        translated = TAG_RE["translated"].search(content)
        subs = inserts = 0
        if translated:
            for no, line in enumerate(translated.group(1).splitlines(), 1):
                if line.count("|") < 5:
                    continue
                cols = line.split("|")
                rtype = cols[0].strip()
                # Skip the canonical CSV header row (type|position|...).
                if rtype.lower() == CSV_HEADER_TYPE:
                    continue
                if rtype not in LEGAL_ROW_TYPES:
                    flag(f"窗口 {cid} 第 {no} 行 type 非法 {rtype!r}")
                    continue
                subs += rtype in ("", "sub")
                inserts += rtype == "insert"
                # translation column is index 3 without start, 4 with start —
                # scan both plausible text columns for meta discourse.
                for col in cols[3:6]:
                    hit = META_DISCOURSE.search(col)
                    if hit:
                        flag(
                            f"窗口 {cid} 字幕文本列含元话语「{hit.group(0)}»: {col[:40]!r}"
                        )
                        break
        else:
            flag(f"窗口 {cid} 提交内容里没有 <translated> 块")
        out.append(
            f"  - {cid}: sub={subs} insert={inserts}"
            + (" +reasoning" if has_reasoning else "")
        )

# --- SRT stats -----------------------------------------------------------------
CJK_RE = re.compile(r"[一-鿿]")


def _traditional_ratio(text: str) -> float | None:
    """Share of CJK chars that change under 简体归一; None if zhconv missing."""

    try:
        from zhconv import convert
    except ImportError:
        return None
    cjk = CJK_RE.findall(text)
    if not cjk:
        return 0.0
    changed = sum(1 for ch in cjk if convert(ch, "zh-hans") != ch)
    return changed / len(cjk)


def audit_language_setting(layout: dict, out: list[str]) -> None:
    stable = layout.get("stable_json")
    if not stable:
        return
    try:
        metadata = json.loads(Path(stable).read_text(encoding="utf-8")).get("metadata", {})
        language = (metadata.get("asr_align") or {}).get("language")
    except Exception:
        return
    if language:
        out.append(f"- ASR language 设置: {language}（核对是否符合素材语言，错设会让全部 ASR 证据失真）")


def audit_srt(layout: dict, refined_path: Path | None, out: list[str]) -> None:
    raw = parse_srt(layout["raw_srt"]) if layout.get("raw_srt") else []
    final = parse_srt(layout["final_srt"]) if layout.get("final_srt") else []
    if raw and final:
        ratio = 100 * (1 - len(final) / len(raw))
        out.append(f"- SRT: raw {len(raw)} 条 -> final {len(final)} 条（压缩 {ratio:.0f}%）")
        if ratio > 70:
            flag(
                f"压缩率 {ratio:.0f}% 偏高（嫌疑，不能代替质量判定）——"
                "对照精修/merge gold；见 docs/tools/prompt-iterate.md §2/§4"
            )
    elif final:
        out.append(f"- SRT: final {len(final)} 条（未找到 raw）")

    long_subs = [e for e in final if e["end"] - e["start"] > LONG_SUB_SECONDS]
    for e in sorted(long_subs, key=lambda e: e["start"] - e["end"])[:5]:
        flag(
            f"单条字幕跨度 {e['end'] - e['start']:.1f}s @{fmt_ts(e['start'])}: "
            f"{e['text'][:30]!r}"
        )
    gaps = []
    for a, b in zip(final, final[1:]):
        if b["start"] - a["end"] > LONG_GAP_SECONDS:
            gaps.append((a["end"], b["start"]))
    for start, end in gaps[:5]:
        note = ""
        if refined_path:
            refined = parse_srt(refined_path)
            inside = [e for e in refined if start <= e["start"] < end]
            if len(inside) >= 2:
                note = f"——精修版同区间有 {len(inside)} 条台词，疑似真实台词被连坐删除"
                flag(f"空档 {fmt_ts(start)}–{fmt_ts(end)}{note}")
        out.append(f"  - 空档 {fmt_ts(start)}–{fmt_ts(end)}（{end - start:.0f}s）{note}")
    if final:
        ratio = _traditional_ratio("\n".join(e["text"] for e in final))
        if ratio is None:
            out.append("- 繁体检测: 跳过（zhconv 未安装）")
        elif ratio > 0.01:
            flag(f"成品字幕疑似繁体：{ratio:.0%} 的汉字在简体归一下发生变化（要求简体）")
    if refined_path:
        refined = parse_srt(refined_path)
        out.append(f"- 精修 SRT: {len(refined)} 条（对照 final {len(final)} 条）")
        if refined and final:
            refined_end, final_end = refined[-1]["end"], final[-1]["end"]
            if final_end < refined_end * 0.8:
                flag(
                    f"覆盖范围可疑：成品止于 {fmt_ts(final_end)}，精修延伸到 "
                    f"{fmt_ts(refined_end)}——检查媒体/音频是否被截断或语言设置错误"
                )


# --- correction / harness timeline --------------------------------------------
def audit_correction_timeline(records: list[dict], out: list[str]) -> None:
    """Surface attempt chains, validation errors, and multi-final races."""

    responses = [r for r in records if r.get("kind") == "correction_window_response"]
    retries = [r for r in records if r.get("kind") == "correction_window_retry"]
    finals = [r for r in records if r.get("kind") == "final_srt"]
    if not (responses or retries or finals):
        return

    out.append("- correction 时间线（attempt / validation；并发双跑时会出现多个 attempt=0）:")
    attempt0_by_chunk: dict[str, list[str]] = {}
    for r in responses:
        pl = r.get("payload") or {}
        chunk = str(pl.get("chunk_id", "?"))
        attempt = pl.get("attempt")
        ok = pl.get("validation_ok")
        errs = pl.get("validation_errors") or []
        err0 = errs[0] if isinstance(errs, list) and errs else ""
        if isinstance(err0, dict):
            err0 = err0.get("message") or err0.get("error") or str(err0)
        err0 = str(err0).replace("\n", " ")
        if len(err0) > 100:
            err0 = err0[:97] + "..."
        ts = str(r.get("created_at") or "")
        extra = f" | {err0}" if err0 and ok is False else ""
        out.append(
            f"  - {ts} chunk={chunk} attempt={attempt} ok={ok}{extra}"
        )
        if attempt == 0:
            attempt0_by_chunk.setdefault(chunk, []).append(ts)

    for r in retries[:12]:
        pl = r.get("payload") or {}
        out.append(
            f"  - retry {r.get('created_at', '')} chunk={pl.get('chunk_id', '?')} "
            f"attempt={pl.get('attempt')} reason={pl.get('reason', '')}"
        )
    if len(retries) > 12:
        out.append(f"  - … +{len(retries) - 12} more retries")

    if len(finals) >= 2:
        times = [str(r.get("created_at") or "") for r in finals]
        flag(
            f"final_srt×{len(finals)}（{', '.join(times)}）——同目录多次收尾，"
            "磁盘成品通常是最后一次；核对是否并发双跑"
        )
        out.append(f"- final_srt artifacts: {len(finals)} 次 @ {', '.join(times)}")
    elif len(finals) == 1:
        out.append(f"- final_srt artifacts: 1 次 @ {finals[0].get('created_at', '')}")

    for chunk, times in sorted(attempt0_by_chunk.items()):
        if len(times) >= 2:
            flag(
                f"chunk {chunk} 出现 {len(times)} 次 attempt=0（{', '.join(times)}）——"
                "同进程重试会递增 attempt；多重 attempt=0 高度可疑为并发写同目录"
            )


# --- task report ---------------------------------------------------------------
def audit_task_report(path: Path, out: list[str]) -> None:
    keywords = ("retries", "retry", "fallback", "validation", "warning", "Fallback")
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(k.lower() in line.lower() for k in keywords) and line.strip()
    ]
    out.append(f"- task-report 关键行 ({path.name}):")
    out.extend(f"  - {line}" for line in lines[:15])


# --- knowledge base health (--kb) ----------------------------------------------
def audit_kb(root: Path, out: list[str]) -> None:
    out.append(f"# 知识库健康检查: {root}")
    for category in ("streamer", "common"):
        cat_dir = root / category
        index = cat_dir / "index.md"
        if not index.exists():
            flag(f"{category}/index.md 不存在")
            continue
        keys = []
        for line in index.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            # Entries are bullet lines: `- 主key [类型] | 别名… | 简介`
            if not line.startswith("- "):
                continue
            key = re.split(r"[\[|]", line[2:], 1)[0].strip()
            if key:
                keys.append(key)
        files = {p.stem for p in cat_dir.glob("*.md")} - {"index"}
        out.append(f"- {category}: index {len(keys)} 条, 条目文件 {len(files)} 个")
        for key in keys:
            if key not in files:
                flag(f"index 有 {category}/{key} 但条目文件缺失")
        for stem in sorted(files - set(keys)):
            flag(f"条目文件 {category}/{stem}.md 不在 index 里")
        for stem in sorted(files):
            text = (cat_dir / f"{stem}.md").read_text(encoding="utf-8")
            approx_tokens = len(text) // 2 + text.count(" ")
            if approx_tokens > KB_ENTRY_TOKEN_LIMIT:
                flag(
                    f"{category}/{stem}.md 约 {approx_tokens} token，超 kb_entries 注入上限 "
                    f"{KB_ENTRY_TOKEN_LIMIT}——replace_section 有截断丢尾风险（压缩 feature 未落地）"
                )
            if category == "streamer":
                for section in ("## 档案", "## 经历"):
                    if section not in text:
                        flag(f"{category}/{stem}.md 缺固定小节 {section}")
    ledger = root / "translation" / "common-mistake.md"
    if ledger.exists():
        text = ledger.read_text(encoding="utf-8")
        ids = set(re.findall(r"^###\s+(M\d{4,})", text, re.MULTILINE))
        featured = re.findall(r"^-\s+(M\d{4,})\s*$", text, re.MULTILINE)
        out.append(f"- common-mistake: 条目 {len(ids)}，精选 {len(featured)}")
        if len(featured) > 10:
            flag(f"精选 {len(featured)} 条，超过 10 条上限")
        for mid in featured:
            if mid not in ids:
                flag(f"精选引用不存在的条目 {mid}")


# --- main ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", help="Run directory (e.g. out/reference/<id>)")
    parser.add_argument("--refined", help="User-refined SRT to compare against")
    parser.add_argument("--kb", help="Knowledge base root to health-check instead")
    args = parser.parse_args()

    out: list[str] = []
    if args.kb:
        audit_kb(Path(args.kb), out)
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            print(f"not a directory: {run_dir}", file=sys.stderr)
            return 2
        layout = resolve_layout(run_dir)
        out.append(f"# Run digest: {run_dir}")
        for key in ("artifact_dir", "raw_srt", "final_srt", "annotated_csv"):
            out.append(f"- {key}: {layout.get(key) or '（未找到）'}")
        artifact_dir = layout.get("artifact_dir")
        if artifact_dir:
            ta = artifact_dir / "task-artifacts.jsonl"
            if ta.exists():
                records = load_jsonl(ta)
                kinds: dict[str, int] = {}
                for r in records:
                    kinds[r.get("kind", "?")] = kinds.get(r.get("kind", "?"), 0) + 1
                out.append("- artifact kinds: " + ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items())))
                audit_feedback(records, out)
                audit_knowledge_updates(records, out)
                audit_correction_timeline(records, out)
            cw = artifact_dir / "correction-windows.jsonl"
            if cw.exists():
                audit_windows(cw, out)
            tr = artifact_dir / "task-report.md"
            if tr.exists():
                audit_task_report(tr, out)
            notes = sorted(artifact_dir.glob("knowledge-update-harness-notes-*.md"))
            if notes:
                out.append("- harness notes 文件: " + ", ".join(p.name for p in notes))
        audit_language_setting(layout, out)
        audit_srt(layout, Path(args.refined) if args.refined else None, out)
    else:
        parser.error("provide a run_dir or --kb")

    print("\n".join(out))
    print()
    if FLAGS:
        print(f"## 自动标记 FLAGS（{len(FLAGS)} 条，逐条核实，不要照单全收）")
        for i, f in enumerate(FLAGS, 1):
            print(f"{i}. {f}")
    else:
        print("## 自动标记 FLAGS：无")
    return 0


if __name__ == "__main__":
    sys.exit(main())
