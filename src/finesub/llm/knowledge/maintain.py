"""Human maintenance CLI for the knowledge store (plan §8, the step between
3 and 4): ``python -m finesub.llm.knowledge <command>``.

The store is the truth and ``rendered/`` is a cache, so every mutation here
goes through the same machinery the harness uses — the edit round-trip and the
``new``/``retire`` commands feed the shared apply engine (``kind=user``
revisions), ``revert``/``restore`` are the compensating transactions of plan
§2.5. Nothing in this module edits markdown in place.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from .base import knowledge_root_path, knowledge_write_lock
from .node.edit import EditError, edit_subject
from .node.envelope import Envelope
from .node.history import RevertError, restore_node, revert_revision
from .node.model import CATEGORIES
from .node.proposals import translate_model_proposals
from .node.apply import apply_envelope
from .node.repo import AmbiguousName, KnowledgeRepo
from .node.store import NotFoundError

_Lines = list[str]


def _repo(args: argparse.Namespace) -> KnowledgeRepo:
    return KnowledgeRepo.open(knowledge_root_path(args.root))


def _role_client(llm_model: list[str] | None):  # type: ignore[no-untyped-def]
    """Production client behind the process-local ``--llm-model`` override.

    ``--llm-model [任务组=]值``（可重复；裸值 = default）installs the runtime
    preferred-bindings overlay on the memoized route loader — config.toml is
    never touched, and every routing consumer in this process (planning,
    preflight, internal clients, resume identity) sees the same override.
    Values are model groups or ROUTE targets (no fallback chain behind the
    pin); bare catalog fact ids are rejected at load. The default ``RoleClient``
    construction keeps loading execution settings itself, which is what already
    guards the round-14 dropped-settings failure."""

    from ..client import RoleClient
    from ..routing.model_routes import install_runtime_preferred, parse_llm_model_args

    install_runtime_preferred(parse_llm_model_args(llm_model))
    return RoleClient()


def _resolve_subject(repo: KnowledgeRepo, name: str):  # type: ignore[no-untyped-def]
    """The single human-facing name->entry step. Ambiguity is reported, never
    resolved by preference: a style and the streamer it is named after are two
    different entries and the user has to say which."""

    try:
        resolved = repo.resolve_qualified(name)
    except AmbiguousName as exc:
        raise SystemExit(f"error: {exc}") from exc
    if resolved is None:
        raise SystemExit(f"error: no entry matches {name!r}")
    return resolved


def _apply_user_proposal(repo: KnowledgeRepo, proposal: dict[str, Any], note: str) -> _Lines:
    ops, report, _drafts, bindings = translate_model_proposals(
        # the human CLI may address every category, including the ones no
        # prompt is wired to yet (`allow_categories` defaults to matchable)
        [proposal], repo=repo, knowledge_read_rev=repo.rev, bindings=[],
        allow_categories=CATEGORIES,
    )
    lines = [f"skipped: {r.op} {r.entry} — {r.reason}" for r in report.skipped]
    engine_ops = [{k: v for k, v in op.items() if k != "_meta"} for op in ops]
    if not engine_ops:
        return lines or ["nothing to do"]
    envelope = Envelope(
        task_id="knowledge-cli",
        assignment_id="cli",
        context_epoch=0,
        knowledge_read_rev=repo.rev,
        ops=engine_ops,
        handle_bindings=bindings,
        draft_bindings=sorted({op["handle"] for op in engine_ops if op.get("op") == "create" and op.get("handle")}),
    )
    result = apply_envelope(repo.store, envelope, revision_kind="user", note=note)
    if result.rolled_back:
        raise SystemExit(f"error: rolled back — {result.rollback_reason}")
    lines.extend(f"rejected: {reason}" for _, reason in result.rejected_ops)
    repo.refresh_rendered()
    lines.append(f"rev {result.rev}")
    return lines


def _cmd_log(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    rows = repo.store.conn.execute(
        "SELECT * FROM revisions ORDER BY rev DESC LIMIT ?", (args.limit,)
    ).fetchall()
    lines = []
    for row in rows:
        extra = row["note"] or row["task_id"] or ""
        lines.append(f"{row['rev']:>5}  {row['created_at']}  {row['kind']:<8} {extra}")
    return lines or ["(empty store)"]


def _cmd_show(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    resolved = _resolve_subject(repo, args.entry)
    return [repo.entry_text(resolved.subject_id, args.rev).rstrip("\n")]


def _spawn_editor(path: Path) -> None:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    command = shlex.split(editor) if editor and " " in editor else ([editor] if editor else [])
    if not command:
        command = ["notepad"] if os.name == "nt" else ["vi"]
    subprocess.run([*command, str(path)], check=False)


def _cmd_edit(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    resolved = _resolve_subject(repo, args.entry)
    subject = repo.store.node(resolved.subject_id)
    if args.file:
        new_text = Path(args.file).read_text(encoding="utf-8")
    else:
        current = repo.entry_text(resolved.subject_id)
        with tempfile.TemporaryDirectory(prefix="finesub-kb-edit-") as tmp:
            scratch = Path(tmp) / f"{resolved.key}.md"
            scratch.write_text(current, encoding="utf-8")
            _spawn_editor(scratch)
            new_text = scratch.read_text(encoding="utf-8")
    report = edit_subject(repo, subject, new_text)
    if not report.changed:
        return ["no change"]
    lines = [f"rejected: {reason}" for reason in report.rejected]
    lines.append(
        f"rev {report.rev}: +{report.created} lines, ~{report.updated} updated, -{report.removed} removed"
    )
    return lines


def _cmd_new(args: argparse.Namespace) -> _Lines:
    proposal = {
        "op": "create_entry",
        "category": args.category,
        "entry": args.key,
        "intro": args.intro,
        "entry_type": args.type or "",
        "aliases": args.alias or [],
        "reason": "cli",
    }
    return _apply_user_proposal(_repo(args), proposal, f"cli:new:{args.key}")


def _cmd_retire(args: argparse.Namespace) -> _Lines:
    # resolve here rather than let the proposal do it: the translator resolves
    # by name, and a bare name may belong to two namespaces (`retire 某字幕组`
    # once retired the same-named common entry instead — review 2026-09-02).
    repo = _repo(args)
    resolved = _resolve_subject(repo, args.entry)
    proposal = {
        "op": "retire_entry", "entry": resolved.key, "category": resolved.category,
        "reason": args.reason or "cli",
    }
    if args.merged_into:
        # same treatment as the entry itself: `<类别>/<名字>` is a CLI notion,
        # and content merges into a sibling — never across namespaces
        target = _resolve_subject(repo, args.merged_into)
        if target.category != resolved.category:
            raise SystemExit(
                f"error: 不能并入另一个命名空间的条目"
                f"（{resolved.category}/{resolved.key} → {target.category}/{target.key}）"
            )
        proposal["merged_into"] = target.key
    return _apply_user_proposal(repo, proposal, f"cli:retire:{resolved.key}")


def _cmd_revert(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    try:
        rev = revert_revision(repo.store, args.rev)
    except (RevertError, NotFoundError) as exc:
        raise SystemExit(f"error: {exc}")
    repo.refresh_rendered()
    return [f"rev {rev}: reverted revision {args.rev}"]


def _cmd_restore(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    try:
        rev = restore_node(repo.store, args.local_id)
    except (NotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}")
    repo.refresh_rendered()
    return [f"rev {rev}: restored {args.local_id}"]


def _cmd_refresh(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    repo.refresh_rendered()
    return [f"rendered/ rebuilt at rev {repo.rev}"]


def _cmd_phase_b(args: argparse.Namespace) -> _Lines:
    from .node.phase_b import build_plan, execute_plan, render_plan, write_report

    repo = _repo(args)
    plan = build_plan(repo.store)
    lines = render_plan(plan)
    if args.report:
        lines.append(f"report written: {write_report(plan, args.report)}")
    if not args.execute:
        lines.append("dry-run (pass --execute to apply the conversion)")
        return lines
    if not plan.ops:
        lines.append("nothing to convert")
        return lines
    rev = execute_plan(repo.store, plan)
    repo.refresh_rendered()
    lines.append(f"rev {rev}: converted {len(plan.rows)} row(s) via {len(plan.ops)} op(s)")
    return lines


def _cmd_candidates(args: argparse.Namespace) -> _Lines:
    from .node.candidates import pending_human_reconciled, resolve_candidate

    repo = _repo(args)
    if args.resolve:
        from .node.scan import scan_candidates

        ok = resolve_candidate(
            repo.store, args.resolve, reason=args.reason,
            candidates=scan_candidates(repo.store, repo.rev).candidates,
        )
        return (
            [f"resolved {args.resolve}（resolution=human）"]
            if ok
            else [f"error: no standing decision for key {args.resolve!r}"]
        )
    rows = pending_human_reconciled(repo.store)
    if not rows:
        return ["no pending_human candidates"]
    lines = [f"{len(rows)} pending human decision(s):"]
    for row in rows:
        freshness = row.get("freshness", "current")
        lines.append(
            f"  {row['candidate_key']}  task={row['task_id'] or '?'}  {row['created_at']}"
            + (f"  [{freshness}]" if freshness != "current" else "")
        )
        if row.get("candidate"):
            lines.append(f"    候选：{row['candidate']}")
        if row["reason"]:
            lines.append(f"    判断：{row['reason']}")
        if row.get("missing"):
            lines.append(f"    缺证据：{row['missing']}")
    lines.append("resolve one with: candidates --resolve <key> [--reason …]")
    return lines


def _cmd_verify(args: argparse.Namespace) -> _Lines:
    from .verify import (
        book_verify_results,
        render_verify_prompt,
        run_verify_session,
        unverified_claims,
    )

    repo = _repo(args)
    claims = unverified_claims(repo.store, limit=args.limit)
    lines: _Lines = [f"{len(claims)} unverified externally-checkable claim(s)"]
    for claim in claims:
        lines.append(f"  {claim['claim_id']} [{claim['kind']}] {claim['line']}")
    if not claims:
        return lines
    if not args.execute:
        prompt = render_verify_prompt(claims)
        lines.append(f"dry-run: verify prompt rendered ({len(prompt)} chars); --execute 跑校验会话")
        return lines
    rows = run_verify_session(claims, client=_role_client(args.llm_model))
    for row in rows:
        lines.append(f"  {row['claim_id']} -> {row['verdict']}" + (f" ({row['url']})" if row["url"] else ""))
        # The evidence table stores verdict + source, not the reasoning, so a
        # `refuted` row is only actionable if the contradiction is printed
        # while the session's answer is still in hand.
        if row["verdict"] != "confirmed" and row.get("note"):
            lines.append(f"       note: {row['note']}")
        if row["verdict"] != "confirmed":
            lines.append(f"       line: {row['line']}")
    if args.apply:
        booked = book_verify_results(repo.store, rows, task_id=args.task_id)
        lines.append(f"booked {booked} evidence row(s)")
    else:
        lines.append("not booked (--apply to record the evidence rows)")
    return lines


def _cmd_repair(args: argparse.Namespace) -> _Lines:
    from .node.repair import render_repair_prompt, repair_targets, run_repair_session

    # Material mode moved out to `ingest` (2026-09-01): it is a different
    # task -- arbitrary text distilled into an entry -- and sharing a command
    # with the candidate pass made both harder to describe.
    repo = _repo(args)
    targets = repair_targets(repo)
    if args.subject:
        resolved = _resolve_subject(repo, args.subject)
        targets = {k: v for k, v in targets.items() if k == resolved.subject_id}
    if not targets:
        return ["no repair candidates (run `phase-b` for the deterministic half first)"]
    lines: _Lines = []
    client = None
    if args.execute:
        client = _role_client(args.llm_model)
    for subject_id, candidates in targets.items():
        subject = repo.store.node(subject_id)
        surface = subject.payload.get("surface", subject_id) if subject else subject_id
        lines.append(f"— {surface}: {len(candidates)} candidate(s)")
        if not args.execute:
            prompt = render_repair_prompt(repo, subject_id, candidates=candidates)
            lines.append(f"  dry-run: repair prompt rendered ({len(prompt)} chars); --execute 跑修复会话")
            continue
        result = run_repair_session(
            repo, subject_id, client=client, candidates=candidates, apply=args.apply,
        )
        if args.apply:
            report = result["apply_report"]
            lines.append(
                f"  applied rev {report.get('rev')}: {len(report.get('applied', []))} op(s),"
                f" {len(report.get('skipped', []))} skipped"
            )
            for row in result.get("candidate_ledger", []):
                resolution = f"（{row['resolution']}）" if row.get("resolution") else ""
                lines.append(f"  candidate {row['candidate']}: {row['status']}{resolution}"
                             + (f" — {row['note']}" if row.get("note") else ""))
        else:
            lines.append("  proposals (not applied; --apply to run them through validate→apply):")
            lines.extend(f"    {row}" for row in result["proposals_text"].splitlines() if row.strip())
    return lines


def _cmd_ingest(args: argparse.Namespace) -> _Lines:
    """Distil a material into one entry.

    The general form of what migration degrades into (owner 2026-09-01): a
    format the mechanical importer cannot read is not force-aligned by special
    cases, it becomes material for this task. Same three stages as every other
    LLM entry point -- no flag renders the prompt, ``--execute`` runs the
    session, ``--apply`` writes through the ordinary validate→apply path.

    ``--prompt`` steers what gets picked up; it does not relax the recording
    standard (``fragment_kb_judgment_v1.md`` rule 0), and the prompt says so.

    ⚠ One entry per run, named by ``--subject``. Material about an entry that
    does not exist yet has no route here -- create it with ``new`` first. That
    is a limit, not a design: routing a material to the right subject (or to
    several) is a judgement call nobody has specified yet.
    """

    from .node.repair import render_repair_prompt, run_repair_session

    repo = _repo(args)
    try:
        resolved = repo.resolve_qualified(args.subject)
    except AmbiguousName as exc:
        raise SystemExit(f"error: {exc}") from exc
    if resolved is None:
        raise SystemExit(
            f"error: no entry matches {args.subject!r}"
            " — create it with `new` before ingesting material about it"
        )
    if args.material == "-":
        material = sys.stdin.read()
    else:
        material = Path(args.material).expanduser().read_text(encoding="utf-8")
    if not material.strip():
        raise SystemExit("error: the material is empty")

    lines: _Lines = [
        f"— {resolved.key}: {len(material)} chars of material"
        + ("（带用户交代）" if args.prompt.strip() else "")
    ]
    if not args.execute:
        prompt = render_repair_prompt(
            repo, resolved.subject_id, material=material, user_prompt=args.prompt
        )
        lines.append(
            f"  dry-run: ingest prompt rendered ({len(prompt)} chars); --execute 跑蒸馏会话"
        )
        return lines
    result = run_repair_session(
        repo, resolved.subject_id, client=_role_client(args.llm_model),
        material=material, user_prompt=args.prompt, apply=args.apply,
        task_id="kb-ingest",
    )
    if not args.apply:
        lines.append("  proposals (not applied; --apply to run them through validate→apply):")
        lines.extend(
            f"    {row}" for row in result["proposals_text"].splitlines() if row.strip()
        )
        return lines
    report = result["apply_report"]
    lines.append(
        f"  applied rev {report.get('rev')}: {len(report.get('applied', []))} op(s),"
        f" {len(report.get('skipped', []))} skipped"
    )
    for row in report.get("skipped", [])[:10]:
        lines.append(f"    skipped: {row}")
    return lines


# ``verify`` is deliberately NOT here: it only inserts evidence rows, which
# are lock-free telemetry by design (busy_timeout), and holding the write
# lock through a web-search session would block real writers for minutes.
_MUTATING = {"edit", "new", "retire", "revert", "restore", "phase-b", "repair",
             "candidates", "ingest"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m finesub.llm.knowledge",
        description="Human maintenance for the knowledge store (SQLite truth, rendered/ cache).",
    )
    parser.add_argument("--root", default=None, help="knowledge root (default: resolved runtime root)")
    sub = parser.add_subparsers(dest="command", required=True)

    log = sub.add_parser("log", help="list revisions, newest first")
    log.add_argument("-n", "--limit", type=int, default=20)
    log.set_defaults(func=_cmd_log)

    show = sub.add_parser("show", help="print one entry (human projection)")
    show.add_argument("entry")
    show.add_argument("--rev", type=int, default=None)
    show.set_defaults(func=_cmd_show)

    edit = sub.add_parser("edit", help="edit one entry via $EDITOR round-trip")
    edit.add_argument("entry")
    edit.add_argument("--file", help="take the edited text from this file instead of an editor")
    edit.set_defaults(func=_cmd_edit)

    new = sub.add_parser("new", help="create an entry from the preset skeleton")
    new.add_argument("category", choices=CATEGORIES)
    new.add_argument("key")
    new.add_argument("--intro", required=True)
    new.add_argument("--type", help="common entry type（游戏/动画/社区/其他）")
    new.add_argument("--alias", action="append")
    new.set_defaults(func=_cmd_new)

    retire = sub.add_parser("retire", help="retire an entry (memberships close with it)")
    retire.add_argument("entry")
    retire.add_argument("--merged-into")
    retire.add_argument("--reason")
    retire.set_defaults(func=_cmd_retire)

    revert = sub.add_parser("revert", help="compensating transaction undoing one revision")
    revert.add_argument("rev", type=int)
    revert.set_defaults(func=_cmd_revert)

    restore = sub.add_parser("restore", help="bring a retired node back under the same id")
    restore.add_argument("local_id")
    restore.set_defaults(func=_cmd_restore)

    refresh = sub.add_parser("refresh", help="rebuild the rendered/ markdown cache")
    refresh.set_defaults(func=_cmd_refresh)

    phase = sub.add_parser(
        "phase-b",
        help="§8 Phase B: deterministic conversion of the v1 archive shape into the v3 grammar",
    )
    phase.add_argument("--execute", action="store_true", help="apply the conversion")
    phase.add_argument("--report", help="write phase-b.json / phase-b.md into this directory")
    phase.set_defaults(func=_cmd_phase_b)

    candidates = sub.add_parser(
        "candidates",
        help="candidate decision ledger: list pending_human rows / resolve one",
    )
    candidates.add_argument("--resolve", metavar="KEY", default="",
                            help="mark this candidate_key resolved (resolution=human)")
    candidates.add_argument("--reason", default="", help="optional human note for --resolve")
    candidates.set_defaults(func=_cmd_candidates)

    verify = sub.add_parser(
        "verify",
        help="§11.4 verification task: web-check unevidenced term/note claims (dry-run by default)",
    )
    verify.add_argument("--limit", type=int, default=20)
    verify.add_argument("--execute", action="store_true", help="run the native-search session (spends quota)")
    verify.add_argument("--llm-model", dest="llm_model", action="append", metavar="[TASKGROUP=]VALUE",
                        help="runtime routing override（可重复；值=模型组或 route target，整体换绑无回退；裸值=default）；仅本进程，config.toml 不动")
    verify.add_argument("--apply", action="store_true", help="book the session's evidence rows")
    verify.add_argument("--task-id", default="kb-verify")
    verify.set_defaults(func=_cmd_verify)

    repair = sub.add_parser(
        "repair",
        help="§11.3 repair task: the LLM half of the second pass (dry-run by default)",
    )
    repair.add_argument("--subject", help="restrict to one entry")
    repair.add_argument("--execute", action="store_true", help="run the repair session (spends quota)")
    repair.add_argument("--llm-model", dest="llm_model", action="append", metavar="[TASKGROUP=]VALUE",
                        help="runtime routing override（可重复；值=模型组或 route target，整体换绑无回退；裸值=default）；仅本进程，config.toml 不动")
    repair.add_argument("--apply", action="store_true", help="apply the proposals through validate→apply")
    repair.set_defaults(func=_cmd_repair)

    ingest = sub.add_parser(
        "ingest",
        help="distil a material (notes, a page you saved) into one entry (dry-run by default)",
    )
    ingest.add_argument("--subject", required=True, help="the entry the material is about")
    ingest.add_argument(
        "--material", required=True, help="path to the material, or - to read stdin"
    )
    ingest.add_argument(
        "--prompt", default="",
        help="what you want out of it; steers WHAT is picked up, never the recording standard",
    )
    ingest.add_argument("--execute", action="store_true", help="run the session (spends quota)")
    ingest.add_argument("--llm-model", dest="llm_model", action="append", metavar="[TASKGROUP=]VALUE",
                        help="runtime routing override（可重复；值=模型组或 route target，整体换绑无回退；裸值=default）；仅本进程，config.toml 不动")
    ingest.add_argument("--apply", action="store_true", help="apply the proposals through validate→apply")
    ingest.set_defaults(func=_cmd_ingest)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """The maintenance CLI.

    The reporter is bound HERE rather than in the ``__main__`` block below:
    the documented invocation is ``python -m finesub.llm.knowledge``, which
    enters through the package's ``__main__.py`` and calls this function, so a
    binding in this module's own guard would never run. Without it the
    knowledge layer's warnings -- an unparseable hand edit, a skipped harvest
    -- are handed to the silent thread-local default and vanish.
    """

    from finesub.reporting import reporting_to, terminal_reporter

    with reporting_to(terminal_reporter()):
        return _main(argv)


def _main(argv: list[str] | None) -> int:
    args = parse_args(argv)
    # Unconditional overlay install at the entry (review 2026-08-28 P2-5):
    # commands without --llm-model pass an empty table, which CLEARS any
    # override a previous in-interpreter invocation installed — the lifecycle
    # rule every entry point follows.
    from ..routing.model_routes import install_runtime_preferred, parse_llm_model_args

    install_runtime_preferred(parse_llm_model_args(getattr(args, "llm_model", None)))
    handler: Callable[[argparse.Namespace], _Lines] = args.func
    try:
        if args.command in _MUTATING:
            root = knowledge_root_path(args.root)
            with knowledge_write_lock(root) as acquired:
                if not acquired:
                    print("error: knowledge base is locked by another process", file=sys.stderr)
                    return 1
                lines = handler(args)
        else:
            lines = handler(args)
    except EditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 1
        raise
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
