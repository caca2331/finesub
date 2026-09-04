"""Shadow migration CLI (plan §8 step 2).

    python -m finesub.llm.knowledge.node.migrate --source knowledge --store tmp/kb-shadow.sqlite --report tmp/kb-shadow

Read-only with respect to the markdown tree; writes only the target store
and the report directory. Refuses to reuse a store whose recorded source
lock differs from the current tree (plan §7.1 step 3) unless ``--force``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .importer import _source_lock, import_knowledge_root, write_report_files
from .parity import check_parity, write_parity_report
from .store import KnowledgeStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shadow-import a markdown knowledge tree into a node store.")
    parser.add_argument("--source", required=True, help="knowledge root (contains streamer/ and common/)")
    parser.add_argument("--store", required=True, help="target sqlite file (created; must not exist unless --force)")
    parser.add_argument("--report", required=True, help="report directory")
    parser.add_argument("--force", action="store_true", help="overwrite an existing store file")
    args = parser.parse_args(argv)

    source = Path(args.source)
    store_path = Path(args.store)
    if store_path.exists():
        if not args.force:
            print(f"error: {store_path} exists (use --force to overwrite)", file=sys.stderr)
            return 2
        store_path.unlink()
        for suffix in ("-wal", "-shm"):
            sidecar = store_path.with_name(store_path.name + suffix)
            if sidecar.exists():
                sidecar.unlink()
    store_path.parent.mkdir(parents=True, exist_ok=True)

    with KnowledgeStore(store_path) as store:
        report = import_knowledge_root(source, store)
        write_report_files(report, args.report)
        parity = check_parity(store, source, rev=report.rev)
        write_parity_report(parity, args.report)

    print(
        f"imported rev={report.rev} subjects={report.subjects} nodes={report.nodes} "
        f"items={report.items} merge-candidates={len(report.merge_candidates)} "
        f"lock={report.source_lock}"
    )
    print(f"parity legacy={'OK' if parity.legacy_ok else 'FAIL'} human={'OK' if parity.human_ok else 'FAIL'}")
    if _source_lock(source) != report.source_lock:
        print("warning: source tree changed during import", file=sys.stderr)
    return 0 if parity.legacy_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
