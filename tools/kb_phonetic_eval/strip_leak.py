"""Copy a knowledge store and retire the items that test clips taught it.

An item counts as learned from a test clip when the revision that created it carries that clip's
task id: run tasks record the id they were started with, which for most runs contains the clip id, so
pass clip ids and each matches every revision whose ``task_id`` contains it. The script prints how
many revisions and items each id matched: most clips taught the store nothing, so a 0 is normal, but a
0 for a clip you know was ingested means a wrong id. Used by the
phonetic-hints A/B (maintainers only) so the knowledge base has not already seen the
mishearings the experiment scores.

What this cannot find: items imported from the old Markdown store, whose revision is ``auto-import``
and names no source. Some of those were written from the same clips; ``score.py`` flags occurrences
where such a stored spelling sits in the raw ("residual") so they can be read apart.

    python tools/kb_phonetic_eval/strip_leak.py <knowledge.sqlite> <out_dir> <clip or task id>...
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finesub.llm.knowledge.node.store import KnowledgeStore  # noqa: E402


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    source, out_dir, *task_ids = sys.argv[1:]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "knowledge.sqlite"
    if target.exists():
        raise SystemExit(f"{target} exists; remove it first")
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        known = [row[0] or "" for row in src.execute("SELECT task_id FROM revisions")]
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)  # consistent even while the source has an open write-ahead log
        finally:
            dst.close()
    finally:
        src.close()
    with KnowledgeStore(target) as store:
        rows = store.conn.execute(
            "SELECT i.item_id, i.field, v.value, r.task_id FROM items i"
            " JOIN item_versions v ON v.item_id = i.item_id AND v.valid_to_rev IS NULL"
            " JOIN revisions r ON r.rev = i.created_rev"
        ).fetchall()
        leaked = [row for row in rows if any(task in (row[3] or "") for task in task_ids)]
        for task in task_ids:
            revisions = sum(task in seen for seen in known)
            items = sum(task in (row[3] or "") for row in leaked)
            print(f"{task}: {revisions} revisions, {items} items")
        with store.begin("user") as txn:
            for item_id, field, value, task in leaked:
                txn.tombstone_item(item_id)
                print(f"retired {field} {value!r} (from {task})")
    print(f"{len(leaked)} items retired in {target}")


if __name__ == "__main__":
    main()
