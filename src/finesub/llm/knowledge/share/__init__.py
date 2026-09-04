"""Knowledge sharing (plan §6): exchange formats, sync merge, local server.

Client half: :mod:`exchange` (bundle/snapshot wire formats + the untrusted-
content boundary), :mod:`sync` (pull merge with three-way scalars and
canonical-id set union), :mod:`client` (urllib HTTP), :mod:`cli`
(``python -m finesub.llm.knowledge.share``). Server half: :mod:`server`, a
stdlib-only shell over the same protocol
(``python -m finesub.llm.knowledge.share.server --root <dir>``).

The three protocol hardenings (review 2026-08-26) live here: snapshot hash
chain with client-side anti-rollback, client-generated push idempotency keys,
and review-queue leases with verdict CAS.
"""
