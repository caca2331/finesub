"""Knowledge node store (design: ``docs/plans/knowledge-node-plan.md``).

Shadow-phase package (plan §8 steps 1-2): an inactive SQLite store with
versioned rows + pinned reads, presets as data, the three render
projections, and the lossless markdown importer. Nothing in the production
harness reads from here yet; the markdown tree under ``knowledge/`` remains
the source of truth until the cut-over commit (plan §8 step 3).
"""
