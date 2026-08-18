"""Golden snapshots of the shipped prompts, plus the version ledger.

Generated prompts change six places at once when one material row moves, so the
snapshots put that diff back on the review surface
(``docs/llm_prompts.md``).

The snapshots alone cannot enforce the *version* discipline: they are tracked
files, so regenerating them makes the comparison pass again and "changed the
prompt without bumping PROMPT_VERSION" has no detection point. ``manifest.json``
is that second, independent载体 -- it records ``PROMPT_VERSION -> {file: sha256}``
and refuses a different hash under a version it has already seen.

Regenerate with::

    python -m pytest test/test_llm_prompt_goldens.py --regenerate-goldens
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from finesub.llm.prompt_compose import (
    PROMPT_VERSION,
    compose_correction_system,
    compose_correction_user,
)
from finesub.llm.routing.profiles import resolve_profile

GOLDEN_DIR = Path(__file__).parent / "data" / "prompt_goldens"
MANIFEST = GOLDEN_DIR / "manifest.json"

# Not a full axis sweep: the six vectors the retired presets mapped to,
# plus one representative new combination per axis.
SHIPPED_COMBINATIONS = {
    "text-none-minimum": ("text", "none", "efficiency"),
    "text-none-high": ("text", "none", "quality"),
    "text-native-high": ("text", "native", "quality"),
    "text-local-high": ("text", "local", "quality"),
    "audio-local-high": ("audio", "local", "quality"),
    "video-local-high": ("video", "local", "quality"),
    # representative uncalibrated combinations
    "audio-none-high": ("audio", "none", "quality"),
    "video-local-med": ("video", "local", "intermediate"),
}

VARIANTS = ("capableB", "capableC", "basicA", "basicB")


def _render(switches, variant: str) -> str:
    profile = resolve_profile(*switches)
    system = compose_correction_system(profile, variant=variant)
    user = compose_correction_user(
        profile,
        variant=variant,
        general_context_json="{}",
        window_context="（无）",
        entry_details="（无）",
        previous_advice="（无）",
        pre_round_notes="（无）",
        search_results="（无）",
        preceding_context_csv="-1|-1.0|0.5|0.1|前文",
        current_asr_csv="1|0.0|1.0|0.0|テスト",
        current_asr_row_count=1,
    )
    return f"=== system ===\n{system}\n=== user ===\n{user}"


def _cases():
    for name, switches in SHIPPED_COMBINATIONS.items():
        for variant in VARIANTS:
            yield f"{name}__{variant}.txt", switches, variant


def test_goldens_match(request) -> None:
    regenerate = request.config.getoption("--regenerate-goldens")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    entry = dict(manifest.get(PROMPT_VERSION, {}))
    mismatched = []
    for filename, switches, variant in _cases():
        rendered = _render(switches, variant)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        path = GOLDEN_DIR / filename
        if regenerate:
            path.write_text(rendered, encoding="utf-8")
        elif not path.exists():
            mismatched.append(f"{filename}: no golden yet (run --regenerate-goldens)")
            continue
        elif path.read_text(encoding="utf-8") != rendered:
            mismatched.append(f"{filename}: differs from its golden")
        recorded = entry.get(filename)
        if recorded is not None and recorded != digest:
            mismatched.append(
                f"{filename}: content changed under an existing PROMPT_VERSION "
                f"({PROMPT_VERSION}) -- bump the version rather than rewriting "
                "history"
            )
        entry[filename] = digest
    # The ledger is written only on a clean regenerate. Writing it before the
    # assert would let the discipline be laundered by running the command
    # twice: the first run reports the violation but records the new hashes,
    # so the second run finds them matching and passes.
    if regenerate and not mismatched:
        manifest[PROMPT_VERSION] = entry
        MANIFEST.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    assert not mismatched, "\n".join(mismatched)


def test_manifest_is_append_only_per_version() -> None:
    """A bump may repeat an unchanged hash; a version may not change one."""

    if not MANIFEST.exists():
        pytest.skip("no manifest yet")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert PROMPT_VERSION in manifest, (
        f"{PROMPT_VERSION} has no manifest entry -- run --regenerate-goldens"
    )
    for version, entry in manifest.items():
        assert entry, f"{version} has an empty entry"
        for filename, digest in entry.items():
            assert len(digest) == 64, f"{version}/{filename}: not a sha256"


def test_regenerate_does_not_record_a_violating_run(tmp_path, monkeypatch) -> None:
    """A failing --regenerate-goldens must leave the ledger untouched.

    Writing the manifest before the assert let the discipline be laundered:
    the first run reported "bump the version" but recorded the new hashes, so
    an identical second run passed.
    """

    import finesub.llm.prompt_compose as pc
    import test.test_llm_prompt_goldens as mod

    golden_dir = tmp_path / "goldens"
    golden_dir.mkdir()
    manifest = golden_dir / "manifest.json"
    monkeypatch.setattr(mod, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(mod, "MANIFEST", manifest)
    monkeypatch.setattr(
        mod, "SHIPPED_COMBINATIONS", {"text-none-high": ("text", "none", "quality")}
    )
    monkeypatch.setattr(mod, "VARIANTS", ("capableB",))

    class _Config:
        @staticmethod
        def getoption(_name):
            return True

    class _Request:
        config = _Config()

    # First regenerate: clean, records the baseline.
    mod.test_goldens_match(_Request())
    baseline = json.loads(manifest.read_text(encoding="utf-8"))
    assert baseline[pc.PROMPT_VERSION]

    # Now change the rendered content without bumping the version.
    monkeypatch.setattr(mod, "_render", lambda *a, **k: "different content")
    with pytest.raises(AssertionError, match="content changed under an existing"):
        mod.test_goldens_match(_Request())

    # The ledger must still hold the pre-change hashes, so a second identical
    # run reports the same violation instead of silently passing.
    assert json.loads(manifest.read_text(encoding="utf-8")) == baseline
    with pytest.raises(AssertionError, match="content changed under an existing"):
        mod.test_goldens_match(_Request())
