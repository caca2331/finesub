"""The contract tests P18 pre-registered for its own probe.

Not collected by the default suite (`tools/` is maintained on demand); run with
`python -m pytest tools/bench/test_sbd_split.py`.

The point of each one is a failure that already happened, or would have been
invisible if it had:

* **offset unit** -- the first run of this probe measured boundaries in BYTES
  and matched them against SentencePiece spans, which are CHARACTERS. On
  Japanese that is a ~3x shift, and it flipped the verdict from PASS to FAIL
  with no error anywhere. This is the test that would have caught it.
* **mapping determinism** -- the reviewer's point: "不先固定对齐规则，不同实现会
  得到不同的标签、AUC 和召回率".
* **no reference punctuation** -- the model card's own SBD numbers condition on
  reference punctuation; ours must not, or we would be measuring a condition
  that does not exist in production.
* **fold isolation** -- adjacent boundaries are highly correlated, so a window
  must never straddle the train/test split. Two gold windows overlap
  (`BV1nxje63ERi`, `yingtao`), so the same physical boundary can arrive twice
  under two group names; it must not.
* **the negative class** -- `never` is the DEFAULT, not a label to look for
  (`segmentation-gold.md` §2.2). Scoring only the written-out `never` items
  silently shrinks the negative pool to the positions a labeller bothered to
  write down, which are exactly the positions with a boundary signal.
* **the operating point** -- gate 3's fixed FPR is the current segmenter's own
  mis-cut rate. Reading it off the evaluation data instead (a Youden point on
  the baseline) makes the gate a function of the thing being tested.
* **the model pin** -- every number in `bench-baselines.md` 第十九节 is about one
  revision and two digests.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.bench import probe_sbd_split as probe


def _model_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(probe.REPO, "sp.model")
    return isinstance(hit, str)


needs_model = pytest.mark.skipif(
    not _model_cached(), reason="SBD model not in the HF cache"
)


class TestOffsetUnit:
    """The bug that flipped the verdict."""

    @needs_model
    def test_spans_are_characters_not_bytes(self) -> None:
        model = probe.SbdModel()
        text = "でも今日はさこれちょっと"

        _ids, spans = model.tokenize(text)

        assert spans[-1][1] == len(text), (
            "SentencePiece reports CHARACTER offsets; a byte-based boundary "
            "offset silently shifts by ~3x on Japanese"
        )
        assert spans[-1][1] != len(text.encode("utf-8"))
        for begin, end in spans:
            assert text[begin:end], "a span must address real characters"

    @needs_model
    def test_the_exposed_probability_still_is_the_shipped_boolean(self) -> None:
        """We expose a pre-threshold tensor rather than re-deriving anything.
        If `seg_preds == (p > 0.05)` ever stops holding, the tensor we grabbed
        is not the one the model ships its decision from."""

        import numpy as np

        model = probe.SbdModel()
        ids, _spans = model.tokenize("hello world how are you today i am fine")
        arr = np.array([[0] + ids + [2]], dtype=np.int64)
        out = dict(
            zip(model._names, model._session.run(None, {"input_ids": arr}))
        )

        assert (
            out["seg_preds"] == (out[probe.SBD_PROB_TENSOR] > probe.SBD_THRESHOLD)
        ).all()


class TestMappingContract:
    """Same boundary, same answer -- whatever the tokenizer does around it."""

    class _Tokenizer:
        """Spans in characters, chosen to exercise all three shapes."""

        def __init__(self, spans, probs):
            self._spans, self._probs = spans, probs

        def tokenize(self, text):
            return list(range(len(self._spans))), self._spans

        def probabilities(self, ids):
            return self._probs

    def test_one_word_over_several_tokens_reads_the_token_that_ends_it(self) -> None:
        words = [{"word": "ありがとう"}, {"word": "ございました"}]
        # 'ありがと' + 'う' + 'ございました': the first word ends mid-run.
        model = self._Tokenizer([(0, 4), (4, 5), (5, 11)], [0.1, 0.9, 0.5])

        out = probe.map_boundaries(words, [0], model)

        assert out[0] == pytest.approx(0.9)

    def test_two_words_inside_one_token_are_unevaluable(self) -> None:
        """Pre-registered: they must NOT share a neighbour's probability."""

        words = [{"word": "じゃあ"}, {"word": "んー"}, {"word": "この"}]
        model = self._Tokenizer([(0, 5), (5, 7)], [0.9, 0.2])

        out = probe.map_boundaries(words, [0, 1], model)

        assert out[0] is None
        assert out[1] == pytest.approx(0.9)

    def test_stripping_punctuation_does_not_shift_the_mapping(self) -> None:
        """The `strip` input form deletes characters, so every offset after a
        deleted one moves. The boundary must still land on the same token."""

        plain = [{"word": "です"}, {"word": "ご視聴"}]
        punctuated = [{"word": "です。"}, {"word": "ご視聴"}]
        model = self._Tokenizer([(0, 2), (2, 5)], [0.95, 0.3])

        assert (
            probe.map_boundaries(plain, [0], model)[0]
            == probe.map_boundaries(punctuated, [0], model)[0]
            == pytest.approx(0.95)
        )

    def test_the_input_form_is_the_pre_registered_one(self) -> None:
        assert probe.INPUT_FORM == "strip"
        assert probe.strip_punctuation("です。ご視聴、Ah!") == "ですご視聴ah"


class TestNoReferencePunctuation:
    """The evaluation may not read punctuation the corrector or the reference
    supplied -- the SBD head must condition on its own prediction."""

    def test_the_text_handed_to_the_model_carries_no_punctuation(self) -> None:
        words = [{"word": "必要です。"}, {"word": "ご視聴、"}]

        assert "".join(probe.strip_punctuation(w["word"]) for w in words) == (
            "必要ですご視聴"
        )

    def test_the_probe_never_loads_the_reference_subtitle(self) -> None:
        """`gold.load_cues` is the only door a reference SRT can come through,
        and the model must only ever see text that went through
        `strip_punctuation`. Asserted on identifiers, not prose."""

        import ast

        source = pathlib.Path(probe.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else
            getattr(node.func, "id", "")
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }

        assert "load_cues" not in called, "a reference subtitle reached the probe"
        # The one call site that feeds the tokenizer builds its text from
        # `strip_punctuation`; nothing else may call `tokenize`.
        feeder = next(
            f for f in ast.walk(tree)
            if isinstance(f, ast.FunctionDef) and f.name == "map_boundaries"
        )
        inner = {
            node.func.attr if isinstance(node.func, ast.Attribute) else
            getattr(node.func, "id", "")
            for node in ast.walk(feeder)
            if isinstance(node, ast.Call)
        }
        assert {"strip_punctuation", "tokenize"} <= inner
        assert source.count(".tokenize(") == 1


class TestFoldIsolation:
    def test_the_gate_splitter_is_group_aware(self) -> None:
        """Behavioural, not a source grep: the splitter the gate actually uses
        must keep every window whole."""

        import numpy as np

        # Interleaved on purpose: with groups in contiguous blocks a plain
        # `KFold` produces disjoint groups by accident, and the test would
        # pass against the very mutation it exists to catch.
        groups = np.array([f"w{i % 10}" for i in range(70)])
        y = np.array([i % 3 == 0 for i in range(70)], dtype=int)
        features = np.random.default_rng(0).normal(size=(70, 2))

        for train, test in probe.fold_splitter(5).split(features, y, groups):
            assert not set(groups[train]) & set(groups[test])


class TestReproducible:
    def test_the_same_table_gives_the_same_verdict(self, capsys) -> None:
        rows = [
            {"group": f"w{i % 4}", "label": "must" if i % 5 == 0 else "never",
             "declared": True, "seam": None, "prod_cut": int(i % 7 == 0),
             "pause": 0.4 if i % 5 == 0 else 0.05,
             "vad": 0.0, "p_sbd": 0.9 if i % 5 == 0 else 0.01}
            for i in range(60)
        ]

        probe.cmd_gate(rows)
        first = capsys.readouterr().out
        probe.cmd_gate(rows)
        second = capsys.readouterr().out

        assert first == second
        assert "VERDICT" in first


def _fixture(tmp_path, windows, *, words=40):
    """A minimal gold bed: one substrate, one worksheet, N label files.

    `windows` is a list of `(name, lo, hi, {k: label})`. Everything not
    mentioned inside `[lo, hi]` is an undeclared position -- the presumed
    `never` the negative class is made of.
    """

    import hashlib
    import json

    substrate = tmp_path / "fake-aligned.json"
    step = 0.5
    substrate.write_text(json.dumps({
        # `segment_split` present on purpose: `production_cuts` then reads the
        # artifact's own segment starts instead of importing the segmenter.
        "metadata": {"asr_align": {"segment_split": {}}},
        "segments": [{
            "words": [
                {"word": "あ", "start": i * step, "end": i * step + 0.4}
                for i in range(words)
            ],
        }],
    }), encoding="utf-8")
    digest = hashlib.sha256(substrate.read_bytes()).hexdigest()[:12]

    labels = tmp_path / "labels"
    sheets = tmp_path / "worksheets"
    labels.mkdir()
    sheets.mkdir()
    for name, lo, hi, declared in windows:
        rows = ["| i | k | t | pause | vad | punct |", "| - | - | - | - | - | - |"]
        for i, k in enumerate(sorted(declared), start=1):
            rows.append(f"| {i} | {k} | {k * step:.2f} | 0.40 | 0.00 |  |")
        (sheets / f"ws-{name}-{lo}-{hi}.md").write_text(
            "\n".join(rows), encoding="utf-8")
        (labels / f"{name}-{lo}-{hi}.json").write_text(json.dumps({
            "clip": name,
            "window": [lo, hi],
            "substrate_path": str(substrate),
            "substrate_sha": digest,
            "items": [{"k": k, "label": v} for k, v in sorted(declared.items())],
        }), encoding="utf-8")
    return labels, sheets


class TestNegativeUniverse:
    """`never` is the default; the probe must build it, not look it up."""

    def test_undeclared_in_window_boundaries_are_presumed_never(
        self, tmp_path, monkeypatch
    ) -> None:
        labels, sheets = _fixture(
            tmp_path, [("clipa", 0, 20, {5: "must", 7: "never", 9: "ok"})]
        )
        monkeypatch.setattr(probe, "LABELS_DIR", labels)
        monkeypatch.setattr(probe, "WORKSHEETS_DIR", sheets)
        rows = probe.load_boundaries()

        by_k = {r["k"]: r for r in rows}
        # [5, 7] is the labelled range; 9 is `ok` and drops out, but it is
        # outside the range anyway -- what matters is that 6 is present.
        assert by_k[5]["label"] == "must"
        assert by_k[6]["label"] == "never" and by_k[6]["declared"] is False
        assert by_k[7]["label"] == "never" and by_k[7]["declared"] is True

    def test_ok_and_unknown_are_excluded_from_both_classes(
        self, tmp_path, monkeypatch
    ) -> None:
        labels, sheets = _fixture(
            tmp_path,
            [("clipa", 0, 20, {4: "must", 6: "ok", 8: "unknown", 10: "must"})],
        )
        monkeypatch.setattr(probe, "LABELS_DIR", labels)
        monkeypatch.setattr(probe, "WORKSHEETS_DIR", sheets)
        ks = {r["k"] for r in probe.load_boundaries()}

        assert 6 not in ks and 8 not in ks
        assert {4, 5, 7, 9, 10} <= ks

    def test_the_same_physical_boundary_arrives_once(
        self, tmp_path, monkeypatch
    ) -> None:
        """Overlapping windows on one clip -- `BV1nxje63ERi`'s k=782 shape."""

        labels, sheets = _fixture(tmp_path, [
            ("clipa", 0, 10, {2: "must", 8: "must"}),
            ("clipa", 8, 20, {8: "must", 14: "must"}),
        ])
        monkeypatch.setattr(probe, "LABELS_DIR", labels)
        monkeypatch.setattr(probe, "WORKSHEETS_DIR", sheets)
        rows = probe.load_boundaries()

        keys = [(r["clip"], r["k"]) for r in rows]
        assert len(keys) == len(set(keys))
        # And the surviving row must not carry a per-window group, or the two
        # halves of one clip could still land on both sides of a fold.
        assert {r["group"] for r in rows} == {"clipa"}


class TestOperatingPoint:
    """Gate 3's FPR is production's, not one optimized on this data."""

    def test_the_fixed_fpr_follows_the_production_cuts(self, capsys) -> None:
        def table(cut_every):
            return [
                {"group": f"c{i % 4}", "clip": f"c{i % 4}", "k": i,
                 "label": "must" if i % 5 == 0 else "never",
                 "declared": True, "seam": None,
                 "pause": 0.4 if i % 5 == 0 else 0.05, "vad": 0.0,
                 "p_sbd": 0.9 if i % 5 == 0 else 0.01,
                 "prod_cut": int(i % cut_every == 0)}
                for i in range(80)
            ]

        probe.cmd_gate(table(5))
        tight = capsys.readouterr().out
        probe.cmd_gate(table(3))
        loose = capsys.readouterr().out

        def fpr(text):
            line = next(l for l in text.splitlines() if "production cuts" in l)
            return float(line.split("FPR ")[1].split(",")[0])

        # Same labels, same features, same folds -- only the production cut set
        # differs. A gate reading its FPR off the evaluation data would print
        # the same number twice.
        assert fpr(tight) != fpr(loose)

    def test_no_youden_style_optimum_survives_in_the_module(self) -> None:
        source = pathlib.Path(probe.__file__).read_text(encoding="utf-8")
        assert "tpr - fpr" not in source
        assert "argmax" not in source


class TestModelPin:
    def test_the_download_is_pinned_to_one_revision(self, tmp_path) -> None:
        seen = {}

        def download(*, repo_id, filename, revision):
            seen["revision"] = revision
            path = tmp_path / filename
            path.write_bytes(b"not the model")
            return str(path)

        with pytest.raises(SystemExit):
            probe.fetch_pinned("sp.model", download)
        assert seen["revision"] == probe.REVISION

    def test_a_matching_digest_is_accepted(self, tmp_path) -> None:
        import hashlib

        payload = b"pretend this is sp.model"
        digest = hashlib.sha256(payload).hexdigest()

        def download(*, repo_id, filename, revision):
            path = tmp_path / filename
            path.write_bytes(payload)
            return str(path)

        probe.FILE_SHA256["sp.model"], saved = digest, probe.FILE_SHA256["sp.model"]
        try:
            assert probe.fetch_pinned("sp.model", download).endswith("sp.model")
        finally:
            probe.FILE_SHA256["sp.model"] = saved
