"""The joint backends' text-side sentence-boundary model (`docs/joint-asr.md`; the P18 model, `docs/bench-baselines.md`).

35t's lines lost to production's on fragmenting; the probability that a sentence ends after each word, taken
off the cost of cutting there, put 83% of the readers' windows on its side, Scribe's 81% and MAI2's 68% (2026-11
Scribe report). Fixed on for all three (owner 2026-11-11: light enough, and no switch to add) and for both MOSS
backends, in every language (owner 2026-11-19 for Japanese, 2026-11-20 for the rest: no reason to differ from the
other backends; measured on Chinese and English in the moss-word-cut report); `FINESUB_SBD=off`
builds the lines without it, for the "without" arm of a comparison (`README_DEV.md`). A load that fails warns once
per process and the lines are built without it, as `metadata.joint_asr.sentence_boundary_model` then records.

The model (xlm-roberta punctuation / sentence boundary, ONNX, about 1.1 GB, CPU) is the manifest's
`sentence-boundary` entry: fetched and verified on the project's download route (`docs/download-routes.md`). Its
graph only returns a thresholded boolean; the probability before the threshold is exposed
as one more graph output, which adds no computation and changes no weights. That graph is written once beside
the download (`EXPOSED`) and loaded from there: rebuilding it in memory on every run held the model three times
over (5.3 GB at the peak against 1.9 GB, measured 2026-10-30 on a 30-minute source; 1 s to load, 4 s to run).
"""

from __future__ import annotations

import os
import threading
import unicodedata
from pathlib import Path
from typing import Sequence

from ...reporting import current_reporter
from ..postprocessing.segmentation import SENTENCE_END_BONUS_KEY

MANIFEST_ID = "sentence-boundary"
#: `off` (or 0 / false / no): no model, the lines as the DP alone builds them. A developer's switch, not an option.
OFF_ENV = "FINESUB_SBD"
SBD_PROB_TENSOR = "/Gather_output_0"
#: `max_length: 256` in the model's own config, minus <s> and </s>.
MAX_TOKENS = 254
#: Overlapping windows: the last token of a window always reads as the end of the text, so each token takes
#: its probability from the window where it sits furthest from both edges.
CHUNK_STRIDE = 127
#: Named by the first digits of the manifest's `model.onnx` digest (an earlier route named it so; kept, so a machine
#: that has it does not write 1.1 GB again).
EXPOSED = "model.p-sbd.c43ca686dabc.onnx"
#: The weight of the probability against the boundary cost (the experiment's lam; 1.0 makes a certain
#: sentence end worth what sentence punctuation is worth).
WEIGHT = 1.0


class SbdUnavailable(RuntimeError):
    pass


def _directory() -> Path:
    from ..runtime import hf_weights

    directory = hf_weights.prepare_directory(MANIFEST_ID)
    if directory is None:
        raise SbdUnavailable("could not fetch the sentence-boundary model")
    return directory


class SbdModel:
    def __init__(self) -> None:
        try:
            import onnx
            import onnxruntime as ort
            import sentencepiece as spm
        except ImportError as error:
            raise SbdUnavailable(f"{error.name} is not installed (pip install -e \".[cloud]\")") from error
        directory = _directory()
        original, exposed = directory / "model.onnx", directory / EXPOSED
        if not exposed.exists():
            model = onnx.load(original)
            model.graph.output.append(
                onnx.helper.make_tensor_value_info(SBD_PROB_TENSOR, onnx.TensorProto.FLOAT, None))
            temporary = exposed.with_name(f".{exposed.name}.{os.getpid()}.part")  # another process may be writing too
            onnx.save(model, temporary)
            del model
            temporary.replace(exposed)
        self._session = ort.InferenceSession(str(exposed), providers=["CPUExecutionProvider"])
        self._names = [o.name for o in self._session.get_outputs()]
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(str(directory / "sp.model"))

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """(ids, spans) with spans in characters (`offset_mapping`; the `proto` form counts UTF-8 bytes)."""
        encoded = self._sp.encode(text, out_type="offset_mapping")
        return list(encoded["ids"]), [tuple(span) for span in encoded["offsets"]]

    def probabilities(self, ids: list[int]):
        import numpy as np

        best = np.zeros(len(ids), dtype=np.float32)
        margin = np.full(len(ids), -1.0, dtype=np.float32)
        for start in range(0, max(1, len(ids)), CHUNK_STRIDE):
            chunk = ids[start:start + MAX_TOKENS]
            if not chunk:
                continue
            arr = np.array([[0] + chunk + [2]], dtype=np.int64)
            out = dict(zip(self._names, self._session.run(None, {"input_ids": arr})))
            for offset, value in enumerate(out[SBD_PROB_TENSOR][0][1:1 + len(chunk)]):
                edge = min(offset, len(chunk) - 1 - offset)
                if edge > margin[start + offset]:
                    margin[start + offset], best[start + offset] = edge, value
            if start + MAX_TOKENS >= len(ids):
                break
        return best


def strip_punctuation(text: str) -> str:
    """The model's input form: no punctuation, no spaces, lowercased (it was trained so)."""
    return "".join(ch.lower() for ch in text if not unicodedata.category(ch).startswith("P") and not ch.isspace())


_MODEL: SbdModel | None = None
#: Why the process's one load failed: not tried again (a missing download would cost every source its timeout).
_FAILED: str | None = None
#: Batch items on the network lane may reach the first load together; the session itself is thread-safe.
_LOAD = threading.Lock()


def loaded() -> SbdModel | None:
    """The process's model, loaded on first use; None when `OFF_ENV` turns it off or it could not load (warned once)."""
    global _MODEL, _FAILED
    if os.environ.get(OFF_ENV, "").strip().lower() in ("off", "0", "false", "no"):
        return None
    with _LOAD:
        if _MODEL is None and _FAILED is None:
            try:
                _MODEL = SbdModel()
            except Exception as error:  # noqa: BLE001 -- a fetch, a pinned digest, an ONNX load: the lines go on without it
                _FAILED = str(error)[:300] or type(error).__name__
                current_reporter().warning(
                    "joint-sbd", f"分句模型加载失败（{_FAILED}），本次运行的行不带它（只试一次）",
                    impact="行在句中切开的会多一些", action="检查网络或下载路线后重跑这一步")
    return _MODEL


def sentence_ends(model: SbdModel, words: Sequence[dict]) -> None:
    """`SENTENCE_END_BONUS_KEY` on every DP word: WEIGHT x the probability that a sentence ends after it (0
    inside a subword, where there is no boundary to speak of)."""
    text, ends = "", []
    for w in words:
        surface = strip_punctuation(w["word"])
        if text and w.get("space_before") and surface:
            text += " "
        text += surface
        ends.append(len(text))
    ids, spans = model.tokenize(text)
    at = {end: k for k, (_begin, end) in enumerate(spans)}
    probs = model.probabilities(ids)
    for w, end in zip(words, ends):
        k = at.get(end)
        w[SENTENCE_END_BONUS_KEY] = 0.0 if k is None else WEIGHT * float(probs[k])
