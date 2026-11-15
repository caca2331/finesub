"""35t's text-side sentence-boundary model (`docs/joint-asr.md`; the P18 model, `docs/bench-baselines.md`).

35t's lines lost to production's on fragmenting; the probability that a sentence ends after each word, taken
off the cost of cutting there, put 83% of the readers' windows on its side. Only 35t gets it: the other two
did not need it in the experiment.

The model (xlm-roberta punctuation / sentence boundary, ONNX, about 1.1 GB, CPU) is pinned by revision and
file digest. Its graph only returns a thresholded boolean; the probability before the threshold is exposed
as one more graph output, which adds no computation and changes no weights. That graph is written once beside
the download (`EXPOSED`) and loaded from there: rebuilding it in memory on every run held the model three times
over (5.3 GB at the peak against 1.9 GB, measured 2026-10-30 on a 30-minute source; 1 s to load, 4 s to run).
"""

from __future__ import annotations

import hashlib
import os
import threading
import unicodedata
from pathlib import Path
from typing import Sequence

from ..postprocessing.segmentation import SENTENCE_END_BONUS_KEY

REPO = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
REVISION = "d1769a597ce8dfaa070d436bc67d4ee761f58884"
FILE_SHA256 = {
    "model.onnx": "c43ca686dabc237c3b06be834b9423c07580fef7e2b1a6c09976f7d60caa5d89",
    "sp.model": "7f944d0be93b275f62e1913fd409f378ddbba108e57fe4a9cb47e8c047f6bef1",
}
SBD_PROB_TENSOR = "/Gather_output_0"
#: `max_length: 256` in the model's own config, minus <s> and </s>.
MAX_TOKENS = 254
#: Overlapping windows: the last token of a window always reads as the end of the text, so each token takes
#: its probability from the window where it sits furthest from both edges.
CHUNK_STRIDE = 127
EXPOSED = f"model.p-sbd.{FILE_SHA256['model.onnx'][:12]}.onnx"
#: The weight of the probability against the boundary cost (the experiment's lam; 1.0 makes a certain
#: sentence end worth what sentence punctuation is worth).
WEIGHT = 1.0


class SbdUnavailable(RuntimeError):
    pass


def _fetch(name: str) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=REPO, filename=name, revision=REVISION)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != FILE_SHA256[name]:
        raise SbdUnavailable(f"{name} of {REPO} is {digest}, pinned {FILE_SHA256[name]}")
    return path


class SbdModel:
    def __init__(self) -> None:
        try:
            import onnx
            import onnxruntime as ort
            import sentencepiece as spm
        except ImportError as error:
            raise SbdUnavailable(f"{error.name} is not installed (pip install -e \".[cloud]\")") from error
        original = Path(_fetch("model.onnx"))
        exposed = original.with_name(EXPOSED)
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
        self._sp.load(_fetch("sp.model"))

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
#: Batch items on the network lane may reach the first load together; the session itself is thread-safe.
_LOAD = threading.Lock()


def sentence_ends(words: Sequence[dict]) -> None:
    """`SENTENCE_END_BONUS_KEY` on every DP word: WEIGHT x the probability that a sentence ends after it (0
    inside a subword, where there is no boundary to speak of)."""
    global _MODEL
    with _LOAD:
        if _MODEL is None:
            _MODEL = SbdModel()
    text, ends = "", []
    for w in words:
        surface = strip_punctuation(w["word"])
        if text and w.get("space_before") and surface:
            text += " "
        text += surface
        ends.append(len(text))
    ids, spans = _MODEL.tokenize(text)
    at = {end: k for k, (_begin, end) in enumerate(spans)}
    probs = _MODEL.probabilities(ids)
    for w, end in zip(words, ends):
        k = at.get(end)
        w[SENTENCE_END_BONUS_KEY] = 0.0 if k is None else WEIGHT * float(probs[k])
