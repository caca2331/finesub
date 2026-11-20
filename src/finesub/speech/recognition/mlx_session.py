"""Request-local bindings for the pinned mlx-whisper 0.4.3 seek loop.

No upstream globals or model methods are mutated. The released function is
compiled in a private namespace with one bounded timestamp-split adaptation
(mlx_seek); all other code is the MIT-licensed mlx_whisper/transcribe.py
(Copyright 2023 Apple Inc.). Its unmodified source digest is provenance.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
from types import SimpleNamespace

import numpy as np

from .encoder_cache import EncoderCache


class CachedEncoder:
    def __init__(self, encoder, entries=0, max_bytes=64 * 1024 * 1024):
        self.encoder = encoder
        self.entries = entries
        self.max_bytes = max_bytes
        self.cache = EncoderCache(entries)
        self.bytes = 0
        self.hits = self.misses = self.evictions = 0

    def __call__(self, features):
        if not self.entries:
            return self.encoder(features)
        # Snapshot the exact post-preprocessing mel. Never key on audio paths.
        key = np.asarray(features).copy()
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        output = self.encoder(features)
        import mlx.core as mx
        mx.eval(output)
        cost = key.nbytes + output.nbytes
        if self.bytes + cost > self.max_bytes or len(self.cache) >= self.entries:
            self.evictions += len(self.cache)
            self.cache.clear()
            self.bytes = 0
        if cost <= self.max_bytes:
            self.cache.put(key, output)
            self.bytes += cost
        return output

    def clear(self):
        self.cache.clear()
        self.bytes = 0


class RequestModel:
    """A model view whose decode and encoder hooks are local to one owner."""
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.owner.model, name)

    def encoder(self, features):
        return self.owner.encoder(features)

    @property
    def keep_attention_on_device(self):
        return self.owner.alignment_mode == "metal"

    @property
    def admission_workers(self):
        return getattr(self.owner, "admission_workers", 1)

    def decode(self, mel, options):
        return self.owner._decode(self, mel, options)

    def detect_language(self, mel, tokenizer=None):
        decoding = importlib.import_module("mlx_whisper.decoding")
        return decoding.detect_language(self, mel, tokenizer)


def bind_transcribe(model, timing):
    from .mlx_seek import adapt_seek_source, split_window

    module = importlib.import_module("mlx_whisper.transcribe")
    upstream = module.transcribe
    source = inspect.getsource(upstream)
    namespace = dict(upstream.__globals__)
    namespace.update(
        ModelHolder=SimpleNamespace(get_model=lambda *_args, **_kwargs: model),
        add_word_timestamps=timing,
        _finesub_split_window=split_window,
    )
    exec(compile(adapt_seek_source(source), "<finesub-mlx-whisper-0.4.3>", "exec"), namespace)
    digest = hashlib.sha256(source.encode()).hexdigest()
    return namespace[upstream.__name__], digest
