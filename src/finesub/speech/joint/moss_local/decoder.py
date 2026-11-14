"""Open MOSS-Transcribe-Diarize decoded in-process by a static-KV, CUDA-graph step instead of `model.generate`, with
token-level control (rewind, forced tokens, logits rules) for the rescue.

Why: a `generate` step is CPU-bound -- 51-63 ms per token at 16-30 k-token prompts on an RTX 5070 Ti, the GPU at about
9%: launch overhead of 28 small layers under Python. Here transformers runs the prefill (audio encoder + prompt), each
layer's K/V go straight into fixed buffers, and each decode step is one CUDA-graph replay of the Qwen3 decoder written
out by hand: transformers' projections and MLP, with attention, RMSNorm and RoPE as fused Triton kernels
(`kernels.py`, rounding to bf16 where transformers' Qwen3 does). Attention reads only the first `bucket` positions (a
multiple of BUCKET), one graph per bucket. Without Triton, when it fails to compile, or with MOSS_FAST_KERNELS=0, the
step is the same decoder in plain PyTorch ops (about half as fast at 24-40 k K/V).

Compute capability 8.0+ only (`device.moss_local_unusable_reason`): below it bf16 has no tensor cores and the
prefill's bf16 SDPA falls back to the math kernel, whose score matrix does not fit.

Rewinding is free: the K/V of an accepted prefix is already in the buffers, so continuing from token k means setting
the position back -- the rescue (`rescue.py`) relies on it.

Everything here works within a VRAM budget the caller passes in (`budget_gib`: what PyTorch's allocator may hold);
the caller also caps the allocator at it (`set_per_process_memory_fraction`).
"""

from __future__ import annotations

import contextlib
import functools
import os
from typing import Iterator

import numpy as np
import torch

from ....reporting import current_reporter
from ...runtime import hf_weights

MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
#: Its entry in `finesub_bootstrap/model-manifest.json` (revision pinned, every file's digest).
MANIFEST_ID = "moss-transcribe-diarize"
SR = 16000
BUCKET = 2048
#: VRAM the process uses outside PyTorch's allocator (CUDA context, cuBLAS workspace, Triton modules): 0.27-0.29 GiB
#: after a decode on the RTX 5070 Ti under Windows. A tier's figure is what the whole process may use
#: (`resources.usable_gib`), so the allocator gets the rest.
CONTEXT_GIB = 0.3
#: One-forward prefill peak above weights + buffers: fixed + per prompt token (0.65-0.91 GiB at 23.8 k tokens,
#: 1.31-1.35 at 47.7 k; the peak is in attention, chunking the MLP did not lower it).
PREFILL_FIXED_GIB = 0.35
PREFILL_PER_TOKEN = 24_000
#: Reserved runs 0.25-0.5 GiB above allocated on Windows (no expandable segments); a second pass in the same process
#: went out of memory at 0.3.
MARGIN_GIB = 0.5
#: Prompt tokens per second of audio (50 048 for an hour), and the prompt's text around them.
AUDIO_TOKENS_PER_SECOND = 13.9
PROMPT_TEXT_TOKENS = 100
#: Upstream's prompt and its English recipe (`moss_transcribe_diarize.inference_utils.DEFAULT_PROMPT` and
#: `examples/prompts.md`, github.com/OpenMOSS/MOSS-Transcribe-Diarize, Apache-2.0). Other languages get the default:
#: MOSS has no language parameter, and under the Chinese prompt an English talk came out in German.
DEFAULT_PROMPT = ("请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
                  "并在段末标注结束时间戳，以清晰标明该段语音范围。")
PROMPTS = {
    "en": "Transcribe the audio. For each segment, start with the timestamp and speaker ID ([S01], [S02], [S03], ...), "
          "then the spoken text, and end with the segment timestamp.",
}
#: MOSS_FAST_KV forces the K/V format (bf16 | fp8), to test the path the card would not pick.
KV_FORMATS = ("bf16", "fp8")


def prompt_for(language: str | None) -> str:
    return PROMPTS.get((language or "").lower(), DEFAULT_PROMPT)


def load(plan: hf_weights.HfLoad):
    """Model and processor on the GPU in bf16, at the manifest's revision."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    def fetch(p: hf_weights.HfLoad):
        kw = {"revision": p.revision, "local_files_only": p.local_files_only, "trust_remote_code": True}
        processor = AutoProcessor.from_pretrained(MODEL, **kw)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", **kw)
        return model.to("cuda").eval(), processor

    return hf_weights.offline_first(fetch, plan, what="moss-local")


def kv_row_bytes(model) -> dict[str, int]:
    """Bytes one K/V position takes over all layers, per format."""
    cfg = model.model.language_model.config
    row = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads
    return {"bf16": row * cfg.head_dim * 2, "fp8": row * (cfg.head_dim + 4)}


def fits(model, prompt_tokens: int, budget_gib: float) -> dict[str, int]:
    """K/V positions each format fits beside what is allocated now (the weights) and the prefill's peak."""
    cfg = model.model.language_model.config
    free = ((budget_gib - MARGIN_GIB - PREFILL_FIXED_GIB) * 2**30 - torch.cuda.memory_allocated()
            - prompt_tokens * PREFILL_PER_TOKEN)
    return {kv: max(0, min(cfg.max_position_embeddings, int(free // b) // BUCKET * BUCKET))
            for kv, b in kv_row_bytes(model).items()}


def plan_kv(model, prompt_tokens: int, want: int, budget_gib: float) -> tuple[str, int]:
    """How to store K/V for a pass wanting `want` positions (prompt + output), and how many fit: bf16 when they all
    fit, else fp8 (about twice the positions), capped at the model's context. bf16 first because 8-bit K/V is not
    only noise: int8 flipped an English talk into Spanish; fp8 with a one-forward prefill flipped nothing on the 11
    test clips, its divergence the size of the fused kernels'."""
    room = fits(model, prompt_tokens, budget_gib)
    kv = _forced_kv() or ("bf16" if room["bf16"] >= want or not fp8_kernels() else "fp8")
    return kv, room[kv]


def longest_pass(model, out_per_second: float, budget_gib: float, text_tokens: int = PROMPT_TEXT_TOKENS) -> float:
    """Seconds of audio one pass holds within the budget, in whole minutes: its prompt and `out_per_second` output
    tokens (+ 512) in the most compact K/V format the card runs. A longer source is cut into parts."""
    kv = _forced_kv() or ("fp8" if fp8_kernels() else "bf16")
    for minutes in range(90, 0, -1):  # upstream: up to 90 minutes in one pass
        prompt = int(AUDIO_TOKENS_PER_SECOND * 60 * minutes) + text_tokens
        if fits(model, prompt, budget_gib)[kv] >= prompt + out_per_second * 60 * minutes + 512:
            return 60.0 * minutes
    raise RuntimeError(f"the VRAM budget ({budget_gib:.2f} GiB for PyTorch) holds no minute of audio")


def _forced_kv() -> str | None:
    forced = os.environ.get("MOSS_FAST_KV")
    if not forced:
        return None
    if forced not in KV_FORMATS:
        raise ValueError(f"MOSS_FAST_KV is {forced!r}; use one of {', '.join(KV_FORMATS)}")
    return forced if forced == "bf16" or fp8_kernels() else "bf16"


def encode_inputs(processor, audio: np.ndarray, prompt: str):
    """Prompt tensors for one 16 kHz float array, on the GPU. The message is upstream's
    (`build_transcription_messages`); the audio path in it is not read, the array is."""
    messages = [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                             {"type": "text", "text": prompt.strip() or DEFAULT_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return processor(text=text, audio=[audio], max_length=131072, audio_kwargs={"device": "cuda"},
                         return_tensors="pt").to("cuda")


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dt = x.dtype
    x = x.float()
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(dt)


def _rot(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _kernels():
    try:
        from . import kernels
    except ImportError:  # no Triton
        return None
    return kernels


@functools.cache
def fused_kernels() -> bool:
    """Whether the decode step can use `kernels` here: Triton present, not switched off, and a small kernel compiles
    and runs (a broken Triton install falls back with a warning instead of failing)."""
    K = _kernels()
    if K is None or os.environ.get("MOSS_FAST_KERNELS", "1") == "0":
        return False
    try:
        x = torch.ones(1, 1, 64, dtype=torch.bfloat16, device="cuda")
        K.add_rms(x, x, x[0, 0], 1e-6)
        _probe("bf16")
    except Exception as e:  # noqa: BLE001 -- any compile or launch failure means: use the plain step
        current_reporter().warning("moss-local-kernels", f"MOSS fused kernels unavailable ({type(e).__name__}: {e}); "
                                   "using the plain PyTorch step", impact="本地 MOSS 解码约慢一倍")
        return False
    return True


@functools.cache
def fp8_kernels() -> bool:
    """Whether the fused kernels store and read fp8 here (Triton converts e4m3 in software below compute capability
    8.9; a Triton that cannot falls back to bf16 K/V with a warning)."""
    if not fused_kernels():
        return False
    try:
        _probe("fp8")
    except Exception as e:  # noqa: BLE001 -- any compile or launch failure means: keep K/V in bf16
        current_reporter().warning("moss-local-fp8", f"fp8 K/V unavailable ({type(e).__name__}: {e}); K/V stay bf16",
                                   impact="一趟能转的音频变短，长源多切几段")
        return False
    return True


def _probe(kv: str) -> None:
    """Write one position into a `kv` cache and attend over it: both kernels that touch the cache compile and run
    (a failure here falls back; the same failure at graph capture would not)."""
    K = _kernels()
    cache = KVCache(1, 1, BUCKET, 128, kv)
    x = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    q = torch.empty(2, 128, dtype=torch.bfloat16, device="cuda")  # two query heads per K/V head, as MOSS's decoder
    pos = torch.zeros(1, dtype=torch.long, device="cuda")
    K.qk_rope_store(x.repeat(2), x, x, x, x, torch.ones(64, device="cuda"), pos, q, cache, 0, 1e-6)
    K.DecodeAttention(1, 2, 128)(q, cache, 0, pos, BUCKET, 1.0)
    torch.cuda.synchronize()


class KVCache:
    """Per-layer K/V buffers `[1, KVH, MAX, D]`: bf16, or fp8 (e4m3) with one fp32 scale per (head, position) -- the
    row's absmax over 448."""

    QMAX = 448.0
    DTYPE = {"bf16": torch.bfloat16, "fp8": torch.float8_e4m3fn}

    def __init__(self, layers: int, kv_heads: int, max_pos: int, head_dim: int, kv: str):
        self.kvh, self.kv = kv_heads, kv
        self.quant = kv != "bf16"
        shape = (1, kv_heads, max_pos, head_dim)
        self.k = [torch.zeros(shape, dtype=self.DTYPE[kv], device="cuda") for _ in range(layers)]
        self.v = [torch.zeros(shape, dtype=self.DTYPE[kv], device="cuda") for _ in range(layers)]
        if self.quant:
            self.ks = [torch.zeros(shape[:3], device="cuda") for _ in range(layers)]
            self.vs = [torch.zeros(shape[:3], device="cuda") for _ in range(layers)]

    def layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """K, V and their scales (bf16: K and V again, never read)."""
        return (self.k[i], self.v[i], self.ks[i], self.vs[i]) if self.quant else (self.k[i], self.v[i], self.k[i], self.v[i])

    def fill(self, i: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Prefill K, V `[1, KVH, n, D]` (bf16) into positions 0..n-1 of layer i."""
        n = k.shape[2]
        if not self.quant:
            self.k[i][:, :, :n], self.v[i][:, :, :n] = k, v
            return
        for x, buf, sbuf in ((k, self.k[i], self.ks[i]), (v, self.v[i], self.vs[i])):
            xf = x.float()
            sc = xf.abs().amax(-1).clamp_min(1e-8) / self.QMAX
            buf[:, :, :n] = (xf / sc[..., None]).to(torch.float8_e4m3fn)
            sbuf[:, :, :n] = sc


def _prompt_embeds(model, inputs, batch: int = 16) -> torch.Tensor:
    """The language model's input embeddings for all prompt tokens but the last: the model's own audio path
    (`get_audio_features` + `inject_audio_features`), but the Whisper encoder runs `batch` 30 s chunks at a time
    instead of all at once (an hour of audio is 120 chunks; all at once peaks at several GiB)."""
    m = model.model
    feats, lens = inputs["input_features"], inputs["audio_feature_lengths"].tolist()
    if int(inputs["audio_chunk_mapping"].max()) != 0:
        raise ValueError("one audio per prompt")
    enc = m.whisper_encoder
    dt = next(enc.parameters()).dtype
    parts = []
    for b in range(0, feats.shape[0], batch):
        out = enc(feats[b:b + batch].to(dt), return_dict=True).last_hidden_state
        parts += [out[j:j + 1, :lens[b + j] * 4] for j in range(out.shape[0])]
    audio = m.vq_adaptor(m.time_merge(torch.cat(parts, dim=1).to(m.dtype))).squeeze(0)
    ids = inputs["input_ids"][:, :-1]
    emb = m.get_input_embeddings()(ids)
    mask = (ids == m.config.audio_token_id)[..., None].expand_as(emb)
    return emb.masked_scatter(mask, audio.to(emb.dtype))


@functools.cache
def _into_buffers():
    """A prefill cache layer that writes into the decoder's buffers and keeps no copy: transformers' own cache would
    hold all of them in bf16 at once (5 GiB at an hour of audio)."""
    from transformers.cache_utils import DynamicLayer

    class IntoBuffers(DynamicLayer):
        def __init__(self, cache: KVCache, i: int):
            super().__init__()
            self.cache, self.i, self.seen = cache, i, 0

        def get_seq_length(self) -> int:
            return self.seen

        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
            if self.seen:
                raise RuntimeError("the prompt is prefilled in one forward")
            self.cache.fill(self.i, key_states, value_states)
            self.seen = key_states.shape[2]
            return key_states, value_states

    return IntoBuffers


def check_transformers() -> None:
    """Fail with a sentence, not a traceback, when transformers is missing or its internals moved: the prefill
    writes through `Cache` / `DynamicLayer` and swaps `use_gqa_in_sdpa` (verified on transformers 5.14.1)."""
    try:
        import transformers
    except ImportError as e:
        raise RuntimeError('--asr moss-local needs the [asr] extra: pip install -e ".[asr]"') from e
    try:
        import transformers.integrations.sdpa_attention as hf_sdpa
        from transformers.cache_utils import Cache, DynamicLayer  # noqa: F401

        hf_sdpa.use_gqa_in_sdpa  # noqa: B018
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"--asr moss-local does not run on transformers {transformers.__version__}: {e} "
                           "(verified on 5.14.1)") from e


@contextlib.contextmanager
def _expanded_kv_sdpa() -> Iterator[None]:
    """SDPA with K/V expanded to the query heads instead of `enable_gqa`, so the memory-efficient kernel stays
    eligible for the long prompt (with `enable_gqa` the prefill's peak did not fit). transformers asks
    `use_gqa_in_sdpa` at every call; it is swapped only while this prefill runs."""
    import transformers.integrations.sdpa_attention as hf_sdpa

    original = hf_sdpa.use_gqa_in_sdpa
    hf_sdpa.use_gqa_in_sdpa = lambda *a, **k: False
    try:
        yield
    finally:
        hf_sdpa.use_gqa_in_sdpa = original


class Decoder:
    """Greedy-capable single-sequence decoder over one prompt. `logits(token)` appends `token` at `self.pos` and
    returns the next-token logits (float32, on the GPU); `self.pos = k` rewinds."""

    def __init__(self, model, inputs, budget: int, budget_gib: float):
        from transformers.cache_utils import Cache

        lm = model.model.language_model
        cfg = lm.config
        self.model, self.lm, self.eps = model, lm, cfg.rms_norm_eps
        self.h, self.kvh, self.d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.layers = list(lm.layers)
        self.inv_freq = lm.rotary_emb.inv_freq.float()
        self.scaling = self.d ** -0.5
        n = inputs["input_ids"].shape[1]
        kv, room = plan_kv(model, n, n + budget, budget_gib)
        self.max = min(room, -(-(n + budget) // BUCKET) * BUCKET)
        if self.max - n < 256:
            raise RuntimeError(f"prompt {n} tokens leaves no room in the VRAM budget ({room} K/V positions)")
        self._fused = fused_kernels()
        # Prefill all prompt tokens but the last; the last goes through the graph like every later token, so the
        # first generated token comes from the same code path as the rest. The audio is embedded before the buffers
        # exist (the two peaks do not stack).
        with _expanded_kv_sdpa(), torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb = _prompt_embeds(model, inputs)
            torch.cuda.empty_cache()
            self.cache = KVCache(len(self.layers), self.kvh, self.max, self.d, kv)
            past = Cache(layers=[_into_buffers()(self.cache, i) for i in range(len(self.layers))])
            lm(inputs_embeds=emb, attention_mask=inputs["attention_mask"][:, :emb.shape[1]], use_cache=True,
               past_key_values=past)
        del emb, past
        torch.cuda.empty_cache()
        self.prompt = inputs["input_ids"][0].tolist()
        self.pos = n - 1  # position of the next token to feed
        self._tok = torch.zeros(1, dtype=torch.long, device="cuda")
        self._p = torch.zeros(1, dtype=torch.long, device="cuda")
        if self._fused:
            self._attn = _kernels().DecodeAttention(self.kvh, self.h // self.kvh, self.d)
            self._q = torch.empty(self.h, self.d, dtype=torch.bfloat16, device="cuda")
        self._graphs: dict[int, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self._pool = None

    def _step(self, bucket: int) -> torch.Tensor:
        return self._step_fused(bucket) if self._fused else self._step_plain(bucket)

    @torch.inference_mode()
    def _step_plain(self, bucket: int) -> torch.Tensor:
        x = self.lm.embed_tokens(self._tok)[None]  # [1, 1, hidden]
        freqs = self._p.float()[:, None] * self.inv_freq[None]  # [1, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype)[None, None], emb.sin().to(x.dtype)[None, None]  # [1, 1, 1, D]
        keep = (torch.arange(bucket, device="cuda") <= self._p)[None, None, None]  # [1, 1, 1, bucket]
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            r = x
            y = _rms(x, layer.input_layernorm.weight, self.eps)
            q = _rms(a.q_proj(y).view(1, 1, self.h, self.d), a.q_norm.weight, self.eps).transpose(1, 2)
            k = _rms(a.k_proj(y).view(1, 1, self.kvh, self.d), a.k_norm.weight, self.eps).transpose(1, 2)
            v = a.v_proj(y).view(1, 1, self.kvh, self.d).transpose(1, 2)
            q, k = q * cos + _rot(q) * sin, k * cos + _rot(k) * sin
            self.cache.k[i].index_copy_(2, self._p, k)
            self.cache.v[i].index_copy_(2, self._p, v)
            kk, vv = self.cache.k[i][:, :, :bucket], self.cache.v[i][:, :, :bucket]
            s = torch.matmul(q.reshape(1, self.kvh, self.h // self.kvh, self.d), kk.transpose(-1, -2)).float()
            s = (s * self.scaling).masked_fill(~keep, float("-inf"))
            o = torch.matmul(torch.softmax(s, dim=-1).to(vv.dtype), vv)  # [1, KVH, g, D]
            x = r + a.o_proj(o.reshape(1, 1, self.h * self.d))
            r = x
            x = r + layer.mlp(_rms(x, layer.post_attention_layernorm.weight, self.eps))
        x = _rms(x, self.lm.norm.weight, self.eps)
        return self.model.lm_head(x)[0, 0].float()

    @torch.inference_mode()
    def _step_fused(self, bucket: int) -> torch.Tensor:
        K = _kernels()
        x, delta = None, self.lm.embed_tokens(self._tok)[None]  # residual stream and what the last block adds to it
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            x, y = K.add_rms(delta, x, layer.input_layernorm.weight, self.eps)
            K.qk_rope_store(a.q_proj(y), a.k_proj(y), a.v_proj(y), a.q_norm.weight, a.k_norm.weight, self.inv_freq,
                            self._p, self._q, self.cache, i, self.eps)
            x, y = K.add_rms(a.o_proj(self._attn(self._q, self.cache, i, self._p, bucket, self.scaling)), x,
                             layer.post_attention_layernorm.weight, self.eps)
            delta = layer.mlp(y)
        _, y = K.add_rms(delta, x, self.lm.norm.weight, self.eps)
        return self.model.lm_head(y)[0, 0].float()

    def logits(self, token: int) -> torch.Tensor:
        if self.pos >= self.max:
            raise RuntimeError(f"KV buffer full at {self.pos}")
        bucket = min(self.max, -(-(self.pos + 1) // BUCKET) * BUCKET)
        self._tok.fill_(token)
        self._p.fill_(self.pos)
        if bucket not in self._graphs:
            with torch.amp.autocast("cuda", enabled=False):
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        self._step(bucket)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self._pool):
                    out = self._step(bucket)
                self._pool = g.pool()
            self._graphs[bucket] = (g, out)
        g, out = self._graphs[bucket]
        g.replay()
        self.pos += 1
        return out
