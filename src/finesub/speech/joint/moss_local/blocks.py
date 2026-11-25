"""Known tokens fed to the decoder in one forward instead of one CUDA-graph step each (the forced decode, `forced`).

The forced decode knows most of its tokens in advance (a row's start time, `]`, `[S`, then after the pick the
label's digits, the text and the end time), so they go through the decoder the way the prompt does: one forward over
a block of tokens at positions pos .. pos+n-1, their K/V written into the decoder's buffers (`decoder.KVCache`, bf16
only: the forced decode asks for bf16), attention causal over the buffers. The block is `Decoder._step_plain`
widened to n tokens: transformers' projections and MLP, RMSNorm, RoPE and attention in plain PyTorch over the cache.
Numerics differ from the fused step by rounding only (an execution-shape change: the exploration's 27 episodes read
the same after the merge by voice, one bf16 tie in 11 windows flipped).

**This is a second hand-written forward beside `Decoder`'s step**: a change to one (the model, transformers'
modules, RoPE, the norms) has to be made to the other; the heavy-resource test compares their logits.

Eager, a block costs about 50 ms whatever its length (the launches of 28 layers), more than ten graph steps; so a
block is a CUDA graph too, one per (attention bucket, block size): the tokens are padded to the next of SIZES (the
padding's K/V land past the stream's end and are overwritten before anything reads them, as after a rewind).

Memory: each graph holds its intermediates (the attention scores grow with the bucket) for as long as it lives, so
blocks use buckets of SPAN positions and drop the graphs of buckets the stream has left. Every warm-up runs on one
side stream and the cache is emptied after each capture: with a new stream per capture the warm-ups' blocks stayed
cached on dead streams and a long part spilled into shared memory. Attention runs QUERIES tokens at a time, which
bounds the float scores at QUERIES x 16 heads x bucket. A graph keeps only the logits after the block's last token.
"""

from __future__ import annotations

import torch

from .decoder import BUCKET, Decoder, _rms, _rot

SIZES = (16, 32, 64, 128)
QUERIES = 16
#: a block's attention bucket: coarser than the step's BUCKET, so a 10-minute part makes few block graphs
SPAN = 4 * BUCKET


class Blocks:
    """`feed(ids)`: the ids appended at the decoder's position; -> the logits after the last (float32, on the GPU)."""

    def __init__(self, dec: Decoder):
        if dec.cache.quant:
            raise ValueError("blocks write bf16 K/V; the decoder's cache is " + dec.cache.kv)
        self.dec = dec
        self.graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.stream = torch.cuda.Stream()
        self.tok = {c: torch.zeros(c, dtype=torch.long, device="cuda") for c in SIZES}
        self.pos = {c: torch.zeros(c, dtype=torch.long, device="cuda") for c in SIZES}
        self.last = torch.zeros(1, dtype=torch.long, device="cuda")

    @torch.inference_mode()
    def _forward(self, c: int, bucket: int) -> torch.Tensor:
        d = self.dec
        h, kvh, hd = d.h, d.kvh, d.d
        g = h // kvh
        tok, pos = self.tok[c], self.pos[c]
        x = d.lm.embed_tokens(tok)[None]  # [1, c, hidden]
        freqs = pos.float()[:, None] * d.inv_freq[None]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(x.dtype)[None, None], emb.sin().to(x.dtype)[None, None]  # [1, 1, c, D]
        keep = torch.arange(bucket, device="cuda")[None] <= pos[:, None]  # [c, bucket]
        cache = d.cache
        for i, layer in enumerate(d.layers):
            a = layer.self_attn
            r = x
            y = _rms(x, layer.input_layernorm.weight, d.eps)
            q = _rms(a.q_proj(y).view(1, c, h, hd), a.q_norm.weight, d.eps).transpose(1, 2)
            k = _rms(a.k_proj(y).view(1, c, kvh, hd), a.k_norm.weight, d.eps).transpose(1, 2)
            v = a.v_proj(y).view(1, c, kvh, hd).transpose(1, 2)
            q, k = q * cos + _rot(q) * sin, k * cos + _rot(k) * sin
            cache.k[i].index_copy_(2, pos, k)
            cache.v[i].index_copy_(2, pos, v)
            kk, vv = cache.k[i][:, :, :bucket], cache.v[i][:, :, :bucket]
            q = q.view(1, kvh, g, c, hd)
            outs = []
            for j in range(0, c, QUERIES):
                qj = q[:, :, :, j:j + QUERIES].reshape(1, kvh, g * QUERIES, hd)
                s = torch.matmul(qj, kk.transpose(-1, -2)).float() * d.scaling  # [1, kvh, g*QUERIES, bucket]
                s = s.masked_fill(~keep[j:j + QUERIES].repeat(g, 1)[None, None], float("-inf"))
                outs.append(torch.matmul(torch.softmax(s, -1).to(vv.dtype), vv).view(1, kvh, g, QUERIES, hd))
            o = torch.cat(outs, 3).permute(0, 3, 1, 2, 4).reshape(1, c, h * hd)
            x = r + a.o_proj(o)
            r = x
            x = r + layer.mlp(_rms(x, layer.post_attention_layernorm.weight, d.eps))
        x = _rms(x[0].index_select(0, self.last), d.lm.norm.weight, d.eps)
        return d.model.lm_head(x)[0].float()

    def _graph(self, c: int, bucket: int) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        key = (c, bucket)
        if key not in self.graphs:
            d = self.dec
            with torch.amp.autocast("cuda", enabled=False):
                s = self.stream
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        self._forward(c, bucket)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, pool=d._pool):
                    out = self._forward(c, bucket)
                d._pool = gr.pool()
            self.graphs[key] = (gr, out)
            torch.cuda.empty_cache()
        return self.graphs[key]

    def _drop_below(self, bucket: int) -> None:
        """Free the graphs of buckets the stream has left for good (rewinds go back a row at most)."""
        old = [k for k in self.graphs if k[1] < bucket - SPAN]
        for k in old:
            del self.graphs[k]
        if old:
            torch.cuda.empty_cache()

    def feed(self, ids: list[int]) -> torch.Tensor:
        """`ids` at the decoder's position, in blocks of at most SIZES[-1]; -> the logits after the last."""
        d = self.dec
        out = None
        for at in range(0, len(ids), SIZES[-1]):
            part = ids[at:at + SIZES[-1]]
            n, p0 = len(part), d.pos
            c = next(s for s in SIZES if s >= n)
            if p0 + c > d.max:  # the padding would run past the buffer: step these tokens
                for t in part:
                    out = d.logits(t)
                continue
            bucket = min(d.max, -(-(p0 + c) // SPAN) * SPAN)
            self._drop_below(bucket)
            self.tok[c].copy_(torch.tensor(part + [0] * (c - n)))
            self.pos[c].copy_(torch.arange(p0, p0 + c))
            self.last.fill_(n - 1)
            gr, last = self._graph(c, bucket)
            gr.replay()
            out = last
            d.pos = p0 + n
        return out
