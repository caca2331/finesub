"""Fused Triton kernels for `decoder.Decoder`'s decode step.

Why: at 24-40 k K/V positions most of a step went to the attention's weighted sum over V -- an output of 8 heads x
2 queries x 128 reduced along a 40 k-long axis, which cuBLAS cannot spread over the card -- and about 1.7 ms of every
step to RMSNorm and RoPE written as about a thousand tiny elementwise kernels. Fused, the decode takes 41% less wall
time on an RTX 5070 Ti.

- `DecodeAttention`: flash-decoding. Each (KV head, split) program walks its slice of the cache once, reading K and V
  together with an online softmax; a second kernel merges the splits. GQA without expanding K/V: one program serves
  the G query heads that share a KV head, padded to 16 rows so `tl.dot` runs on tensor cores.
- `add_rms`: residual add + RMSNorm in one kernel; `qk_rope_store`: q/k RMSNorm, RoPE and the K/V cache write in one.
  Both round to bf16 where transformers' Qwen3 does, so their output matches the unfused ops.

K/V are bf16, or fp8 (e4m3) with one fp32 scale per (head, position): the row's absmax over 448.
All kernels read the position from a device tensor, so a CUDA graph captured once per bucket stays valid as it moves.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

#: Cache positions per inner step. Its shared memory fits the 99 KB per block of every card with compute capability
#: 8.0+ (sm_80 / 86 / 89 / 90 / 120); 128 does not fit sm_86 / 89 / 120.
BLOCK = 64


def splits_for(kv_heads: int) -> int:
    """Splits per KV head: about two programs per SM (16-64, a power of two for the merge): 32 on the 5070 Ti's 70
    SMs, where 32 and 64 timed within 10% of each other."""
    from ...runtime.device import multiprocessor_count

    want = 2 * multiprocessor_count() / kv_heads
    return min(64, max(16, 1 << max(0, math.ceil(math.log2(want)))))


@triton.jit
def _partial(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, pos_ptr, m_ptr, l_ptr, acc_ptr, scale, stride_kh, stride_sh, chunk,
             G: tl.constexpr, D: tl.constexpr, BN: tl.constexpr, FP8: tl.constexpr):
    h = tl.program_id(0)
    sp = tl.program_id(1)
    nsp = tl.num_programs(1)
    pos = tl.load(pos_ptr)
    d = tl.arange(0, D)
    g = tl.arange(0, 16)
    q = tl.load(q_ptr + (h * G + g)[:, None] * D + d[None, :], mask=(g < G)[:, None], other=0.0)
    m = tl.full([16], float("-inf"), tl.float32)
    l = tl.zeros([16], tl.float32)
    acc = tl.zeros([16, D], tl.float32)
    start = sp * chunk
    for n0 in range(start, start + chunk, BN):
        n = n0 + tl.arange(0, BN)
        ok = n <= pos
        k = tl.load(k_ptr + h * stride_kh + n[:, None] * D + d[None, :], mask=ok[:, None], other=0.0)
        if FP8:  # fp8 values are exact in bf16; the per-position scale goes onto the scores
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * (tl.load(ks_ptr + h * stride_sh + n, mask=ok, other=0.0) * scale)[None, :]
        else:
            s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m - m_safe)
        v = tl.load(v_ptr + h * stride_kh + n[:, None] * D + d[None, :], mask=ok[:, None], other=0.0)
        if FP8:  # and onto the weights for V
            pv = p * tl.load(vs_ptr + h * stride_sh + n, mask=ok, other=0.0)[None, :]
            acc = acc * alpha[:, None] + tl.dot(pv.to(tl.bfloat16), v.to(tl.bfloat16))
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new
    o = (h * nsp + sp) * G + g
    tl.store(m_ptr + o, m, mask=g < G)
    tl.store(l_ptr + o, l, mask=g < G)
    tl.store(acc_ptr + o[:, None] * D + d[None, :], acc, mask=(g < G)[:, None])


@triton.jit
def _merge(m_ptr, l_ptr, acc_ptr, out_ptr, G: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
    """One program per query head: combine the S splits' partial softmax sums into the bf16 output row."""
    qh = tl.program_id(0)
    h, g = qh // G, qh % G
    sp = tl.arange(0, S)
    d = tl.arange(0, D)
    row = (h * S + sp) * G + g
    m = tl.load(m_ptr + row)
    l = tl.load(l_ptr + row)
    top = tl.max(m, axis=0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp(m - top))
    acc = tl.load(acc_ptr + row[:, None] * D + d[None, :])
    o = tl.sum(w[:, None] * acc, axis=0) / tl.sum(w * l, axis=0)
    tl.store(out_ptr + qh * D + d, o.to(tl.bfloat16))


class DecodeAttention:
    """Scratch buffers sized once; `__call__` writes `[1, 1, H * D]` bf16 (query-head-major, as `o_proj` expects)."""

    def __init__(self, kv_heads: int, group: int, head_dim: int):
        self.kvh, self.g, self.d = kv_heads, group, head_dim
        self.splits = splits_for(kv_heads)
        self.m = torch.empty(kv_heads, self.splits, group, device="cuda")
        self.l = torch.empty_like(self.m)
        self.acc = torch.empty(kv_heads, self.splits, group, head_dim, device="cuda")

    def __call__(self, q: torch.Tensor, cache, i: int, pos: torch.Tensor, bucket: int,
                 scale: float) -> torch.Tensor:
        """q: `[KVH * G, D]` bf16, contiguous; attends to layer i of `cache` (`decoder.KVCache`) at positions
        <= pos[0]."""
        k, v, ks, vs = cache.layer(i)
        chunk = -(-bucket // self.splits // BLOCK) * BLOCK
        _partial[(self.kvh, self.splits)](q, k, v, ks, vs, pos, self.m, self.l, self.acc, scale, k.stride(1),
                                          ks.stride(1), chunk, G=self.g, D=self.d, BN=BLOCK, FP8=cache.quant)
        out = torch.empty(1, 1, self.kvh * self.g * self.d, dtype=torch.bfloat16, device="cuda")
        _merge[(self.kvh * self.g,)](self.m, self.l, self.acc, out, G=self.g, D=self.d, S=self.splits)
        return out


@triton.jit
def _add_rms(d_ptr, r_ptr, w_ptr, sum_ptr, out_ptr, eps, N: tl.constexpr, HAS_RES: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(d_ptr + i).to(tl.float32)
    if HAS_RES:
        x = (x + tl.load(r_ptr + i).to(tl.float32)).to(tl.bfloat16)
        tl.store(sum_ptr + i, x)
        x = x.to(tl.float32)
    y = (x * tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + i, (tl.load(w_ptr + i).to(tl.float32) * y).to(tl.bfloat16))


def add_rms(delta: torch.Tensor, res: torch.Tensor | None, w: torch.Tensor, eps: float):
    """`(res + delta, rms(res + delta) * w)`, bf16, as transformers rounds them; with `res` None, `(delta, rms(delta) * w)`."""
    n = delta.shape[-1]
    out = torch.empty_like(delta)
    total = torch.empty_like(delta) if res is not None else delta
    _add_rms[(1,)](delta, delta if res is None else res, w, total, out, eps, N=n, HAS_RES=res is not None)
    return total, out


@triton.jit
def _store_row(c_ptr, s_ptr, x, at, sat, FP8: tl.constexpr):
    """One head's row into the cache: bf16 as is, or fp8 (e4m3) with the row's absmax / 448 as its scale."""
    if FP8:
        xf = x.to(tl.float32)
        sc = tl.maximum(tl.max(tl.abs(xf), axis=0), 1e-8) / 448.0
        tl.store(c_ptr + at, (xf / sc).to(tl.float8e4nv))
        tl.store(s_ptr + sat, sc)
    else:
        tl.store(c_ptr + at, x)


@triton.jit
def _qk_rope_store(q_ptr, k_ptr, v_ptr, qw_ptr, kw_ptr, freq_ptr, pos_ptr, q_out, kc_ptr, vc_ptr, ks_ptr, vs_ptr,
                   stride_kh, stride_sh, eps, H: tl.constexpr, KVH: tl.constexpr, D: tl.constexpr,
                   FP8: tl.constexpr):
    r = tl.program_id(0)
    pos = tl.load(pos_ptr)
    d = tl.arange(0, D)
    if r >= H + KVH:  # V: straight into the cache
        h = r - H - KVH
        _store_row(vc_ptr, vs_ptr, tl.load(v_ptr + h * D + d), h * stride_kh + pos * D + d, h * stride_sh + pos, FP8)
    else:
        if r < H:
            src, w_ptr = q_ptr + r * D, qw_ptr
        else:
            src, w_ptr = k_ptr + (r - H) * D, kw_ptr
        half: tl.constexpr = D // 2
        rd = (d + half) % D
        x = tl.load(src + d).to(tl.float32)
        xr = tl.load(src + rd).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        y = (tl.load(w_ptr + d).to(tl.float32) * (x * inv).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        yr = (tl.load(w_ptr + rd).to(tl.float32) * (xr * inv).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        yr = tl.where(d < half, -yr.to(tl.float32), yr.to(tl.float32))
        ang = pos.to(tl.float32) * tl.load(freq_ptr + d % half)
        cos = tl.cos(ang).to(tl.bfloat16).to(tl.float32)
        sin = tl.sin(ang).to(tl.bfloat16).to(tl.float32)
        a = (y.to(tl.float32) * cos).to(tl.bfloat16).to(tl.float32)
        b = (yr * sin).to(tl.bfloat16).to(tl.float32)
        o = (a + b).to(tl.bfloat16)
        if r < H:
            tl.store(q_out + r * D + d, o)
        else:
            _store_row(kc_ptr, ks_ptr, o, (r - H) * stride_kh + pos * D + d, (r - H) * stride_sh + pos, FP8)


def qk_rope_store(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qw: torch.Tensor, kw: torch.Tensor,
                  inv_freq: torch.Tensor, pos: torch.Tensor, q_out: torch.Tensor, cache, i: int,
                  eps: float) -> None:
    """Per-head RMSNorm and RoPE of the projected q `[H*D]` and k `[KVH*D]` at position pos[0]; q goes to `q_out`
    `[H, D]`, k and v into layer i's cache at that position."""
    h, d = q_out.shape
    kc, vc, ks, vs = cache.layer(i)
    _qk_rope_store[(h + 2 * cache.kvh,)](q, k, v, qw, kw, inv_freq, pos, q_out, kc, vc, ks, vs, kc.stride(1),
                                         ks.stride(1), eps, H=h, KVH=cache.kvh, D=d, FP8=cache.quant)
