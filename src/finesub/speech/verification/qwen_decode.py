"""Fixed-shape greedy decode for the Qwen3-ASR referee.

Why this exists (docs/bench-baselines.md 21.6): the referee's decode step is
launch-bound (~5 ms of GPU work in a ~60 ms step), so compiling the step into
a CUDA graph is the lever -- but transformers' own static-cache compile path
grows the 2-D attention mask by one column per token, and under
`reduce-overhead` every distinct mask length records its own graph. A full
re-check of one file meets hundreds of lengths, runs no faster than eager and
holds ~11 GiB of graph pools.

Here every tensor the decode step sees has one shape per (batch size, cache
length): the token `(B, 1)`, the padding mask `(B, L)`, the position `(B, 1)`,
the cache position `(1,)` and the static KV cache `(B, L)`, with `L` 512 when
the prompt and budget fit and the full length otherwise. The 4-D causal mask
is built inside the step from the cache position, so nothing grows. One graph
per shape, recorded once per process; the prefill (audio encoder + prompt)
stays eager, as it does in `generate`, but encodes the audio one clip at a
time -- the batched conv front-end was the referee's largest transient.

Semantics mirror `GenerationMixin.generate` for greedy decoding as this model
configures it: argmax, stop when every row has produced an EOS, `pad_token_id`
after a row's EOS, and positions counted per row from its first real token
(`generate` derives them from the padding mask -- `attention_mask.cumsum() - 1`
-- so a short row in a left-padded batch starts at 0, not at the pad width;
the cache slot is the global `cache_position`, the RoPE position is the
row's own). Output layout is `generate`'s: prompt followed by the new tokens.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch

__all__ = ["FixedShapeDecoder"]


class FixedShapeDecoder:
    """Greedy decoder whose per-step shapes depend only on the batch size."""

    def __init__(
        self,
        model,
        *,
        max_cache_len: int,
        compile_step: bool,
    ) -> None:
        self._model = model
        self._max = int(max_cache_len)
        self._compile = bool(compile_step)
        self._caches: Dict[tuple, object] = {}
        self._steps: Dict[tuple, Callable] = {}
        # The uncompiled step per batch size: Dynamo caches compiled code on
        # the function's code object, and those entries keep the closure (the
        # model and the cache) alive until they are reset -- see `close`.
        self._inner: Dict[tuple, Callable] = {}

    @property
    def compiled_shapes(self) -> set[tuple[int, int]]:
        """`(batch, cache_len)` pairs with a step graph built in this process.

        Both halves matter: a graph for eight rows over a 512-slot cache says
        nothing about eight rows over 1024, and a gate that keyed on the batch
        size alone would call the second warm and skip its own threshold.
        """

        return set(self._steps)

    def fits(self, prompt_len: int, max_new_tokens: int) -> bool:
        return prompt_len + max_new_tokens <= self._max

    # A shorter cache for the common case: with the batch cap of 120 s of
    # padded audio and ~13 prompt tokens per second, a compiled batch's prompt
    # plus the token budget fits in 512, at half the KV memory of `max`. A
    # lone long clip still gets the full length. Each (batch, length) pair is
    # its own graph.
    SHORT_CACHE_LEN = 512

    @classmethod
    def cache_len_for(cls, prompt_len: int, max_new_tokens: int, max_cache_len: int) -> int:
        """The static cache length a prompt takes; the referee's gate uses it
        to name a batch's shape before the decoder exists."""

        need = prompt_len + max_new_tokens
        return cls.SHORT_CACHE_LEN if need <= cls.SHORT_CACHE_LEN < max_cache_len else max_cache_len

    def _cache_len_for(self, prompt_len: int, max_new_tokens: int) -> int:
        return self.cache_len_for(prompt_len, max_new_tokens, self._max)

    def _cache_for(self, batch: int, cache_len: int):
        cache = self._caches.get((batch, cache_len))
        if cache is None:
            from transformers import StaticCache

            cache = StaticCache(config=self._model.config, max_cache_len=cache_len)
            self._caches[(batch, cache_len)] = cache
        else:
            cache.reset()
        return cache

    def _prefill_embeddings(self, inputs):
        """Prompt embeddings with the audio features scattered in, the way
        `Qwen3ASRModel.forward` does it -- except that the encoder sees one
        clip at a time (see `generate`)."""

        inner = self._model.model
        input_ids = inputs["input_ids"]
        inputs_embeds = inner.get_input_embeddings()(input_ids)
        features = inputs.get("input_features")
        if features is None:
            return inputs_embeds
        mask = inputs.get("input_features_mask")
        parts = [
            inner.get_audio_features(
                features[i : i + 1], mask[i : i + 1], return_dict=True
            ).pooler_output
            for i in range(features.shape[0])
        ]
        audio = torch.cat(parts, dim=0)
        placeholder = inner.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, audio_features=audio
        )
        return inputs_embeds.masked_scatter(placeholder, audio.to(inputs_embeds.dtype))

    def _step_for(self, batch: int, cache_len: int, cache) -> Callable:
        key = (batch, cache_len)
        step = self._steps.get(key)
        if step is not None:
            return step
        model = self._model
        max_len = cache_len

        def step(token, padding_mask, position_ids, cache_position):
            # 4-D boolean mask, True = attend: the row's padding mask AND the
            # causal constraint against the cache position. Fixed (B, 1, 1, MAX).
            allowed = padding_mask[:, None, None, :] & (
                torch.arange(max_len, device=token.device)[None, None, None, :]
                <= cache_position[0]
            )
            out = model(
                input_ids=token,
                attention_mask={"full_attention": allowed},
                position_ids=position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1,
            )
            return out.logits[:, -1, :]

        self._inner[key] = step
        if self._compile:
            step = torch.compile(step, mode="reduce-overhead", fullgraph=False)
        self._steps[key] = step
        return step

    @torch.no_grad()
    def generate(
        self,
        inputs,
        *,
        max_new_tokens: int,
        eos_token_ids: Sequence[int],
        pad_token_id: int,
    ) -> torch.LongTensor:
        model = self._model
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch, prompt_len = input_ids.shape
        device = input_ids.device
        if not self.fits(prompt_len, max_new_tokens):
            raise ValueError(
                f"prompt {prompt_len} + {max_new_tokens} new tokens exceeds the "
                f"{self._max} static cache"
            )
        cache_len = self._cache_len_for(prompt_len, max_new_tokens)
        cache = self._cache_for(batch, cache_len)

        # Prefill, eager. The audio encoder runs one clip at a time: batched,
        # its conv front-end is the single largest transient of the whole
        # referee (+0.87 GiB for eight 15 s clips, +1.73 GiB for sixteen,
        # against +0.12 GiB clip by clip at the same speed). The merged
        # embeddings then go through the language model in one pass, filling
        # the static cache. Positions are per row, the way `generate` derives
        # them from the padding mask: a left-padded row counts from its first
        # real token (pad slots get 0 and are masked anyway).
        inputs_embeds = self._prefill_embeddings(inputs)
        prefill_slots = torch.arange(prompt_len, device=device)
        mask_long = attention_mask.to(torch.long)
        prefill_positions = (mask_long.cumsum(-1) - 1).masked_fill(mask_long == 0, 0)
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=prefill_positions,
            past_key_values=cache,
            cache_position=prefill_slots,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = out.logits[:, -1, :].argmax(dim=-1)

        # Fixed-shape step buffers. The padding mask covers the whole cache:
        # prompt padding stays 0, everything else is 1 and the causal term in
        # the step decides what is visible. The step position is `(B, 1)` --
        # each row continues its own count -- while the cache slot is shared.
        padding_mask = torch.ones(batch, cache_len, dtype=torch.bool, device=device)
        padding_mask[:, :prompt_len] = attention_mask.to(torch.bool)
        position_ids = mask_long.sum(dim=-1, keepdim=True)
        cache_position = torch.tensor([prompt_len], dtype=torch.long, device=device)
        token = torch.empty(batch, 1, dtype=torch.long, device=device)

        eos = torch.tensor(list(eos_token_ids), dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        generated = torch.full(
            (batch, max_new_tokens), pad_token_id, dtype=torch.long, device=device
        )
        step = self._step_for(batch, cache_len, cache)
        produced = 0
        for index in range(max_new_tokens):
            next_token = torch.where(finished, pad_token_id, next_token)
            generated[:, index] = next_token
            produced = index + 1
            finished |= torch.isin(next_token, eos)
            if produced == max_new_tokens or bool(finished.all()):
                break
            token.copy_(next_token[:, None])
            logits = step(token, padding_mask, position_ids, cache_position)
            next_token = logits.argmax(dim=-1)
            position_ids += 1
            cache_position += 1
        # Same width as `generate` would return: up to the longest row.
        return torch.cat([input_ids, generated[:, :produced]], dim=1)

    def close(self) -> None:
        """Drop the caches and everything that would keep them alive.

        Two things outlive a plain `del`: Dynamo's per-code-object cache of
        the compiled step (its guards reference the closure, i.e. the model
        and the static cache), and the CUDA-graph pools recorded for it.
        Both are reset explicitly. The graph-tree reset is global, which is
        safe here: the separator's JIT compiles in the default mode and owns
        no graphs.
        """

        compiled = bool(self._steps) and self._compile
        self._steps.clear()
        self._caches.clear()
        inner = list(self._inner.values())
        self._inner.clear()
        if not compiled:
            return
        try:
            import torch._dynamo

            for function in inner:
                torch._dynamo.reset_code(function.__code__)
        except Exception:  # noqa: BLE001 - best effort, the process may exit soon
            pass
        try:
            from torch._inductor import cudagraph_trees

            cudagraph_trees.reset_cudagraph_trees()
        except Exception:  # noqa: BLE001
            pass
