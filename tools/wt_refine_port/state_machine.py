"""Pure WT 1.15.9 decoder flush, repair, and confidence contracts.

The production implementation will consume events exported by CTranslate2.
Keeping the policy free of Torch and CT2 types makes the behavioral boundary
testable before the generator API is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class AttentionRows(str, Enum):
    """Which collected decoder queries WT passes to word alignment."""

    ALL = "all"
    DROP_LAST = "drop_last"


@dataclass(frozen=True)
class FlushDecision:
    flush: bool
    discard: bool = False


@dataclass
class FlushTracker:
    """Mirror WT's ``must_flush_segment`` consecutive-timestamp state."""

    timestamp_begin: int
    sot: int
    trust_whisper_timestamps: bool = True
    modern_whisper: bool = True
    saw_consecutive_timestamps: bool = False

    def observe(
        self,
        current_tokens: Sequence[int] | None,
        collected_segment_tokens: Sequence[int],
        *,
        last_chunk_token: int | None = None,
    ) -> FlushDecision:
        is_sot = (
            current_tokens is None
            or len(current_tokens) > 1
            or current_tokens[0] == self.sot
        )
        if not is_sot:
            current_is_timestamp = current_tokens[0] >= self.timestamp_begin
            previous_is_timestamp = bool(collected_segment_tokens) and (
                collected_segment_tokens[-1] >= self.timestamp_begin
            )
            consecutive = current_is_timestamp and previous_is_timestamp
            self.saw_consecutive_timestamps |= consecutive
            return FlushDecision(flush=consecutive)

        must_flush = (
            len(collected_segment_tokens) > 1
            and not self.saw_consecutive_timestamps
        )
        if not must_flush and self.modern_whisper:
            if last_chunk_token is None:
                must_flush = (
                    len(collected_segment_tokens) > 2
                    and collected_segment_tokens[-1] >= self.timestamp_begin
                )
            else:
                must_flush = last_chunk_token >= self.timestamp_begin
        discard = not must_flush and self.trust_whisper_timestamps
        self.saw_consecutive_timestamps = False
        return FlushDecision(flush=must_flush, discard=discard)


def reached_decoding_limit(
    *,
    chunk_tokens_without_sot: int,
    initial_prompt_tokens: int,
    max_sample_len: int,
    text_context: int,
) -> bool:
    """Mirror WT's token and context-limit test at the current decoder step."""

    next_length = chunk_tokens_without_sot + 1
    context_length = next_length + initial_prompt_tokens
    return next_length + 1 >= max_sample_len or context_length > text_context


@dataclass(frozen=True)
class RepairPlan:
    """Tokens and query selection produced by WT's ``align_last_segment``."""

    alignment_tokens: tuple[int, ...]
    recorded_segment_tokens: tuple[int, ...]
    attention_rows: AttentionRows
    last_logprobs_index: int
    unfinished_decoding: bool
    last_token_reliable: bool
    appended_token: int | None = None
    reestimated_end_token: int | None = None


def _argmax_after(values: Sequence[float], start: int) -> int:
    first = start + 1
    if first >= len(values):
        raise ValueError("no token is available after the start timestamp")
    return first + max(range(len(values) - first), key=lambda index: values[first + index])


def plan_alignment_repair(
    collected_segment_tokens: Sequence[int],
    *,
    timestamp_begin: int,
    eot: int,
    unfinished_decoding: bool,
    last_logprobs: Sequence[float],
    fallback_token: int | None = None,
    fallback_reliable: bool = True,
) -> RepairPlan:
    """Plan the exact token repair and attention-row policy used by WT 1.15.9.

    ``collected_segment_tokens[0]`` is the decoder query retained before the
    segment start and is intentionally excluded from alignment. WT appends a
    fallback/EOT to its recorded segment, but a constrained end-timestamp
    re-estimate only changes the temporary alignment token list. Both views are
    therefore returned explicitly.
    """

    if not collected_segment_tokens:
        raise ValueError("collected_segment_tokens must include the retained query")
    tokens = [int(token) for token in collected_segment_tokens[1:]]
    recorded = [int(token) for token in collected_segment_tokens]
    if not tokens:
        raise ValueError("segment has no token after the retained query")

    appended_token: int | None = None
    if unfinished_decoding:
        if fallback_token is None:
            raise ValueError("unfinished decoding requires a fallback token")
        appended_token = int(fallback_token)
        tokens.append(appended_token)
        recorded.append(appended_token)
        attention_rows = AttentionRows.ALL
        last_logprobs_index = -1
        last_token_reliable = fallback_reliable
    elif tokens[-1] < timestamp_begin:
        appended_token = eot
        tokens.append(eot)
        recorded.append(eot)
        attention_rows = AttentionRows.ALL
        last_logprobs_index = -1
        last_token_reliable = True
    else:
        attention_rows = AttentionRows.DROP_LAST
        last_logprobs_index = -2
        last_token_reliable = True

    reestimated_end_token: int | None = None
    if tokens[-1] >= timestamp_begin:
        start_token = tokens[0]
        if start_token < timestamp_begin:
            raise ValueError("timestamp-delimited segment is missing its start timestamp")
        if tokens[-1] <= start_token:
            reestimated_end_token = _argmax_after(last_logprobs, start_token)
            tokens[-1] = reestimated_end_token

    return RepairPlan(
        alignment_tokens=tuple(tokens),
        recorded_segment_tokens=tuple(recorded),
        attention_rows=attention_rows,
        last_logprobs_index=last_logprobs_index,
        unfinished_decoding=unfinished_decoding,
        last_token_reliable=last_token_reliable,
        appended_token=appended_token,
        reestimated_end_token=reestimated_end_token,
    )


@dataclass(frozen=True)
class ConfidencePartition:
    average_logprob: float
    segment_logprobs: tuple[tuple[float, ...], ...]


def partition_confidence(
    *,
    chosen_tokens: Sequence[int],
    chosen_logprobs: Sequence[float],
    segment_tokens: Sequence[Sequence[int]],
    unfinished_last_segment: bool = False,
    average_denominator: int | None = None,
) -> ConfidencePartition:
    """Apply WT's segment slicing after per-step chosen logprobs are known.

    The caller supplies the final token stream WT constructs for the 30-second
    chunk, including its fallback or EOT. Start timestamps are excluded from
    every segment; end timestamps are excluded except for an unfinished final
    segment, where the appended fallback remains part of confidence.
    """

    indices = tuple(int(token) for token in chosen_tokens)
    values = tuple(float(value) for value in chosen_logprobs)
    if len(indices) != len(values):
        raise ValueError("chosen token and logprob counts differ")
    denominator = len(values) if average_denominator is None else average_denominator
    if denominator <= 0:
        raise ValueError("average denominator must be positive")

    partitions: list[tuple[float, ...]] = []
    cursor = 0
    for index, raw_segment in enumerate(segment_tokens):
        segment = tuple(int(token) for token in raw_segment)
        end = cursor + len(segment)
        if indices[cursor:end] != segment:
            raise ValueError("segment tokens do not match the chosen decoder stream")
        value_start = cursor + 1
        value_end = end
        if not unfinished_last_segment or index != len(segment_tokens) - 1:
            value_end -= 1
        partitions.append(values[value_start:value_end])
        cursor = end
    if cursor != len(indices):
        raise ValueError("chosen decoder stream has unassigned tokens")

    return ConfidencePartition(
        average_logprob=sum(values) / denominator,
        segment_logprobs=tuple(partitions),
    )
