"""Three passes' pass 1 (the speaker judge's call for one unit) and pass 3 (proofreading one window).

The prompt is rebuilt with the current templates from what the run dumped beside its other artifacts
(`speaker-judge-input.json`, `proofread-input.json`) and the run's own rows, then sent through production's
call for one unit or window (`speaker_judge.run._judge_unit`, `proofread.run._ask`): its routing, parsing,
retries and agent validator, so a sample here fails exactly where production would. One sample is one such
call, retries inside it included. Pass 2 is the `correction` session: its fixture regroups a three-pass
window by pass 1's sidecar.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from finesub.llm.client import RoleClient, VALIDATION_BASE_TEMPERATURE
from finesub.llm.prompts import ContextPack
from finesub.llm.routing.config import LLMRole, role_config_for
from finesub.llm.routing.profiles import parse_profile_id
from finesub.llm.stages.proofread import run as proofread_run
from finesub.llm.stages.proofread.prompt import Line, Window, proofread_prompt
from finesub.llm.stages.speaker_judge import run as judge_run
from finesub.llm.stages.speaker_judge.prompt import Candidate, rows_of
from finesub.llm.stages.speaker_judge.windows import JudgeWindow
from finesub.speech.diarization import judgments as sidecar
from finesub.subtitles.stable_rows import read_rows

from ..fixture import resolve_run_layout
from .base import ReplaySample, _write_summary, pin_client_role_to_free_model, reject_unsupported_variant


@dataclass
class ThreePassReplayResult:
    out_dir: Path
    prompt_system_path: Path
    prompt_user_path: Path
    successes: List[ReplaySample] = field(default_factory=list)
    failures: List[ReplaySample] = field(default_factory=list)
    summary_path: Path | None = None
    dry_run: bool = False


class _DryRun(Exception):
    """Raised by a recording session in a dry run, once the prompt is in hand."""


def _dump(artifact_dir: Path, name: str, what: str) -> Dict[str, Any]:
    path = artifact_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"{path} 不存在：{what}的输入只在跑过一次生产之后才有（这一版之前的运行没有它）")
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_overrides(session: str, *, variant: str | None, force_tier: str | None, thinking_level: str | None,
                      profile: str | None, temperature: float, agentic: bool) -> None:
    reject_unsupported_variant(session, variant=variant, force_tier=force_tier)
    # production's call for this pass sets none of these: a replay that took them would not be production's
    if thinking_level or profile or temperature != VALIDATION_BASE_TEMPERATURE or agentic:
        raise NotImplementedError(
            f"session '{session}' 走生产的调用，不接受 --thinking-level / --profile / --temperature / --agentic")


def _client(role: LLMRole, group: str, difficulty: str, *, test_profile: bool, model: str | None) -> RoleClient:
    client = RoleClient(role_configs={role: role_config_for(group, difficulty, role=role)}, test_profile=test_profile)
    if model:
        pin_client_role_to_free_model(client, role, model)
    return client


def _write_prompts(out_dir: Path, system: str, user: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    system_path, user_path = out_dir / "prompt.system.txt", out_dir / "prompt.user.txt"
    system_path.write_text(system, encoding="utf-8")
    user_path.write_text(user, encoding="utf-8")
    return system_path, user_path


def _sample(out_dir: Path, *, n: int, max_attempts: int, ask: Callable[[], Dict[str, Any]],
            usage: Callable[[], Dict[str, int]], stopped: type[Exception]) -> tuple[list, list]:
    """Up to `n` successful calls in `max_attempts`; each entry (production's record of the unit or window)
    goes to `reply-NN.json` or `failed-NN.json`. A stop (quota, rate limit) ends the loop."""
    successes: List[ReplaySample] = []
    failures: List[ReplaySample] = []
    attempt = 0
    while len(successes) < n and attempt < max_attempts:
        attempt += 1
        before = dict(usage())
        try:
            entry = ask()
        except stopped as stop:
            failures.append(ReplaySample(ok=False, index=len(failures) + 1, attempt=attempt, content="",
                                         validation_errors=[f"stopped: {stop}"]))
            break
        spent = {key: value - before.get(key, 0) for key, value in usage().items()}
        ok = entry.get("status") == "success"
        bucket = successes if ok else failures
        sample = ReplaySample(ok=ok, index=len(bucket) + 1, attempt=attempt,
                              content=json.dumps(entry, ensure_ascii=False, indent=1),
                              validation_errors=[] if ok else [str(entry.get("error") or "failed")],
                              model=str(entry.get("model") or ""), call_meta={"usage": spent})
        sample.path = out_dir / f"{'reply' if ok else 'failed'}-{sample.index:02d}.json"
        sample.path.write_text(sample.content + "\n", encoding="utf-8")
        bucket.append(sample)
    return successes, failures


class SpeakerJudgeSessionAdapter:
    """Pass 1 (and the judge alone): one judge unit (`--chunk` is its id: `0001`, or a half `0001-a`)."""

    name = "speaker-judge"

    def run(self, *, run: Path, chunk_id: str, out_dir: Path, n: int = 2, max_attempts: int = 5,
            label: str = "baseline", note: str = "", dry_run: bool = False, test_profile: bool = False,
            thinking_level: str | None = None, profile: str | None = None,
            temperature: float = VALIDATION_BASE_TEMPERATURE, model: str | None = None,
            force_tier: str | None = None, variant: str | None = None, agentic: bool = False,
            **_kwargs: Any) -> ThreePassReplayResult:
        _reject_overrides(self.name, variant=variant, force_tier=force_tier, thinking_level=thinking_level,
                          profile=profile, temperature=temperature, agentic=agentic)
        layout = resolve_run_layout(run)
        dump = _dump(layout["artifact_dir"], judge_run.INPUT_FILENAME, "说话人裁判")
        stable = Path(dump["stable_json"])
        stable = stable if stable.is_file() else layout["stable_json"]
        segments = read_rows(stable)
        rows = rows_of(segments)
        ids = [row.id for row in rows]
        stored = json.loads(sidecar.sidecar_path(stable).read_text(encoding="utf-8"))
        entries = list(stored.get("windows") or [])
        found = sidecar.units(entries, ids) or []
        unit = next((unit for unit in found if unit.id == chunk_id), None)
        if unit is None:
            raise ValueError(f"裁判 sidecar 里没有单元 {chunk_id!r}；有：{', '.join(unit.id for unit in found)}")
        chunk = next(str(entry.get("chunk")) for entry, (low, high) in zip(entries, sidecar.tiling(entries, ids))
                     if low <= unit.low < high)
        task_profile = parse_profile_id(dump["profile_id"])
        shape = dump.get("shape")
        if shape not in (None, sidecar.SHAPE_TRACKED):
            raise ValueError(f"裁判记的 pass 1 是 {shape!r}，生产里只有带轨号的 pass 1（{sidecar.SHAPE_TRACKED!r}），重放不了")
        serial = task_profile.continuity == "serial"
        carried = judge_run._carried(segments, found, unit) if serial else []
        summaries = dump.get("summaries") or {}
        inputs = judge_run.JudgeInputs(dump["title"],
                                       tuple(Candidate(c["name"], tuple(c.get("others") or ())) for c in dump["pool"]),
                                       summary=lambda _ids: summaries.get(unit.id, summaries.get(chunk, "")))
        audio = Path(dump["audio"]) if dump.get("audio") else None
        if audio is not None and not audio.is_file():
            raise FileNotFoundError(f"裁判听的音频不在了：{audio}")
        out_dir = Path(out_dir).expanduser().resolve()
        client = None if dry_run else _client(LLMRole.AUDIO_MULTIMODAL, judge_run.judge_group(dump["media"]),
                                              task_profile.difficulty, test_profile=test_profile, model=model)
        session = _RecordingJudgeSession(client, task_profile, out_dir, str(dump.get("task_id") or ""),
                                         judge_run.judge_group(dump["media"]), dry=dry_run)

        def ask() -> Dict[str, Any]:
            previous = os.environ.get(judge_run.MARKS_SWITCH)
            os.environ[judge_run.MARKS_SWITCH] = "1" if dump.get("marks", True) else "0"
            try:
                return judge_run._judge_unit(
                    session, rows, JudgeWindow(unit.id, chunk, unit.low, unit.high), inputs,
                    tracks=sidecar.profile_tracks([row.speaker for row in rows]), carried=carried, audio=audio,
                    limits=None, counter=None, may_halve=False, shape=shape, segments=segments)
            finally:
                if previous is None:
                    os.environ.pop(judge_run.MARKS_SWITCH, None)
                else:
                    os.environ[judge_run.MARKS_SWITCH] = previous

        out_dir.mkdir(parents=True, exist_ok=True)
        if dry_run:
            try:
                ask()
            except judge_run.JudgeStopped:
                pass
            system, user = session.prompts[0]
            system_path, user_path = _write_prompts(out_dir, system, user)
            summary = _write_summary(out_dir, self.name, label, note or "dry-run（未调用 API）", [], [],
                                     temperature, dry_run=True)
            return ThreePassReplayResult(out_dir, system_path, user_path, summary_path=summary, dry_run=True)
        successes, failures = _sample(out_dir, n=n, max_attempts=max_attempts, ask=ask, usage=lambda: session.usage,
                                      stopped=judge_run.JudgeStopped)
        system, user = session.prompts[0] if session.prompts else ("", "")
        system_path, user_path = _write_prompts(out_dir, system, user)
        summary = _write_summary(out_dir, self.name, label, note, successes, failures, temperature, dry_run=False)
        return ThreePassReplayResult(out_dir, system_path, user_path, successes, failures, summary)


class _RecordingJudgeSession(judge_run._Session):
    """Production's judge session that keeps each prompt it is asked to send; a dry run stops at the first."""

    def __init__(self, *args: Any, dry: bool, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dry, self.prompts = dry, []

    def call(self, name: str, system: str, user: str, **kwargs: Any) -> Any:
        """Production's `call` with every keyword it takes (unit, repair, validator...) passed through."""
        self.prompts.append((system, user))
        if self.dry:
            raise judge_run.JudgeStopped("dry-run")
        return super().call(name, system, user, **kwargs)

    def upload(self, path: Path) -> Any:
        return None if self.dry else super().upload(path)


class ProofreadSessionAdapter:
    """Pass 3: one proofreading window (`--chunk` is the correction window's id)."""

    name = "proofread"

    def run(self, *, run: Path, chunk_id: str, out_dir: Path, n: int = 2, max_attempts: int = 5,
            label: str = "baseline", note: str = "", dry_run: bool = False, test_profile: bool = False,
            thinking_level: str | None = None, profile: str | None = None,
            temperature: float = VALIDATION_BASE_TEMPERATURE, model: str | None = None,
            force_tier: str | None = None, variant: str | None = None, agentic: bool = False,
            **_kwargs: Any) -> ThreePassReplayResult:
        _reject_overrides(self.name, variant=variant, force_tier=force_tier, thinking_level=thinking_level,
                          profile=profile, temperature=temperature, agentic=agentic)
        dump = _dump(resolve_run_layout(run)["artifact_dir"], proofread_run.INPUT_NAME, "校对（pass 3）")
        windows = [Window(w["chunk_id"], tuple(w["rows"]), tuple(w["before"]), tuple(w["after"]), tuple(w["source_ids"]))
                   for w in dump["windows"]]
        window = next((window for window in windows if window.chunk_id == chunk_id), None)
        if window is None:
            raise ValueError(f"校对的输入里没有窗 {chunk_id!r}；有：{', '.join(w.chunk_id for w in windows)}")
        lines = [Line(**line) for line in dump["lines"]]
        system, user = proofread_prompt(window, lines, title=dump["title"],
                                        context_pack=ContextPack.from_dict(dump["context_pack"]).with_source_order(
                                            dump["source_order"]),
                                        style=dump["style"], kb_terms=(dump.get("kb_terms") or {}).get(chunk_id, ""))
        out_dir = Path(out_dir).expanduser().resolve()
        system_path, user_path = _write_prompts(out_dir, system, user)
        if dry_run:
            summary = _write_summary(out_dir, self.name, label, note or "dry-run（未调用 API）", [], [],
                                     temperature, dry_run=True)
            return ThreePassReplayResult(out_dir, system_path, user_path, summary_path=summary, dry_run=True)
        task_profile = parse_profile_id(dump["profile_id"])
        client = _client(LLMRole.AUDIO_MULTIMODAL, proofread_run.TASK_GROUP, task_profile.difficulty,
                         test_profile=test_profile, model=model)
        session = proofread_run._Session(client, task_profile, out_dir, str(dump.get("task_id") or ""),
                                         retries=proofread_run._retries(None))
        successes, failures = _sample(out_dir, n=n, max_attempts=max_attempts,
                                      ask=lambda: proofread_run._ask(session, window, system, user, ""),  # no sidecar, so no digest
                                      usage=lambda: session.usage, stopped=proofread_run.ProofreadStopped)
        summary = _write_summary(out_dir, self.name, label, note, successes, failures, temperature, dry_run=False)
        return ThreePassReplayResult(out_dir, system_path, user_path, successes, failures, summary)
