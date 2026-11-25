"""The preparation task's descriptions in the person table (v27): where a candidate's description comes from
(`prompt.candidates`: the preparation task's `speakers` only), how the table shows it (`people_io.render_table`:
`〔准备任务〕`, whole, before the judges' notes and outside `NOTES_SHOWN`), the prompt's two templates, and v25's
prompt byte for byte when no candidate has a description."""

from __future__ import annotations

import hashlib
import json

from finesub.llm.client import LLMCallResult
from finesub.llm.prompt_tables import PROMPT_TEMPLATE_DIR
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.speaker_judge import people_io as PIO
from finesub.llm.stages.speaker_judge import prompt as P
from finesub.llm.stages.speaker_judge import run as R
from finesub.speech.diarization import judgments as jd
from finesub.speech.diarization import track_people as T

from .test_speaker_judge_merges import write_stable
from .test_speaker_judge_pieces import write_voices
from .test_speaker_judge_stage import correction

X, Y = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
XY = [0.8, 0.6, 0.0]  # cos with X 0.8 (mid), with Y 0.6


class Recording:
    """A judge client that records every prompt and answers each unit in turn (`replies`, by unit order)."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def complete(self, role, messages, **kwargs):
        self.seen.append([message["content"] for message in messages])
        return LLMCallResult(content=self.replies.pop(0) if self.replies else "", role=role, model="fake",
                             fallback_used=False, raw_response={})


RELAY_LABELS = ["w0:A", "w0:B", "w0:A", "w0:B", "w0:A", "w0:B", "w0:A", "w0:B", "w1:C", "w0:B", "w1:C", "w0:B"]


def keep_all(ids, tracks="", people=""):
    rows = "\n".join(json.dumps({"id": i, "track": "keep", "why": "w"}) for i in ids)
    return f"<judge>\n{rows}\n</judge>\n<tracks>\n{tracks}\n</tracks>\n<people>\n{people}\n</people>"


def relay_run(tmp_path, pool=(P.Candidate("Ema"), P.Candidate("Coco")), **options):
    """A serial cloud relay over three units (a cut in the second), every prompt recorded."""
    stable = write_stable(tmp_path, RELAY_LABELS)
    write_voices(stable, [X if label == "w0:A" else Y if label == "w0:B" else XY for label in RELAY_LABELS],
                 {"w0:A": X, "w0:B": Y, "w1:C": XY})
    client = Recording([keep_all(["q1", "q2", "q3", "q4"], "s0 = p1（q1 被叫エマ）", "s1 补：自称「僕」"),
                        keep_all(["q5", "q6", "q7", "q8"], "s1 自q6 = ?（q6 起自称「私」）"),
                        keep_all(["q9", "q10", "q11", "q12"], "s2 = s0（q11 接着 s0 说）")])
    from finesub.speech.joint import voices as V
    from finesub.subtitles.stable_rows import read_rows

    R.run_speaker_judge(stable_json=stable, correction_windows=correction((1, 4), (5, 8), (9, 12)),
                        profile=__import__("dataclasses").replace(resolve_profile("text", "none", "quality"),
                                                                  continuity="serial"),
                        client=client, inputs=R.JudgeInputs("t", pool=pool,
                                                            voices=V.read(stable, read_rows(stable))),
                        media="text", audio=None, task_artifact_dir=tmp_path / "art", task_id="t", **options)
    return stable, client


#: the digest of every prompt `relay_run` renders, taken on `58827b8f8`'s code (v25's relay): v27 with no
#: description renders it byte for byte
RELAY_DIGEST = "261af5ffbdae6f66c92f7fbcc170fdb25a4f74f50020bdabbb2cda52529d1307"

LONG = "星穹铁道节目的主持人，负责串场与提问；被叫：フク郎さん、フクロウさん；自称「僕」，语尾带「〜ですね」；" \
       "开场自我介绍「AKAフローキング、ミスターフク郎」，全片串场的都是他，所以列他。"
NOTED = (P.Candidate("Ema", note="主角；被叫：エマ；自称「私」；开场自我介绍"), P.Candidate("Coco", note=LONG))


def test_a_description_comes_from_the_preparation_task_speakers_only():
    pool = P.candidates([{"name": "さとう", "type": "人物", "note": "依据：标题"},
                         {"name": "Ema", "type": "人物", "zh": "艾玛", "note": "依据：片头"}],
                        [("ひな", (), "")], [{"name": "Ema", "note": "  主角；被叫：エマ  "}, {"name": "Coco"}])
    by = {c.name: c for c in pool}
    assert by["Ema"].note == "主角；被叫：エマ" and by["Coco"].note == ""
    assert by["さとう"].note == "" and by["ひな"].note == "", "an entity's note is evidence for its name, not a description"


def test_the_description_follows_the_candidate_whole_and_outside_the_notes_limit():
    assert len(LONG) > 80, "the preparation task's length is its own business: the judge does not cut it"
    notes = [{"op": "add", "key": "c:Coco", "text": "x" * 200}, {"op": "add", "key": "c:Coco", "text": "y" * 200}]
    entry = {"prompt_version": "speaker-judge-v27", "segment": 2, "stage": [], "writes": [], "people": notes,
             "persons": {}, "mentions": ["c:Coco"]}
    table = T.replay(["Ema", "Coco"], [("u1", entry)], prepared={c.name: c.note for c in NOTED})
    lines = sorted(PIO.render_table(table).splitlines())  # the latest mentioned first: Coco, then Ema
    assert lines[0] == f"- p1 Ema：〔准备任务〕{NOTED[0].note}"
    # the judges' notes still fit `NOTES_SHOWN` on their own (the latest only here), after the description
    assert lines[1] == f"- p2 Coco：〔准备任务〕{LONG} 〔第2段〕{'y' * 200}"
    plain = T.replay(["Ema", "Coco"], [("u1", entry)])
    assert sorted(PIO.render_table(plain).splitlines())[1] == f"- p2 Coco：〔第2段〕{'y' * 200}"


def test_v27_says_who_wrote_the_descriptions_and_records_its_version(tmp_path):
    stable, client = relay_run(tmp_path, pool=NOTED)
    system, user = client.seen[0]
    assert R.PROMPT_VERSION == "speaker-judge-v27" and R.prepared_notes(NOTED)
    assert f"- p1 Ema：〔准备任务〕{NOTED[0].note}" in user
    assert "人物表（整份共用；认作、连轨与线索都是前面各段的裁判写的，〔准备任务〕那段是准备任务写的，都可能有错" in user
    assert ("人物表里候选那一行的〔准备任务〕是准备任务从节目资料与全片台词写的，可能写错：其中「被叫：」列的当面叫法与"
            "自报姓名，和候选名一样可以当认作这位候选的名字凭据") in system
    assert "；候选那一行的〔准备任务〕是准备任务写的，「台词里叫」汇总了" in system
    assert ("（不是候选那一行汇总的「台词里叫」），与候选的名字或候选那一行〔准备任务〕「被叫：」列的叫法对得上。"
            in system), "the identification gate takes the description's 「被叫」 names"
    v6 = (PROMPT_TEMPLATE_DIR / P.PEOPLE_SYSTEM_TEMPLATE).read_text(encoding="utf-8")
    assert "〔准备任务〕" not in v6 and "（不是候选那一行汇总的叫法），与候选的名字对得上。" in v6, "v6 untouched"
    assert all(f"〔准备任务〕{NOTED[0].note}" in seen[1] for seen in client.seen), "every unit's table shows it"
    windows = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]
    assert {w["prompt_version"] for w in windows} == {"speaker-judge-v27"}


def test_with_no_description_on_any_candidate_the_prompt_is_v25_s_byte_for_byte(tmp_path):
    stable, client = relay_run(tmp_path, pool=(P.Candidate("Ema"), P.Candidate("Coco")))
    assert hashlib.sha256(json.dumps(client.seen, ensure_ascii=False).encode("utf-8")).hexdigest() == RELAY_DIGEST
    windows = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]
    assert {w["prompt_version"] for w in windows} == {R.PROMPT_VERSION}
