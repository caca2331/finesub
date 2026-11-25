"""A cloud task's judge merging tracks it says are one person (`judgments.merges`), the voice comparison it reads
(`speech.joint.voices`), and what the judge is shown of both (`speaker_judge.prompt.profiles`)."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from finesub.llm.stages.speaker_judge import prompt as P
from finesub.llm.stages.speaker_judge.windows import JudgeWindow
from finesub.speech.diarization import judgments as jd
from finesub.speech.diarization.labels import speaker_of
from finesub.speech.diarization.product import product
from finesub.speech.joint import voices as V

NEW = "speaker-judge-v27"


def cloud(track, label):
    return {"track": track, "tracks": [track], "block": 0, "labels": [], "joint_speaker": label}


def write_stable(tmp_path, labels, times=None, backend="mai2"):
    """One row per label (`w0:A`...), each 1 s long and apart unless `times` says otherwise; tracks by first
    appearance, as `write_joint_rows` numbers them; transcribed by `backend`."""
    tracks: dict[str, int] = {}
    segments = []
    for index, label in enumerate(labels, start=1):
        start, end = times[index - 1] if times else (index * 2.0, index * 2.0 + 1.0)
        row = {"id": str(index), "start": start, "end": end, "text": f"line {index}", "line_speaker": label}
        row["speaker"] = cloud(tracks.setdefault(label, len(tracks)), label)
        segments.append(row)
    path = tmp_path / "clip-stable.json"
    path.write_text(json.dumps({"segments": segments, "metadata": {"joint_asr": {"backend": backend}}}),
                    encoding="utf-8")
    return path


#: MAI2's tiers (`voices.TIERS`): 高 from 0.90, 中 from 0.75
TWO_TIERS = {"high": 0.90, "mid": 0.75}


def write_sidecar(stable, units):
    """`units`: (first id, last id, names, results[, extra entry fields])."""
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    ids = jd.segment_ids(segments)
    tracks = jd.profile_tracks([speaker_of(s) for s in segments])
    entries = []
    for first, last, names, results, *extra in units:
        low, high = ids.index(first), ids.index(last) + 1
        entries.append({"id": f"000{len(entries) + 1}", "chunk": f"000{len(entries) + 1}", "first": first,
                        "last": last, "key": jd.window_key(segments[low:high], ids[low:high], tracks),
                        "status": "success", "prompt_version": NEW, "results": results, "names": names,
                        **(extra[0] if extra else {})})
    payload = {"version": jd.SIDECAR_VERSION, "route": "separate", "rows_hash": jd.rows_hash(segments),
               "windows": entries}
    jd.sidecar_path(stable).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_voices(stable, pairs):
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    labels = sorted({s["line_speaker"] for s in segments})
    payload = {"version": V.VERSION, "key": V.identity(segments), "tiers": TWO_TIERS,
               "labels": {label: {"talk": 1.0, "printed": True} for label in labels},
               "pairs": [{"a": a, "b": b, "cos": cos, "overlap": 0.0} for a, b, cos in pairs]}
    V.voices_path(stable).write_text(json.dumps(payload), encoding="utf-8")


def wrote(*writes, stage=None, **fields):
    """A unit's person-table record (`track_people`): it wrote `writes`, `(track, track)` a link (`s1 = s0`) or
    `(track, name)` an identification (a candidate's), and showed the tracks they name, or `stage`."""
    def key(side):
        return f"t:{side}" if isinstance(side, int) else side if side.startswith("t:") else f"c:{side}"

    pairs = [{"who": key(a), "to": key(b)} for a, b in writes]
    shown = stage if stage is not None else sorted({side for pair in writes for side in pair if isinstance(side, int)})
    return {"stage": list(shown), "writes": pairs, "people": [], "persons": {},
            "mentions": list(dict.fromkeys(side for pair in pairs for side in pair.values())), **fields}


def verdict(stable, index):
    """The verdict a later unit carrying one of unit `index`'s new tracks names (`judgments.verdict_of`)."""
    return jd.verdict_of(json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][index])


def judged(stable):
    return jd.read_judgments(stable, warn=lambda message: None)


def spk(stable):
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    return {key: speaker.track for key, speaker in product(segments, judged(stable)).speakers().items()}


# --- merging --------------------------------------------------------------------------------------------


def test_a_link_merges_the_track_into_the_one_with_more_talk(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w0:A", "w0:B", "w1:C"])
    write_sidecar(stable, [("1", "4", {}, [], wrote((2, 0)))])
    result = judged(stable)
    assert result.merged == {2: 0} and [(m.track, m.into, m.by) for m in result.merges] == [(2, 0, "link")]
    assert spk(stable) == {"1": 0, "2": 0, "3": 1, "4": 0}


def test_the_same_name_in_two_windows_merges(tmp_path):
    """A seam split: the person is named in one window on each side, never both in one."""
    stable = write_stable(tmp_path, ["w0:A", "w0:B", "w0:A", "w1:C", "w1:B"])
    write_sidecar(stable, [("1", "3", {}, [], wrote((0, "钟离"), (1, "嘉宾"))),
                           ("4", "5", {}, [], wrote((2, "钟离"), (1, "嘉宾")))])
    result = judged(stable)
    assert result.merged == {2: 0}
    assert (result.merges[0].by, result.merges[0].name) == ("name", "钟离")


def test_older_prompts_and_a_single_pass_never_merge(tmp_path):
    """`dev`'s v11 units and a single pass's windows (the correction's prompt version) cast no vote."""
    from finesub.llm.prompt_compose import TRACKED_PROMPT_VERSION

    stable = write_stable(tmp_path, ["w0:A", "w0:A", "w1:C"])
    write_sidecar(stable, [("1", "2", {}, [], wrote((0, "钟离"), prompt_version="speaker-judge-v11")),
                           ("3", "3", {}, [], wrote((1, "钟离"), source="single-pass",
                                                    prompt_version=TRACKED_PROMPT_VERSION))])
    assert judged(stable).merged == {}


def test_in_a_resumed_mix_only_the_new_prompt_votes(tmp_path):
    """A resume reuses an older unit beside newly judged ones: its name for s0 is no vote."""
    stable = write_stable(tmp_path, ["w0:A", "w0:A", "w1:C", "w1:C"])
    write_sidecar(stable, [("1", "2", {}, [], wrote((0, "钟离"), prompt_version="speaker-judge-v11")),
                           ("3", "4", {}, [], wrote((1, "钟离")))])
    assert judged(stable).merged == {}
    write_sidecar(stable, [("1", "2", {}, [], wrote((0, "钟离"))), ("3", "4", {}, [], wrote((1, "钟离")))])
    assert judged(stable).merged == {1: 0}, "both on the new prompt: they merge"


def test_tracks_that_talk_at_once_are_not_merged(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w1:C"], times=[(0.0, 4.0), (1.0, 3.0)])
    write_sidecar(stable, [("1", "2", {}, [], wrote((1, 0)))])
    result = judged(stable)
    assert result.merged == {} and result.refused == ((0, 1, "overlap"),)


def test_voices_far_apart_are_not_merged_and_close_ones_are(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w0:A", "w1:C", "w1:D"])
    write_sidecar(stable, [("1", "4", {}, [], wrote((1, 0), (2, 0)))])
    write_voices(stable, [("w0:A", "w1:C", 0.7), ("w0:A", "w1:D", 0.3), ("w1:C", "w1:D", 0.3)])
    result = judged(stable)
    assert result.merged == {1: 0} and (0, 2, "voice") in result.refused


def test_a_merge_checks_every_pair_across_the_two_groups(tmp_path):
    """s1 joins s0 first; s2 is linked to s1, but its voice is far from s0's, so it stays apart."""
    stable = write_stable(tmp_path, ["w0:A", "w0:A", "w1:C", "w2:D"])
    write_sidecar(stable, [("1", "4", {}, [], wrote((1, 0), (2, 1)))])
    write_voices(stable, [("w0:A", "w1:C", 0.9), ("w1:C", "w2:D", 0.8), ("w0:A", "w2:D", 0.2)])
    result = judged(stable)
    assert result.merged == {1: 0} and any(why == "voice" for *_pair, why in result.refused)


def test_the_unknown_track_never_merges(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "unknown", "w0:A"])
    write_sidecar(stable, [("1", "3", {}, [], wrote((1, 0)))])
    assert judged(stable).merged == {}


def test_a_link_to_a_new_track_of_the_unit(tmp_path):
    """`s0 = @甲`: a link to the unit's own new track, the person it opened."""
    stable = write_stable(tmp_path, ["w0:A", "w0:B", "w0:A", "w0:B"])
    write_sidecar(stable, [("1", "4", {}, [{"row": "2", "label": "甲", "from": 1}],
                            wrote((0, "t:0001:甲"), stage=[0, 1], opened_people=["甲"]))])
    assert judged(stable).merged == {2: 0}


def renumbered_link(stable, of):
    """Unit 1 opens 丙 then 甲 (now s2, s3); unit 2 was shown 甲 as `s2` (before a resume renumbered it), opens 乙 on
    row 4 and links 乙 to that `s2`."""
    opener = ("1", "2", {}, [{"row": "1", "label": "丙", "from": 0}, {"row": "2", "label": "甲", "from": 1}])
    write_sidecar(stable, [opener, ("3", "4", {}, []), ("5", "6", {}, [])])
    shown = {"s2": {"window": "0001", "label": "甲", "of": of(stable)}}
    write_sidecar(stable, [opener, ("3", "4", {"@乙": "s2"}, [{"row": "4", "label": "乙", "from": 1}],
                                    {"shown_tracks": shown}), ("5", "6", {}, [])])


def carried_names(stable):
    from finesub.llm.stages.speaker_judge import run as R

    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    windows = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]
    found = jd.units(windows, jd.segment_ids(segments))
    return {(c.window, c.label): c.name for c in R.carried_tracks(segments, found, found[2])}


def test_a_carried_link_is_shown_by_the_number_its_track_has_now(tmp_path):
    """The next unit is told 乙 is the person of 甲's track as numbered now (s3), not the stale `s2` (丙 now)."""
    stable = write_stable(tmp_path, ["w0:A", "w0:B"] * 3)
    renumbered_link(stable, lambda stable: verdict(stable, 0))
    assert carried_names(stable) == {("0002", "乙"): "s3"}


def test_a_carried_link_whose_source_no_longer_holds_is_no_name(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w0:B"] * 3)
    renumbered_link(stable, lambda stable: "an older verdict")
    assert carried_names(stable) == {("0002", "乙"): ""}


def test_a_carried_track_is_named_by_what_the_carrying_unit_called_it(tmp_path):
    """Unit 2 was judged while 甲 and 乙 were s2 and s3 and named them so; a resume put 丙 first, so they are s3 and s4
    now. The next unit is told 甲 is Alice, not what unit 2 said of the track now numbered s3."""
    stable = write_stable(tmp_path, ["w0:A", "w0:B"] * 3)
    opener = ("1", "3", {}, [{"row": "1", "label": "丙", "from": 0}, {"row": "2", "label": "甲", "from": 1},
                             {"row": "3", "label": "乙", "from": 0}])
    write_sidecar(stable, [opener, ("4", "5", {}, []), ("6", "6", {}, [])])
    of = verdict(stable, 0)
    carries = [{"row": "4", "carried": {"window": "0001", "label": "甲", "of": of}, "from": 1},
               {"row": "5", "carried": {"window": "0001", "label": "乙", "of": of}, "from": 0}]
    shown = {"s2": {"window": "0001", "label": "甲", "of": of}, "s3": {"window": "0001", "label": "乙", "of": of}}
    write_sidecar(stable, [opener, ("4", "5", {"s2": "Alice", "s3": "Bob"}, carries, {"shown_tracks": shown}),
                           ("6", "6", {}, [])])
    assert carried_names(stable) == {("0001", "甲"): "Alice", ("0001", "乙"): "Bob"}


def test_a_new_track_opened_in_the_unit_merges(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w0:B", "w0:A", "w0:B"])
    write_sidecar(stable, [("1", "4", {}, [{"row": "2", "label": "甲", "from": 1}],
                            wrote(("t:0001:甲", 0), stage=[0, 1], opened_people=["甲"]))])
    result = judged(stable)
    assert result.merged == {2: 0}
    assert spk(stable) == {"1": 0, "2": 0, "3": 0, "4": 1}


def test_every_line_on_a_merged_track_reads_as_the_track_kept(tmp_path):
    """A second half and an inserted line on the merged track move with it, and so does a line heard missing that has
    no room yet."""
    stable = write_stable(tmp_path, ["w0:A", "w0:A", "w1:C"])
    payload = json.loads(stable.read_text(encoding="utf-8"))
    payload["segments"][0]["words"] = [{"word": "a", "start": 2.0, "end": 2.4}, {"word": "b", "start": 2.5, "end": 3.0}]
    stable.write_text(json.dumps(payload), encoding="utf-8")
    write_sidecar(stable, [("1", "3", {}, [], {
        **wrote((1, 0)), "splits": [{"row": "1", "from": 0, "at": 1, "track": 1}],
        "inserts": [{"row": "2", "text": "嗯", "start": 4.2, "end": 4.8, "track": 1},
                    {"row": "3", "text": "啊", "start": 6.0, "end": 6.1, "track": 1}]})])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    result = judged(stable)
    # the kept track is the one with more talk in the product layer: s1 has the second half and both inserts
    assert result.merged == {0: 1}
    made = product(segments, result)
    tracks = {key: speaker.track for key, speaker in made.judged.items()}
    assert set(tracks.values()) == {1}
    assert all(speaker.extras["tracks"] == [1] for speaker in made.judged.values())
    assert all(speaker.track == 1 for speaker in made.deferred.values())
    assert made.speakers()["1"].track == 1


def test_a_second_half_talking_over_the_other_track_refuses_the_merge(tmp_path):
    """The gate reads the product layer: s0's row cut at 5 s gives its second half to a new track the judge names
    as s1, who talks from 5 to 6 s -- the two talk at once over all of the shorter one's talk."""
    stable = write_stable(tmp_path, ["w0:A", "w0:B"], times=[(0.0, 10.0), (5.0, 6.0)])
    payload = json.loads(stable.read_text(encoding="utf-8"))
    payload["segments"][0]["words"] = [{"word": "a", "start": 0.0, "end": 4.9}, {"word": "b", "start": 5.0, "end": 10.0}]
    stable.write_text(json.dumps(payload), encoding="utf-8")
    write_sidecar(stable, [("1", "2", {}, [], {**wrote(("t:0001:甲", 1), stage=[0, 1], opened_people=["甲"]),
                                               "splits": [{"row": "1", "from": 0, "at": 1, "label": "甲"}]})])
    result = judged(stable)
    assert result.merged == {}
    assert [why for *_pair, why in result.refused] == ["overlap"]


# --- the voice comparison -------------------------------------------------------------------------------


def fake_embed(piece, mask):
    """A voiceprint from the piece's level: two labels whose audio has the same level compare as one voice."""
    level = float(np.abs(piece).mean())
    vector = np.array([1.0, level * 50, 0.0])
    return vector / np.linalg.norm(vector)


def test_the_voice_file_is_for_the_rows_and_their_labels(tmp_path):
    """The same rows labelled again sound different per label: a stored comparison for the old labels is not read."""
    stable = write_stable(tmp_path, ["w0:A", "w0:B", "w0:A"])
    write_voices(stable, [("w0:A", "w0:B", 0.9)])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    assert V.read(stable, segments) is not None
    segments[2]["line_speaker"] = "w0:B"
    assert jd.rows_hash(segments) == jd.rows_hash(json.loads(stable.read_text(encoding="utf-8"))["segments"])
    assert V.read(stable, segments) is None


def test_compare_scores_each_pair_and_skips_unknown_and_short_labels():
    sr = 16000
    pcm = np.zeros(sr * 20, np.int16)
    pcm[0:sr * 8] = 3000      # A, 0-8 s
    pcm[sr * 10:sr * 18] = 3000  # C, 10-18 s, same level
    lines = [{"start": 0.0, "end": 8.0, "line_speaker": "w0:A"}, {"start": 10.0, "end": 18.0, "line_speaker": "w1:C"},
             {"start": 18.5, "end": 19.5, "line_speaker": "w1:D"}, {"start": 8.5, "end": 9.5, "line_speaker": "unknown"}]
    out = V.compare(lines, pcm, fake_embed)
    assert set(out["labels"]) == {"w0:A", "w1:C", "w1:D"} and out["labels"]["w1:D"]["printed"] is False
    pair_of = V.pairs(out)
    pair = pair_of("w1:C", "w0:A")
    assert pair["cos"] > 0.99 and pair["overlap"] == 0.0 and V.tier(pair, TWO_TIERS) == "high"
    assert pair_of("w0:A", "w1:D")["cos"] is None


def test_overlap_seconds_matches_the_double_loop():
    rng = random.Random(7)
    for _ in range(200):
        spans = [[(start := rng.uniform(0, 50), start + rng.uniform(0.1, 6)) for _ in range(rng.randint(0, 12))]
                 for _ in range(2)]
        expected = sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in spans[0] for s2, e2 in spans[1])
        assert abs(V.overlap_seconds(*spans) - expected) < 1e-9


@pytest.mark.parametrize("backend, tiers", [("mai2", TWO_TIERS), ("moss-local", {"high": 0.75, "mid": None}),
                                            ("gemini-35t", {"high": 0.95, "mid": 0.75}),
                                            ("moss-diarization-aligner", {"high": 0.75, "mid": None})])
def test_without_audio_the_record_is_written_with_overlaps_only(tmp_path, backend, tiers):
    """...and with its backend's tiers (owner 2026-11-21), so a resumed task shows what it first showed."""
    stable = write_stable(tmp_path, ["w0:A", "w1:C"], backend=backend)
    warnings = []
    out = V.ensure(stable, None, warn=warnings.append)
    assert out["unavailable"] and all(pair["cos"] is None for pair in out["pairs"]) and warnings
    assert out["tiers"] == tiers
    assert V.ensure(stable, None, warn=warnings.append) == out and len(warnings) == 1  # read back, not made again


def test_audio_that_does_not_read_leaves_the_record_unavailable(tmp_path):
    """A hint never stops the judge: a failure while comparing is written as unavailable, and read back on resume."""
    stable = write_stable(tmp_path, ["w0:A", "w1:C"])
    warnings = []
    voiceprints = SimpleNamespace(embed=lambda: fake_embed, error="")
    out = V.ensure(stable, tmp_path / "missing.wav", warn=warnings.append, voiceprints=voiceprints)
    assert "声纹没算成" in out["unavailable"] and warnings
    assert V.read(stable, json.loads(stable.read_text(encoding="utf-8"))["segments"]) == out


# --- what the judge sees --------------------------------------------------------------------------------


def rows_of(stable):
    return P.rows_of(json.loads(stable.read_text(encoding="utf-8"))["segments"])


def test_a_cloud_profile_says_when_a_track_speaks_and_where_its_label_came_from(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w1:C", "c1:w0:B", "unknown"])
    rows = rows_of(stable)
    text = P.profiles(rows, JudgeWindow("0001", "0001", 0, 4), merging=True, people=True).text
    assert "- s0：占语音 25%，2–3 秒出现。" in text
    assert "转写分块的第 2 块里才有的标签" in text and "补转回来的一小段里给的标签" in text
    assert "- s3：**不是一个人**" in text
    assert "声音相似度" not in text


def test_the_single_pass_window_keeps_the_plain_profile(tmp_path):
    """Only the judge's own windows merge by `<tracks>`: the single-pass window's profile is as it was."""
    stable = write_stable(tmp_path, ["w0:A", "w1:C", "unknown"])
    voices = {"labels": {}, "pairs": [{"a": "w0:A", "b": "w1:C", "cos": 0.9, "overlap": 0.0}]}
    text = P.profiles(rows_of(stable), JudgeWindow("0001", "0001", 0, 3), voices=voices).text
    assert text.splitlines() == [P.PHRASES["profile_line"].format(track=track, share=100 / 3, examples=P.PHRASES["no_examples"])
                                 for track in range(3)]
    carried = [P.Carried(track=3, window="0001", label="甲", name="s0", rows=(1,))]
    assert "s0 那个人" not in P.profiles(rows_of(stable), JudgeWindow("0002", "0002", 2, 3), carried).text


def test_the_voice_section_lists_the_two_tiers_and_the_tracks_too_short(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w1:C", "w2:D", "w3:E"])
    voices = {"labels": {"w0:A": {"printed": True}, "w1:C": {"printed": True}, "w2:D": {"printed": True},
                         "w3:E": {"printed": False}},
              "pairs": [{"a": "w0:A", "b": "w1:C", "cos": 0.9, "overlap": 0.0},
                        {"a": "w0:A", "b": "w2:D", "cos": 0.8, "overlap": 0.0},
                        {"a": "w1:C", "b": "w2:D", "cos": 0.8, "overlap": 0.5},
                        {"a": "w0:A", "b": "w3:E", "cos": None, "overlap": 0.0}], "tiers": TWO_TIERS}
    text = P.profiles(rows_of(stable), JudgeWindow("0001", "0001", 0, 4), voices=voices, merging=True, people=True).text
    assert "- s0 与 s1：高\n- s0 与 s2：中\n" in text and "s1 与 s2" not in text
    # a backend with no 中 (local MOSS: 高 from 0.75): 0.8 is 高, nothing is 中
    moss = {**voices, "pairs": [{"a": "w0:A", "b": "w1:C", "cos": 0.8, "overlap": 0.0},
                                {"a": "w0:A", "b": "w2:D", "cos": 0.7, "overlap": 0.0}], "tiers": {"high": 0.75, "mid": None}}
    text = P.profiles(rows_of(stable), JudgeWindow("0001", "0001", 0, 4), voices=moss, merging=True, people=True).text
    assert "- s0 与 s1：高\n" in text and "s0 与 s2" not in text
    assert "说话太少、没比声音的轨：s3" in text and "**不代表**不是同一个人" in text
    assert "声音相似度" not in P.profiles(rows_of(stable), JudgeWindow("0001", "0001", 0, 4),
                                          voices={**voices, "unavailable": "没有源音频"}, merging=True, people=True).text


def test_a_carried_track_named_as_another_track(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w0:B", "w0:A"])
    carried = [P.Carried(track=2, window="0001", label="甲", name="s0", rows=(1,))]
    text = P.profiles(rows_of(stable), JudgeWindow("0002", "0002", 2, 3), carried, merging=True).text
    assert "前面的窗口认为是 s0 那个人，未确认" in text


def test_a_cloud_task_without_audio_still_judges_and_reports_its_merges(tmp_path):
    """The task-level judge (`task.judge_task`): the voice comparison is written as unavailable with one warning,
    the judge gets no hints, and two tracks it names as one person (the second unit links s2 to s0, which the
    first unit described, with its evidence) merge and are listed in the task's record."""
    from finesub.llm.stages.speaker_judge.task import judge_task

    from .test_speaker_judge_stage import FakeClient, correction, task_request

    stable = write_stable(tmp_path, ["w0:A", "w0:B", "w0:A", "w0:B", "w1:A", "w0:B", "w1:A", "w0:B"])

    def judge(ids):
        rows = "\n".join(json.dumps({"id": i, "track": "keep", "why": "w"}) for i in ids)
        first = "q1" in ids
        tracks = "" if first else "s2 = s0（q5 自称佐藤，接着 s0 说）"
        people = "s0 补：自称佐藤" if first else ""
        return f"<judge>\n{rows}\n</judge>\n<tracks>\n{tracks}\n</tracks>\n<people>\n{people}\n</people>"

    warnings = []
    judge_task(task_request(tmp_path, stable), correction((1, 4), (5, 8)), None, client=FakeClient(judge=judge),
               artifact_dir=tmp_path / "art", task_id="t", knowledge_root=None, fallback_title="clip",
               warn=warnings.append)
    stored = json.loads(V.voices_path(stable).read_text(encoding="utf-8"))
    assert stored["unavailable"] == "没有源音频" and all(pair["cos"] is None for pair in stored["pairs"])
    assert sum("声音相似度" in message for message in warnings) == 1
    records = [json.loads(line) for line in (tmp_path / "art" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    report = next(record["payload"] for record in records if record["kind"] == "speaker_judge_report")
    assert [(merge["track"], merge["into"]) for merge in report["merged"]] == [(2, 0)], report["merged"]
    assert spk(stable) == {"1": 0, "2": 1, "3": 0, "4": 1, "5": 0, "6": 1, "7": 0, "8": 1}


def test_the_rows_earlier_units_moved_off_a_track_are_counted_by_destination(tmp_path):
    """Moves: per transcription track, the rows before the unit that it leads and those the earlier units moved
    elsewhere, by destination as numbered now, said with what the units before identified it as; under MOVED_LEAST
    moves nothing is said."""
    from finesub.llm.stages.speaker_judge import run as R

    stable = write_stable(tmp_path, ["w0:A", "w0:B"] * 4)
    write_sidecar(stable, [("1", "4", {},
                            [{"row": "1", "track": 1, "from": 0}, {"row": "3", "track": 1, "from": 0},
                             {"row": "2", "label": "甲", "from": 1}]),
                           ("5", "8", {}, [])])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    windows = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]
    found = jd.units(windows, jd.segment_ids(segments))
    moved = R.moved_out(segments, found, found[1], R.CutState())
    assert moved == {0: (2, 2, [(1, 2)]), 1: (1, 2, [(2, 1)])}
    names = {1: ("认作 p2 派蒙", "前面认作 p2 派蒙（第1段）")}
    assert P.moved_phrase(moved[0], names) == "；前面的段从这条轨挪走 2 行（占它前面 2 行的 100%）：2 行到 s1（那条轨前面认作 p2 派蒙）"
    assert P.moved_phrase(moved[1], names) == ""
    assert R.moved_out(segments, found, found[0], R.CutState()) == {}


def test_a_cloud_profile_lists_only_the_tracks_on_stage(tmp_path):
    """The judge's profiles list the tracks a shown row's cell names, context rows included; a move to a track off
    stage is said without that track's number."""
    stable = write_stable(tmp_path, ["w0:A", "w0:B", "w0:C", "w0:A", "w0:C"], times=[(i * 100.0, i * 100.0 + 1.0) for i in range(1, 6)])
    rows = rows_of(stable)
    window = JudgeWindow("0002", "0002", 3, 5)
    stage = P.stage_tracks(rows, window, marks=False)
    assert 2 in stage and 1 not in stage
    relayed = {0: ("连到 s1", "前面连到 s1（第1段）"), 1: ("认作 p2 派蒙", "前面认作 p2 派蒙（第1段）")}
    moved = {0: (3, 3, [(1, 2), (2, 1)])}
    text = P.profiles(rows, window, merging=True, people=True, relayed=relayed, moved=moved, stage=stage).text
    assert "- s1：" not in text and "- s2：" in text
    assert "2 行到画像里没有的一条轨（那条轨前面认作 p2 派蒙）、1 行到 s2" in text
    assert "- s1：" in P.profiles(rows, window, merging=True, people=True).text  # no stage: every track


def test_the_judge_is_not_shown_off_stage_or_carried_tracks(tmp_path):
    """In a run, a cloud unit's prompt lists neither an earlier track that has no row on stage nor a carried new
    track, and answering an off-stage number does not read (it is sent back)."""
    from .test_speaker_judge_stage import FakeClient, run

    far = [(i * 100.0, i * 100.0 + 1.0) for i in range(1, 7)]
    stable = write_stable(tmp_path, ["w0:A", "w0:C", "w0:A", "w0:B", "w0:A", "w0:B"], times=far)
    prompts = []

    def judge(ids):
        rows = "\n".join(json.dumps({"id": i, "track": "@甲" if i == "q2" else "keep", "why": "w"}) for i in ids)
        return f"<judge>\n{rows}\n</judge>\n<tracks>\n@甲 = 钟离\n</tracks>" if "q2" in ids else f"<judge>\n{rows}\n</judge>\n<tracks>\n</tracks>"

    class Seeing(FakeClient):
        def complete(self, role, messages, **kwargs):
            prompts.append(messages[1]["content"])
            return super().complete(role, messages, **kwargs)

    run(stable, Seeing(judge=judge), tmp_path, cut=((1, 3), (4, 6)), parallel=1)
    second = next(p for p in prompts if "▶ q4" in p)
    profile = second.split("这一段字幕")[0]
    assert "- s0：" in profile and "- s2：" in profile and "- s1：" not in profile and "新建的轨" not in profile


def test_every_joint_backend_with_labels_has_tiers():
    from finesub.speech.joint import BACKENDS, LOCAL

    assert set(V.TIERS) == set(BACKENDS) - {LOCAL}
