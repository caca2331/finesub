"""P13 (docs/plans/argon-audio-judge-plan.md 「P13 先纯文本、再给录音补丁」): back in the session of each judge unit of
a round's run, tell it what its patch turn changed against its own text-only draft -- rows it broke, rows it fixed --
and the rows the draft moved wrongly that the patch left (short ones too, with whether the patch reply named them), then ask what it heard and how the patch note should
be rewritten.

    python tools/diarization_explore/p13_ask.py <round> show            # what each session will be told
    python tools/diarization_explore/p13_ask.py <round> ask <label,...>

Rows are told by whole-recording seconds, with the attachment and its own seconds (agy hears long audio in pieces);
the piece map is read back from the session's agy store. Basis for the ruler as in judge_ask.py (owner-heard rows,
pitch and MAI2 for other-sex moves, the ruler alone for same-sex ones); other-sex rows no witness backs are left out,
and so are patch moves to a wrong person on rows whose acoustic track is wrong too (the ruler's person is then on no
track the session can be pointed to). Rows whose ruler is in doubt (SUSPECT) say so instead of giving a basis.
Answers and agy's per-request ledger go to <MAIN>/tmp/<round>-ask/answers.json.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from collections import Counter
from string import Template

import judge_ask as A
from agy_media_resume_probe_ledger import CONV as CONV_DIR
from judge_ask import HERE, MAIN, ledger

#: round -> (patched run, draft run, run dir under tmp/spk-merge/accept, readout dir holding f0.csv, template,
#:           {label: (window key, patch exchange file, agy session)})
ROUNDS = {
    "p13a": ("38F p13a r1", "38F p13a draft", "audio-rp13a", MAIN / "tmp" / "p13a" / "readout", "ask_patch_v2.md", {
        "h22-u1": ("hina-22-c0102", "002-speaker_judge-0001-attempt0-patch.md", "5089b0fd-3f6a-4f95-bd6d-edf1b8a2e06f"),
        "h22-u2": ("hina-22-c0102", "004-speaker_judge-0002-attempt0-patch.md", "bbc45714-5cc1-4429-8039-d19f2f932e77"),
        "h46-u1": ("hina-46-c0001", "002-speaker_judge-0001-attempt0-patch.md", "9d398d37-4e81-4a98-b2ba-d5636dcbce9d"),
        "h46-u2": ("hina-46-c0001", "004-speaker_judge-0002-attempt0-patch.md", "cf5c256d-8525-4b1f-8d7e-cd35c0954167"),
    }),
    "p13a2": ("38F p13a2 r1", "38F p13a2 draft", "audio-rp13a2", MAIN / "tmp" / "p13a2" / "readout", "ask_patch_v2.md", {
        "h22-u1": ("hina-22-c0102", "002-speaker_judge-0001-attempt0-patch.md", "016859e0-9bdd-4272-88b3-31b5005c73ea"),
        "h22-u2": ("hina-22-c0102", "004-speaker_judge-0002-attempt0-patch.md", "62bc9ccf-0fc7-4133-8d45-de75a2b784af"),
        "h46-u1": ("hina-46-c0001", "002-speaker_judge-0001-attempt0-patch.md", "35a51c11-329b-477f-a376-97c02b7bba1a"),
        "h46-u2": ("hina-46-c0001", "004-speaker_judge-0002-attempt0-patch.md", "0d2dc0a8-8aef-41d8-a0f7-7e20c7424710"),
    }),
    "p13b": ("38F p13b r1", "38F p13b draft", "audio-rp13b", MAIN / "tmp" / "p13b2" / "readout", "ask_patch_v2.md", {
        "h22-u1": ("hina-22-c0102", "002-speaker_judge-0001-attempt0-patch.md", "e52e0779-8d16-4215-8f91-39ebf1f7bb92"),
        "h22-u2": ("hina-22-c0102", "004-speaker_judge-0002-attempt0-patch.md", "52c58a05-b2ba-4295-afa5-11e70394fa51"),
        "h46-u1": ("hina-46-c0001", "002-speaker_judge-0001-attempt0-patch.md", "11c82501-7ff4-4f0e-8bda-523901eba0d6"),
        "h46-u2": ("hina-46-c0001", "004-speaker_judge-0002-attempt0-patch.md", "4d6b74a1-5e83-4adb-8236-feb61b1a62e9"),
    }),
    "p13d": ("38F p13d r1", "38F p13d draft", "audio-rp13d", MAIN / "tmp" / "p13d" / "readout", "ask_patch_v3.md", {
        "h22-u1": ("hina-22-c0102", "002-speaker_judge-0001-attempt0-patch.md", "108028df-2786-48b9-8e38-2b6928552ca1"),
        "h22-u2": ("hina-22-c0102", "004-speaker_judge-0002-attempt0-patch.md", "fcccb1d2-5ab6-4faa-a35c-8061fb27b83d"),
        "h46-u1": ("hina-46-c0001", "002-speaker_judge-0001-attempt0-patch.md", "5dc7a423-840e-4d03-afd6-7236566acaa0"),
        "h46-u2": ("hina-46-c0001", "004-speaker_judge-0002-attempt0-patch.md", "3464f20a-d01d-48ce-9097-d304a17a455d"),
    }),
    "p13e": ("38F p13e r1", "38F p13e draft", "audio-rrp13e", MAIN / "tmp" / "p13e" / "readout", "ask_patch_v4.md", {
        "h22-u2": ("hina-22-c0102", "004-speaker_judge-0002-attempt0-patch.md", "e4b64230-6899-4a11-b5d8-b2b91a71ee7f"),
        "h46-u2": ("hina-46-c0001", "004-speaker_judge-0002-attempt0-patch.md", "4448aaf0-1dca-4ed0-9b22-fa1d17d3fe0b"),
    }),
    "p13b2": ("38F p13b2 r1", "38F p13b2 draft", "audio-rp13b2", MAIN / "tmp" / "p13b2" / "readout", "ask_patch_v2.md", {
        "h22-u1": ("hina-22-c0102", "002-speaker_judge-0001-attempt0-patch.md", "e2310084-3f06-468e-89a2-d0a4f7812ee1"),
        "h22-u2": ("hina-22-c0102", "005-speaker_judge-0002-attempt1-patch.md", "2544798c-3ca0-41c7-baea-b21a0852dfb5"),
        "h46-u1": ("hina-46-c0001", "002-speaker_judge-0001-attempt1-patch.md", "f425f133-0025-44df-833d-75fbae03cfe0"),
        "h46-u2": ("hina-46-c0001", "004-speaker_judge-0002-attempt0-patch.md", "c559e5ed-441f-4f70-871d-45f95866b3dd"),
    }),
}
PIECE = re.compile(r"附件 (\d+) `[^`]+` 是整段的 ([\d.]+)–([\d.]+) 秒")
LONG = 1.5
_OFTEN = "各版几乎每遍都把这一行改走、旁证又弱，可能是标注错了，请照实说"
#: rows whose ruler is in doubt -> what the session is told instead of a basis (every round; the witnesses read the
#: row's own sound, whatever the round)
SUSPECT = {("hina-22-c0102", "55"): _OFTEN, ("hina-22-c0102", "140"): _OFTEN,  # moved in nearly all of P7's and P12's runs
           # a round-5 draft meant a woman (its reason names 古賀葵) but wrote s3, a man's track in the ruler's map;
           # MAI2 hears a woman, pitch (269 Hz) sits between the ruler's man and her, so "only the ruler" would mislead
           ("hina-22-c0102", "453"): "另一套分人系统听成女声；基频介于标注的人与女声之间、判不了；可能是标注错了，请照实说"}
#: (round, label) -> a question put after question 1 (templates with `$extra`)
EXTRA = {("p13d", "h22-u1"): "1b. 第一张表 q210–q217 之间的 q211、q215，补丁没有改，仍在 s11。当时这两行你听到的是谁、为什么没改？\n",
         ("p13e", "h22-u2"): "1b. q385、q514、q547 分别长 1.44、1.26、1.14 秒，说明里写改给初稿没给的人要长于约 1.5 秒。你当时怎么判这几行的长度，为什么还是改了？\n"
                             "1c. q542、q547 改前改后都是女声（s8、s6）。你区分这两位女声时听的是音高、音色还是发声方式？两条轨的样本之间，你听得出差别吗？\n"
                             "1d. 初稿把 q453 归到 s3，理由说是古贺葵的轨。s3 是谁，你当时是怎么认定的？\n",
         ("p13e", "h46-u2"): "1b. 第二张表的 q254 长 1.02 秒，说明里写改给初稿没给的人要长于约 1.5 秒。你当时怎么判这一行的长度，为什么还是改了？\n"}


def pieces(sid):
    """[(attachment number, start, end)] the session was told; one piece when the audio was not split (no map)."""
    con = sqlite3.connect(CONV_DIR / f"{sid}.db")
    maps = set()
    for (table,) in con.execute("select name from sqlite_master where type='table'"):
        for row in con.execute(f"select * from {table}"):
            for v in row:
                text = v.decode("utf-8", "replace") if isinstance(v, bytes) else v if isinstance(v, str) else ""
                found = tuple(sorted({(int(k), float(s), float(e)) for k, s, e in PIECE.findall(text)}))
                if found:
                    maps.add(found)
    if len(maps) > 1:
        raise SystemExit(f"{sid}: {len(maps)} different piece maps")
    return list(maps.pop()) if maps else [(1, 0.0, float("inf"))]


def where(s, e, cut):
    k, start, end = next(p for p in reversed(cut) if p[1] <= s)
    assert e <= end + 0.05, (s, e, cut)
    return f"{s:.1f}–{e:.1f}（附件 {k} 第 {s - start:.1f}–{e - start:.1f} 秒）" if len(cut) > 1 else f"{s:.1f}–{e:.1f}"


def classify(win, d, p, i):
    """What the patch did on row i against the draft: broke (draft right, patch wrong), moved (draft wrong, patch
    moved it to another wrong person, the ruler's person on the acoustic track), other (the same with the ruler's
    person on no track the session was given: not told), fixed, left / left_short (the draft's wrong move kept)."""
    dr, pr = d[0].get(i), p[0].get(i)
    acoustic = win.acoustic[i][0]
    draft_said, patch_said = (dr or {}).get("new_label", acoustic), (pr or {}).get("new_label", acoustic)
    if draft_said == patch_said:
        assert d[1][i] == p[1][i], (win.ids[i], draft_said)
        if dr and dr["kind"] == "wrong":
            return "left" if win.seconds(i) > LONG else "left_short"
        return None
    if p[1][i]:
        return None if d[1][i] else "fixed"
    if d[1][i]:
        return "broke"
    return "moved" if pr and pr["kind"] == "wrong" else "other"


def told(round_, label):
    import judge_moves_readout as M
    import p9_readout as P
    patched, draft, run, readout, template, sessions = ROUNDS[round_]
    key, exchange, sid = sessions[label]
    wins, results = M.load(None)
    keys = {k for k, _, _ in sessions.values()}
    units = [(k, u) for k in keys for w in results[patched, k][2]["windows"] for u in (w.get("parts") or [w])]
    dead = [(k, int(u["first"]), int(u["last"])) for k, u in units if (u.get("patch") or {}).get("status") != "success"]
    failed = len(dead)
    total, unpatched = Counter(), Counter()  # unpatched: rows of units whose patch failed (counted at the draft)
    for k in keys:
        for i in wins[k].scored:
            kind = classify(wins[k], results[draft, k], results[patched, k], i)
            total[kind] += 1
            unpatched[kind] += any(k == dk and a <= int(wins[k].ids[i]) <= b for dk, a, b in dead)
    drafted = Counter(r["kind"] for k in keys for r in results[draft, k][0].values())
    win = wins[key]
    d, p = results[draft, key], results[patched, key]
    witness = P.witnesses(readout)
    index = {k: i for i, k in enumerate(win.ids)}
    text = (M.SPK / "accept" / run / key / "llm" / "exchanges" / exchange).read_text(encoding="utf-8")
    at = {m.group(1): (float(m.group(2)), float(m.group(3)))
          for m in map(re.compile(r"^q(\d+)｜([\d.]+)–([\d.]+)$").match, text.split("## 模型响应", 1)[0].splitlines()) if m}
    cut = pieces(sid)
    reply = text.split("## 模型响应", 1)[1]
    said = {}  # the patch's own reasons by id (a `keep` back to the acoustic track leaves no move in the readout)
    for block in re.findall(r"<judge>(.*?)</judge>", reply, re.S)[-1:]:
        for row in block.splitlines():
            try:
                said[str(json.loads(row)["id"])] = str(json.loads(row).get("why", ""))
            except (ValueError, KeyError, TypeError):
                pass
    broke, fixed, left = [], [], []
    for rid, (s, e) in at.items():
        i = index[rid]
        if i not in win.scored:
            continue
        what = classify(win, d, p, i)
        if what not in ("broke", "moved", "fixed", "left", "left_short"):
            continue
        dr, pr = d[0].get(i), p[0].get(i)
        acoustic = win.acoustic[i][0]
        line = f"q{rid}｜{where(s, e, cut)}｜{win.segs[i]['text']}"
        suspect = SUSPECT.get((key, rid))
        if what == "fixed":
            fixed.append(f"{line}｜{(dr or {}).get('new_label', acoustic)} → {(pr or {}).get('new_label', acoustic)}"
                         + ("｜（这一行的标注可疑）" if suspect else ""))
            continue
        drafted_label = (dr or {}).get("new_label", acoustic)
        if suspect:
            why = suspect
        elif pr and pr["kind"] == "wrong" or what.startswith("left"):  # the ruler's person is the acoustic track's
            why = A.basis(witness, key, i, dict(pr, id=rid), win.truth[i].person)
        else:  # the draft fixed a wrong acoustic track; pitch and MAI2 were only read against the acoustic person
            why = "标注认为是初稿给的那个人，没有标注以外的旁证，可能是标注错了，请照实说"
        if not why:
            continue
        if what.startswith("left"):
            length = f"{win.seconds(i):.2f} 秒" + ("（时间戳可能不准）" if win.seconds(i) < 0.3 else "")
            named = "写进了 <judge>" if f"q{rid}" in said else \
                "只在回复正文里提到" if re.search(rf"(?<![0-9A-Za-z])q{rid}(?!\d)", reply) else "没提到"
            left.append(f"{line}｜{length}｜{acoustic}｜{dr['new_label']}｜{named}｜{dr['why']}｜{why}")
            continue
        if what == "broke":
            right = "初稿对" + (f"（初稿的 {drafted_label} 与原来的 {acoustic} 是同一个人）"
                                if drafted_label != acoustic and pr and pr["kind"] == "wrong" else "")
        else:
            right = "初稿也错（标注认为是原来那条轨的人）"
        broke.append(f"{line}｜{acoustic}｜{drafted_label}｜{(pr or {}).get('new_label', acoustic)}｜{right}"
                     f"｜{said.get(f'q{rid}') or (pr or {}).get('why', '')}｜{why}")
    patched_kinds = Counter(r["kind"] for k in keys for r in results[patched, k][0].values())
    return dict(d_wrong=drafted["wrong"], d_fix=drafted["fix"], d_net=f"{drafted['fix'] - drafted['wrong']:+d}",
                p_net=f"{patched_kinds['fix'] - patched_kinds['wrong']:+d}",
                p_broke=total["broke"], p_moved=total["moved"], p_other=total["other"], p_fixed=total["fixed"],
                failed=failed,
                d_left=total["left"] + total["left_short"] - unpatched["left"] - unpatched["left_short"],
                d_left_short=total["left_short"] - unpatched["left_short"],
                d_left_unpatched=unpatched["left"] + unpatched["left_short"],
                broke_count=len(broke), broke="\n".join(broke) or "（没有）", fixed="\n".join(fixed) or "（没有）",
                left_count=len(left), left="\n".join(left) or "（没有）", extra=EXTRA.get((round_, label), "")), sid


def render(round_, label):
    fields, sid = told(round_, label)
    template = ROUNDS[round_][4]
    return Template((HERE / "judge_ask_templates" / template).read_text(encoding="utf-8")).substitute(fields), sid


def ask(round_, labels):
    from finesub.llm.agent.local_agent import AgyDriverConfig, AgyLocalAgentDriver
    work = MAIN / "tmp" / f"{round_}-ask"
    work.mkdir(parents=True, exist_ok=True)
    for label in labels:
        text, sid = render(round_, label)
        started = time.time()
        driver = AgyLocalAgentDriver(AgyDriverConfig(model="gemini-3.8-flash", timeout_seconds=1200))
        result = driver.run([{"role": "user", "content": text}], task=f"{round_}-ask", session_scope="assignment",
                            conversation_key=f"{round_}-ask-{label}-{int(started)}", conversation_handle=sid,
                            continuation=True)
        row = {"label": label, "conversation": result.conversation_handle, "same_session": result.conversation_handle == sid,
               "answer": result.content.strip(), "seconds": round(time.time() - started, 1), "ledger": ledger(sid),
               "prompt": text}
        out = work / "answers.json"
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        out.write_text(json.dumps(old + [row], ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in row.items() if k not in ("answer", "prompt")}, ensure_ascii=False), flush=True)


def main():
    round_, verb = sys.argv[1], sys.argv[2]
    if verb == "show":
        for label in ROUNDS[round_][5]:
            text, sid = render(round_, label)
            print(f"===== {label} ({sid})\n{text}\n")
    else:
        ask(round_, sys.argv[3].split(","))


if __name__ == "__main__":
    main()
