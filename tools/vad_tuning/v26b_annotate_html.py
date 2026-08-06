"""Build a single-file annotation page from v26_step0 output.

One row per snippet that needs a human ear: inline audio player, the feature
columns that matter, the snippet-ASR text, and one radio group per row. Labels
live in localStorage as you click; the export button downloads a CSV that can be
joined back onto regions.csv by (clip, idx).

Usage:
  python v26b_annotate_html.py --outdir <tmp/vad-step0> [--clips a,b,c]
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
from pathlib import Path

LABELS = ["真语音", "轻语/低语", "语气词", "噪声/抖动", "幻觉", "听不清"]

CSS = """
body{font-family:system-ui,'Segoe UI',sans-serif;margin:16px;background:#111;color:#ddd}
table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #333;padding:4px 8px;font-size:13px;text-align:left}
tr.head th{position:sticky;top:0;background:#1b1b1b}
audio{height:28px;width:230px}
.v-real{color:#7bd88f}.v-ambiguous{color:#ffd866}.v-delegate{color:#78dce8}
.v-jitter_silero,.v-jitter_certain{color:#909090}
.k-added{color:#ff9d68}.k-empty{color:#ab9df2}
button{margin:8px 4px;padding:6px 14px}
label{margin-right:6px;white-space:nowrap}
.done{opacity:0.45}
"""

JS = """
function save(k,v,tr){localStorage.setItem('step0-'+k,v);tr.classList.add('done');count();}
function count(){
  let n=0,rows=document.querySelectorAll('tr[data-k]');
  rows.forEach(tr=>{if(localStorage.getItem('step0-'+tr.dataset.k))n++;});
  document.getElementById('prog').textContent=n+' / '+rows.length;
}
function restore(){
  document.querySelectorAll('tr[data-k]').forEach(tr=>{
    const v=localStorage.getItem('step0-'+tr.dataset.k);
    if(v){tr.classList.add('done');
      const r=tr.querySelector('input[value="'+v+'"]');if(r)r.checked=true;}
  });count();
}
function exportCsv(){
  let out='clip,idx,label\\n';
  document.querySelectorAll('tr[data-k]').forEach(tr=>{
    const v=localStorage.getItem('step0-'+tr.dataset.k);
    if(v)out+=tr.dataset.k.replace(':',',')+','+v+'\\n';
  });
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([out],{type:'text/csv'}));
  a.download='step0-labels.csv';a.click();
}
window.addEventListener('load',restore);
"""


def snip_map(clip_dir: Path) -> dict:
    out = {}
    for f in (clip_dir / "snips").glob("*.wav"):
        idx = int(f.name.split("-", 1)[0])
        out[idx] = f
    return out


GROUPS = [
    ("A 丢recall直接证据(新增,真文本,从未解码)", None),
    ("B 新增未解码,自动判不了", None),
    ("C 新增已解码抽查(creep在压什么)", 8),
    ("D empty区间但切片ASR有文本(下游丢失?)", 15),
    ("E empty区间,自动判不了", 10),
    ("F 自动判为抖动的抽查", 5),
]


def group_of(row: dict) -> int:
    kind, v = row["kind"], row["verdict"]
    ndf = float(row["never_decoded_frac"])
    if kind == "added":
        if v == "real" and ndf > 0.5:
            return 0
        if v in ("ambiguous", "delegate") and ndf > 0.3:
            return 1
        if v == "real":
            return 2
        if v.startswith("jitter"):
            return 5
        return 2
    if kind == "empty":
        if v == "real":
            return 3
        if v.startswith("jitter"):
            return 5
        return 4
    return -1


def build(outdir: Path, clips: list[str]) -> Path:
    import random

    picked: list[tuple[int, str, dict, Path]] = []
    for clip in clips:
        d = outdir / clip
        if not (d / "regions.csv").exists():
            continue
        snips = snip_map(d)
        by_group: dict[int, list] = {}
        with open(d / "regions.csv", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                idx = int(row["idx"])
                if idx not in snips:
                    continue
                g = group_of(row)
                if g < 0:
                    continue
                by_group.setdefault(g, []).append((g, clip, row, snips[idx]))
        rng = random.Random(7)
        for g, items in by_group.items():
            cap = GROUPS[g][1]
            if cap is not None and len(items) > cap:
                items = rng.sample(items, cap)
            picked.extend(items)

    picked.sort(key=lambda t: (t[0], t[1], float(t[2]["start"])))
    rows_html = []
    total_bytes = 0
    cur_group = -1
    for g, clip, row, snip in picked:
        if g != cur_group:
            cur_group = g
            rows_html.append(f'<tr><th colspan="8" style="background:#222;'
                             f'padding:10px">{GROUPS[g][0]}</th></tr>')
        idx = int(row["idx"])
        wav = snip.read_bytes()
        total_bytes += len(wav)
        b64 = base64.b64encode(wav).decode()
        key = f"{clip}:{idx}"
        radios = "".join(
            f'<label><input type="radio" name="r{key}" value="{v}" '
            f'onclick="save(\'{key}\',\'{v}\',this.closest(\'tr\'))">{v}</label>'
            for v in LABELS)
        rows_html.append(
            f'<tr data-k="{key}">'
            f'<td class="k-{row["kind"]}">{clip}<br>#{idx} {row["kind"]}</td>'
            f'<td>{row["start"]}s<br>d={row["dur"]}s</td>'
            f'<td>pk {row["peak_db"]}<br>sil {row["silero_peak"]}</td>'
            f'<td class="v-{row["verdict"]}">{row["verdict"]}<br>'
            f'未解码 {row["never_decoded_frac"]}</td>'
            f'<td>{html.escape(row["asr_text"] or "(无)")}</td>'
            f'<td><audio controls preload="none" '
            f'src="data:audio/wav;base64,{b64}"></audio></td>'
            f'<td>{radios}</td></tr>')
    page = (f"<!doctype html><meta charset='utf-8'><title>step0 标注</title>"
            f"<style>{CSS}</style><script>{JS}</script>"
            f"<h2>step0 人工标注 <span id='prog'></span>"
            f"<button onclick='exportCsv()'>导出 CSV</button></h2>"
            f"<p>判断该片段里有没有真实语音。真语音=有语义内容;轻语=确实是人声但很轻/"
            f"听不出完整语义;语气词=えー/うん类;噪声/抖动=非人声。标完点导出,"
            f"把 step0-labels.csv 放回本目录。</p>"
            f"<table><tr class='head'><th>片段</th><th>时间</th><th>能量/silero</th>"
            f"<th>自动判定</th><th>切片ASR</th><th>音频(前后各多含0.15s)</th>"
            f"<th>标注</th></tr>{''.join(rows_html)}</table>")
    out = outdir / "annotate.html"
    out.write_text(page, encoding="utf-8")
    print(f"{out} ({len(rows_html)} rows, {total_bytes/1e6:.1f} MB audio)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--clips", default=None)
    args = ap.parse_args()
    clips = (args.clips.split(",") if args.clips else
             sorted(p.parent.name for p in args.outdir.glob("*/regions.csv")))
    build(args.outdir, clips)


if __name__ == "__main__":
    main()
