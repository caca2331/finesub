"use client";

const BASE_FONTS = [
  "Maple Mono NL NF CN",
  "Maple Mono NF CN",
  "Maple Mono CN",
  "Microsoft YaHei UI",
  "Segoe UI",
  "SimSun",
  "SimHei",
  "KaiTi",
  "FangSong",
  "Arial",
  "Helvetica Neue",
  "Times New Roman",
  "Consolas",
  "Courier New",
  "Georgia",
  "Verdana",
  "Tahoma",
  "Trebuchet MS",
  "Comic Sans MS",
  "Impact",
  "Noto Sans SC",
  "Noto Serif SC",
  "Source Han Sans SC",
  "Source Han Serif SC",
  "PingFang SC",
  "Hiragino Sans GB",
  "WenQuanYi Micro Hei",
];


export function detectAvailableFonts(): string[] {
  if (typeof document === "undefined") {
    return BASE_FONTS;
  }

  const probe = document.createElement("span");
  probe.style.position = "absolute";
  probe.style.left = "-9999px";
  probe.style.fontSize = "72px";
  probe.style.visibility = "hidden";
  probe.textContent = "mmmmmmmmmmlli";

  const fallbacks = ["monospace", "sans-serif", "serif"] as const;
  const fallbackWidths: Record<string, number> = {};

  for (const fallback of fallbacks) {
    probe.style.fontFamily = fallback;
    document.body.appendChild(probe);
    fallbackWidths[fallback] = probe.offsetWidth;
    document.body.removeChild(probe);
  }

  const available: string[] = [];
  for (const font of BASE_FONTS) {
    let detected = false;
    for (const fallback of fallbacks) {
      probe.style.fontFamily = `"${font}", ${fallback}`;
      document.body.appendChild(probe);
      const width = probe.offsetWidth;
      document.body.removeChild(probe);
      if (width !== fallbackWidths[fallback]) {
        detected = true;
        break;
      }
    }
    if (detected) {
      available.push(font);
    }
  }

  return available.length > 0 ? available : BASE_FONTS.slice(0, 6);
}
