"use client";

import { useCallback, useEffect, useState } from "react";


export type ThemeMode = "light" | "dark" | "system" | "marisa" | "reimu";
export type FontScale = "xs" | "sm" | "md" | "lg" | "xl";

export interface AppearanceSettings {
  theme: ThemeMode;
  fontFamily: string;
  fontScale: FontScale;
  glassOpacity: number;
}

const STORAGE_KEY = "finesub-appearance";

export const FONT_SCALE_MAP: Record<FontScale, number> = {
  xs: 0.85,
  sm: 0.925,
  md: 1,
  lg: 1.1,
  xl: 1.2,
};

export const FONT_SCALE_LABELS: Record<FontScale, string> = {
  xs: "最小",
  sm: "小",
  md: "标准",
  lg: "大",
  xl: "最大",
};

const DEFAULTS: AppearanceSettings = {
  theme: "system",
  fontFamily: "",
  fontScale: "md",
  glassOpacity: 75,
};

const THEME_MODES = new Set<ThemeMode>([
  "light",
  "dark",
  "system",
  "marisa",
  "reimu",
]);
const FONT_SCALES = new Set<FontScale>(Object.keys(FONT_SCALE_MAP) as FontScale[]);


export function fontSizeForScale(scale: FontScale): string {
  return `${15 * FONT_SCALE_MAP[scale]}px`;
}


function loadSettings(): AppearanceSettings {
  if (typeof window === "undefined") {
    return DEFAULTS;
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULTS;
    }
    const parsed = JSON.parse(raw) as Partial<AppearanceSettings>;
    return {
      theme:
        parsed.theme && THEME_MODES.has(parsed.theme)
          ? parsed.theme
          : DEFAULTS.theme,
      fontFamily:
        typeof parsed.fontFamily === "string"
          ? parsed.fontFamily
          : DEFAULTS.fontFamily,
      fontScale:
        parsed.fontScale && FONT_SCALES.has(parsed.fontScale)
          ? parsed.fontScale
          : DEFAULTS.fontScale,
      glassOpacity:
        typeof parsed.glassOpacity === "number" &&
        Number.isFinite(parsed.glassOpacity)
          ? Math.min(100, Math.max(40, parsed.glassOpacity))
          : DEFAULTS.glassOpacity,
    };
  } catch {
    return DEFAULTS;
  }
}


function applyToDom(settings: AppearanceSettings) {
  const root = document.documentElement;
  const scale = FONT_SCALE_MAP[settings.fontScale];
  root.style.setProperty("--font-scale", String(scale));
  root.style.setProperty("--base-font-size", `${15 * scale}px`);
  root.style.setProperty("--glass-opacity", String(settings.glassOpacity / 100));
  // 模糊半径与不透明度联动：透明度越低，模糊越强（视觉补偿）
  const blurValue = 10 + (100 - settings.glassOpacity) * 0.4;
  root.style.setProperty("--glass-blur", `${blurValue}px`);

  if (settings.fontFamily) {
    root.style.setProperty("--user-font", settings.fontFamily);
    document.body.style.fontFamily = `"${settings.fontFamily}", "Microsoft YaHei UI", "Segoe UI", sans-serif`;
  } else {
    root.style.removeProperty("--user-font");
    document.body.style.fontFamily = "";
  }

  // 魔理沙/灵梦是基于 dark/light 基底的角色主题：复用现有明暗切换逻辑，
  // 通过额外的 data-accent 属性承载角色配色（金色 / 绯红）。
  let resolved: "light" | "dark";
  if (settings.theme === "system") {
    resolved = window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  } else if (settings.theme === "marisa") {
    resolved = "dark";
  } else if (settings.theme === "reimu") {
    resolved = "light";
  } else {
    resolved = settings.theme;
  }
  root.setAttribute("data-theme", resolved);

  const accent =
    settings.theme === "marisa" || settings.theme === "reimu"
      ? settings.theme
      : "default";
  root.setAttribute("data-accent", accent);
}


export function useAppearance() {
  const [settings, setSettings] = useState<AppearanceSettings>(DEFAULTS);

  useEffect(() => {
    const loaded = loadSettings();
    setSettings(loaded);
    applyToDom(loaded);
  }, []);

  useEffect(() => {
    if (settings.theme !== "system") {
      return;
    }
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => applyToDom(settings);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [settings]);

  const update = useCallback((changes: Partial<AppearanceSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...changes };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      applyToDom(next);
      return next;
    });
  }, []);

  return { settings, update };
}
