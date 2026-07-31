"use client";

import { useCallback, useEffect, useState } from "react";
import { type Language, translations } from "./translations";

const STORAGE_KEY = "finesub-language";

const DEFAULT_LANGUAGE: Language = "zh";

function loadLanguage(): Language {
  if (typeof window === "undefined") {
    return DEFAULT_LANGUAGE;
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return DEFAULT_LANGUAGE;
    }
    return raw === "en" ? "en" : "zh";
  } catch {
    return DEFAULT_LANGUAGE;
  }
}

export function useLanguage() {
  const [language, setLanguage] = useState<Language>(DEFAULT_LANGUAGE);

  useEffect(() => {
    const loaded = loadLanguage();
    setLanguage(loaded);
  }, []);

  const updateLanguage = useCallback((newLanguage: Language) => {
    setLanguage(newLanguage);
    localStorage.setItem(STORAGE_KEY, newLanguage);
  }, []);

  const t = translations[language];

  return { language, setLanguage: updateLanguage, t };
}

export type { Language };