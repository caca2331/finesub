"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import { saveUi, subscribePreferences, uiValue } from "@/lib/preferences";
import { type Language, translations } from "@/lib/translations";


const DEFAULT_LANGUAGE: Language = "zh";

interface LanguageContextValue {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (typeof translations)[Language];
}

const LanguageContext = createContext<LanguageContextValue>({
  language: DEFAULT_LANGUAGE,
  setLanguage: () => {},
  t: translations.zh,
});

export function useLanguage() {
  return useContext(LanguageContext);
}

function loadLanguage(): Language {
  if (typeof window === "undefined") {
    return DEFAULT_LANGUAGE;
  }
  return uiValue<Language>("language", DEFAULT_LANGUAGE) === "en" ? "en" : "zh";
}

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguageState] = useState<Language>(DEFAULT_LANGUAGE);

  useEffect(() => {
    const apply = () => setLanguageState(loadLanguage());
    apply();
    return subscribePreferences(apply);
  }, []);

  const setLanguage = useCallback((newLanguage: Language) => {
    setLanguageState(newLanguage);
    saveUi({ language: newLanguage });
  }, []);

  const t = translations[language];

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export type { Language };