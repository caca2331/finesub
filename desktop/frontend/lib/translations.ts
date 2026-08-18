import { zh } from "./translations.zh";
import { en } from "./translations.en";

export type Language = "zh" | "en";

/** The shape of one language's copy: whatever Chinese says, in strings.
 *
 * Chinese is where a new key is added, so it defines the shape and English has
 * to match it. `DeepStringly` rewrites the literal types `as const` gives us
 * back to `string`, or every English value would have to equal the Chinese one.
 */
type DeepStringly<T> = {
  [K in keyof T]: T[K] extends string ? string : DeepStringly<T[K]>;
};

export type Translations = DeepStringly<typeof zh>;

export const translations: Record<Language, Translations> = { zh, en };

export type TranslationKey = keyof Translations;
