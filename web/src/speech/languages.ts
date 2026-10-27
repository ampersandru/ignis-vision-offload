// The 30 languages the speech model recognises, as the server lists them
// (crates/server/src/speech/languages.rs, from Qwen3-ASR's model card): the
// model's English name and the ISO 639 code a request sends. The Transcribe
// view picks among them, and names a language the server reports by them.

export type SpeechLanguage = { name: string; code: string };

/** In the model card's order. */
export const SPEECH_LANGUAGES: SpeechLanguage[] = [
  { name: "Chinese", code: "zh" },
  { name: "English", code: "en" },
  { name: "Cantonese", code: "yue" },
  { name: "Arabic", code: "ar" },
  { name: "German", code: "de" },
  { name: "French", code: "fr" },
  { name: "Spanish", code: "es" },
  { name: "Portuguese", code: "pt" },
  { name: "Indonesian", code: "id" },
  { name: "Italian", code: "it" },
  { name: "Korean", code: "ko" },
  { name: "Russian", code: "ru" },
  { name: "Thai", code: "th" },
  { name: "Vietnamese", code: "vi" },
  { name: "Japanese", code: "ja" },
  { name: "Turkish", code: "tr" },
  { name: "Hindi", code: "hi" },
  { name: "Malay", code: "ms" },
  { name: "Dutch", code: "nl" },
  { name: "Swedish", code: "sv" },
  { name: "Danish", code: "da" },
  { name: "Finnish", code: "fi" },
  { name: "Polish", code: "pl" },
  { name: "Czech", code: "cs" },
  { name: "Filipino", code: "fil" },
  { name: "Persian", code: "fa" },
  { name: "Greek", code: "el" },
  { name: "Hungarian", code: "hu" },
  { name: "Macedonian", code: "mk" },
  { name: "Romanian", code: "ro" },
];

/** A code's English name, or the code itself for one the model does not list. */
export const languageName = (code: string): string => SPEECH_LANGUAGES.find((l) => l.code === code)?.name ?? code;

export const isSpeechLanguage = (code: unknown): code is string => SPEECH_LANGUAGES.some((l) => l.code === code);
