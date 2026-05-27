# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "nl": "Dutch",
    "it": "Italian",
    "es": "Spanish",
    "pt": "Portuguese",
    "fr": "French",
    "de": "German",
    "pl": "Polish",
    "sv": "Swedish",
    "ro": "Romanian",
    "da": "Danish",
    "sl": "Slovenian",
    "sk": "Slovak",
    "et": "Estonian",
    "fi": "Finnish",
    "hu": "Hungarian",
    "mt": "Maltese",
    "hr": "Croatian",
    "cs": "Czech",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "bg": "Bulgarian",
    "ru": "Russian",
    "uk": "Ukrainian",
    "el": "Greek",
    "ar": "Arabic",
    "he": "Hebrew",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "th": "Thai",
}

# Reverse map: display name → ISO 639-1 code (e.g. "German" → "de").
# Built once at import time; raises on duplicate display names (shouldn't happen).
NAME_TO_CODE: dict[str, str] = {name: code for code, name in LANGUAGE_MAP.items()}


def _normalize_code(code: str) -> str:
    """Normalize an ISO 639-1 code (e.g. ``"pt_BR"`` -> ``"pt"``)."""
    return code.split("_", 1)[0].strip().lower()


def lang_code_to_name(lang_code: str) -> str:
    """Map an ISO 639-1 code to its display name via ``LANGUAGE_MAP``.

    Splits on ``_`` (so ``"pt_BR"`` -> ``"pt"``) and lowercases. Raises
    ``KeyError`` if the resolved code is not in ``LANGUAGE_MAP`` — bad
    codes should fail loudly at pipeline start rather than silently
    producing nonsense prompts.
    """
    if not lang_code:
        msg = "lang_code is empty"
        raise ValueError(msg)
    code = _normalize_code(lang_code)
    if code not in LANGUAGE_MAP:
        msg = (
            f"Language code '{lang_code}' not found in LANGUAGE_MAP. "
            "Add it to nemo_curator/stages/audio/translation/language_map.py."
        )
        raise KeyError(msg)
    return LANGUAGE_MAP[code]


def name_to_code(display_name: str) -> str:
    """Map a language display name to its ISO 639-1 code via ``NAME_TO_CODE``.

    Raises ``KeyError`` for unknown display names so callers fail loudly
    rather than silently emitting wrong language codes.
    """
    if display_name not in NAME_TO_CODE:
        msg = (
            f"Display name '{display_name}' not found in NAME_TO_CODE. "
            "Add it to nemo_curator/stages/audio/translation/language_map.py."
        )
        raise KeyError(msg)
    return NAME_TO_CODE[display_name]
