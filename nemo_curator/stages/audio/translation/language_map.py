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

"""Curated language code -> human-readable name mapping.

Consulted before falling back to ``pycountry`` for translation prompt
language resolution. Keys are lowercase ISO 639-1 alpha-2 codes.
"""


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
