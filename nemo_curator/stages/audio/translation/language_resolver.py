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

"""Language resolution for the translation pipeline.

Reads each row's source-language code and emits two new fields:

- ``source_lang_name`` — display name resolved via ``LANGUAGE_MAP``.
- ``translate_to``     — list of display names for the row's target
                         languages, computed direction-aware:
                         only ``En->X`` and ``X->En`` pairs are produced,
                         where ``X`` ranges over ``target_lang_codes``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.stages.audio.translation.language_map import LANGUAGE_MAP
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


def _normalize_code(code: str) -> str:
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


@dataclass
class LanguageResolverStage(ProcessingStage[AudioTask, AudioTask]):
    """Populate display-name language fields and per-row target lists.

    Writes ``source_lang_name`` (display name of the source) and
    ``translate_to`` (list of target display names) onto each task,
    overwriting any pre-existing values under those keys.

    Direction rules (``T`` = normalized set of ``target_lang_codes``):
        * ``source_lang == "en"``           -> translate_to = T \\ {en}  (En->X)
        * ``source_lang in T \\ {"en"}``      -> translate_to = ["English"] (X->En)
        * otherwise                          -> translate_to = []         (skipped)
    """

    target_lang_codes: list[str] = field(default_factory=list)
    source_lang_key: str = "source_lang"
    source_lang_name_key: str = "source_lang_name"
    translate_to_key: str = "translate_to"
    name: str = "LanguageResolver"

    _target_codes_norm: list[str] = field(default_factory=list, init=False, repr=False)
    _target_names: list[str] = field(default_factory=list, init=False, repr=False)
    _target_set: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)
    _en_to_x_names: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.target_lang_codes:
            msg = "LanguageResolverStage: target_lang_codes must be non-empty"
            raise ValueError(msg)

        self._target_codes_norm = [_normalize_code(c) for c in self.target_lang_codes]
        self._target_names = [lang_code_to_name(c) for c in self.target_lang_codes]
        self._target_set = frozenset(self._target_codes_norm)
        self._en_to_x_names = [
            name
            for code, name in zip(self._target_codes_norm, self._target_names, strict=True)
            if code != "en"
        ]

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.source_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.source_lang_name_key, self.translate_to_key]

    def process(self, task: AudioTask) -> AudioTask:
        raw_src = task.data.get(self.source_lang_key, "")
        src_norm = _normalize_code(raw_src) if raw_src else ""

        if raw_src:
            task.data[self.source_lang_name_key] = lang_code_to_name(raw_src)

        if src_norm == "en":
            targets = list(self._en_to_x_names)
        elif src_norm and src_norm in self._target_set:
            targets = [LANGUAGE_MAP["en"]]
        else:
            targets = []
            if not src_norm:
                logger.warning("LanguageResolver: row with empty/missing source_lang skipped")
            else:
                logger.warning(
                    "LanguageResolver: source_lang '{}' not in target set; row skipped",
                    raw_src,
                )

        task.data[self.translate_to_key] = targets
        return task
