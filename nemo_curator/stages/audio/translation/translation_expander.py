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

"""Expand a single translated AudioTask into one task per direction.

``LLMTranslationStage`` accumulates all translations inside
``data["translations"]`` as ``{display_name: text}``.  This stage fans
that dict out into one ``AudioTask`` per target language, using the flat
schema expected by downstream consumers:

    {"text": "...", "source_lang": "en", "target_lang": "de", "translation": "..."}

The ``translations`` dict and the resolver scratch fields
(``source_lang_name``, ``translate_to``) are removed from every output task.
Shard metadata (``shard_id``, ``manifest_stem``, ``shard_item_idx``,
``shard_total``) is forwarded unchanged in ``_metadata``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from nemo_curator.stages.audio.translation.language_map import name_to_code
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask

# Scratch keys written by LanguageResolverStage that must not appear in output.
_RESOLVER_SCRATCH_KEYS = ("source_lang_name", "translate_to")


@dataclass
class TranslationExpanderStage(ProcessingStage[AudioTask, AudioTask]):
    """Fan-out: emit one ``AudioTask`` per translation direction.

    Reads ``data[translations_key]`` (a ``{display_name: text}`` dict
    produced by ``LLMTranslationStage``) and emits one task per entry,
    setting ``target_lang`` (ISO code) and ``translation`` (translated
    text) while copying all other source fields.

    Args:
        source_lang_key:  Key holding the source language ISO code (default: ``"source_lang"``).
        translations_key: Key holding the ``{display_name: text}`` dict (default: ``"translations"``).
        target_lang_key:  Key to write the target language ISO code into (default: ``"target_lang"``).
        translation_key:  Key to write the translated text into (default: ``"translation"``).
    """

    name: str = "TranslationExpander"
    source_lang_key: str = "source_lang"
    translations_key: str = "translations"
    target_lang_key: str = "target_lang"
    translation_key: str = "translation"

    _n_expanded: int = field(default=0, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.translations_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.target_lang_key, self.translation_key]

    def process(self, task: AudioTask) -> list[AudioTask]:  # type: ignore[override]
        translations: dict[str, str] = task.data.get(self.translations_key) or {}
        if not translations:
            logger.warning(
                "TranslationExpander: task {} has empty translations dict; dropping",
                task.task_id,
            )
            return []

        # Build clean source payload: remove translations dict and resolver scratch.
        source_data = {
            k: v
            for k, v in task.data.items()
            if k != self.translations_key and k not in _RESOLVER_SCRATCH_KEYS
        }

        results: list[AudioTask] = []
        for display_name, translated_text in translations.items():
            try:
                tgt_code = name_to_code(display_name)
            except KeyError:
                logger.warning(
                    "TranslationExpander: unknown display name '{}' for task {}; skipping",
                    display_name,
                    task.task_id,
                )
                continue

            if not translated_text or not translated_text.strip():
                logger.warning(
                    "TranslationExpander: empty translation for target '{}' in task {}; "
                    "emitting row with empty translation to preserve row counts",
                    display_name,
                    task.task_id,
                )
                translated_text = ""

            output_data = dict(source_data)
            output_data[self.target_lang_key] = tgt_code
            output_data[self.translation_key] = translated_text

            results.append(
                AudioTask(
                    task_id=f"{task.task_id}_{tgt_code}",
                    dataset_name=task.dataset_name,
                    data=output_data,
                    _metadata=dict(task._metadata),
                    _stage_perf=list(task._stage_perf),
                )
            )

        self._n_expanded += len(results)
        logger.debug(
            "TranslationExpander: task {} → {} direction task(s)", task.task_id, len(results)
        )
        return results

    def teardown(self) -> None:
        if self._n_expanded:
            logger.info("TranslationExpander: total expanded tasks: {}", self._n_expanded)
