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


from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from nemo_curator.backends.base import NodeInfo, WorkerMetadata

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import AudioTask

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "translation_prompt.md"
_DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt.md"


@dataclass
class LLMTranslationStage(ProcessingStage[AudioTask, AudioTask]):
    """Translate source text to a target language via batched vLLM inference.

    Reads source text plus pre-resolved language **display names** from each
    ``AudioTask.data`` dict (``source_lang_key`` and ``target_lang_key``;
    populated upstream by ``LanguageResolverStage``) and writes the result
    into ``data[translations_key]`` as a ``{display_name: translation}``
    mapping, so multiple target languages accumulate without overwriting
    prior entries. The scratch fields ``source_lang_key`` and
    ``target_lang_key`` are removed from each row after the prompts are
    built, so they don't bleed into the output manifest.

    The prompt template uses fixed semantic placeholders ``{target_lang}``,
    ``{source_lang}``, ``{text}`` regardless of the actual manifest key
    names. ``{source_lang}`` is always populated from ``source_lang_key``,
    which is mandatory and assumed pre-resolved by ``LanguageResolverStage``.

    Both the translation prompt and the system prompt have bundled defaults
    in ``prompts/`` (``translation_prompt.md`` and ``system_prompt.md``).
    Override either via the inline-string fields (``translation_prompt``,
    ``system_prompt``) or the ``*_file`` fields (``translation_prompt_file``,
    ``system_prompt_file``); passing both inline and file for the same
    prompt raises.
    """

    name: str = "LLMTranslation"
    model_id: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    translation_prompt: str | None = None
    translation_prompt_file: str | None = None
    system_prompt: str | None = None
    system_prompt_file: str | None = None
    text_key: str = "text"
    source_lang_key: str = "source_lang_name"
    target_lang_key: str = "translate_to"
    translations_key: str = "translations"
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 1024
    max_model_len: int = 4096
    max_num_seqs: int = 16
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "fp8"
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 1.5
    repetition_penalty: float = 1.0
    seed: int = 1234
    log_inputs: int = 5
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 64

    _llm: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _sampling_params: Any = field(default=None, init=False, repr=False)
    _translation_prompt: str = field(default="", init=False, repr=False)
    _system_prompt: str | None = field(default=None, init=False, repr=False)
    _prompt_placeholders: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)
    _n_inputs_logged: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.tensor_parallel_size or self.tensor_parallel_size <= 0:
            from nemo_curator.utils.gpu_utils import get_gpu_count
            self.tensor_parallel_size = get_gpu_count()
        self.resources = Resources(gpus=float(self.tensor_parallel_size))

        self._translation_prompt = self._resolve_prompt(
            inline=self.translation_prompt,
            file_path=self.translation_prompt_file,
            default_path=_DEFAULT_PROMPT_PATH,
            label="translation",
        )
        self._system_prompt = self._resolve_prompt(
            inline=self.system_prompt,
            file_path=self.system_prompt_file,
            default_path=_DEFAULT_SYSTEM_PROMPT_PATH,
            label="system",
        )
        self._prompt_placeholders = frozenset(
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self._translation_prompt)
            if field_name
        )

    # ------------------------------------------------------------------
    # Prompt resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_prompt(
        inline: str | None,
        file_path: str | None,
        default_path: Path | None,
        label: str,
    ) -> str | None:
        if inline and file_path:
            raise ValueError(
                f"LLMTranslation: pass either {label}_prompt or {label}_prompt_file, not both."
            )
        if inline:
            return inline
        path = Path(file_path) if file_path else default_path
        if path is None:
            return None
        logger.info("LLMTranslation: loading {} prompt from {}", label, path)
        if not path.exists():
            raise FileNotFoundError(f"{label.capitalize()} prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for LLMTranslationStage. pip install vllm")

        max_num_batched_tokens = self.max_num_batched_tokens or max(self.max_model_len, 8192)

        logger.info(
            "LLMTranslation: loading {} (tp={}, max_model_len={}, "
            "max_num_batched_tokens={}, kv_cache_dtype={})",
            self.model_id,
            self.tensor_parallel_size,
            self.max_model_len,
            max_num_batched_tokens,
            self.kv_cache_dtype,
        )

        self._llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="xxhash",
            kv_cache_dtype=self.kv_cache_dtype,
            enforce_eager=False,
            seed=self.seed,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_output_tokens,
            seed=self.seed,
        )

        logger.info(
            "LLMTranslation: model ready (prefix_caching=True, prompt={} chars)",
            len(self._translation_prompt),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        self._init_model()

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if self._llm is None:
            self._init_model()

    def teardown(self) -> None:
        if self._n_processed:
            logger.info("LLMTranslation: processed {} entries", self._n_processed)
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._tokenizer = None
            self._sampling_params = None

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.text_key, self.target_lang_key, self.source_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.translations_key]

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _build_prompt_values(
        self, data: dict, target_lang: str, source_lang: str
    ) -> dict[str, str]:

        semantic: dict[str, str] = {
            "target_lang": target_lang,
            "source_lang": source_lang,
        }

        values: dict[str, str] = {}
        missing: list[str] = []
        for placeholder in self._prompt_placeholders:
            if placeholder in semantic:
                value = semantic[placeholder]
            else:
                raw = data.get(placeholder, None)
                value = "" if raw is None else str(raw)
            if not value.strip():
                missing.append(placeholder)
            values[placeholder] = value

        if missing:
            raise ValueError(f"Translation prompt placeholders not filled: {sorted(missing)}")
        return values

    def _format_prompt(
        self, data: dict, target_lang: str, source_lang: str
    ) -> str:
        user_content = self._translation_prompt.format_map(
            self._build_prompt_values(data, target_lang, source_lang)
        )
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_content})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[AudioTask]) -> list[AudioTask]:
        if not tasks:
            return []

        if self._llm is None:
            msg = "Model not initialised — setup() was not called"
            raise RuntimeError(msg)

        prompts: list[str] = []
        prompt_owners: list[tuple[int, str]] = []

        for task_idx, task in enumerate(tasks):
            data = task.data

            # Pop scratch fields written by LanguageResolverStage up-front so
            # they never reach the output manifest, regardless of which skip
            # branch (if any) the task hits below.
            raw_targets = data.pop(self.target_lang_key, None) or []
            source_lang = data.pop(self.source_lang_key, "")

            # Skip tasks with empty text
            text = data.get(self.text_key, "")
            if not text or not text.strip():
                continue

            # Normalize target language(s): accept str or list[str].
            if isinstance(raw_targets, str):
                raw_targets = [raw_targets]
            targets = list(dict.fromkeys(raw_targets)) # Remove duplicates

            # Skip tasks with no targets
            if not targets:
                continue

            for target_lang in targets:
                prompt = self._format_prompt(data, target_lang, source_lang)
                prompts.append(prompt)
                prompt_owners.append((task_idx, target_lang))

                if self._n_inputs_logged < self.log_inputs:
                    self._n_inputs_logged += 1
                    logger.info("\nInput example {}: {}", self._n_inputs_logged, prompt)

        if prompts:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )

            for seq_idx, (task_idx, target_lang) in enumerate(prompt_owners):
                task = tasks[task_idx]
                translation = outputs[seq_idx].outputs[0].text.strip()

                if not translation:
                    logger.warning(
                        "LLMTranslation: empty translation for target={}", target_lang,
                    )

                translations = task.data.get(self.translations_key) or {}
                translations[target_lang] = translation
                task.data[self.translations_key] = translations
                self._n_processed += 1

        logger.debug("LLMTranslation: batch of {} tasks ({} translations)", len(tasks), len(prompts))
        return tasks
