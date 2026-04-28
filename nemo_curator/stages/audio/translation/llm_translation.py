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

"""LLM-based translation stage using vLLM.

Reads source-language text plus per-task ``source_lang`` and
``target_lang`` tags from each ``AudioTask``, runs a text LLM
(Qwen3.5 by default), and stores the translation under
``data["translations"][target_lang]`` so multiple target languages
accumulate across stage invocations.
"""

from __future__ import annotations

from collections import defaultdict
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


@dataclass
class LLMTranslationStage(ProcessingStage[AudioTask, AudioTask]):
    """Translate source text to a target language via batched LLM inference.

    Reads ``text_key`` (source text), ``source_lang_key`` and
    ``target_lang_key`` (language tags, e.g. ``"en"``, ``"de"``) from
    each ``AudioTask.data`` dict.  Writes the result into
    ``data[translations_key]`` as a ``{target_lang: translation}``
    mapping so the same source text can collect translations for
    multiple languages across runs without overwriting prior entries.

    The stage ships with a default translation prompt template at
    ``prompts/translation_prompt.md``.  Override with
    ``translation_prompt`` (inline string) or
    ``translation_prompt_file`` (path to a markdown file).  An
    optional ``system_prompt`` can be supplied separately.

    Uses ``process_batch`` for efficient batched GPU inference via
    vLLM with prefix caching enabled (shared prompt prefixes are
    cached across requests in the batch).

    Args:
        model_id: HuggingFace model identifier for the text LLM.
        translation_prompt: Inline translation prompt template with
            ``{text}``, ``{source_lang}`` and ``{target_lang}``
            placeholders.  Takes precedence over
            ``translation_prompt_file``.
        translation_prompt_file: Path to a file containing the
            translation prompt template.  Falls back to the bundled
            default if neither ``translation_prompt`` nor
            ``translation_prompt_file`` is set.
        system_prompt: Optional system prompt for the LLM.
        text_key: Input manifest key holding the source text.
        source_lang_key: Input manifest key holding the source
            language tag.
        target_lang_key: Input manifest key holding the target
            language tag.
        translations_key: Output manifest key under which the
            ``{target_lang: translation}`` dict is stored.
        skip_me_key: Key used to flag entries to skip (consistent
            with PnC / ITN stages).
        tensor_parallel_size: GPUs for tensor parallelism (``None``
            = auto-detect).
        max_output_tokens: Maximum tokens to generate per sample.
        max_model_len: Maximum context length passed to vLLM.
        max_num_seqs: Maximum concurrent sequences in vLLM.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        kv_cache_dtype: KV-cache dtype (``fp8`` halves memory, 2x
            concurrent sequences on Hopper).
        temperature: Sampling temperature (0.0 = greedy).
    """

    name: str = "LLMTranslation"
    model_id: str = "Qwen/Qwen3.5-35B-A3B-FP8"
    translation_prompt: str | None = None
    translation_prompt_file: str | None = None
    system_prompt: str | None = None
    text_key: str = "text"
    source_lang_key: str = "source_lang"
    target_lang_key: str = "target_lang"
    translations_key: str = "translations"
    skip_me_key: str = "_skip_me"
    tensor_parallel_size: int | None = None
    max_output_tokens: int = 1024
    max_model_len: int = 4096
    max_num_seqs: int = 16
    gpu_memory_utilization: float = 0.95
    kv_cache_dtype: str = "fp8"
    temperature: float = 0.0
    resources: Resources = field(default_factory=lambda: Resources(gpus=1.0))
    batch_size: int = 64

    _llm: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _sampling_params: Any = field(default=None, init=False, repr=False)
    _translation_prompt: str = field(default="", init=False, repr=False)
    _n_processed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        tp = self.tensor_parallel_size
        if tp and tp > 0:
            self.resources = Resources(gpus=float(tp))

    # ------------------------------------------------------------------
    # Prompt resolution
    # ------------------------------------------------------------------

    def _resolve_translation_prompt(self) -> str:
        if self.translation_prompt:
            return self.translation_prompt
        path = Path(self.translation_prompt_file) if self.translation_prompt_file else _DEFAULT_PROMPT_PATH
        logger.info("LLMTranslation: loading prompt from {}", path)
        if not path.exists():
            raise FileNotFoundError(f"Translation prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _init_model(self) -> None:
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for LLMTranslationStage. pip install vllm")

        self._translation_prompt = self._resolve_translation_prompt()

        from nemo_curator.utils.gpu_utils import get_gpu_count

        tp = self.tensor_parallel_size or get_gpu_count()

        logger.info(
            "LLMTranslation: loading {} (tp={}, max_model_len={}, kv_cache_dtype={})",
            self.model_id,
            tp,
            self.max_model_len,
            self.kv_cache_dtype,
        )

        self._llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=tp,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=8192,
            enable_prefix_caching=True,
            prefix_caching_hash_algo="xxhash",
            kv_cache_dtype=self.kv_cache_dtype,
            enforce_eager=False,
            seed=1234,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
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
        return [], [self.text_key, self.source_lang_key, self.target_lang_key, self.skip_me_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.translations_key]

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format_prompt(self, data: dict) -> str:
        user_content = self._translation_prompt.format_map(defaultdict(str, data))
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
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

        valid_indices: list[int] = []
        prompts: list[str] = []

        for i, task in enumerate(tasks):
            data = task.data
            if data.get(self.skip_me_key, ""):
                continue
            text = data.get(self.text_key, "")
            source_lang = data.get(self.source_lang_key, "")
            target_lang = data.get(self.target_lang_key, "")
            if not text or not text.strip() or not source_lang or not target_lang:
                continue
            valid_indices.append(i)
            prompts.append(self._format_prompt(dict(data)))

        if prompts:
            outputs = self._llm.generate(
                prompts,
                sampling_params=self._sampling_params,
                use_tqdm=False,
            )

            for seq_idx, task_idx in enumerate(valid_indices):
                task = tasks[task_idx]
                target_lang = task.data[self.target_lang_key]
                translation = outputs[seq_idx].outputs[0].text.strip()

                translations = task.data.get(self.translations_key) or {}
                translations[target_lang] = translation
                task.data[self.translations_key] = translations
                self._n_processed += 1

        logger.debug("LLMTranslation: batch of {} tasks ({} translated)", len(tasks), len(prompts))
        return tasks
