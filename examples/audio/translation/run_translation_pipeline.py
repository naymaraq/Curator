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

"""LLM-based translation pipeline.

Runs ``LLMTranslationStage`` on a JSONL manifest whose entries already
carry per-row ``text``, ``source_lang`` and ``target_lang`` fields.

Architecture:
    ManifestReader (CPU)
        -> reads JSONL manifest(s), emits one AudioTask per line
    LLMTranslationStage (GPU)
        -> batched vLLM inference, writes data["translations"][target_lang]
    ManifestWriterStage (CPU)
        -> appends each translated entry to a single JSONL output
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

import argparse
import time

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.audio import (
    LLMTranslationStage,
    ManifestReader,
    ManifestWriterStage,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LLM translation pipeline (text-only, vLLM).")
    ap.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Input JSONL manifest path (file or directory). Multiple paths can be passed space-separated.",
        nargs="+",
    )
    ap.add_argument("--output_manifest", type=str, required=True, help="Output JSONL manifest path.")

    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-35B-A3B-FP8", help="Translation LLM model id.")
    ap.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Path to translation prompt template. Falls back to the stage's bundled default if unset.",
    )
    ap.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt. Inline string or path to a file.",
    )

    ap.add_argument("--text_key", type=str, default="text", help="Manifest key holding the source text.")
    ap.add_argument(
        "--source_lang_key", type=str, default="source_lang", help="Manifest key holding the source language tag."
    )
    ap.add_argument(
        "--target_lang_key", type=str, default="target_lang", help="Manifest key holding the target language tag."
    )
    ap.add_argument(
        "--translations_key",
        type=str,
        default="translations",
        help="Output manifest key under which the {target_lang: translation} dict is stored.",
    )

    ap.add_argument("--tensor_parallel_size", type=int, default=None, help="GPUs for tensor parallelism (auto if unset).")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_output_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_num_seqs", type=int, default=16)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--kv_cache_dtype", type=str, default="fp8", help="KV-cache dtype passed to vLLM.")
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument(
        "--execution_mode",
        type=str,
        default="streaming",
        choices=["streaming", "batch"],
        help="Xenna execution mode.",
    )
    return ap


def _resolve_system_prompt(value: str | None) -> str | None:
    if not value:
        return None
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            return f.read().strip()
    return value


def main() -> None:
    args = _build_arg_parser().parse_args()

    system_prompt = _resolve_system_prompt(args.system_prompt)

    manifest_path: str | list[str] = args.manifest if len(args.manifest) > 1 else args.manifest[0]

    stages = [
        ManifestReader(manifest_path=manifest_path),
        LLMTranslationStage(
            model_id=args.model_id,
            translation_prompt_file=args.prompt_file,
            system_prompt=system_prompt,
            text_key=args.text_key,
            source_lang_key=args.source_lang_key,
            target_lang_key=args.target_lang_key,
            translations_key=args.translations_key,
            tensor_parallel_size=args.tensor_parallel_size,
            max_output_tokens=args.max_output_tokens,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_dtype=args.kv_cache_dtype,
            temperature=args.temperature,
            batch_size=args.batch_size,
        ),
        ManifestWriterStage(output_path=args.output_manifest),
    ]

    pipeline = Pipeline(name="translation_pipeline", stages=stages)
    logger.info(f"Pipeline: {pipeline.describe()}")

    executor = XennaExecutor(config={"execution_mode": args.execution_mode})

    t0 = time.time()
    pipeline.run(executor=executor)
    elapsed = time.time() - t0
    logger.info(f"Pipeline finished in {elapsed / 60:.1f} min. Output: {args.output_manifest}")


if __name__ == "__main__":
    main()

