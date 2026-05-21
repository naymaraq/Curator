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

"""LLM-based translation pipeline (text-only, vLLM).

Reads a JSONL manifest of ``text`` + ``source_lang`` (code) entries and
produces ``data["translations"]`` keyed by target language display name.
Direction is restricted to ``En->X`` and ``X->En``, where ``X`` ranges
over the codes passed via ``--target_langs``.

Architecture:
    ManifestReader (CPU)
        -> reads JSONL manifest(s), emits one Task per line
    LanguageResolverStage (CPU)
        -> resolves source_lang code -> source_lang_name (display name)
           and writes per-row translate_to list (En->X / X->En only)
    LLMTranslationStage (GPU)
        -> batched vLLM inference, writes data["translations"][lang_name]
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
    LanguageResolverStage,
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
    ap.add_argument(
        "--output_manifest", 
        type=str, 
        required=True, 
        help="Output JSONL manifest path."
    )
    ap.add_argument(
        "--model_id", 
        type=str, 
        default="Qwen/Qwen3.5-35B-A3B-FP8", 
        help="Translation LLM model id."
    )
    translation_prompt_group = ap.add_mutually_exclusive_group()
    translation_prompt_group.add_argument(
        "--translation_prompt",
        type=str,
        default=None,
        help="Optional inline translation prompt template. Mutually exclusive with --translation_prompt_file.",
    )
    translation_prompt_group.add_argument(
        "--translation_prompt_file",
        type=str,
        default=None,
        help=(
            "Path to translation prompt template. Falls back to the stage's bundled default if unset. "
            "Mutually exclusive with --translation_prompt."
        ),
    )
    system_prompt_group = ap.add_mutually_exclusive_group()
    system_prompt_group.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional inline system prompt string. Mutually exclusive with --system_prompt_file.",
    )
    system_prompt_group.add_argument(
        "--system_prompt_file",
        type=str,
        default=None,
        help="Optional path to a system prompt file. Mutually exclusive with --system_prompt.",
    )

    ap.add_argument(
        "--text_key", 
        type=str, 
        default="text", 
        help="Manifest key holding the source text."
    )
    ap.add_argument(
        "--target_langs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Target language codes (e.g. 'en sv fr'). LanguageResolverStage resolves them "
            "to display names and writes per-row translate_to lists, restricted to En->X "
            "and X->En pairs."
        ),
    )
    ap.add_argument(
        "--source_lang_code_key",
        type=str,
        default="source_lang",
        help=(
            "Input manifest key holding the source language CODE. The resolver reads this "
            "and writes the resolved display name to --source_lang_key."
        ),
    )
    ap.add_argument(
        "--source_lang_key",
        type=str,
        default="source_lang_name",
        help=(
            "Manifest key holding the source language DISPLAY NAME (resolver output, "
            "translator input)."
        ),
    )
    ap.add_argument(
        "--target_lang_key",
        type=str,
        default="translate_to",
        help=(
            "Manifest key holding the per-row list of target language display names "
            "(resolver output, translator input)."
        ),
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
    ap.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=None,
        help="vLLM max batched tokens per step. Defaults to max(max_model_len, 8192).",
    )
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--kv_cache_dtype", type=str, default="fp8", help="KV-cache dtype passed to vLLM.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--presence_penalty", type=float, default=1.5)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234, help="Seed for vLLM and sampling params.")

    ap.add_argument(
        "--execution_mode",
        type=str,
        default="streaming",
        choices=["streaming", "batch"],
        help="Xenna execution mode.",
    )
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    stages = [
        ManifestReader(manifest_path=args.manifest),
        LanguageResolverStage(
            target_lang_codes=args.target_langs,
            source_lang_key=args.source_lang_code_key,
            source_lang_name_key=args.source_lang_key,
            translate_to_key=args.target_lang_key,
        ),
        LLMTranslationStage(
            model_id=args.model_id,
            translation_prompt=args.translation_prompt,
            translation_prompt_file=args.translation_prompt_file,
            system_prompt=args.system_prompt,
            system_prompt_file=args.system_prompt_file,
            text_key=args.text_key,
            source_lang_key=args.source_lang_key,
            target_lang_key=args.target_lang_key,
            translations_key=args.translations_key,
            tensor_parallel_size=args.tensor_parallel_size,
            max_output_tokens=args.max_output_tokens,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_dtype=args.kv_cache_dtype,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
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

