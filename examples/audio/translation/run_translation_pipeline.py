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

Architecture
------------
::

    ShardedManifestReaderStage   (CPU, _EmptyTask → AudioTask)
        Reads one or more JSONL manifests in shards; skips shards that are
        already complete from a prior run (.done markers).  Also resolves
        source_lang ISO code → display name and writes the per-row
        translate_to list (En→X / X→En only).

    LLMTranslationStage          (GPU, AudioTask → AudioTask)
        Batched vLLM inference; writes data["translations"]
        as {display_name: translated_text}.

    TranslationExpanderStage     (CPU, AudioTask → list[AudioTask])
        Fan-out: one task per direction with flat schema
        {text, source_lang, target_lang (ISO), translation}.

    DirectionalShardedWriterStage (CPU, AudioTask → AudioTask)
        Writes tasks to {output_dir}/shards/{shard_id}_{src}-{tgt}.jsonl;
        renames to .done when the shard is complete.

After the pipeline, ``reconcile_manifests`` merges completed shards into::

    {output_dir}/{manifest_stem}_{src}-{tgt}.jsonl

e.g. ``m1_en-de.jsonl``, ``m1_en-fr.jsonl``, ``m2_en-de.jsonl``, …

Resume behaviour
----------------
Re-running with the same ``--output_dir`` automatically skips shards whose
``.done`` files are present.  Only failed or partial shards are re-processed.

Example
-------
::

    python run_translation_pipeline.py \\
        --manifest m1.jsonl m2.jsonl \\
        --output_dir /data/translations \\
        --target_langs de fr ru ja \\
        --model_id Qwen/Qwen3-8B \\
        --shard_size 500
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
    DirectionalShardedWriterStage,
    LLMTranslationStage,
    ShardedManifestReaderStage,
    TranslationExpanderStage,
    all_shards_done,
    mark_complete_shards,
    reconcile_manifests,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LLM translation pipeline (text-only, vLLM).")

    # ------------------------------------------------------------------ I/O
    ap.add_argument(
        "--manifest",
        type=str,
        required=True,
        nargs="+",
        help="Input JSONL manifest path(s). Multiple paths can be passed space-separated.",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "Output directory. Shard files land in {output_dir}/shards/; "
            "reconciled per-manifest per-direction files land directly in {output_dir}/."
        ),
    )
    ap.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Lines per shard (default: 1000). Smaller shards = finer resume granularity.",
    )

    # ------------------------------------------------------------------ Languages
    ap.add_argument(
        "--target_langs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Target language ISO codes (e.g. 'de fr ru ja'). "
            "ShardedManifestReaderStage generates En→X and X→En pairs."
        ),
    )
    ap.add_argument(
        "--source_lang_code_key",
        type=str,
        default="source_lang",
        help="Input manifest key holding the source language ISO code.",
    )
    ap.add_argument(
        "--source_lang_name_key",
        type=str,
        default="source_lang_name",
        help="Intermediate key for resolved source language display name (written by the reader).",
    )
    ap.add_argument(
        "--target_lang_key",
        type=str,
        default="translate_to",
        help="Intermediate key for per-row list of target language display names (written by the reader).",
    )

    # ------------------------------------------------------------------ Model
    ap.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3.5-35B-A3B-FP8",
        help="Translation LLM model ID.",
    )

    # Prompt overrides (mutually exclusive pairs)
    tpg = ap.add_mutually_exclusive_group()
    tpg.add_argument("--translation_prompt", type=str, default=None)
    tpg.add_argument("--translation_prompt_file", type=str, default=None)

    spg = ap.add_mutually_exclusive_group()
    spg.add_argument("--system_prompt", type=str, default=None)
    spg.add_argument("--system_prompt_file", type=str, default=None)

    ap.add_argument("--text_key", type=str, default="text", help="Manifest key for source text.")
    ap.add_argument(
        "--translations_key",
        type=str,
        default="translations",
        help="Intermediate key for the {lang_name: text} dict produced by LLMTranslationStage.",
    )

    # vLLM params
    ap.add_argument("--tensor_parallel_size", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_output_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_num_seqs", type=int, default=16)
    ap.add_argument("--max_num_batched_tokens", type=int, default=None)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    ap.add_argument("--kv_cache_dtype", type=str, default="fp8")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--presence_penalty", type=float, default=1.5)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)

    # ------------------------------------------------------------------ Executor
    ap.add_argument(
        "--execution_mode",
        type=str,
        default="streaming",
        choices=["streaming", "batch"],
    )
    return ap


def main() -> None:
    args = _build_arg_parser().parse_args()

    stages = [
        ShardedManifestReaderStage(
            manifest_paths=args.manifest,
            output_dir=args.output_dir,
            target_lang_codes=args.target_langs,
            source_lang_key=args.source_lang_code_key,
            source_lang_name_key=args.source_lang_name_key,
            translate_to_key=args.target_lang_key,
            shard_size=args.shard_size,
        ),
        LLMTranslationStage(
            model_id=args.model_id,
            translation_prompt=args.translation_prompt,
            translation_prompt_file=args.translation_prompt_file,
            system_prompt=args.system_prompt,
            system_prompt_file=args.system_prompt_file,
            text_key=args.text_key,
            source_lang_key=args.source_lang_name_key,
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
        TranslationExpanderStage(
            source_lang_key=args.source_lang_code_key,
            translations_key=args.translations_key,
        ),
        DirectionalShardedWriterStage(
            output_dir=args.output_dir,
            source_lang_key="source_lang",
            target_lang_key="target_lang",
        ),
    ]

    pipeline = Pipeline(name="translation_pipeline", stages=stages)
    logger.info("Pipeline:\n{}", pipeline.describe())

    t0 = time.time()
    if all_shards_done(
        manifest_paths=args.manifest,
        output_dir=args.output_dir,
        target_lang_codes=args.target_langs,
        shard_size=args.shard_size,
    ):
        logger.info("All shards are already complete — skipping pipeline.run().")
    else:
        executor = XennaExecutor(config={"execution_mode": args.execution_mode})
        pipeline.run(executor=executor)
        logger.info("Pipeline finished in {:.1f} min.", (time.time() - t0) / 60)

        # Write .shard.done markers for every fully-written shard.  Xenna does
        # not call teardown() on our stages, so this must be done here.
        n_marked = mark_complete_shards(output_dir=args.output_dir)
        logger.info("mark_complete_shards: {} new shard marker(s) written.", n_marked)

    logger.info("Reconciling shards → per-manifest per-direction manifests …")
    t1 = time.time()
    reconcile_manifests(output_dir=args.output_dir)
    logger.info("Reconciliation finished in {:.1f} s.", time.time() - t1)

    logger.info(
        "Done. Output files are in: {}",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
