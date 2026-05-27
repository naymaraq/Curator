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

"""Sharded manifest reader with resume support for the translation pipeline.

Each input manifest is split into virtual chunks of ``shard_size`` lines.
A shard is considered *done* when a single ``{shard_id}.shard.done`` marker
file (written by ``mark_complete_shards()`` after ``pipeline.run()`` returns)
exists in ``{output_dir}/shards/``.  Done shards are skipped on subsequent
pipeline calls, enabling cheap resume after failures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.audio.translation.language_map import LANGUAGE_MAP, _normalize_code, lang_code_to_name
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask, _EmptyTask


def shard_done_marker_path(output_dir: str, shard_id: str) -> str:
    """Return the path of the shard-level completion marker file."""
    return os.path.join(output_dir, "shards", f"{shard_id}.shard.done")


def _shard_is_done(output_dir: str, shard_id: str) -> bool:
    """Return True if the shard-level completion marker exists for this shard."""
    return os.path.exists(shard_done_marker_path(output_dir, shard_id))


def mark_complete_shards(output_dir: str) -> int:
    """Write ``{shard_id}.shard.done`` for every shard with no orphan ``.jsonl`` files.

    Called from the pipeline runner after ``pipeline.run()`` returns.  At that
    point all tasks have been processed, so any shard whose direction files have
    all been renamed to ``.jsonl.done`` (no bare ``.jsonl`` remaining) is fully
    complete and can be marked for skipping on the next run.

    Args:
        output_dir: Root output directory (same as passed to the pipeline stages).

    Returns:
        Number of new ``.shard.done`` markers written.
    """
    shards_dir = os.path.join(output_dir, "shards")
    if not os.path.isdir(shards_dir):
        logger.warning("mark_complete_shards: shards dir not found: {}", shards_dir)
        return 0

    done_shards: set[str] = set()
    orphan_shards: set[str] = set()

    for fname in os.listdir(shards_dir):
        # Direction done files: {shard_id}_{src}-{tgt}.jsonl.done
        if fname.endswith(".jsonl.done"):
            base = fname[: -len(".jsonl.done")]
            parts = base.rsplit("_", 1)
            if len(parts) == 2 and "-" in parts[1]:
                done_shards.add(parts[0])
        # Orphan direction files: {shard_id}_{src}-{tgt}.jsonl  (incomplete)
        elif fname.endswith(".jsonl"):
            base = fname[: -len(".jsonl")]
            parts = base.rsplit("_", 1)
            if len(parts) == 2 and "-" in parts[1]:
                orphan_shards.add(parts[0])

    written = 0
    for shard_id in done_shards - orphan_shards:
        marker = shard_done_marker_path(output_dir, shard_id)
        if not os.path.exists(marker):
            try:
                with open(marker, "w") as _f:
                    pass
                written += 1
                logger.info("mark_complete_shards: shard marker written → {}", marker)
            except OSError as exc:
                logger.warning("mark_complete_shards: could not write marker {}: {}", marker, exc)

    logger.info(
        "mark_complete_shards: {}/{} shard(s) marked done ({} already had marker)",
        written,
        len(done_shards - orphan_shards),
        len(done_shards - orphan_shards) - written,
    )
    if orphan_shards:
        logger.warning(
            "mark_complete_shards: {} shard(s) still have incomplete direction files: {}",
            len(orphan_shards),
            sorted(orphan_shards),
        )
    return written


def all_shards_done(
    manifest_paths: list[str],
    output_dir: str,
    target_lang_codes: list[str],
    shard_size: int = 1000,
) -> bool:
    """Return True if every shard across all manifests is already complete.

    Useful as a pre-flight check before starting the pipeline — if this
    returns True the caller can skip ``pipeline.run()`` entirely and go
    straight to ``reconcile_manifests()``, avoiding unnecessary Ray worker
    initialisation (and potential segfaults during setup).

    Args:
        manifest_paths:    Same list passed to ``ShardedManifestReaderStage``.
        output_dir:        Same ``output_dir`` passed to the stage.
        target_lang_codes: Unused — kept for API compatibility.
        shard_size:        Same shard size passed to the stage.
    """
    for manifest_path in manifest_paths:
        for _path, shard_idx, _start, _end in _collect_shard_descriptors(manifest_path, shard_size):
            shard_id = f"{Path(manifest_path).stem}_{shard_idx}"
            if not _shard_is_done(output_dir, shard_id):
                return False
    return True


def _collect_shard_descriptors(
    manifest_path: str,
    shard_size: int,
) -> list[tuple[str, int, int, int]]:
    """Return list of (manifest_path, shard_idx, start_line, end_line) tuples."""
    fs, resolved = url_to_fs(manifest_path)
    total = 0
    with fs.open(resolved, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1

    descriptors = []
    shard_idx = 0
    start = 0
    while start < total:
        end = min(start + shard_size, total)
        descriptors.append((manifest_path, shard_idx, start, end))
        shard_idx += 1
        start = end
    return descriptors


@dataclass
class ShardedManifestReaderStage(ProcessingStage[_EmptyTask, AudioTask]):
    """Read JSONL manifests in shards; skip already-completed shards.

    Splits every input manifest into virtual chunks of ``shard_size`` lines.
    For each shard, checks for a ``{shard_id}.shard.done`` marker written by
    ``mark_complete_shards()`` after the pipeline has finished.  If the marker
    exists the shard is skipped entirely (resume behaviour).

    Per-row enrichment
    ------------------
    In addition to reading and sharding, this stage also writes the
    fields that the LLM translation stage consumes (formerly produced by
    the now-deleted ``LanguageResolverStage``):

    - ``source_lang_name``: display name of the source language
      (e.g. ``"German"``), resolved from ``source_lang`` via ``LANGUAGE_MAP``.
    - ``translate_to``:     list of target display names per the direction
      rules below.

    Direction rules (``T`` = normalized set of ``target_lang_codes``):
        * ``source_lang == "en"``           -> translate_to = T \\ {"en"}  (En->X)
        * ``source_lang in T \\ {"en"}``     -> translate_to = ["English"] (X->En)
        * otherwise                          -> translate_to = []          (skipped)

    Each emitted ``AudioTask`` carries the following ``_metadata`` keys:

    - ``shard_id``:          ``"{manifest_stem}_{shard_idx}"``
    - ``manifest_stem``:     stem of the source manifest filename
    - ``shard_item_idx``:    0-based index of this row within the shard
    - ``shard_total``:       total source rows in the shard
    - ``direction_counts``:  ``dict[str, int]`` mapping ``"{src}-{tgt}"`` to the
                             number of rows in the shard that will produce tasks
                             for that direction.  ``DirectionalShardedWriterStage``
                             uses this to know when a (shard, direction) is
                             fully written and can be renamed to ``.jsonl.done``.

    Args:
        manifest_paths:       One or more JSONL manifest paths (local or cloud).
        output_dir:           Output directory passed to ``DirectionalShardedWriterStage``;
                              used to locate shard completion markers.
        target_lang_codes:    ISO 639-1 codes of all target languages.  Used to
                              compute per-direction expected counts and to
                              populate per-row ``translate_to``.
        source_lang_key:      Row key holding the source-language ISO code
                              (default: ``"source_lang"``).
        source_lang_name_key: Row key to write the resolved source display name
                              into (default: ``"source_lang_name"``).
        translate_to_key:     Row key to write the per-row list of target display
                              names into (default: ``"translate_to"``).
        shard_size:           Number of lines per shard (default: 1 000).
    """

    manifest_paths: list[str] = field(default_factory=list)
    output_dir: str = ""
    target_lang_codes: list[str] = field(default_factory=list)
    source_lang_key: str = "source_lang"
    source_lang_name_key: str = "source_lang_name"
    translate_to_key: str = "translate_to"
    shard_size: int = 1000
    name: str = "ShardedManifestReader"

    _target_codes_norm: list[str] = field(default_factory=list, init=False, repr=False)
    _target_set: frozenset[str] = field(default_factory=frozenset, init=False, repr=False)
    _en_to_x_codes: list[str] = field(default_factory=list, init=False, repr=False)
    _en_to_x_names: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.manifest_paths:
            msg = "ShardedManifestReaderStage: manifest_paths must be non-empty"
            raise ValueError(msg)
        if not self.output_dir:
            msg = "ShardedManifestReaderStage: output_dir must be set"
            raise ValueError(msg)
        if not self.target_lang_codes:
            msg = "ShardedManifestReaderStage: target_lang_codes must be non-empty"
            raise ValueError(msg)

        self._target_codes_norm = [_normalize_code(c) for c in self.target_lang_codes]
        self._target_set = frozenset(self._target_codes_norm)
        self._en_to_x_codes = [c for c in self._target_codes_norm if c != "en"]
        # Display-name version of _en_to_x_codes, used to populate translate_to
        # for English source rows.  Fails loudly if a configured target code is
        # not in LANGUAGE_MAP.
        self._en_to_x_names = [LANGUAGE_MAP[c] for c in self._en_to_x_codes]

    def _row_targets(self, src_norm: str) -> list[str]:
        """Direction rules: enumerate per-row target ISO codes."""
        if src_norm == "en":
            return list(self._en_to_x_codes)
        if src_norm and src_norm in self._target_set:
            return ["en"]
        return []

    def _row_target_names(self, src_norm: str) -> list[str]:
        """Direction rules: enumerate per-row target display names."""
        if src_norm == "en":
            return list(self._en_to_x_names)
        if src_norm and src_norm in self._target_set:
            return [LANGUAGE_MAP["en"]]
        return []

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], [self.source_lang_name_key, self.translate_to_key]

    def process(self, task: _EmptyTask) -> list[AudioTask]:  # type: ignore[override]
        results: list[AudioTask] = []
        total_skipped_shards = 0
        total_read_shards = 0

        for manifest_path in self.manifest_paths:
            manifest_stem = Path(manifest_path).stem
            descriptors = _collect_shard_descriptors(manifest_path, self.shard_size)
            logger.info(
                "ShardedManifestReader: {} → {} shard(s) of up to {} lines",
                manifest_path,
                len(descriptors),
                self.shard_size,
            )

            for manifest_path_inner, shard_idx, start_line, end_line in descriptors:
                shard_id = f"{manifest_stem}_{shard_idx}"

                if _shard_is_done(self.output_dir, shard_id):
                    logger.info("ShardedManifestReader: skipping done shard {}", shard_id)
                    total_skipped_shards += 1
                    continue

                shard_results = self._read_shard(
                    manifest_path_inner, shard_id, manifest_stem, start_line, end_line
                )
                results.extend(shard_results)
                total_read_shards += 1

        logger.info(
            "ShardedManifestReader: {} shard(s) read ({} tasks), {} shard(s) skipped (done)",
            total_read_shards,
            len(results),
            total_skipped_shards,
        )
        return results

    def _read_shard(
        self,
        manifest_path: str,
        shard_id: str,
        manifest_stem: str,
        start_line: int,
        end_line: int,
    ) -> list[AudioTask]:
        shard_total = end_line - start_line
        fs, resolved = url_to_fs(manifest_path)
        rows: list[dict[str, Any]] = []
        line_no = 0

        with fs.open(resolved, "r", encoding="utf-8") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                if line_no < start_line:
                    line_no += 1
                    continue
                if line_no >= end_line:
                    break
                rows.append(json.loads(raw_line.strip()))
                line_no += 1

        # Single pass over rows: enrich with source_lang_name + translate_to
        # (formerly done by LanguageResolverStage) and accumulate the
        # per-direction expected row counts the writer relies on.
        direction_counts: dict[str, int] = {}
        for row in rows:
            src_raw = row.get(self.source_lang_key, "")
            src_norm = _normalize_code(src_raw) if src_raw else ""

            # Per-row enrichment.  source_lang_name is only set when the source
            # is non-empty so empty rows do not silently get a wrong name.
            if src_raw:
                row[self.source_lang_name_key] = lang_code_to_name(src_raw)
            row[self.translate_to_key] = self._row_target_names(src_norm)

            for tgt_norm in self._row_targets(src_norm):
                key = f"{src_norm}-{tgt_norm}"
                direction_counts[key] = direction_counts.get(key, 0) + 1

        tasks: list[AudioTask] = []
        for item_idx, row in enumerate(rows):
            metadata: dict[str, Any] = {
                "shard_id": shard_id,
                "manifest_stem": manifest_stem,
                "shard_item_idx": item_idx,
                "shard_total": shard_total,
                "direction_counts": dict(direction_counts),
            }
            tasks.append(
                AudioTask(
                    task_id=f"{shard_id}_{item_idx}",
                    dataset_name=manifest_stem,
                    data=row,
                    _metadata=metadata,
                )
            )

        logger.debug(
            "ShardedManifestReader: shard {} lines [{}, {}) → {} tasks, direction_counts={}",
            shard_id,
            start_line,
            end_line,
            len(tasks),
            direction_counts,
        )
        return tasks

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
