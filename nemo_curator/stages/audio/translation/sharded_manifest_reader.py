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
A shard is considered *done* when all per-direction ``.done`` files produced
by ``DirectionalShardedWriterStage`` exist for it.  Done shards are skipped
on subsequent pipeline calls, enabling cheap resume after failures.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.audio.translation.language_resolver import _normalize_code
from nemo_curator.tasks import AudioTask, _EmptyTask


def _direction_done_path(output_dir: str, shard_id: str, src: str, tgt: str) -> str:
    return os.path.join(output_dir, "shards", f"{shard_id}_{src}-{tgt}.jsonl.done")


def _shard_is_done(output_dir: str, shard_id: str, directions: list[tuple[str, str]]) -> bool:
    """Return True if every direction's .done file exists for this shard."""
    return all(
        os.path.exists(_direction_done_path(output_dir, shard_id, src, tgt))
        for src, tgt in directions
    )


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
        target_lang_codes: Same target language codes passed to the stage.
        shard_size:        Same shard size passed to the stage.
    """
    target_codes = [_normalize_code(c) for c in target_lang_codes]
    directions: list[tuple[str, str]] = []
    for code in target_codes:
        if code != "en":
            directions.append(("en", code))
            directions.append((code, "en"))
    directions = list(dict.fromkeys(directions))

    for manifest_path in manifest_paths:
        for _path, shard_idx, _start, _end in _collect_shard_descriptors(manifest_path, shard_size):
            shard_id = f"{Path(manifest_path).stem}_{shard_idx}"
            if not _shard_is_done(output_dir, shard_id, directions):
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
    For each shard, checks whether ``DirectionalShardedWriterStage`` has
    already produced ``.done`` files for every expected direction; if so,
    the shard is skipped entirely (resume behaviour).

    Each emitted ``AudioTask`` carries the following ``_metadata`` keys:

    - ``shard_id``:        ``"{manifest_stem}_{shard_idx}"``
    - ``manifest_stem``:   stem of the source manifest filename
    - ``shard_item_idx``:  0-based index of this row within the shard
    - ``shard_total``:     total non-empty lines in the shard

    Args:
        manifest_paths:   One or more JSONL manifest paths (local or cloud).
        output_dir:       Output directory passed to ``DirectionalShardedWriterStage``;
                          used to locate ``.done`` marker files.
        target_lang_codes: ISO 639-1 codes of all target languages (used to
                           determine which direction ``.done`` files to check).
        shard_size:       Number of lines per shard (default: 1 000).
    """

    manifest_paths: list[str] = field(default_factory=list)
    output_dir: str = ""
    target_lang_codes: list[str] = field(default_factory=list)
    shard_size: int = 1000
    name: str = "ShardedManifestReader"

    _directions: list[tuple[str, str]] = field(default_factory=list, init=False, repr=False)

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

        # Pre-compute all expected directions so shard-done checks are fast.
        # Mirrors LanguageResolverStage direction rules:
        #   en -> X (for each non-en target)
        #   X -> en (for each non-en target that appears in sources)
        # We conservatively include both directions for every non-en target.
        target_codes = [_normalize_code(c) for c in self.target_lang_codes]
        directions: list[tuple[str, str]] = []
        for code in target_codes:
            if code != "en":
                directions.append(("en", code))
                directions.append((code, "en"))
        self._directions = list(dict.fromkeys(directions))  # deduplicate, preserve order

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

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

                if _shard_is_done(self.output_dir, shard_id, self._directions):
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
        tasks: list[AudioTask] = []
        item_idx = 0
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

                metadata: dict[str, Any] = {
                    "shard_id": shard_id,
                    "manifest_stem": manifest_stem,
                    "shard_item_idx": item_idx,
                    "shard_total": shard_total,
                }
                tasks.append(
                    AudioTask(
                        task_id=f"{shard_id}_{item_idx}",
                        dataset_name=manifest_stem,
                        data=json.loads(raw_line.strip()),
                        _metadata=metadata,
                    )
                )
                item_idx += 1
                line_no += 1

        logger.debug(
            "ShardedManifestReader: shard {} lines [{}, {}) → {} tasks",
            shard_id,
            start_line,
            end_line,
            len(tasks),
        )
        return tasks

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
