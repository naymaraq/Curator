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

"""Per-direction, per-shard JSONL writer with .done marker support.

Writes each expanded ``AudioTask`` to::

    {output_dir}/shards/{shard_id}_{src}-{tgt}.jsonl

Per-direction completion
------------------------
Each ``AudioTask`` carries ``direction_counts`` in its ``_metadata`` — a
``dict[str, int]`` produced by ``ShardedManifestReaderStage`` that maps
``"{src}-{tgt}"`` to the exact number of tasks that direction will receive
for the originating shard.

The writer tracks a per-direction row counter (``_seen_counts``); when the
counter reaches the expected count for that direction, the file handle is
closed and the file is immediately renamed to ``{handle_key}.jsonl.done``
inside ``process()``.  This avoids relying on ``teardown()`` being called by
the execution framework (Xenna currently does not call it).

Shard-level completion markers (``{shard_id}.shard.done``) are written
externally by ``mark_complete_shards()`` after ``pipeline.run()`` returns.

Resume behaviour
----------------
If a ``.done`` file already exists for a given (shard, direction) pair,
that direction is silently skipped — its data is already safe.

On a retry after a crash, new shard files are opened with ``"w"`` (truncate)
so stale partial data from the previous run is discarded and replaced.

Safety net
----------
``teardown()`` is kept as a safety net: if the runtime ever does call it, it
closes and renames any direction files still open (e.g. if some upstream
tasks were dropped so the per-direction counter never reached its expected
value).  In normal operation it is a no-op because every direction is
already renamed inline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.audio.translation.language_map import _normalize_code
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class DirectionalShardedWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Write expanded translation tasks to per-direction, per-shard JSONL files.

    Output path per task::

        {output_dir}/shards/{shard_id}_{source_lang}-{target_lang}.jsonl

    Per-direction completion happens inline in ``process()``: once a
    ``handle_key`` has received as many rows as ``task._metadata['direction_counts']``
    says it should, the file is closed and renamed to ``.jsonl.done`` immediately.
    ``teardown()`` is a safety net only.

    If a direction's ``.done`` file already exists (from a prior successful
    run), that direction is skipped entirely — its data is kept as-is.

    Runs with a single worker (``num_workers() → 1``) to guarantee that
    file handles are not split across workers.

    Args:
        output_dir:       Root output directory.
        source_lang_key:  Key holding the source language ISO code (default: ``"source_lang"``).
        target_lang_key:  Key holding the target language ISO code (default: ``"target_lang"``).
    """

    output_dir: str
    name: str = "DirectionalShardedWriter"
    source_lang_key: str = "source_lang"
    target_lang_key: str = "target_lang"

    # Open file handles: "{shard_id}_{src}-{tgt}" → file-like object.
    _handles: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Rows written per handle_key (for per-direction completion detection).
    _seen_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _shards_dir: str = field(default="", init=False, repr=False)
    _n_written: int = field(default=0, init=False, repr=False)
    _n_skipped_done: int = field(default=0, init=False, repr=False)
    _n_directions_completed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.output_dir:
            msg = "DirectionalShardedWriterStage: output_dir must be set"
            raise ValueError(msg)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        self._shards_dir = os.path.join(self.output_dir, "shards")
        os.makedirs(self._shards_dir, exist_ok=True)
        logger.info("DirectionalShardedWriter: shard output dir → {}", self._shards_dir)

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        self._shards_dir = os.path.join(self.output_dir, "shards")
        os.makedirs(self._shards_dir, exist_ok=True)

    def teardown(self) -> None:
        """Safety-net: close and rename any direction files still open after processing.

        Under normal operation all direction files are completed inline in
        ``process()`` once their row counter reaches the expected count from
        ``task._metadata['direction_counts']``.  This method only handles edge
        cases where the expected count was missing or some upstream tasks were
        dropped, leaving open handles.  It does NOT write
        ``{shard_id}.shard.done`` markers — that is the responsibility of
        ``mark_complete_shards()`` called from the pipeline runner.

        Note: the Xenna executor currently does not invoke ``teardown()`` on
        stage adapters, so do not rely on this method for correctness.
        """
        safety_net_renamed: list[str] = []
        for key in list(self._handles.keys()):
            self._complete_direction(key)
            safety_net_renamed.append(key)

        logger.info(
            "DirectionalShardedWriter: teardown — {} direction(s) completed inline, "
            "{} direction(s) completed in safety-net teardown, "
            "{} rows written, {} rows skipped (already done)",
            self._n_directions_completed,
            len(safety_net_renamed),
            self._n_written,
            self._n_skipped_done,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _complete_direction(self, handle_key: str) -> None:
        """Close the open file handle for ``handle_key`` and rename to ``.done``."""
        fh = self._handles.pop(handle_key, None)
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        src_path = os.path.join(self._shards_dir, f"{handle_key}.jsonl")
        dst_path = f"{src_path}.done"
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            os.rename(src_path, dst_path)
            self._n_directions_completed += 1
            logger.info("DirectionalShardedWriter: direction complete → {}", dst_path)

    # ------------------------------------------------------------------
    # ProcessingStage interface
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.source_lang_key, self.target_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: AudioTask) -> AudioTask:
        shard_id: str = task._metadata.get("shard_id", "unknown_shard")
        direction_counts: dict[str, int] = task._metadata.get("direction_counts", {})

        src_raw = task.data.get(self.source_lang_key, "")
        tgt_raw = task.data.get(self.target_lang_key, "")

        if not src_raw or not tgt_raw:
            logger.warning(
                "DirectionalShardedWriter: task {} missing source/target lang keys; skipping write",
                task.task_id,
            )
            return task

        # Normalize source/target codes so handle_key matches the keys produced
        # by ShardedManifestReaderStage in `direction_counts`.
        src = _normalize_code(src_raw)
        tgt = _normalize_code(tgt_raw)
        direction = f"{src}-{tgt}"
        handle_key = f"{shard_id}_{direction}"

        # If this direction's .done file already exists from a prior run, skip.
        done_path = os.path.join(self._shards_dir, f"{handle_key}.jsonl.done")
        if os.path.exists(done_path):
            self._n_skipped_done += 1
            return task

        if handle_key not in self._handles:
            shard_path = os.path.join(self._shards_dir, f"{handle_key}.jsonl")
            fs, resolved = url_to_fs(shard_path)
            # "w" truncates any stale partial file from a previous crashed run.
            self._handles[handle_key] = fs.open(resolved, "w", encoding="utf-8")
            self._seen_counts[handle_key] = 0

        self._handles[handle_key].write(json.dumps(task.data, ensure_ascii=False) + "\n")
        self._handles[handle_key].flush()
        self._n_written += 1
        self._seen_counts[handle_key] += 1

        # Complete the direction immediately once all expected rows have been written.
        expected = direction_counts.get(direction, -1)
        if expected > 0 and self._seen_counts[handle_key] >= expected:
            self._complete_direction(handle_key)

        return AudioTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=dict(task._metadata),
            _stage_perf=list(task._stage_perf),
        )

    # ------------------------------------------------------------------
    # Executor hints
    # ------------------------------------------------------------------

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
