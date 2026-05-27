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

When every unique ``shard_item_idx`` for a given shard has been observed
(i.e. ``len(seen_indices) == shard_total``), the writer renames every
direction file for that shard from ``*.jsonl`` → ``*.jsonl.done``.

``ShardedManifestReaderStage`` uses the presence of all ``.done`` files to
decide whether to skip a shard on the next pipeline call.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from fsspec.core import url_to_fs
from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class DirectionalShardedWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Write expanded translation tasks to per-direction, per-shard JSONL files.

    Output path per task::

        {output_dir}/shards/{shard_id}_{source_lang}-{target_lang}.jsonl

    When a shard is complete (all ``shard_total`` unique item indices seen),
    every direction file for that shard is renamed to ``*.jsonl.done``.

    Runs with a single worker (``num_workers() → 1``) to guarantee correct
    shard-completion accounting without cross-worker coordination.

    Args:
        output_dir:       Root output directory.
        source_lang_key:  Key holding the source language ISO code (default: ``"source_lang"``).
        target_lang_key:  Key holding the target language ISO code (default: ``"target_lang"``).
    """

    output_dir: str
    name: str = "DirectionalShardedWriter"
    source_lang_key: str = "source_lang"
    target_lang_key: str = "target_lang"

    # Per-shard accounting: shard_id → set of seen shard_item_idx values.
    _seen: dict[str, set[int]] = field(default_factory=dict, init=False, repr=False)
    # shard_id → shard_total (first time we see a task for that shard).
    _totals: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    # shard_id → set of direction strings written so far ("src-tgt").
    _shard_directions: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    # Open file handles: "{shard_id}_{src}-{tgt}" → file-like object.
    _handles: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _shards_dir: str = field(default="", init=False, repr=False)

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
        for key, fh in list(self._handles.items()):
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass
        self._handles.clear()
        n_done = sum(1 for s in self._seen if len(self._seen[s]) >= self._totals.get(s, 0))
        logger.info(
            "DirectionalShardedWriter: teardown — {} shard(s) marked done out of {} seen",
            n_done,
            len(self._seen),
        )

    # ------------------------------------------------------------------
    # ProcessingStage interface
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.source_lang_key, self.target_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def process(self, task: AudioTask) -> AudioTask:
        shard_id: str = task._metadata.get("shard_id", "unknown_shard")
        shard_item_idx: int = task._metadata.get("shard_item_idx", 0)
        shard_total: int = task._metadata.get("shard_total", 1)

        src = task.data.get(self.source_lang_key, "")
        tgt = task.data.get(self.target_lang_key, "")
        direction = f"{src}-{tgt}"

        if not src or not tgt:
            logger.warning(
                "DirectionalShardedWriter: task {} missing source/target lang keys; skipping write",
                task.task_id,
            )
            return task

        # Open file handle on first write for this (shard_id, direction) pair.
        handle_key = f"{shard_id}_{direction}"
        if handle_key not in self._handles:
            shard_path = os.path.join(self._shards_dir, f"{handle_key}.jsonl")
            fs, resolved = url_to_fs(shard_path)
            # Append so partial shards accumulate across workers.
            self._handles[handle_key] = fs.open(resolved, "a", encoding="utf-8")

        self._handles[handle_key].write(json.dumps(task.data, ensure_ascii=False) + "\n")
        self._handles[handle_key].flush()

        # Track shard_total (idempotent — same value every time for a given shard).
        if shard_id not in self._totals:
            self._totals[shard_id] = shard_total

        # Track seen item indices and direction files for this shard.
        self._seen.setdefault(shard_id, set()).add(shard_item_idx)
        self._shard_directions.setdefault(shard_id, set()).add(direction)

        # Mark shard complete when all items have been written.
        if len(self._seen[shard_id]) >= self._totals[shard_id]:
            self._mark_shard_done(shard_id)

        return AudioTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=task.data,
            _metadata=dict(task._metadata),
            _stage_perf=list(task._stage_perf),
        )

    # ------------------------------------------------------------------
    # Shard completion
    # ------------------------------------------------------------------

    def _mark_shard_done(self, shard_id: str) -> None:
        """Close and rename every direction file for *shard_id* to ``.done``."""
        directions = self._shard_directions.get(shard_id, set())
        renamed: list[str] = []
        for direction in directions:
            handle_key = f"{shard_id}_{direction}"
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
                renamed.append(direction)

        if renamed:
            logger.info(
                "DirectionalShardedWriter: shard {} complete — marked done: {}",
                shard_id,
                renamed,
            )

    # ------------------------------------------------------------------
    # Executor hints
    # ------------------------------------------------------------------

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers": 1}
