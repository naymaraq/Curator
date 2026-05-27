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

All ``.done`` renaming happens in ``teardown()``, which is called after
every task has been processed.  This avoids the race where per-task
completion checks fire before all direction tasks from a shard have been
written (tasks from the same original item but different directions can
arrive interleaved due to parallel ``TranslationExpanderStage`` workers).

After renaming all direction files, ``teardown()`` writes a shard-level
``{shard_id}.shard.done`` marker for every shard that had at least one
direction processed.  ``ShardedManifestReaderStage`` checks only this
single marker — not per-direction ``.done`` files — so the skip logic
works regardless of which directions actually appeared in the data.

Resume behaviour
----------------
If a ``.done`` file already exists for a given (shard, direction) pair,
that direction is silently skipped — its data is already safe.  At the next
``teardown()`` only the *newly written* ``.jsonl`` files are renamed to
``.done``; existing ``.done`` files are never touched.

On a retry after a crash, new shard files are opened with ``"w"`` (truncate)
so stale partial data from the previous run is discarded and replaced.
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
from nemo_curator.stages.audio.translation.sharded_manifest_reader import shard_done_marker_path
from nemo_curator.tasks import AudioTask


@dataclass
class DirectionalShardedWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Write expanded translation tasks to per-direction, per-shard JSONL files.

    Output path per task::

        {output_dir}/shards/{shard_id}_{source_lang}-{target_lang}.jsonl

    All ``.done`` markers are written at ``teardown()`` time, after every
    task has been processed, ensuring no direction file is marked complete
    before all its rows have been flushed.

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
    # All shard IDs seen during this worker's lifetime (for shard-level markers).
    _seen_shard_ids: set[str] = field(default_factory=set, init=False, repr=False)
    _shards_dir: str = field(default="", init=False, repr=False)
    _n_written: int = field(default=0, init=False, repr=False)
    _n_skipped_done: int = field(default=0, init=False, repr=False)

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
        """Flush + close all open handles, then rename each new shard file to ``.done``.

        After all direction files are renamed, writes a ``{shard_id}.shard.done``
        marker for every shard seen during this worker's lifetime so that
        ``ShardedManifestReaderStage`` can skip the shard on the next run.

        Existing ``.done`` files are never overwritten.
        """
        renamed: list[str] = []
        handle_keys = list(self._handles.keys())
        for key in handle_keys:
            fh = self._handles.pop(key, None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass

            src_path = os.path.join(self._shards_dir, f"{key}.jsonl")
            dst_path = f"{src_path}.done"
            # Guard: never overwrite an existing .done file.
            if os.path.exists(src_path) and not os.path.exists(dst_path):
                os.rename(src_path, dst_path)
                renamed.append(key)

        # Write a single shard-level marker for each completed shard so the
        # reader only needs to check one file per shard (not one per direction).
        for shard_id in self._seen_shard_ids:
            marker = shard_done_marker_path(self.output_dir, shard_id)
            if not os.path.exists(marker):
                try:
                    with open(marker, "w") as _f:
                        pass
                    logger.info("DirectionalShardedWriter: shard marker written → {}", marker)
                except OSError as exc:
                    logger.warning("DirectionalShardedWriter: could not write shard marker {}: {}", marker, exc)

        logger.info(
            "DirectionalShardedWriter: teardown — {}/{} shard-direction file(s) marked .done, "
            "{} shard marker(s) written, {} rows written, {} rows skipped (already done)",
            len(renamed),
            len(handle_keys),
            len(self._seen_shard_ids),
            self._n_written,
            self._n_skipped_done,
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
        self._seen_shard_ids.add(shard_id)

        src = task.data.get(self.source_lang_key, "")
        tgt = task.data.get(self.target_lang_key, "")
        direction = f"{src}-{tgt}"

        if not src or not tgt:
            logger.warning(
                "DirectionalShardedWriter: task {} missing source/target lang keys; skipping write",
                task.task_id,
            )
            return task

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

        self._handles[handle_key].write(json.dumps(task.data, ensure_ascii=False) + "\n")
        self._handles[handle_key].flush()
        self._n_written += 1

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
