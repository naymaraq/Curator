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

"""Per-direction shard writer for the translation pipeline.

For each task this stage appends a single JSON line to::

    {output_dir}/{shard_key}_{src}-{tgt}.jsonl

where ``shard_key`` comes from ``task._metadata["_shard_key"]`` (set by
``TranslationManifestReaderStage``) and ``(src, tgt)`` come from the row's
language fields (normalised via ``_normalize_code``).

Per-direction completion
------------------------
Each ``AudioTask`` carries ``direction_counts`` in its ``_metadata`` — a
``dict[str, int]`` mapping ``"{src}-{tgt}"`` to the exact number of tasks
that direction will receive for this shard.  The writer keeps a per-handle
counter; the moment ``_seen_counts[handle_key]`` reaches the expected
count, the bare ``.jsonl`` is renamed to ``.jsonl.done`` — and that file
*is* the final per-(manifest, direction) output (no separate reconciliation
step).

Open-append-close per row
-------------------------
The writer does not hold persistent file handles.  Each ``process()`` call
opens the per-direction file in ``"a"`` mode, writes one line, and closes
it.  This keeps the actor's state minimal and survives mid-shard actor
restarts cleanly when combined with the disk-based counter recovery in
``setup()``.

Resume / restart
----------------
* Cross-run resume: ``TranslationManifestReaderStage`` deletes any partial
  ``.jsonl`` (no ``.done`` sibling) for an expected direction before
  re-emitting rows, so the writer always sees a clean per-direction file
  on the next run.
* Intra-run actor restart: ``setup()`` walks ``output_dir``, finds every
  ``*.jsonl`` without a sibling ``*.jsonl.done``, counts its lines, and
  seeds ``_seen_counts``.  Subsequent rows append correctly and the rename
  fires once the expected count is reached.
* If a direction's ``.jsonl.done`` already exists, ``process()`` short-
  circuits — that direction is final and untouchable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from nemo_curator.backends.base import NodeInfo, WorkerMetadata
from nemo_curator.backends.utils import RayStageSpecKeys
from nemo_curator.stages.audio.translation.language_map import _normalize_code
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.tasks import AudioTask


@dataclass
class DirectionalShardedWriterStage(ProcessingStage[AudioTask, AudioTask]):
    """Write expanded translation tasks to per-(shard, direction) JSONL files.

    Output path per task::

        {output_dir}/{shard_key}_{source_lang}-{target_lang}.jsonl

    Runs as a single actor (``num_workers() == 1``) so the per-handle
    counter cannot be split across workers.

    Args:
        output_dir:      Root output directory.  Final ``*.jsonl.done``
                         files land directly under this path (no
                         ``shards/`` subdir).
        source_lang_key: Row key holding the source language ISO code
                         (default: ``"source_lang"``).
        target_lang_key: Row key holding the target language ISO code
                         (default: ``"target_lang"``).
    """

    output_dir: str
    name: str = "DirectionalShardedWriter"
    source_lang_key: str = "source_lang"
    target_lang_key: str = "target_lang"

    # Rows written per "{shard_key}_{src}-{tgt}" handle key.
    _seen_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _n_written: int = field(default=0, init=False, repr=False)
    _n_skipped_done: int = field(default=0, init=False, repr=False)
    _n_directions_completed: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.output_dir:
            msg = "DirectionalShardedWriterStage: output_dir must be set"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup_on_node(
        self,
        _node_info: NodeInfo | None = None,
        _worker_metadata: WorkerMetadata | None = None,
    ) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("DirectionalShardedWriter: output_dir={}", self.output_dir)

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        """Recover per-direction counters from partial ``*.jsonl`` files.

        Walks ``output_dir`` once and seeds ``_seen_counts[handle_key]`` from
        the line count of every ``*.jsonl`` that does not yet have a sibling
        ``*.jsonl.done``.  Shards with a ``.done`` marker are skipped — the
        ``process()`` short-circuit will handle them.

        This is required because Ray/Xenna may kill and replace this actor
        at any time (preemption, OOM, autoscaler scale-down, exception
        retry).  A fresh actor would otherwise start with empty counters
        and the ``.done`` rename for partially-written directions would
        never fire.
        """
        if not os.path.isdir(self.output_dir):
            return

        recovered = 0
        for fname in os.listdir(self.output_dir):
            if not fname.endswith(".jsonl"):
                continue
            full = os.path.join(self.output_dir, fname)
            if os.path.exists(full + ".done"):
                continue
            base = fname[: -len(".jsonl")]
            parts = base.rsplit("_", 1)
            if len(parts) != 2 or "-" not in parts[1]:
                # Not a "{shard_key}_{src}-{tgt}.jsonl" filename — ignore.
                continue
            handle_key = base
            try:
                with open(full, "rb") as f:
                    self._seen_counts[handle_key] = sum(1 for _ in f)
            except OSError as exc:
                logger.warning(
                    "DirectionalShardedWriter: failed to recover line count for {}: {}",
                    full,
                    exc,
                )
                continue
            recovered += 1

        if recovered:
            logger.info(
                "DirectionalShardedWriter: recovered partial counts for {} direction(s)",
                recovered,
            )

    def teardown(self) -> None:
        logger.info(
            "DirectionalShardedWriter: {} rows written, {} skipped (already .done), "
            "{} direction(s) completed inline",
            self._n_written,
            self._n_skipped_done,
            self._n_directions_completed,
        )

    # ------------------------------------------------------------------
    # I/O contract
    # ------------------------------------------------------------------

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], [self.source_lang_key, self.target_lang_key]

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, task: AudioTask) -> AudioTask:
        shard_key: str = task._metadata.get("_shard_key", "unknown_shard")
        direction_counts: dict[str, int] = task._metadata.get("direction_counts", {})

        src_raw = task.data.get(self.source_lang_key, "")
        tgt_raw = task.data.get(self.target_lang_key, "")

        if not src_raw or not tgt_raw:
            logger.warning(
                "DirectionalShardedWriter: task {} missing source/target lang keys; skipping write",
                task.task_id,
            )
            return task

        # Normalise so handle_key matches the keys the reader put in direction_counts.
        src = _normalize_code(src_raw)
        tgt = _normalize_code(tgt_raw)
        direction = f"{src}-{tgt}"
        handle_key = f"{shard_key}_{direction}"

        out_path = os.path.join(self.output_dir, f"{handle_key}.jsonl")
        done_path = out_path + ".done"

        # If this direction is already final, leave it alone.
        if os.path.exists(done_path):
            self._n_skipped_done += 1
            return task

        # Open-append-close per row.  Cheap on local disk and keeps actor
        # state minimal so restarts can be recovered from disk.
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(task.data, ensure_ascii=False) + "\n")

        self._n_written += 1
        self._seen_counts[handle_key] = self._seen_counts.get(handle_key, 0) + 1

        expected = direction_counts.get(direction, -1)
        if expected > 0 and self._seen_counts[handle_key] >= expected:
            os.rename(out_path, done_path)
            self._n_directions_completed += 1
            logger.info(
                "DirectionalShardedWriter: direction complete -> {}",
                done_path,
            )

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

    def ray_stage_spec(self) -> dict[str, Any]:
        # Single persistent actor so the in-memory ``_seen_counts`` sees
        # every row for each shard.  Without this, Ray Data would run the
        # writer as parallel stateless tasks with fresh per-task state and
        # the ``.done`` rename would never fire.
        return {RayStageSpecKeys.IS_ACTOR_STAGE: True}
