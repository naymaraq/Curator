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

"""Reconcile completed translation shards into per-manifest per-direction JSONL files.

After ``DirectionalShardedWriterStage`` finishes, the shard directory contains::

    {output_dir}/shards/{manifest_stem}_{shard_idx}_{src}-{tgt}.jsonl.done

``reconcile_manifests`` groups these by ``(manifest_stem, direction)``, sorts
by ``shard_idx``, and concatenates them into::

    {output_dir}/{manifest_stem}_{src}-{tgt}.jsonl

Incomplete shards (files without a ``.done`` suffix) are reported as warnings
but do not block reconciliation of the completed ones.
"""

from __future__ import annotations

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

from loguru import logger

# Filename pattern: {manifest_stem}_{shard_idx}_{src}-{tgt}.jsonl.done
# manifest_stem may contain underscores, so we match greedily from the left
# and anchor on the final two underscore-delimited tokens before the direction.
_DONE_PATTERN = re.compile(
    r"^(?P<manifest_stem>.+)_(?P<shard_idx>\d+)_(?P<direction>[a-z]+-[a-z]+)\.jsonl\.done$"
)


def _parse_done_filename(filename: str) -> tuple[str, int, str] | None:
    """Return ``(manifest_stem, shard_idx, direction)`` or ``None`` if unrecognised."""
    m = _DONE_PATTERN.match(filename)
    if m is None:
        return None
    return m.group("manifest_stem"), int(m.group("shard_idx")), m.group("direction")


def reconcile_manifests(output_dir: str) -> None:
    """Merge completed shard files into per-manifest per-direction JSONL outputs.

    Scans ``{output_dir}/shards/`` for ``*.jsonl.done`` files, groups them by
    ``(manifest_stem, direction)``, and concatenates them (in shard-index order)
    into ``{output_dir}/{manifest_stem}_{direction}.jsonl``.

    Args:
        output_dir: The same ``output_dir`` passed to
                    ``ShardedManifestReaderStage`` and
                    ``DirectionalShardedWriterStage``.
    """
    shards_dir = os.path.join(output_dir, "shards")
    if not os.path.isdir(shards_dir):
        logger.warning("reconcile_manifests: shards dir not found: {}", shards_dir)
        return

    # Collect all .done files.
    done_files = [f for f in os.listdir(shards_dir) if f.endswith(".jsonl.done")]
    if not done_files:
        logger.warning("reconcile_manifests: no .done files found in {}", shards_dir)
        return

    # Report any incomplete (non-.done) shard files.
    partial_files = [f for f in os.listdir(shards_dir) if f.endswith(".jsonl")]
    for pf in partial_files:
        logger.warning(
            "reconcile_manifests: incomplete shard file found (not .done): {}", pf
        )

    # Group: (manifest_stem, direction) → sorted list of (shard_idx, abs_path).
    groups: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    unrecognised: list[str] = []
    for fname in done_files:
        parsed = _parse_done_filename(fname)
        if parsed is None:
            unrecognised.append(fname)
            continue
        manifest_stem, shard_idx, direction = parsed
        groups[(manifest_stem, direction)].append(
            (shard_idx, os.path.join(shards_dir, fname))
        )

    if unrecognised:
        logger.warning(
            "reconcile_manifests: {} file(s) with unrecognised names skipped: {}",
            len(unrecognised),
            unrecognised,
        )

    if not groups:
        logger.warning("reconcile_manifests: no valid .done groups found; nothing to reconcile")
        return

    os.makedirs(output_dir, exist_ok=True)
    n_written = 0

    for (manifest_stem, direction), shard_entries in sorted(groups.items()):
        shard_entries.sort(key=lambda x: x[0])  # sort by shard_idx
        out_path = os.path.join(output_dir, f"{manifest_stem}_{direction}.jsonl")

        logger.info(
            "reconcile_manifests: {} + {} shard(s) → {}",
            manifest_stem,
            len(shard_entries),
            out_path,
        )

        with open(out_path, "w", encoding="utf-8") as out_fh:
            for _shard_idx, shard_path in shard_entries:
                with open(shard_path, encoding="utf-8") as in_fh:
                    shutil.copyfileobj(in_fh, out_fh)

        n_written += 1

    logger.info(
        "reconcile_manifests: done — wrote {} output file(s) to {}",
        n_written,
        output_dir,
    )


def list_pending_shards(output_dir: str) -> list[str]:
    """Return shard IDs that have partial (non-done) output files.

    Useful for diagnostics or targeted retries.

    Args:
        output_dir: Root output directory used by the pipeline.

    Returns:
        Sorted list of shard IDs with at least one incomplete direction file.
    """
    shards_dir = os.path.join(output_dir, "shards")
    if not os.path.isdir(shards_dir):
        return []

    pending: set[str] = set()
    for fname in os.listdir(shards_dir):
        if not fname.endswith(".jsonl"):
            continue
        stem = Path(fname).stem  # strip .jsonl
        # stem is "{manifest_stem}_{shard_idx}_{direction}"
        # Extract shard_id = everything before the last "_"-separated direction token.
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and "-" in parts[1]:
            shard_id = parts[0]
            pending.add(shard_id)

    return sorted(pending)
