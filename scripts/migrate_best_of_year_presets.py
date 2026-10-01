#!/usr/bin/env python3
"""Migrate best-of-year pipeline_settings from the old expanded node layout
to the new composite node layout.

Affects: presets, schedules, and jobs tables in the webgui database.

Old curation pipeline aliases → New composite aliases:
  output.branches.curation.events             → output.branches.curation.near_duplicate  (time_window_minutes, gps_window_meters)
  output.branches.curation.duplicate          → output.branches.curation.duplicate       (threshold → threshold)
  output.branches.curation.compute_embedding  → output.branches.curation.near_duplicate  (concurrency, batch_size)
  output.branches.curation.near_duplicate     → output.branches.curation.near_duplicate  (threshold)
  output.branches.curation.scenes             → output.branches.curation.scenes          (time_window_minutes)
  output.branches.curation.scene_cluster      → output.branches.curation.scenes          (threshold)
  output.branches.curation.scene_cluster.pick → output.branches.curation.scenes          (max_picks, pick_percentage, quality_weight, diversity_weight)

Usage:
    python scripts/migrate_best_of_year_presets.py <path_to_database.db> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any


# Mapping: old alias → (new alias, key renames)
# key renames is a dict of {old_key: new_key} — empty means keys pass through unchanged
CURATION_PREFIX = "output.branches.curation."

ALIAS_MIGRATION: dict[str, tuple[str, dict[str, str]]] = {
    # PartitionTimeGpsAnchor → DedupSimilar
    "events": ("near_duplicate", {}),
    # PartitionPHash → DedupPHashComposite
    "duplicate": ("duplicate", {}),
    # AnalyzeEmbedding → DedupSimilar (merged)
    "compute_embedding": ("near_duplicate", {}),
    # PartitionFaiss → DedupSimilar (merged)
    "near_duplicate": ("near_duplicate", {}),
    # PartitionTime → DedupScenes
    "scenes": ("scenes", {}),
    # PartitionCosine → DedupScenes (merged)
    "scene_cluster": ("scenes", {}),
    # SelectDiversePick → DedupScenes (merged)
    "scene_cluster.pick": ("scenes", {}),
}

# Aliases that had no config and can be dropped
DROP_ALIASES = {
    "duplicate.pick_best",
    "duplicate.merge",
    "near_duplicate.pick_best",
    "near_duplicate.merge",
    "events.merge",
    "scene_cluster.merge",
    "scenes.merge",
}


def migrate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Transform old best-of-year pipeline_settings to new composite layout.

    Args:
        settings: The old pipeline_settings dict (alias → config dict).
            Must be a dict; a non-dict value (corrupt row) is rejected by
            the caller before this is invoked — see SC-03.

    Returns:
        The migrated settings dict. Returns a copy; original is not mutated.

    Note on idempotency (SC-01): several old aliases map to themselves
    (``duplicate`` -> ``duplicate``, ``near_duplicate`` -> ``near_duplicate``,
    ``scenes`` -> ``scenes``). Previously a `changed` flag was set to True
    just from popping+re-adding those aliases, even when nothing about the
    settings dict actually changed — so every re-run of the script (and
    every "would update" dry-run report) treated already-migrated rows as
    needing an update, and would bump their `updated_at` on each live run.
    Comparing the final result against the original input directly fixes
    this: a second run against already-migrated settings now reports (and
    makes) no changes.
    """
    if not isinstance(settings, dict):
        raise TypeError(f"pipeline_settings must be a dict, got {type(settings).__name__}")

    result = deepcopy(settings)

    for old_alias, (new_alias, key_renames) in ALIAS_MIGRATION.items():
        full_old = CURATION_PREFIX + old_alias
        full_new = CURATION_PREFIX + new_alias

        if full_old not in result:
            continue

        old_config = result.pop(full_old)

        if old_config is None:
            old_config = {}
        if not isinstance(old_config, dict):
            # SC-03: a corrupt/unexpected shape (e.g. a list or string where
            # a config dict is expected) previously raised AttributeError
            # from `.items()` below and crashed the whole migration run.
            raise TypeError(
                f"Expected a dict config for alias {full_old!r}, got "
                f"{type(old_config).__name__}: {old_config!r}"
            )

        if not old_config:
            continue

        # Rename keys if needed
        migrated_config: dict[str, Any] = {}
        for k, v in old_config.items():
            new_key = key_renames.get(k, k)
            migrated_config[str(new_key)] = v

        # Merge into the new alias (multiple old aliases may map to the same new one)
        existing_new = result.get(full_new)
        if isinstance(existing_new, dict):
            existing_new.update(migrated_config)
        else:
            result[full_new] = migrated_config

    # Drop aliases that no longer exist (they had no meaningful config)
    for alias in DROP_ALIASES:
        full = CURATION_PREFIX + alias
        result.pop(full, None)

    # SC-01: compare the actual resulting dict rather than tracking a
    # "did we touch anything" flag, so already-migrated (or never-matching)
    # settings are correctly reported/left as unchanged.
    return result if result != settings else settings


def has_updated_at(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table has an updated_at column."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = {row["name"] for row in cursor.fetchall()}
    return "updated_at" in columns


def backup_database(db_path: str) -> str:
    """Snapshot ``db_path`` to a timestamped ``.bak`` file before a live run.

    Uses SQLite's online backup API (safe even if the DB is in WAL mode or
    has a connection open elsewhere) rather than copying the raw file.
    Addresses SC-02: previously the script edited the live database in
    place with no safety net at all — a bug in the migration (or a crash
    partway through, see SC-03) had no way to be undone short of restoring
    an unrelated backup.

    Returns the path to the backup file.
    """
    backup_path = f"{db_path}.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    src = sqlite3.connect(db_path)
    try:
        dst = sqlite3.connect(backup_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return backup_path


def migrate_database(db_path: str, *, dry_run: bool = False, skip_backup: bool = False) -> None:
    """Run the migration against the SQLite database.

    Updates pipeline_settings in presets, schedules, and jobs tables
    for rows where pipeline = 'best-of-year'.

    Args:
        skip_backup: Skip the pre-run backup snapshot (SC-02). Only
            meaningful for a live run; dry runs never write anything.
    """
    if not dry_run and not skip_backup:
        backup_path = backup_database(db_path)
        print(f"Backup written to: {backup_path}\n")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        tables = [
            ("presets", "id"),
            ("schedules", "id"),
            ("jobs", "id"),
        ]

        total_updated = 0
        total_skipped = 0

        for table, pk_col in tables:
            # Check if table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if cursor.fetchone() is None:
                print(f"  Table '{table}' does not exist — skipping.")
                continue

            cursor = conn.execute(
                f"SELECT {pk_col}, pipeline_settings FROM {table} WHERE pipeline = ?",
                ("best-of-year",),
            )
            rows = cursor.fetchall()

            table_has_updated_at = has_updated_at(conn, table)

            updated = 0
            skipped = 0
            for row in rows:
                row_id = row[pk_col]
                try:
                    old_settings = json.loads(row["pipeline_settings"])
                except (json.JSONDecodeError, TypeError) as exc:
                    print(f"  [{table}] {row_id}: skipping — invalid JSON ({exc})")
                    skipped += 1
                    continue

                # SC-03: a row with an unexpected shape (pipeline_settings
                # that parsed as valid JSON but isn't a dict, or an alias
                # value that isn't a dict) previously raised TypeError or
                # AttributeError and crashed the entire run, leaving later
                # tables/rows unprocessed. Report and skip the offending
                # row instead.
                try:
                    new_settings = migrate_settings(old_settings)
                except TypeError as exc:
                    print(f"  [{table}] {row_id}: skipping — unexpected shape ({exc})")
                    skipped += 1
                    continue

                if new_settings is old_settings:
                    # No changes needed
                    continue

                updated += 1

                if dry_run:
                    print(f"  [{table}] {row_id}: would update")
                    print(f"    Old: {json.dumps(old_settings, indent=2)}")
                    print(f"    New: {json.dumps(new_settings, indent=2)}")
                    print()
                else:
                    new_json = json.dumps(new_settings)
                    if table_has_updated_at:
                        now = datetime.now(timezone.utc).isoformat()
                        conn.execute(
                            f"UPDATE {table} SET pipeline_settings = ?, updated_at = ? WHERE {pk_col} = ?",
                            (new_json, now, row_id),
                        )
                    else:
                        conn.execute(
                            f"UPDATE {table} SET pipeline_settings = ? WHERE {pk_col} = ?",
                            (new_json, row_id),
                        )

            if updated:
                print(f"  [{table}] {updated}/{len(rows)} rows {'would be ' if dry_run else ''}updated.")
            else:
                print(f"  [{table}] {len(rows)} rows — no migration needed.")
            if skipped:
                print(f"  [{table}] {skipped} row(s) skipped due to errors (see above).")

            total_updated += updated
            total_skipped += skipped

        if not dry_run and total_updated > 0:
            conn.commit()
            print(f"\n✓ Committed {total_updated} updates.")
        elif dry_run and total_updated > 0:
            print(f"\n(dry-run) {total_updated} rows would be updated. Run without --dry-run to apply.")
        else:
            print("\n✓ No rows needed migration.")

        if total_skipped:
            print(f"⚠ {total_skipped} row(s) were skipped due to errors — see messages above.")
    finally:
        # SC-02: close() was previously a bare call after the main body,
        # so an exception partway through the loop (e.g. a malformed row —
        # see SC-03) left the connection open.
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate best-of-year presets/schedules to composite node config layout."
    )
    parser.add_argument(
        "database",
        help="Path to the webgui SQLite database file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing to the database.",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help=(
            "Skip the automatic pre-run backup snapshot (<database>.bak-<timestamp>). "
            "Only use this if the caller already takes its own backup."
        ),
    )
    args = parser.parse_args()

    print(f"Migrating best-of-year settings in: {args.database}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    migrate_database(args.database, dry_run=args.dry_run, skip_backup=args.skip_backup)


if __name__ == "__main__":
    main()
