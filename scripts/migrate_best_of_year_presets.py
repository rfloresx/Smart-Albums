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

    Returns:
        The migrated settings dict. Returns a copy; original is not mutated.
    """
    result = deepcopy(settings)
    changed = False

    for old_alias, (new_alias, key_renames) in ALIAS_MIGRATION.items():
        full_old = CURATION_PREFIX + old_alias
        full_new = CURATION_PREFIX + new_alias

        if full_old not in result:
            continue

        old_config = result.pop(full_old)
        changed = True

        if not old_config:
            continue

        # Rename keys if needed
        migrated_config: dict[str, Any] = {}
        for k, v in old_config.items():
            new_key = key_renames.get(k, k)
            migrated_config[new_key] = v

        # Merge into the new alias (multiple old aliases may map to the same new one)
        if full_new in result:
            result[full_new].update(migrated_config)
        else:
            result[full_new] = migrated_config

    # Drop aliases that no longer exist (they had no meaningful config)
    for alias in DROP_ALIASES:
        full = CURATION_PREFIX + alias
        if full in result:
            result.pop(full)
            changed = True

    return result if changed else settings


def has_updated_at(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table has an updated_at column."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = {row["name"] for row in cursor.fetchall()}
    return "updated_at" in columns


def migrate_database(db_path: str, *, dry_run: bool = False) -> None:
    """Run the migration against the SQLite database.

    Updates pipeline_settings in presets, schedules, and jobs tables
    for rows where pipeline = 'best-of-year'.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    tables = [
        ("presets", "id"),
        ("schedules", "id"),
        ("jobs", "id"),
    ]

    total_updated = 0

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

        updated = 0
        for row in rows:
            row_id = row[pk_col]
            try:
                old_settings = json.loads(row["pipeline_settings"])
            except (json.JSONDecodeError, TypeError):
                continue

            new_settings = migrate_settings(old_settings)

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
                if has_updated_at(conn, table):
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

        total_updated += updated

    if not dry_run and total_updated > 0:
        conn.commit()
        print(f"\n✓ Committed {total_updated} updates.")
    elif dry_run and total_updated > 0:
        print(f"\n(dry-run) {total_updated} rows would be updated. Run without --dry-run to apply.")
    else:
        print("\n✓ No rows needed migration.")

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
    args = parser.parse_args()

    print(f"Migrating best-of-year settings in: {args.database}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    migrate_database(args.database, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
