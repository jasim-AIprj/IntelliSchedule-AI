"""SQLite persistence for timetable generation runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import Placement, Problem


class RunStore:
    def __init__(self, path: str | Path = "timetable.db"):
        self.path = str(path)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS timetable_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    soft_cost REAL NOT NULL,
                    runtime_seconds REAL NOT NULL,
                    schedule_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS university_datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    name TEXT NOT NULL,
                    tables_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    initial_cost REAL NOT NULL,
                    final_cost REAL NOT NULL,
                    runtime_seconds REAL NOT NULL,
                    iterations INTEGER NOT NULL,
                    accepted_moves INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        algorithm: str,
        soft_cost: float,
        runtime_seconds: float,
        schedule: dict[str, Placement],
    ) -> int:
        payload = {
            meeting_id: {
                "day": placement.day,
                "start": placement.start,
                "room_id": placement.room_id,
            }
            for meeting_id, placement in schedule.items()
        }
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO timetable_runs
                    (created_at, algorithm, soft_cost, runtime_seconds, schedule_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    algorithm,
                    soft_cost,
                    runtime_seconds,
                    json.dumps(payload),
                ),
            )
            return int(cursor.lastrowid)

    def recent(self, limit: int = 20) -> list[dict[str, object]]:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, created_at, algorithm, soft_cost, runtime_seconds FROM timetable_runs "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_dataset(self, name: str, tables: dict[str, object]) -> int:
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                "INSERT INTO university_datasets (created_at, name, tables_json) VALUES (?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    name,
                    json.dumps(tables, default=str),
                ),
            )
            return int(cursor.lastrowid)

    def save_experiment(
        self,
        algorithm: str,
        initial_cost: float,
        final_cost: float,
        runtime_seconds: float,
        iterations: int,
        accepted_moves: int,
        metrics: dict[str, object],
    ) -> int:
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO experiment_results
                    (created_at, algorithm, initial_cost, final_cost, runtime_seconds,
                     iterations, accepted_moves, metrics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    algorithm,
                    initial_cost,
                    final_cost,
                    runtime_seconds,
                    iterations,
                    accepted_moves,
                    json.dumps(metrics, default=str),
                ),
            )
            return int(cursor.lastrowid)
