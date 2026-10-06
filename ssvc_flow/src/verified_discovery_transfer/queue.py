"""Transactional, bounded one-model-per-GPU work queue; no stale-lease guessing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

SUCCESS = {"COMPLETE", "ALIAS", "NO_VERIFIED_TARGETS_RETURN_PARENT"}
TERMINAL = SUCCESS | {"BLOCKED_TECHNICAL"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


class Queue:
    def __init__(self, path, *, maximum=5):
        if not 1 <= maximum <= 5:
            raise ValueError("GPU concurrency must be in 1..5")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.executescript("""CREATE TABLE IF NOT EXISTS metadata(
key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,payload TEXT NOT NULL,
status TEXT NOT NULL,worker TEXT,started REAL,finished REAL,result TEXT);
CREATE TABLE IF NOT EXISTS events(
id INTEGER PRIMARY KEY,job TEXT,event TEXT,at REAL,details TEXT);""")
        existing = self.db.execute("SELECT value FROM metadata WHERE key='maximum'").fetchone()
        if existing and int(existing[0]) != maximum:
            raise ValueError("Queue concurrency cannot change after registration")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES('maximum',?)", (str(maximum),))
        self.maximum = maximum

    def register(self, jobs, identity):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._assert_unreleased()
            old = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if old and old[0] != identity:
                raise ValueError("Frozen queue identity changed")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('identity',?)", (identity,))
            for job in jobs:
                payload = json.dumps(job, sort_keys=True)
                prior = self.db.execute(
                    "SELECT payload FROM jobs WHERE id=?", (job["id"],)
                ).fetchone()
                if prior and prior[0] != payload:
                    raise ValueError("Registered job was changed: " + job["id"])
                self.db.execute(
                    "INSERT OR IGNORE INTO jobs(id,payload,status) VALUES(?,?,?)",
                    (job["id"], payload, "PENDING"),
                )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def rows(self):
        return [
            {
                **dict(r),
                "payload": json.loads(r["payload"]),
                "result": json.loads(r["result"]) if r["result"] else None,
            }
            for r in self.db.execute("SELECT * FROM jobs ORDER BY rowid")
        ]

    def claim(self, worker, *, blocked_kinds=()):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._assert_unreleased()
            rows = self.rows()
            states = {row["id"]: row["status"] for row in rows}
            if sum(r["status"] == "RUNNING" for r in rows) >= self.maximum:
                self.db.execute("COMMIT")
                return None
            running_parents = {
                r["payload"].get("parent")
                for r in rows
                if r["status"] == "RUNNING" and r["payload"]["kind"] == "teacher"
            }

            def priority(row):
                job = row["payload"]
                value = job.get("priority", 10)
                if job["kind"] == "teacher" and job.get("split") in ("V_selection", "T_train"):
                    value = (
                        0
                        if job.get("parent") not in running_parents and len(running_parents) < 2
                        else value + 4
                    )
                return (value, rows.index(row))

            for row in sorted(rows, key=priority):
                if row["status"] != "PENDING" or row["payload"]["kind"] in blocked_kinds:
                    continue
                if any(states.get(dep) not in TERMINAL for dep in row["payload"].get("after", [])):
                    continue
                deps = row["payload"].get("dependencies", [])
                if any(states.get(dep) == "BLOCKED_TECHNICAL" for dep in deps):
                    self.db.execute(
                        "UPDATE jobs SET status='BLOCKED_TECHNICAL',finished=?,result=? WHERE id=?",
                        (
                            time.time(),
                            json.dumps({"reason": "dependency blocked", "dependencies": deps}),
                            row["id"],
                        ),
                    )
                    states[row["id"]] = "BLOCKED_TECHNICAL"
                    continue
                if any(states.get(dep) not in SUCCESS for dep in deps):
                    continue
                self.db.execute(
                    "UPDATE jobs SET status='RUNNING',worker=?,started=? WHERE id=?",
                    (worker, time.time(), row["id"]),
                )
                self.db.execute("COMMIT")
                return row["payload"]
            self.db.execute("COMMIT")
            return None
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def finish(self, job_id, worker, status, result):
        if status not in TERMINAL:
            raise ValueError("Invalid terminal status")
        changed = self.db.execute(
            (
                "UPDATE jobs SET status=?,finished=?,resu"
                "lt=? WHERE id=? AND status='RUNNING' AND"
                " worker=?"
            ),
            (status, time.time(), json.dumps(result, allow_nan=False), job_id, worker),
        ).rowcount
        if changed != 1:
            raise ValueError("Worker does not own running task")

    def all_terminal(self):
        rows = self.rows()
        return bool(rows) and all(row["status"] in TERMINAL for row in rows)

    def retry(self, job_id, *, reason):
        """Explicit operator recovery only, after the old process has been stopped."""
        if not reason:
            raise ValueError("Recovery requires an audit reason")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._assert_unreleased()
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] not in {"BLOCKED_TECHNICAL", "RUNNING"}:
                raise ValueError("Only interrupted/technical tasks can be retried")
            self.db.execute(
                "INSERT INTO events(job,event,at,details) VALUES(?,?,?,?)",
                (
                    job_id,
                    "EXPLICIT_RETRY",
                    time.time(),
                    json.dumps({"reason": reason, "previous": dict(row)}),
                ),
            )
            self.db.execute(
                (
                    "UPDATE jobs SET status='PENDING',worker="
                    "NULL,started=NULL,finished=NULL,result=N"
                    "ULL WHERE id=?"
                ),
                (job_id,),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def _assert_unreleased(self):
        if self.db.execute("SELECT value FROM metadata WHERE key='released'").fetchone():
            raise PermissionError(
                "E/G release is irreversible; no registration, claiming or retry afterward"
            )

    def seal_release(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if not self.all_terminal():
                raise PermissionError("Nonterminal matrix cannot be released")
            matrix_digest = digest(self.rows())
            prior = self.db.execute("SELECT value FROM metadata WHERE key='released'").fetchone()
            if prior and prior[0] != matrix_digest:
                raise ValueError("Released matrix was modified")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('released',?)", (matrix_digest,))
            self.db.execute("COMMIT")
            return matrix_digest
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
