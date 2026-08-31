"""Durable state persistence engine for multi-agent business pipeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional, List
from workflow.schema import LeadState


class DurableStateStore:
    """SQLite-backed durable state store ensuring mid-pipeline crash recovery."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path == ":memory:":
            self._conn = sqlite3.connect(":memory:")
        else:
            self._conn = None
        self._init_db()

    def _get_conn(self):
        if self._conn:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lead_states (
                lead_id TEXT PRIMARY KEY,
                current_stage TEXT NOT NULL,
                state_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        if not self._conn:
            conn.close()

    def save_state(self, state: LeadState):
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO lead_states (lead_id, current_stage, state_json)
            VALUES (?, ?, ?)
            ON CONFLICT(lead_id) DO UPDATE SET
                current_stage = excluded.current_stage,
                state_json = excluded.state_json,
                updated_at = CURRENT_TIMESTAMP;
        """, (state.lead_id, state.current_stage, state.model_dump_json()))
        conn.commit()
        if not self._conn:
            conn.close()

    def load_state(self, lead_id: str) -> Optional[LeadState]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT state_json FROM lead_states WHERE lead_id = ?", (lead_id,))
        row = cursor.fetchone()
        if not self._conn:
            conn.close()
        if row:
            return LeadState.model_validate_json(row[0])
        return None

    def list_pending_approvals(self) -> List[LeadState]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT state_json FROM lead_states WHERE current_stage = 'approval_gate' AND approval_status = 'pending'")
        rows = cursor.fetchall()
        if not self._conn:
            conn.close()
        return [LeadState.model_validate_json(r[0]) for r in rows]
