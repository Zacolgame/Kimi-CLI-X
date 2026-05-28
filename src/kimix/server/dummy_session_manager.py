# -*- coding: utf-8 -*-
"""Dummy session manager: stubs all SessionManager interfaces with no real logic.

Prints each web request and its formatted arguments for debugging purposes.
Matches the interface consumed by `src/kimix/server/app.py`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Minimal data models (replicated for standalone use) ──────────


@dataclass
class SessionInfo:
    id: str = ""
    title: Optional[str] = None
    createdAt: float = 0.0
    updatedAt: float = 0.0
    parentID: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "createdAt": self.createdAt,
            "updatedAt": self.updatedAt,
            "parentID": self.parentID,
        }


@dataclass
class SessionStatus:
    type: str = "idle"
    time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "time": self.time}


# ── ID Helpers ──────────────────────────────────────────────────

_counter = 0


def _gen_id(prefix: str) -> str:
    global _counter
    _counter += 1
    return f"{prefix}_dummy_{_counter:04x}"


def _now_ts() -> float:
    return time.time()


# ── Dummy Session Manager ────────────────────────────────────────


class DummySessionManager:
    """Drop-in replacement for SessionManager with zero real logic.

    Every public method prints the incoming request name and its arguments.
    Return values are minimal, valid stubs so the HTTP layer does not crash.
    """

    def __init__(self) -> None:
        self._dummy_session = SessionInfo(
            id=_gen_id("ses"),
            title="Dummy Session",
            createdAt=_now_ts(),
            updatedAt=_now_ts(),
        )

    # ── Session CRUD ─────────────────────────────────────────────

    async def create_session(self, title: Optional[str] = None) -> SessionInfo:
        """POST /session"""
        print(f"[DummySessionManager] create_session(title={title!r})")
        info = SessionInfo(
            id=_gen_id("ses"),
            title=title or "Dummy Session",
            createdAt=_now_ts(),
            updatedAt=_now_ts(),
        )
        return info

    def get_session(self, session_id: str) -> SessionInfo:
        """GET /session/{sessionID}"""
        print(f"[DummySessionManager] get_session(session_id={session_id!r})")
        return SessionInfo(
            id=session_id,
            title="Dummy Session",
            createdAt=_now_ts(),
            updatedAt=_now_ts(),
        )

    def list_sessions(self) -> List[SessionInfo]:
        """GET /session"""
        print("[DummySessionManager] list_sessions()")
        return [self._dummy_session]

    async def delete_session(self, session_id: str) -> bool:
        """DELETE /session/{sessionID}"""
        print(f"[DummySessionManager] delete_session(session_id={session_id!r})")
        return True

    def get_session_status(self) -> Dict[str, Dict[str, Any]]:
        """GET /session/status"""
        print("[DummySessionManager] get_session_status()")
        return {self._dummy_session.id: SessionStatus().to_dict()}

    # ── Messages ─────────────────────────────────────────────────

    def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """GET /session/{sessionID}/message"""
        print(
            f"[DummySessionManager] get_messages(session_id={session_id!r}, limit={limit!r})"
        )
        return []

    # ── Prompt (fire-and-forget) ──────────────────────────────────

    async def prompt_async(
        self,
        session_id: str,
        text: str,
        agent: Optional[str] = None,
    ) -> None:
        """POST /session/{sessionID}/prompt_async"""
        print(
            f"[DummySessionManager] prompt_async(session_id={session_id!r}, text={text!r}, agent={agent!r})"
        )

    # ── Abort ────────────────────────────────────────────────────

    def abort_session(self, session_id: str) -> bool:
        """POST /session/{sessionID}/abort"""
        print(f"[DummySessionManager] abort_session(session_id={session_id!r})")
        return True

    # ── Options ─────────────────────────────────────────────────

    async def clear_session(self, session_id: str) -> bool:
        """GET /session/{sessionID}/clear"""
        print(f"[DummySessionManager] clear_session(session_id={session_id!r})")
        return True

    async def compact_session(
        self, session_id: str, keep: Optional[int] = None
    ) -> bool:
        """GET /session/{sessionID}/compact"""
        print(
            f"[DummySessionManager] compact_session(session_id={session_id!r}, keep={keep!r})"
        )
        return True

    async def get_session_context(
        self, session_id: str, keep: Optional[int] = None
    ) -> Dict[str, Any]:
        """GET /session/{sessionID}/context"""
        print(
            f"[DummySessionManager] get_session_context(session_id={session_id!r}, keep={keep!r})"
        )
        return {"sessionID": session_id, "context_usage": None}

    async def export_session(
        self, session_id: str, output_path: Optional[str] = None
    ) -> tuple[str, int]:
        """GET /session/{sessionID}/export"""
        print(
            f"[DummySessionManager] export_session(session_id={session_id!r}, output_path={output_path!r})"
        )
        return (output_path or f"dummy_export_{session_id}.json", 0)


# Global singleton (drop-in for `from kimix.server.session_manager import session_manager`)
session_manager = DummySessionManager()
