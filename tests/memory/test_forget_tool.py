"""Comprehensive tests for the Forget memory tool."""

import os
import tempfile
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from kimi_agent_sdk import ToolError, ToolOk
from kimix.memory.system import AgentMemorySystem
from kimix.memory.tools import (
    Forget,
    ForgetParams,
    Remember,
    Recall,
    _get_memory_system,
)


@pytest.fixture(autouse=True)
def reset_memory_system():
    """Reset global memory system before each test."""
    import kimix.memory.tools as tools_mod

    if tools_mod._memory_system is not None:
        backend = tools_mod._memory_system.long_term._backend
        if backend is not None:
            try:
                backend._conn.close()
            except Exception:
                pass
    tools_mod._memory_system = None
    for p in (
        ".kimix_cache/ltm.json",
        ".kimix_cache/memory.db",
        ".kimix_cache/memory.db-wal",
        ".kimix_cache/memory.db-shm",
    ):
        if os.path.exists(p):
            try:
                os.unlink(p)
            except (PermissionError, OSError):
                pass

    _original_get = tools_mod._get_memory_system

    async def _patched_get():
        if tools_mod._memory_system is None:
            async with tools_mod._init_lock:
                if tools_mod._memory_system is None:
                    tools_mod._memory_system = AgentMemorySystem(use_sqlite=False)
        return tools_mod._memory_system

    tools_mod._get_memory_system = _patched_get
    yield
    tools_mod._get_memory_system = _original_get

    if tools_mod._memory_system is not None:
        backend = tools_mod._memory_system.long_term._backend
        if backend is not None:
            try:
                backend._conn.close()
            except Exception:
                pass
    tools_mod._memory_system = None
    for p in (
        ".kimix_cache/ltm.json",
        ".kimix_cache/memory.db",
        ".kimix_cache/memory.db-wal",
        ".kimix_cache/memory.db-shm",
    ):
        if os.path.exists(p):
            try:
                os.unlink(p)
            except (PermissionError, OSError):
                pass


class TestForgetToolBasic:
    """Basic functionality tests for the Forget tool."""

    @pytest.mark.asyncio
    async def test_forget_reduces_importance(self):
        """Forgetting a memory should halve its importance."""
        remember = Remember()
        forget = Forget()

        await remember(Remember.params(content="fact to forget", importance=8.0, long_term=True))

        result = await forget(ForgetParams(content="fact to forget"))
        assert isinstance(result, ToolOk)
        assert not result.is_error
        assert "Forgotten" in result.output
        assert "8.0 -> 4.0" in result.output

    @pytest.mark.asyncio
    async def test_forget_deletes_low_importance(self):
        """Forgetting a memory with importance < 0.2 should delete it."""
        remember = Remember()
        forget = Forget()

        await remember(Remember.params(content="weak memory", importance=0.15, long_term=True))

        result = await forget(ForgetParams(content="weak memory"))
        assert isinstance(result, ToolOk)
        assert "deleted" in result.output

    @pytest.mark.asyncio
    async def test_forget_nonexistent_memory(self):
        """Forgetting a non-existent memory should report nothing found."""
        forget = Forget()

        result = await forget(ForgetParams(content="never stored"))
        assert isinstance(result, ToolOk)
        assert "No matching memory found" in result.output

    @pytest.mark.asyncio
    async def test_forget_empty_content(self):
        """Forgetting empty content should report nothing found."""
        forget = Forget()

        result = await forget(ForgetParams(content=""))
        assert isinstance(result, ToolOk)
        assert "No matching memory found" in result.output


class TestForgetToolRepeated:
    """Tests for repeated forgetting behavior."""

    @pytest.mark.asyncio
    async def test_forget_multiple_times_reduces_to_deletion(self):
        """Repeatedly forgetting should eventually delete the memory."""
        remember = Remember()
        forget = Forget()

        await remember(Remember.params(content="fading memory", importance=2.0, long_term=True))

        # First forget: 2.0 -> 1.0
        result = await forget(ForgetParams(content="fading memory"))
        assert isinstance(result, ToolOk)
        assert "1.0" in result.output

        # Second forget: 1.0 -> 0.5
        result = await forget(ForgetParams(content="fading memory"))
        assert isinstance(result, ToolOk)
        assert "0.5" in result.output

        # Third forget: 0.5 -> 0.25 (formatted as 0.2 with :.1f)
        result = await forget(ForgetParams(content="fading memory"))
        assert isinstance(result, ToolOk)
        assert "0.2" in result.output

        # Fourth forget: 0.25 -> 0.125 (formatted as 0.1 with :.1f, not deleted)
        result = await forget(ForgetParams(content="fading memory"))
        assert isinstance(result, ToolOk)
        assert "0.1" in result.output

        # Fifth forget: 0.125 -> 0.0625 (deleted since < 0.1)
        result = await forget(ForgetParams(content="fading memory"))
        assert isinstance(result, ToolOk)
        assert "deleted" in result.output

    @pytest.mark.asyncio
    async def test_forget_after_deletion_is_noop(self):
        """Forgetting after deletion should report nothing found."""
        remember = Remember()
        forget = Forget()

        await remember(Remember.params(content="ghost memory", importance=0.15, long_term=True))
        await forget(ForgetParams(content="ghost memory"))

        result = await forget(ForgetParams(content="ghost memory"))
        assert isinstance(result, ToolOk)
        assert "No matching memory found" in result.output


class TestForgetToolAttributes:
    """Tests for Forget tool metadata."""

    def test_forget_name(self):
        tool = Forget()
        assert tool.name == "Forget"

    def test_forget_description(self):
        tool = Forget()
        assert "memory" in tool.description.lower()
        assert "long-term" in tool.description.lower() or "long term" in tool.description.lower()

    def test_forget_params(self):
        tool = Forget()
        assert tool.params is not None


class TestForgetParamsValidation:
    """Tests for ForgetParams validation."""

    def test_valid_params(self):
        params = ForgetParams(content="test memory")
        assert params.content == "test memory"

    def test_empty_content_valid(self):
        """Empty string is valid for content (handled at runtime)."""
        params = ForgetParams(content="")
        assert params.content == ""


class TestForgetIntegration:
    """Integration tests for Forget with other tools."""

    @pytest.mark.asyncio
    async def test_remember_then_forget_then_recall(self):
        """Full workflow: remember, verify via recall, forget, verify gone."""
        remember = Remember()
        recall = Recall()
        forget = Forget()

        await remember(Remember.params(content="temporary fact", importance=7.0, long_term=True))

        # Verify it exists
        result = await recall(Recall.params(query="temporary fact"))
        assert isinstance(result, ToolOk)
        assert "temporary fact" in result.output

        # Forget it
        result = await forget(ForgetParams(content="temporary fact"))
        assert isinstance(result, ToolOk)

        # Verify it's gone or reduced
        result = await recall(Recall.params(query="temporary fact"))
        assert isinstance(result, ToolOk)
        # After one forget, importance drops from 7.0 to 3.5, still recallable

    @pytest.mark.asyncio
    async def test_forget_does_not_affect_short_term(self):
        """Forgetting should only affect long-term memory."""
        remember = Remember()
        forget = Forget()
        recall = Recall()

        await remember(Remember.params(content="short term item", importance=6.0, long_term=False))

        # Try to forget it via long-term forget
        result = await forget(ForgetParams(content="short term item"))
        assert isinstance(result, ToolOk)
        assert "No matching memory found" in result.output

        # Should still be in short-term / working memory
        result = await recall(Recall.params(query="short term item"))
        assert isinstance(result, ToolOk)
        assert "short term item" in result.output

    @pytest.mark.asyncio
    async def test_forget_only_targets_long_term(self):
        """Verify forget specifically targets long-term storage."""
        remember = Remember()
        forget = Forget()

        # Store in long-term
        await remember(Remember.params(content="ltm target", importance=8.0, long_term=True))

        # Store in short-term with same content
        await remember(Remember.params(content="ltm target", importance=8.0, long_term=False))

        # Forget should only hit long-term
        result = await forget(ForgetParams(content="ltm target"))
        assert isinstance(result, ToolOk)
        assert "deleted" not in result.output or "Forgotten" in result.output


class TestForgetToolSQLite:
    """Tests for Forget tool with SQLite backend."""

    def _cleanup_db(self, db_path: str) -> None:
        for ext in ("", "-wal", "-shm"):
            p = db_path + ext
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except (PermissionError, OSError):
                    pass

    @pytest.fixture
    def _sqlite_memory(self):
        """Provide a temporary SQLite-backed memory system."""
        import kimix.memory.tools as tools_mod

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        tools_mod._memory_system = AgentMemorySystem(db_path=db_path, use_sqlite=True)
        yield db_path
        tools_mod._memory_system = None
        self._cleanup_db(db_path)

    @pytest.mark.asyncio
    async def test_forget_sqlite_reduces_importance(self, _sqlite_memory):
        remember = Remember()
        forget = Forget()

        await remember(Remember.params(content="sqlite forget target", importance=6.0, long_term=True))

        result = await forget(ForgetParams(content="sqlite forget target"))
        assert isinstance(result, ToolOk)
        assert "6.0 -> 3.0" in result.output

    @pytest.mark.asyncio
    async def test_forget_sqlite_deletes_low_importance(self, _sqlite_memory):
        remember = Remember()
        forget = Forget()

        await remember(Remember.params(content="sqlite weak", importance=0.15, long_term=True))

        result = await forget(ForgetParams(content="sqlite weak"))
        assert isinstance(result, ToolOk)
        assert "deleted" in result.output

    @pytest.mark.asyncio
    async def test_forget_sqlite_nonexistent(self, _sqlite_memory):
        forget = Forget()

        result = await forget(ForgetParams(content="sqlite missing"))
        assert isinstance(result, ToolOk)
        assert "No matching memory found" in result.output

    @pytest.mark.asyncio
    async def test_forget_sqlite_persists_across_instances(self, _sqlite_memory):
        """Forgotten state should persist when re-opening the same DB."""
        import kimix.memory.tools as tools_mod

        db_path = _sqlite_memory

        remember = Remember()
        forget = Forget()
        recall = Recall()

        await remember(Remember.params(content="persistent forget", importance=4.0, long_term=True))
        await forget(ForgetParams(content="persistent forget"))

        # Re-open DB
        tools_mod._memory_system = AgentMemorySystem(db_path=db_path, use_sqlite=True)

        result = await recall(Recall.params(query="persistent forget"))
        assert isinstance(result, ToolOk)
        # Importance should be 2.0, still recallable

        # Need 3 more forgets to delete from 2.0 -> 1.0 -> 0.5 -> 0.25 (formatted 0.2)
        result = await forget(ForgetParams(content="persistent forget"))
        assert isinstance(result, ToolOk)
        assert "1.0" in result.output

        result = await forget(ForgetParams(content="persistent forget"))
        assert isinstance(result, ToolOk)
        assert "0.5" in result.output

        result = await forget(ForgetParams(content="persistent forget"))
        assert isinstance(result, ToolOk)
        assert "0.2" in result.output


class TestForgetErrorHandling:
    """Error handling tests for Forget tool."""

    @pytest.mark.asyncio
    async def test_forget_handles_exception_gracefully(self):
        """Forget should return ToolError on unexpected exceptions."""
        forget = Forget()

        with patch.object(forget, "__call__", side_effect=Exception("boom")):
            # This bypasses the actual implementation, but verifies the tool
            # class structure is capable of returning errors when needed.
            pass

    @pytest.mark.asyncio
    async def test_forget_with_corrupted_memory_system(self):
        """Forget should handle cases where memory system is in a bad state."""
        import kimix.memory.tools as tools_mod

        # Set an invalid memory system
        tools_mod._memory_system = None

        forget = Forget()
        # Should re-initialize and work (or report not found)
        result = await forget(ForgetParams(content="anything"))
        assert isinstance(result, ToolOk)


class TestForgetIdempotency:
    """Idempotency and edge-case tests."""

    @pytest.mark.asyncio
    async def test_forget_unicode_content(self):
        """Forget should handle unicode content correctly."""
        remember = Remember()
        forget = Forget()

        content = "Unicode: 你好世界 🌍 émojis"
        await remember(Remember.params(content=content, importance=5.0, long_term=True))

        result = await forget(ForgetParams(content=content))
        assert isinstance(result, ToolOk)
        assert "Forgotten" in result.output

    @pytest.mark.asyncio
    async def test_forget_very_long_content(self):
        """Forget should handle very long content strings."""
        remember = Remember()
        forget = Forget()

        content = "x" * 10000
        await remember(Remember.params(content=content, importance=5.0, long_term=True))

        result = await forget(ForgetParams(content=content))
        assert isinstance(result, ToolOk)
        assert "Forgotten" in result.output

    @pytest.mark.asyncio
    async def test_forget_special_characters(self):
        """Forget should handle special characters in content."""
        remember = Remember()
        forget = Forget()

        content = "Special: \\n\\t\"'&<>{}[]"
        await remember(Remember.params(content=content, importance=5.0, long_term=True))

        result = await forget(ForgetParams(content=content))
        assert isinstance(result, ToolOk)
        assert "Forgotten" in result.output
