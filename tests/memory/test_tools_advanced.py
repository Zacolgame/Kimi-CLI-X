"""Advanced tests for memory tools: temporal params, deep reflect."""

import pytest

from kimi_agent_sdk import ToolOk
from kimix.memory.tools import (
    Remember,
    Recall,
    GetContext,
    Reflect,
    RememberParams,
    RecallParams,
    ReflectParams,
)


@pytest.mark.asyncio
class TestRememberTemporalParams:
    async def test_remember_with_expires_at(self):
        import time
        tool = Remember()
        result = await tool(
            RememberParams(
                content="temporary fact",
                importance=5.0,
                long_term=True,
                expires_at=time.time() + 3600,
            )
        )
        assert isinstance(result, ToolOk)


@pytest.mark.asyncio
class TestRecallBasic:
    async def test_recall_without_procedural(self):
        tool = Recall()
        result = await tool(RecallParams(query="test"))
        assert isinstance(result, ToolOk)
        # Should not contain procedural section
        assert "PROCEDURAL" not in result.output


@pytest.mark.asyncio
class TestReflectDeep:
    async def test_reflect_default(self):
        tool = Reflect()
        result = await tool(ReflectParams())
        assert isinstance(result, ToolOk)
        assert "Memory System Status Report" in result.output

    async def test_reflect_deep(self):
        tool = Reflect()
        result = await tool(ReflectParams(deep=True))
        assert isinstance(result, ToolOk)
        assert "Self-Reflection Report" in result.output
