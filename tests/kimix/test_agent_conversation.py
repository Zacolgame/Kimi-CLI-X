"""Tests for the conversational Agent system."""

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kimix.base import MessageType
from kimix.tools.agent.store import (
    AgentSessionEntry,
    AgentSessionStore,
    ConversationTurn,
)
from kimix.tools.agent import (
    Agent,
    AgentClose,
    AgentCloseParams,
    AgentList,
    AgentListParams,
    AskParent,
    AskParentParams,
    SubAgentParams,
    _AgentConversationCollector,
    _get_store,
    _register_entry,
    _unregister_entry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_session() -> MagicMock:
    session = MagicMock()
    session.custom_data = {}
    session.custom_config = {}
    return session


@pytest.fixture
def mock_sub_session() -> MagicMock:
    session = MagicMock()
    session.id = "sub-123"
    session.get_custom_config.return_value = {}
    session.close = AsyncMock()
    return session


# ---------------------------------------------------------------------------
# AgentSessionStore tests
# ---------------------------------------------------------------------------
async def test_store_get_put(mock_sub_session: MagicMock) -> None:
    store = AgentSessionStore()
    assert store.get("nonexistent") is None

    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="s1",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=0,
    )
    store.put(entry)
    assert store.get("s1") is entry


async def test_store_close(mock_sub_session: MagicMock) -> None:
    store = AgentSessionStore()
    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="s1",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=0,
    )
    store.put(entry)
    assert store.close("s1") is True
    assert store.get("s1") is None
    assert store.close("s1") is False


async def test_store_list_active(mock_sub_session: MagicMock) -> None:
    store = AgentSessionStore()
    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="s1",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=3,
    )
    store.put(entry)
    active = store.list_active()
    assert len(active) == 1
    assert active[0]["session_id"] == "s1"
    assert active[0]["total_turns"] == 3


async def test_store_lru_eviction(mock_sub_session: MagicMock) -> None:
    store = AgentSessionStore()
    store.MAX_SESSIONS = 3

    for i in range(4):
        entry = AgentSessionEntry(
            session=mock_sub_session,
            session_id=f"s{i}",
            created_at=time.time(),
            last_accessed=time.time() + i,
            conversation_history=[],
            total_turns=0,
        )
        store.put(entry)

    assert len(store.entries) == 4
    with patch(
        "kimix.tools.agent.store.close_session_async", new_callable=AsyncMock
    ) as mock_close:
        await store.evict_lru_if_needed()
        assert mock_close.await_count == 2

    assert len(store.entries) == 2
    assert store.get("s0") is None  # oldest
    assert store.get("s1") is None
    assert store.get("s2") is not None
    assert store.get("s3") is not None


# ---------------------------------------------------------------------------
# _AgentConversationCollector tests
# ---------------------------------------------------------------------------
async def test_collector_text_only() -> None:
    col = _AgentConversationCollector()
    col.finalize_user_turn("hello")
    col.consume("world", MessageType.Text)
    text = col.finalize_assistant_turn()
    assert text == "world"
    assert len(col.turns) == 2
    assert col.turns[0].role == "user"
    assert col.turns[1].role == "assistant"
    assert col.turns[1].metadata == {"type": "text"}


async def test_collector_thinking_excluded_from_output() -> None:
    col = _AgentConversationCollector()
    col.consume("text1", MessageType.Text)
    col.consume("think1", MessageType.Thinking)
    col.consume("text2", MessageType.Text)
    text = col.finalize_assistant_turn()
    assert text == "text1text2"
    assert any(t.metadata == {"type": "thinking"} for t in col.turns)


async def test_collector_tool_call_and_result() -> None:
    col = _AgentConversationCollector()
    col.consume("ToolA args", MessageType.ToolCalling)
    col.consume("[ToolResult] ok", MessageType.ToolResult)
    text = col.finalize_assistant_turn()
    assert text == ""
    roles = [t.role for t in col.turns]
    assert roles == ["tool", "tool"]
    assert col.turns[0].metadata == {"type": "tool_call"}
    assert col.turns[1].metadata == {"type": "tool_result"}


async def test_collector_empty_output() -> None:
    col = _AgentConversationCollector()
    col.finalize_user_turn("prompt")
    text = col.finalize_assistant_turn()
    assert text == ""


# ---------------------------------------------------------------------------
# Agent tool tests
# ---------------------------------------------------------------------------
async def test_agent_recursive_guard(mock_session: MagicMock) -> None:
    mock_session.custom_config = {"is_sub_agent": True}
    agent = Agent(mock_session)
    result = await agent(SubAgentParams(prompt="test"))
    assert result.is_error
    assert "Recursive sub-agent call detected" in result.message


async def test_agent_new_session(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X"))

    assert not result.is_error
    assert result.extras is not None
    assert "session_id" in result.extras
    assert result.extras["status"] == "closed"
    assert result.extras["turn_count"] >= 1
    mock_create.assert_awaited_once()
    mock_prompt.assert_awaited_once()


async def test_agent_keep_alive_stores_session(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X", close_session=False))

    assert not result.is_error
    assert result.extras["status"] == "continued"
    store = _get_store(mock_session)
    assert store.get(result.extras["session_id"]) is not None
    mock_close.assert_not_awaited()


async def test_agent_reuse_session(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="reuse-id",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=0,
        )
    )

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            agent = Agent(mock_session)
            result = await agent(
                SubAgentParams(prompt="follow up", session_id="reuse-id", close_session=False)
            )

    assert not result.is_error
    assert result.extras["session_id"] == "reuse-id"
    mock_create.assert_not_awaited()


async def test_agent_close_session_param(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="close-id",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=0,
        )
    )

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                result = await agent(
                    SubAgentParams(
                        prompt="do X", session_id="close-id", close_session=True
                    )
                )

    assert not result.is_error
    assert result.extras["status"] == "closed"
    assert store.get("close-id") is None
    mock_close.assert_awaited_once()


async def test_agent_return_history(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X", return_history=True))

    assert not result.is_error
    assert "conversation_history" in result.extras
    history = result.extras["conversation_history"]
    assert isinstance(history, list)
    assert any(h["role"] == "user" for h in history)


async def test_agent_error_path(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            mock_prompt.side_effect = RuntimeError("boom")
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                result = await agent(SubAgentParams(prompt="do X", close_session=False))

    assert result.is_error
    assert "boom" in result.message
    assert result.extras["status"] == "closed"
    mock_close.assert_awaited_once()


async def test_agent_lru_eviction(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    store.MAX_SESSIONS = 2

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ):
            with patch(
                "kimix.tools.agent.store.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                for i in range(3):
                    mock_sub = MagicMock()
                    mock_sub.id = f"sub-{i}"
                    mock_sub.get_custom_config.return_value = {}
                    mock_sub.close = AsyncMock()
                    mock_create.return_value = mock_sub
                    await agent(
                        SubAgentParams(prompt=f"task {i}", close_session=False)
                    )

    assert len(store.entries) == 2
    mock_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# Companion tool tests
# ---------------------------------------------------------------------------
async def test_agent_list(mock_session: MagicMock, mock_sub_session: MagicMock) -> None:
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="list-id",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=1,
        )
    )
    agent_list = AgentList(mock_session)
    result = await agent_list(AgentListParams())
    assert not result.is_error
    assert "list-id" in result.output


async def test_agent_close(mock_session: MagicMock, mock_sub_session: MagicMock) -> None:
    store = _get_store(mock_session)
    store.put(
        AgentSessionEntry(
            session=mock_sub_session,
            session_id="close-id",
            created_at=time.time(),
            last_accessed=time.time(),
            conversation_history=[],
            total_turns=1,
        )
    )
    with patch(
        "kimix.tools.agent.close_session_async", new_callable=AsyncMock
    ) as mock_close:
        agent_close = AgentClose(mock_session)
        result = await agent_close(AgentCloseParams(session_id="close-id"))

    assert not result.is_error
    assert store.get("close-id") is None
    mock_close.assert_awaited_once()


async def test_agent_close_not_found(mock_session: MagicMock) -> None:
    agent_close = AgentClose(mock_session)
    result = await agent_close(AgentCloseParams(session_id="missing"))
    assert result.is_error
    assert "Session not found" in result.message


# ---------------------------------------------------------------------------
# Conversation protocol tests
# ---------------------------------------------------------------------------
async def test_ask_parent_tool_sets_pending_question(mock_sub_session: MagicMock) -> None:
    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="ask-id",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=0,
    )
    _register_entry("ask-id", entry)
    mock_sub_session.id = "ask-id"
    ask_parent = AskParent(mock_sub_session)
    result = await ask_parent(AskParentParams(question="What is the color?"))
    assert not result.is_error
    assert entry.pending_question == "What is the color?"
    assert entry.state == "awaiting_response"
    _unregister_entry("ask-id")


async def test_agent_awaiting_response_status(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="conv-id",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=0,
    )
    store.put(entry)
    _register_entry("conv-id", entry)

    async def _mock_prompt_async(*, prompt_str, session, output_function, **kwargs):
        # Simulate sub-agent calling ask_parent during the turn
        entry.pending_question = "What format do you want?"
        entry.state = "awaiting_response"
        if output_function:
            output_function("I need clarification", MessageType.Text)

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            mock_prompt.side_effect = _mock_prompt_async
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ) as mock_close:
                agent = Agent(mock_session)
                result = await agent(
                    SubAgentParams(prompt="do X", session_id="conv-id", close_session=False)
                )

    assert not result.is_error
    assert result.extras["status"] == "awaiting_response"
    assert result.extras["question"] == "What format do you want?"
    mock_close.assert_not_awaited()
    _unregister_entry("conv-id")


async def test_agent_response_injection(
    mock_session: MagicMock, mock_sub_session: MagicMock
) -> None:
    mock_session.custom_config = {"chat_provider": None}
    store = _get_store(mock_session)
    entry = AgentSessionEntry(
        session=mock_sub_session,
        session_id="resp-id",
        created_at=time.time(),
        last_accessed=time.time(),
        conversation_history=[],
        total_turns=0,
        pending_question="What format?",
        state="awaiting_response",
    )
    store.put(entry)
    _register_entry("resp-id", entry)

    captured_prompt = None

    async def _mock_prompt_async(*, prompt_str, session, output_function, **kwargs):
        nonlocal captured_prompt
        captured_prompt = prompt_str
        if output_function:
            output_function("OK", MessageType.Text)

    with patch(
        "kimix.tools.agent._create_session_async", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_sub_session
        with patch(
            "kimix.tools.agent.utils.prompt_async", new_callable=AsyncMock
        ) as mock_prompt:
            mock_prompt.side_effect = _mock_prompt_async
            with patch(
                "kimix.tools.agent.close_session_async", new_callable=AsyncMock
            ):
                agent = Agent(mock_session)
                result = await agent(
                    SubAgentParams(
                        prompt="continue",
                        session_id="resp-id",
                        response="JSON format",
                        close_session=False,
                    )
                )

    assert not result.is_error
    assert captured_prompt is not None
    assert "JSON format" in captured_prompt
    assert "What format?" in captured_prompt
    assert entry.pending_question is None
    assert entry.state == "running"
    _unregister_entry("resp-id")
