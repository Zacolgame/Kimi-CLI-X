from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

from prompt_toolkit.document import Document

from kimix.cli_impl.core import _enable_line_editing
from kimix.cli_impl import utils
from kimix.cli_impl.utils import SlashCommandCompleter


def test_enable_line_editing_imports_readline(monkeypatch):
    imported: list[str] = []
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "readline":
            imported.append(name)
            return SimpleNamespace()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    _enable_line_editing()

    assert imported == ["readline"]


def test_enable_line_editing_ignores_missing_readline(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "readline":
            raise ImportError("readline unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    _enable_line_editing()


def _completion_texts(text: str) -> list[str]:
    completer = SlashCommandCompleter()
    return [item.text for item in completer.get_completions(Document(text), None)]


def test_slash_completer_lists_commands():
    completions = _completion_texts("/")

    assert "help" in completions
    assert "resume" in completions
    assert "clear" in completions


def test_slash_completer_filters_by_prefix():
    completions = _completion_texts("/r")

    assert completions
    assert all(item.startswith("r") for item in completions)
    assert {"ralph", "rename", "resume"}.issubset(set(completions))
    assert "help" not in completions


async def test_slash_completer_supports_async_completion():
    completer = SlashCommandCompleter()
    completions = [
        item.text
        async for item in completer.get_completions_async(Document("/r"), None)
    ]

    assert {"ralph", "rename", "resume"}.issubset(set(completions))


def test_slash_completer_ignores_non_command_input():
    assert _completion_texts("hello") == []


def test_input_falls_back_when_prompt_toolkit_fails(monkeypatch):
    monkeypatch.setattr(utils, "_prompt_with_completion", lambda text: None)
    monkeypatch.setattr(builtins, "input", lambda text: "fallback")

    assert utils._input("prompt", []) == "fallback"
