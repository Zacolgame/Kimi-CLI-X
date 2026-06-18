from collections.abc import Iterator


def _get_slash_commands() -> set[str]:
    """Lazily get slash command names to avoid circular import at module level."""
    from .commands import _command_map_keys  # type: ignore[attr-defined]

    return _command_map_keys


def _get_slash_command_items() -> dict[str, str]:
    """Lazily get slash command descriptions to avoid circular imports."""
    from .commands import get_command_descriptions

    return get_command_descriptions()


class SlashCommandCompleter:
    """prompt_toolkit completer for interactive slash commands."""

    def get_completions(self, document, complete_event) -> Iterator[object]:
        try:
            from prompt_toolkit.completion import Completion
        except Exception:
            return

        text = document.text_before_cursor
        if not text.startswith('/'):
            return

        command_prefix = text[1:]
        if any(ch in command_prefix for ch in ' \t\n:'):
            return

        start_position = -len(command_prefix) if command_prefix else 0
        for name, description in _get_slash_command_items().items():
            if name.startswith(command_prefix):
                yield Completion(
                    name,
                    start_position=start_position,
                    display=f'/{name}',
                    display_meta=description,
                )

    async def get_completions_async(self, document, complete_event) -> Iterator[object]:
        for completion in self.get_completions(document, complete_event):
            yield completion


def _prompt_with_completion(text: str) -> str | None:
    try:
        from prompt_toolkit import prompt
    except Exception:
        return None

    return prompt(
        text,
        completer=SlashCommandCompleter(),
        complete_while_typing=True,
    )


def _input(text: str, text_arr: list[str], multi_line_mode: bool = False) -> str:
    if text_arr is None or len(text_arr) == 0:
        if multi_line_mode:
            return input(text)
        try:
            completed = _prompt_with_completion(text)
        except EOFError:
            raise
        except Exception:
            completed = None
        if completed is not None:
            return completed
        return input(text)
    v = text_arr.pop(0)
    return v


def _split_text(lines: list[str], command_map: set[str] | None = None) -> list[str]:
    text_arr: list[str] = []
    current_text: list[str] = []
    for line in lines:
        strip_line = line.strip()
        if len(strip_line) == 0:
            current_text.append('')
            continue
        if strip_line.startswith('/'):
            if len(strip_line) > 1:
                cmd = strip_line[1:].split()[0]
                if command_map is not None and cmd not in command_map:
                    current_text.append(line)
                    continue
            if current_text:
                text_arr.append('\n'.join(current_text))
                current_text = []
            if len(strip_line) > 1:
                text_arr.append(strip_line)
        else:
            current_text.append(line)
    if current_text:
        text_arr.append('\n'.join(current_text))
    return text_arr
