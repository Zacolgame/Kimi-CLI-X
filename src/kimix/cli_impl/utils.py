def _get_slash_commands() -> set[str]:
    """Lazily get slash command names to avoid circular import at module level."""
    from .commands import _command_map_keys  # type: ignore[attr-defined]
    return _command_map_keys

def _input(text: str, text_arr: list[str], multi_line_mode: bool = False) -> str:
    if text_arr is None or len(text_arr) == 0:
        try:
            from prompt_toolkit import prompt as pt_prompt
            from prompt_toolkit.completion import Completer, Completion, PathCompleter
            # KeyBindings: define custom keyboard shortcuts.
            # merge_key_bindings: combine multiple KeyBindings sets (custom + defaults).
            # load_default_key_bindings: load Emacs-style defaults so standard
            #     shortcuts (arrow keys, Ctrl+A, Ctrl+E, Ctrl+K, etc.) are preserved.
            from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
            from prompt_toolkit.key_binding.bindings.named_commands import get_by_name
            from prompt_toolkit.key_binding.defaults import (
                load_key_bindings as load_default_key_bindings,
            )

            class SlashCompleter(Completer):
                """Auto-complete slash commands when user types '/'."""
                def __init__(self, multi_line_mode: bool = False):
                    self.multi_line_mode = multi_line_mode
                    self.path_completer = PathCompleter(
                        expanduser=True,
                        min_input_len=1,
                    )

                def get_completions(self, document, complete_event):
                    text_before = document.text_before_cursor
                    if text_before.startswith('/'):
                        partial = text_before[1:]
                        if self.multi_line_mode:
                            candidates = ['end', 'cancel']
                        else:
                            candidates = _get_slash_commands()
                        for cmd_name in candidates:
                            if cmd_name.startswith(partial):
                                yield Completion(
                                    '/' + cmd_name,
                                    start_position=-len(text_before),
                                    display='/' + cmd_name,
                                    display_meta='command',
                                )
                    elif text_before and not text_before.startswith(' '):
                        # Path completion using built-in PathCompleter.
                        # Only complete from the last word split by space.
                        last_word = text_before.split(' ')[-1]
                        if last_word:
                            from prompt_toolkit.document import Document
                            sub_doc = Document(last_word, len(last_word))
                            for completion in self.path_completer.get_completions(sub_doc, complete_event):
                                # PathCompleter returns start_position=0 and only
                                # the suffix text; using -len(last_word) would discard
                                # the already-typed characters.
                                yield Completion(
                                    completion.text,
                                    start_position=completion.start_position,
                                    display=completion.display,
                                    display_meta=completion.display_meta,
                                )

            # -- Key bindings --
            kb = KeyBindings()
            
            # Ctrl+W: delete previous word (backward-kill-word)
            kb.add('c-w')(get_by_name('backward-kill-word'))
            
            kb.add('\ue000')(get_by_name('backward-kill-word'))
            # Ctrl+U: delete from cursor to beginning of line (unix-line-discard)
            kb.add('c-u')(get_by_name('unix-line-discard'))

            # Ctrl+C: copy selection if any, otherwise exit with KeyboardInterrupt
            @kb.add('c-c')
            def _(event):
                buffer = event.current_buffer
                if buffer.selection_state is not None:
                    # Copy selected text to system clipboard
                    data = buffer.copy_selection()
                    import pyperclip
                    pyperclip.copy(data.text)
                    buffer.exit_selection()
                else:
                    event.app.exit(exception=KeyboardInterrupt, style='class:aborting')

            # Ctrl+V / Shift+Insert: paste from system clipboard at cursor position.
            # Ctrl+V is often intercepted by the terminal (verbatim-next),
            # so Shift+Insert is also bound as the standard terminal paste shortcut.
            def _paste_from_clipboard(event):
                import pyperclip
                try:
                    text = pyperclip.paste()
                    if isinstance(text, str) and text:
                        event.current_buffer.insert_text(text)
                except Exception:
                    pass

            kb.add('c-v')(_paste_from_clipboard)
            kb.add('s-insert')(_paste_from_clipboard)

            # Ctrl+D: exit if input is empty, otherwise delete character
            @kb.add('c-d')
            def _(event):
                print(222)
                buffer = event.current_buffer
                if buffer.text.strip() == '':
                    event.app.exit(exception=EOFError, style='class:exiting')
                else:
                    get_by_name('delete-char')(event)


            result = pt_prompt(
                text,
                completer=SlashCompleter(multi_line_mode=multi_line_mode),
                complete_while_typing=True,
                complete_in_thread=True,
                key_bindings=kb,
                handle_sigint=True,
            )
        except ImportError:
            # Fallback to built-in input if prompt_toolkit is not installed
            result = input(text)
        return result
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
