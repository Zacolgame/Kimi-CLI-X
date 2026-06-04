# Data Locations

Kimi Code CLI stores all data in the `~/.kimi/` directory under the user's home directory. This page describes the locations and purposes of various data files.

::: tip
You can customize the share directory path by setting the `KIMI_SHARE_DIR` environment variable. See [Environment Variables](./env-vars.md#kimi-share-dir) for details.

Note: `KIMI_SHARE_DIR` only affects the storage location of the runtime data listed above, not the [Agent Skills](../customization/skills.md) search paths. Skills, as cross-tool shared capability extensions, are a different type of data from application runtime data.
:::

## Directory structure

```
~/.kimi/
├── config.toml           # Main configuration file
├── kimi.json             # Metadata
├── mcp.json              # MCP server configuration
├── credentials/          # OAuth credentials
│   └── <provider>.json
├── mcp-oauth/            # MCP OAuth tokens
├── sessions/             # Session data
│   └── <work-dir-hash>/
│       └── <session-id>/
│           ├── context.jsonl
│           ├── wire.jsonl
│           └── state.json
├── imported_sessions/    # Imported session data (via kimi vis)
│   └── <session-id>/
│       ├── context.jsonl
│       ├── wire.jsonl
│       └── state.json
│   └── <slug>.md
├── user-history/         # Input history
│   └── <work-dir-hash>.jsonl
└── logs/                 # Logs
    └── kimi.log
```

## Configuration and metadata

### `config.toml`

Main configuration file, stores providers, models, services, and runtime parameters. See [Config Files](./config-files.md) for details.

You can specify a configuration file at a different location with the `--config-file` flag.

### `kimi.json`

Metadata file, stores Kimi Code CLI's runtime state, including:

- `work_dirs`: List of working directories and their last used session IDs
- `thinking`: Whether thinking mode was enabled in the last session

This file is automatically managed by Kimi Code CLI and typically doesn't need manual editing.

### `mcp.json`

MCP server configuration file, stores MCP servers added via the `kimi mcp add` command. See [MCP](../customization/mcp.md) for details.

Example structure:

```json
{
  "mcpServers": {
    "context7": {
      "url": "https://mcp.context7.com/mcp",
      "transport": "http",
      "headers": {
        "CONTEXT7_API_KEY": "ctx7sk-xxx"
      }
    }
  }
}
```

## Credentials

OAuth credentials are stored in the `~/.kimi/credentials/` directory. After logging in to your Kimi account via `/login`, OAuth tokens are saved in this directory.

OAuth tokens for MCP servers are stored separately in `~/.kimi/mcp-oauth/`. After authorizing an MCP server added with `--auth oauth` via `kimi mcp auth <name>`, later sessions reuse the tokens from this directory. Use `kimi mcp reset-auth <name>` to clear the MCP OAuth token for one server.

Files in the `credentials/` directory have permissions set to read/write for the current user only (600) to protect sensitive information.

## Session data

Session data is grouped by working directory and stored under `~/.kimi/sessions/`. Each working directory corresponds to a subdirectory named with the path's MD5 hash, and each session corresponds to a subdirectory named with the session ID.

### `context.jsonl`

Context history file, stores the session's full context in JSON Lines (JSONL) format. The first line is a system prompt record (`_system_prompt`), followed by messages (user input, model response, tool calls, etc.) and internal records (checkpoints, token usage, etc.).

The system prompt is generated and frozen at session creation time, and reused on session restore instead of being regenerated.

Kimi Code CLI uses this file to restore session context when using `--continue` or `--session`.

### `wire.jsonl`

Wire message log file, stores Wire events during the session in JSON Lines (JSONL) format. Used for session replay and extracting session titles.

### `state.json`

Session state file, stores the session's runtime state, including:

- `title`: User-set session title
- `approval`: Approval decision state (YOLO and AFK mode on/off, auto-approved operation types)
- `subagent_instances`: Subagent instance state and metadata
- `additional_dirs`: Additional workspace directories added via `--add-dir` or `/add-dir`

When resuming a session, Kimi Code CLI reads this file to restore the session state. This file uses atomic writes to prevent data corruption on crash.

### `subagents/<agent_id>/`

Each subagent instance created via the `Agent` tool has its own storage directory under the session directory, containing:

- `context.jsonl`: Subagent conversation history
- `wire.jsonl`: Subagent Wire event log
- `meta.json`: Instance metadata (status, type, creation time, etc.)
- `prompt.txt`: Last executed prompt
- `output`: Execution output

When resuming a session, subagent instance context and state are automatically restored, allowing continuation via the `resume` parameter.

## Input history

User input history is stored in the `~/.kimi/user-history/` directory. Each working directory corresponds to a `.jsonl` file named with the path's MD5 hash.

Input history is used for history browsing (up/down arrow keys) and search (Ctrl-R) in shell mode.

## Logs

Runtime logs are stored in `~/.kimi/logs/kimi.log`. Default log level is INFO, use the `--debug` flag to enable TRACE level.

Log files are used for troubleshooting. When reporting bugs, please include relevant log content.

## Cleaning data

Deleting the share directory (default `~/.kimi/`, or the path specified by `KIMI_SHARE_DIR`) completely clears all Kimi Code CLI data, including configuration, sessions, and history.

To clean only specific data:

| Need | Action |
| --- | --- |
| Reset configuration | Delete `~/.kimi/config.toml` |
| Clear all sessions | Delete `~/.kimi/sessions/` directory |
| Clear sessions for specific working directory | Use `/sessions` in shell mode to view and delete |
| Clear input history | Delete `~/.kimi/user-history/` directory |
| Clear logs | Delete `~/.kimi/logs/` directory |
| Clear MCP configuration | Delete `~/.kimi/mcp.json` or use `kimi mcp remove` |
| Clear login credentials | Delete `~/.kimi/credentials/` directory or use `/logout` |
