# HTTP Server (FastAPI + SSE)

Kimix provides an **OpenCode-compatible** FastAPI + SSE HTTP server. REST API manages sessions; Server-Sent Events stream reasoning and tool-call status in real time.

> Full SSE protocol details: [`docs/server/opencode_style_sse.md`](../server/opencode_style_sse.md)

---

## Quick Start

### 1. Install Dependency

```bash
pip install uvicorn
```

### 2. Start Server

```bash
uv run kimix serve --host 127.0.0.1 --port 4096
```

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `4096` | Bind port |

Startup output:
```
kimix server listening on http://127.0.0.1:4096
API docs (Swagger UI): http://127.0.0.1:4096/docs
OpenAPI schema: http://127.0.0.1:4096/openapi.json
Press Ctrl+C to stop
```

---

## API Endpoints

All endpoints follow the OpenCode standard.

### Health Check

```
GET /global/health
→ 200 {"healthy": true, "version": "0.1.0"}
```

### Session Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/session` | Create session (Body: `{"title": "..."}`) |
| `GET` | `/session` | List active sessions |
| `GET` | `/session/{sessionID}` | Get session metadata |
| `DELETE` | `/session/{sessionID}` | Delete session |
| `GET` | `/session/status` | Get all session statuses (`idle`/`busy`/`error`) |

**Create session:**
```bash
curl -X POST http://127.0.0.1:4096/session \
  -H "Content-Type: application/json" \
  -d '{"title": "My Session"}'
```

Response:
```json
{
  "id": "ses_xxxxxxxxxxxx",
  "title": "My Session",
  "createdAt": 1716883200.0,
  "updatedAt": 1716883200.0,
  "parentID": null
}
```

### Send Message

```
POST /session/{sessionID}/prompt_async
Body: {"parts": [{"type": "text", "text": "..."}], "agent": "...", "model": "..."}
→ 204 No Content
```

Fire-and-forget endpoint. Returns 204 immediately; results stream via SSE `/event`.

**Example:**
```bash
curl -X POST http://127.0.0.1:4096/session/ses_xxx/prompt_async \
  -H "Content-Type: application/json" \
  -d '{"parts": [{"type": "text", "text": "Write a Hello World script"}]}'
```

**Built-in slash commands:**

| Command | Description |
|---------|-------------|
| `/clear` | Clear session |
| `/compact` | Compact context |
| `/context` | Get context usage |
| `/export` | Export session messages |

### Query Messages

```
GET /session/{sessionID}/message?limit=N
→ 200 [Message, ...]
```

### Session Control

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/session/{sessionID}/abort` | Abort current prompt |
| `POST` | `/session/{sessionID}/permissions/{permissionID}` | Grant pending permission |
| `GET` | `/session/{sessionID}/clear` | Clear session |
| `GET` | `/session/{sessionID}/context` | Get context info |
| `GET` | `/session/{sessionID}/compact?keep=N` | Compact history (default keep 10) |
| `GET` | `/session/{sessionID}/export?output_path=PATH` | Export messages to file |

---

## SSE Event Stream

### Global Events

```
GET /event
→ 200 text/event-stream
```

`/event` is **global** — broadcasts all session events. Clients must filter by `sessionID`.

### Event Format

Pure `data:` lines, **no SSE `event:` field** (OpenCode compatible):

```
data: {"type":"<event_type>","properties":{...}}

```

### Core Event Types

| Type | Description | Priority |
|------|-------------|----------|
| `server.connected` | SSE connection established | Ignorable |
| `session.status` | Session status (`busy` / `idle`) | `idle` = termination signal |
| `session.diff` | File diff | Ignorable |
| `session.updated` | Session metadata update | Ignorable |
| `message.updated` | Message metadata | Ignorable |
| `message.part.updated` | **Core: message part updates** | Must handle |

### `message.part.updated` Subtypes

| `part.type` | Description |
|-------------|-------------|
| `step-start` | New reasoning step begins |
| `reasoning` | Model reasoning (incremental: `text` = full, `delta` = increment) |
| `tool` | Tool call (`pending` → `running` → `completed`/`error`) |
| `text` | Final LLM text reply (incremental) |
| `step-finish` | Step ends. `reason=tool-calls` → more steps; `reason=stop` → fully done |

### Termination Signals

1. `session.status` with `status.type == "idle"`
2. `step-finish` with `reason != "tool-calls"` (e.g., `reason == "stop"`)

### Full Interaction Sequence

```
POST /session/{id}/prompt_async → 204 (fire-and-forget)
GET /event                       → SSE stream established

← server.connected               [connection confirm]
← session.status (busy)          [processing started]
← message.part.updated (step-start)
← message.part.updated (reasoning) ×N
← message.part.updated (tool/pending)
← message.part.updated (tool/running)
← message.part.updated (tool/completed)
← message.part.updated (step-finish, reason=tool-calls) [more steps]
← ... (next step) ...
← message.part.updated (text) ×N
← message.part.updated (step-finish, reason=stop)  [termination]
```

---

## SSE CLI Debugger (`ssecli`)

Built-in SSE client for testing `kimix serve`:

```bash
uv run kimix ssecli --host 127.0.0.1 --port 4096 --debug
```

| Flag | Description |
|------|-------------|
| `--host` | Server address (default `127.0.0.1`) |
| `--port` | Server port (default `4096`) |
| `--debug` | Print raw SSE events and save to `sse_log_<timestamp>.txt` |

### Built-in Commands

| Command | Description |
|---------|-------------|
| `/new` | Create new session |
| `/abort` | Abort current prompt |
| `/status` | Show all session statuses |
| `/sessions` | List sessions |
| `/messages` | Show current session messages |
| `/clear` | Clear session |
| `/compact` | Compact context |
| `/export` | Export session |
| `/help` | Help |

Press `Ctrl+C` or EOF (`Ctrl+D` / `Ctrl+Z`) to exit.

---

## Dummy Mode (Testing)

`src/kimix/server/dummy_app.py` provides a stub server. All endpoints return stub responses without a real LLM backend. Useful for frontend dev and integration testing.

Differences from real server:
- Uses `DummySessionManager` (no actual logic)
- SSE `/event` only pushes `server.connected` + heartbeat
- `prompt_async` only logs request params, no inference

```python
import uvicorn
from kimix.server.dummy_app import create_app

uvicorn.run(create_app(), host="127.0.0.1", port=4096)
```

---

## Client Implementation

### Must-Handle Events

| Priority | Event | Action |
|----------|-------|--------|
| P0 | `text` (delta) | Stream output to user |
| P0 | `step-finish` (reason=stop) | Stop SSE listener |
| P1 | `tool` (running/completed/error) | Show tool status |
| P1 | `reasoning` (delta) | Optionally show thinking |

### Session Filtering

`/event` is global. Filter by `sessionID` via:
```
properties.sessionID
properties.part.sessionID
properties.info.sessionID
```

### Reconnect

- Auto-reconnect on disconnect (max ~5 attempts, increasing delay)
- Reconnect triggers `server.connected` again

---

## Architecture Overview

```
┌──────────┐  REST API   ┌─────────────┐     ┌──────────────────┐
│  Client  │ ◄─────────► │  FastAPI    │────►│ SessionManager   │
│ (curl /  │  POST/GET   │  (app.py)   │     │ (create/delete/  │
│  Web UI) │             │             │     │  prompt_async)   │
└──────────┘             └──────┬──────┘     └──────────────────┘
                                │
                           SSE  │  /event
                                │
                         ┌──────▼───────┐
                         │   Bus (queue)│
                         │  Broadcasts  │
                         │  to all SSE  │
                         │  clients     │
                         └──────────────┘
```

- **`app.py`**: FastAPI factory, routes, SSE stream
- **`session_manager.py`**: Session lifecycle (create, delete, prompt execution)
- **`bus.py`**: Event bus, broadcasts SSE events to all clients
- **`dummy_app.py`** / **`dummy_session_manager.py`**: Stub implementations for testing
- **`serve.py`**: `kimix serve` CLI entry, launches uvicorn
