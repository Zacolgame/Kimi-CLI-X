# Long-Running Tasks

Kimix provides two main approaches for complex or time-consuming tasks:

1. **`/plan`**: Sequential step-by-step execution with resume support
2. **`/swarm`**: DAG-based parallel scheduling with multiple sub-agents

---

## `/plan`

Two-phase mode: plan generation + step execution. Best for sequential, resumable tasks.

### Basic Usage

```
/plan
>>>> Make a task-list: input multiple-lines, end with /end, cancel with /cancel
Add complete error handling to this project:
1. Add parameter validation to all functions
2. Unify exception types and error codes
3. Add logging
/end
```

### Execution Flow

**Phase 1: Plan Generation**
1. Creates a dedicated session with `TodoMaker` system prompt
2. LLM breaks task into steps, writes to `~/.kimi/plan/plan_<uuid>.md` (up to 4 retries)
3. Plan metadata cached in `~/.kimi/plan/.cache.json` with SHA256 validation

**Phase 2: Step Execution**
1. Clears context via `clear_default_context()`
2. Resumes from checkpoint if cached and incomplete
3. Executes steps sequentially; calls `SetTodoList` after each step
4. Updates `finished_step_count` in cache
5. Deletes cache on completion

### Load from File

```
/plan:path/to/task_description.md
```

> No cache check or confirmation prompt in file mode.

### Cache & Resume

Interactive `/plan` checks cache first:
- `y`: resume from last completed step
- `n`: generate new plan, overwrite cache

Resume requires: plan file exists with matching SHA256, and `0 < finished < total`.

### Execution Confirmation

After plan generation, optional confirmation prompt:
- `y`: print remaining steps and confirm once before execution
- `n`: auto-execute all steps

---

## Long Prompts & Error Handling

### Auto-Truncate

Prompts exceeding **65536 chars** are exported to a temp file and replaced with `read and execute: <temp_file>`.

### Auto-Retry

`prompt_async` retries up to **5 times** with exponential backoff:

| Status | Behavior |
|--------|----------|
| `429` | Wait `min(2^attempt, 60)`s, retry |
| `400`, `500`, `502`, `503` | Exponential backoff, retry |
| Other | Wait 1s, retry; throw on last attempt |

---

## `SetTodoList`

Progress tracking tool, auto-invoked during execution.

- **Read mode** (`todos` = `null`): returns current todo list
- **Write mode** (list provided): updates and persists todos

```python
class Todo:
    title: str
    status: str  # "pending" | "in_progress" | "done"
```

**Persistence:**
- Root Agent: `state.todos`
- Sub Agent: `state.json` in agent directory

---

## Agent Swarm (`/swarm`)

Coordinator breaks complex tasks into a DAG and schedules sub-agents in parallel.

### Usage

```
/swarm
>>>> Start input multiple-lines for swarm task, end with /end, cancel with /cancel
Generate unit tests covering core functions in src/utils.py and src/core.py
/end
```

### Execution Flow

1. Collect multi-line text into `task_prompt`
2. `create_swarm_session(task_prompt)` — coordinator plans DAG nodes via `agent_swarm.json`
3. `Executor().execute(dag)` schedules nodes by dependencies
4. Print results; errors included in final output

### Status Messages

| Scenario | Output |
|----------|--------|
| DAG created | `Swarm session created, DAG has N node(s).` |
| Completed | `Swarm execution completed. Results: ...` |
| Empty input | `Empty task prompt, skipping swarm command.` |
| Create fail | `Failed to create swarm session: ...` |
| Execution fail | `Swarm execution failed: ...` |
| Cancelled | `Swarm command cancelled.` |

### Examples

**Batch code review:**
```
/swarm
Review all Python files in src/ for style, performance, and security issues.
/end
```

**Multi-module refactor:**
```
/swarm
Replace print-based logging with standard logging in utils.py, core.py, and cli.py.
/end
```

### Notes

1. Swarm tasks involve multiple LLM calls — be patient
2. File changes are merged via VFS; conflicts resolved by coordinator
3. Single node failure does not block independent nodes

---

## `/plan` vs `/swarm`

| Feature | `/plan` | `/swarm` |
|---------|---------|----------|
| Execution | Serial | Parallel (DAG) |
| Task split | Linear steps | Directed acyclic graph |
| Resume | Yes | No |
| Progress | Step visualization | DAG node status |
| Use case | Ordered, dependent tasks | Parallel, independent tasks |
| Overhead | Lower (single session) | Higher (multi-agent) |
| Examples | Feature implementation | Batch review, multi-module analysis |

**Choose `/plan`** when steps have clear order and must complete sequentially, or resumability is needed.
**Choose `/swarm`** when subtasks are independent and can benefit from parallel execution.
