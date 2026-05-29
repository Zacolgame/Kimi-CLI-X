# Writing Custom Skills

Skills are modular knowledge packs that extend the agent's capabilities. A skill bundles domain-specific workflows, scripts, docs, and templates.

---

## Why Skills

- **Modular**: write once, reuse everywhere
- **On-demand loading**: only loaded when relevant, saving tokens
- **Carries resources**: binds scripts, docs, templates
- **Distributable**: package as `.skill` files

---

## Design Principles

### 1. Brevity
Context is limited. Include only what the agent doesn't already know. Use concise examples over long explanations. Keep `SKILL.md` under 500 lines.

### 2. Match Freedom to Fragility

| Freedom | Scenario | Form |
|---------|----------|------|
| High | Multiple valid approaches | Descriptions & principles |
| Medium | Recommended pattern with flexibility | Pseudocode / parameterized scripts |
| Low | High-consistency, error-prone operations | Exact executable scripts |

### 3. Progressive Disclosure
Three-tier loading:
1. **Metadata** (`name` + `description`) — always in context, for triggering
2. **SKILL.md body** — loaded only when triggered
3. **Resources** (scripts, docs) — loaded only on demand

Core workflows go in `SKILL.md`; details go in `references/` with clear read triggers.

---

## Directory Structure

```
skill-name/
├── SKILL.md       # Required: core skill file
├── scripts/       # Optional: executable scripts
├── references/    # Optional: on-demand reference docs
└── assets/        # Optional: templates, images, fonts
```

| Directory | Purpose |
|-----------|---------|
| `SKILL.md` | YAML frontmatter (`name`, `description`). Description must state **what it does** and **when to use it**. Body: imperative Markdown instructions. |
| `scripts/` | Deterministic, reusable code. Code is more token-efficient than prose. |
| `references/` | Large docs (>10k words), JSON Schema, API docs. Linked from `SKILL.md` with read triggers. |
| `assets/` | Templates, sample images, fonts for output generation. |

> No `README.md`, `CHANGELOG.md`, or `.git` — they waste context.

---

## Creation Workflow

```
Understand need → Plan resources → Init directory → Write content → Test scripts → Package → Iterate
```

1. **Understand need**: When does the agent need this? What do users typically ask? What's the output format and quality standard?
2. **Plan resources**: What belongs in prose, scripts, or external docs?
3. **Init directory**: Folder name must exactly match skill `name`.
4. **Write content**: Write scripts and references first, then distill core instructions into `SKILL.md`.
5. **Test scripts**: All scripts must be run before packaging. Spot-check representative samples if many.
6. **Package**: Zip the folder as `.skill`.
7. **Iterate**: Refine based on usage feedback.

---

## Writing SKILL.md

### Frontmatter

```yaml
---
name: skill-name
description: Brief function. Use when: (1) condition A, (2) condition B...
---
```

**Naming rules:**
- Lowercase letters, numbers, hyphens only
- Max 64 chars
- Verb-first recommended: `gh-address-comments`, `linear-address-issue`

### Body

Use **imperative sentences**. Recommended content:
- Multi-step workflows
- Decision branches with clear conditions
- Output format and quality standards
- Resource links (`references/`, `scripts/`)

Example:
```markdown
## Workflow

1. Extract repo name and PR number from user input
2. Run `scripts/fetch_pr.sh <repo> <pr_number>`
3. If unresolved comments exist, group by file
4. Output: Markdown table with file path, line, comment, suggested fix
```

---

## Organizing Resources

### `scripts/`: Let Code Speak

When a step is deterministic, a script is more efficient and reliable than prose.

```bash
# scripts/generate-report.sh
python3 -m report_generator --input "$1" --output report.md
```

Reference in `SKILL.md`:
```markdown
Run `scripts/generate-report.sh <input_file>` to generate the report.
```

### `references/`: Large Documents

Put API docs, full schemas, and domain knowledge here. Link with read triggers:
```markdown
For full API parameters, read `references/api-docs.md`.
```

### `assets/`: Output Templates

Place PPT templates, HTML templates, fonts here. Describe usage in `SKILL.md`.

---

## Testing & Packaging

### Checklist

- [ ] `name` and `description` are accurate
- [ ] Folder name matches `name`
- [ ] All script paths in `SKILL.md` are correct and tested
- [ ] No extraneous files (README, CHANGELOG, .git, etc.)
- [ ] `references/` docs have clear read triggers

### Package

```bash
cd <skills-root>
zip -r my-skill.skill my-skill
```

---

## Installation

| Level | Path |
|-------|------|
| User | `~/.config/agents/skills/`, `~/.kimi/skills/`, `~/.claude/skills/` |
| Project | `.agents/skills/` |
| Custom | `--skills-dir` flag |

Place the skill folder or `.skill` package in any of these paths.

---

## Best Practices

### Do
1. **Design the trigger first**: A clear `description` reduces false triggers and misses.
2. **One skill, one job**: Small, focused skills are easier to maintain and combine.
3. **Use relative paths**: All internal links relative to `SKILL.md`.
4. **Keep references flat**: Avoid deep nesting in `references/`.

### Don't
- **Vague description**: `description: helps write code` — triggers everywhere, wastes context.
- **Overly long body**: Pasting entire spec books into `SKILL.md` explodes context.
- **Missing decision points**: Only stating the final goal without intermediate steps.
- **Untested scripts**: Packaged scripts fail in target environments.

### Full Example

**Directory:**
```
gh-address-comments/
├── SKILL.md
└── scripts/
    └── fetch_comments.py
```

**SKILL.md:**
```markdown
---
name: gh-address-comments
description: Organize pending PR comments. Use when: (1) user mentions replying to or processing PR comments, (2) need to summarize code review feedback.
---

# gh-address-comments

## Workflow

1. Extract repo full name (e.g., `owner/repo`) and PR number from user input
2. Run `scripts/fetch_comments.py <repo> <pr_number>`
3. Filter open, non-bot comments
4. Group by reviewed file path
5. For each comment: file path, line number, original text, suggested code change
6. Output as Markdown table

## Output Standard

- Table must have `File`, `Line`, `Comment`, `Suggestion` columns
- If no clear suggestion, last column = "needs discussion"
- Output table only, no extra summary
```

---

## Summary

A good skill needs **precise triggers**, **concise instructions**, and **reliable resources**:

1. `description` must state "what it does" and "when to use it"
2. Keep `SKILL.md` under 500 lines; details in `references/`
3. Test scripts; clean extraneous files before packaging
