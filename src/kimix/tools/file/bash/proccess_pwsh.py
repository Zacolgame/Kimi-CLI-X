"""Transform PowerShell 7.x syntax to PowerShell 5.1 compatible syntax."""

from __future__ import annotations

import re


def _find_string_regions(code: str) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    i = 0
    n = len(code)
    while i < n:
        c = code[i]
        if c == "<" and i + 1 < n and code[i + 1] == "#":
            start = i
            depth = 1
            i += 2
            while i < n and depth > 0:
                if code[i] == "<" and i + 1 < n and code[i + 1] == "#":
                    depth += 1; i += 2
                elif code[i] == "#" and i + 1 < n and code[i + 1] == ">":
                    depth -= 1; i += 2
                else:
                    i += 1
            regions.append((start, i))
        elif c == "#":
            start = i
            while i < n and code[i] != "\n":
                i += 1
            regions.append((start, i))
        elif c == "@" and i + 1 < n and code[i + 1] in ("'", '"'):
            # Here-string start must be at end of line (only whitespace after)
            j = i + 2
            while j < n and code[j] in " \t\r":
                j += 1
            if j < n and code[j] != "\n":
                # Not a real here-string start; skip the @ and let normal parsing continue
                i += 1
                continue
            start = i
            quote_char = code[i + 1]
            i += 2
            while i < n:
                if code[i] == quote_char and i + 1 < n and code[i + 1] == "@":
                    line_start = code.rfind("\n", 0, i)
                    line_start = 0 if line_start == -1 else line_start + 1
                    if code[line_start:i].strip() == "":
                        i += 2; break
                i += 1
            regions.append((start, i))
        elif c == "'":
            start = i
            i += 1
            while i < n:
                if code[i] == "'":
                    if i + 1 < n and code[i + 1] == "'":
                        i += 2
                    else:
                        i += 1; break
                else:
                    i += 1
            regions.append((start, i))
        elif c == '"':
            start = i
            i += 1
            while i < n:
                ch = code[i]
                if ch == "`" and i + 1 < n:
                    i += 2
                elif ch == '"':
                    i += 1; break
                elif ch == "$" and i + 1 < n and code[i + 1] == "(":
                    i = _skip_subexpression(code, i)
                else:
                    i += 1
            regions.append((start, i))
        else:
            i += 1
    return regions

def _skip_subexpression(code: str, start: int) -> int:
    assert code[start] == "$"
    i = start + 2
    depth = 1
    n = len(code)
    while i < n and depth > 0:
        c = code[i]
        if c == "(":
            depth += 1; i += 1
        elif c == ")":
            depth -= 1; i += 1
        elif c == "'":
            i += 1
            while i < n:
                if code[i] == "'":
                    if i + 1 < n and code[i + 1] == "'":
                        i += 2
                    else:
                        i += 1; break
                else:
                    i += 1
        elif c == '"':
            i += 1
            while i < n:
                if code[i] == "`" and i + 1 < n:
                    i += 2
                elif code[i] == '"':
                    i += 1; break
                else:
                    i += 1
        elif c == "$" and i + 1 < n and code[i + 1] == "(":
            i = _skip_subexpression(code, i)
        else:
            i += 1
    return i

def _find_line_regions(line: str) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "<" and i + 1 < n and line[i + 1] == "#":
            start = i
            depth = 1
            i += 2
            while i < n and depth:
                if line[i] == "<" and i + 1 < n and line[i + 1] == "#":
                    depth += 1; i += 2
                elif line[i] == "#" and i + 1 < n and line[i + 1] == ">":
                    depth -= 1; i += 2
                else:
                    i += 1
            regions.append((start, i))
        elif c == "#":
            regions.append((i, n)); break
        elif c in "'\"":
            start = i
            quote = c
            i += 1
            while i < n:
                if line[i] == quote:
                    if quote == "'" and i + 1 < n and line[i + 1] == "'":
                        i += 2; continue
                    i += 1; break
                if quote == '"' and line[i] == "`" and i + 1 < n:
                    i += 2
                elif quote == '"' and line[i] == "$" and i + 1 < n and line[i + 1] == "(":
                    i = _skip_subexpression(line, i)
                else:
                    i += 1
            regions.append((start, i))
        else:
            i += 1
    return regions

def _outside_line_regions(regions: list[tuple[int, int]], pos: int) -> bool:
    return not any(start <= pos < end for start, end in regions)

def _compute_depth_array(line: str) -> list[int]:
    depths: list[int] = []
    depth = 0
    for ch in line:
        depths.append(depth)
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    depths.append(depth)
    return depths

def _depth_at(line: str, pos: int, depth_array: list[int] | None = None) -> int:
    if depth_array is not None:
        return depth_array[pos] if pos < len(depth_array) else 0
    depth = 0
    for i in range(pos):
        c = line[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
    return depth

def _join_continuation_lines(code: str) -> str:
    regions = _find_string_regions(code)
    result: list[str] = []
    i = 0
    n = len(code)
    while i < n:
        if code[i] == "`":
            in_region = any(start <= i < end for start, end in regions)
            if not in_region:
                j = i + 1
                while j < n and code[j] in " \t\r":
                    j += 1
                if j < n and code[j] == "\n":
                    j += 1
                    while j < n and code[j] in " \t\r":
                        j += 1
                    result.append(" ")
                    i = j
                    continue
        result.append(code[i])
        i += 1
    return "".join(result)

_ASSIGN_RE = re.compile(r"(.*?)(\$\w+(?:\.\w+)*)\s*=\s*$")
def _match_assignment(before: str) -> tuple[str, str] | None:
    m = _ASSIGN_RE.match(before)
    if m:
        return m.group(1), m.group(2)
    return None

_NCA_RE = re.compile(r"(\$\w+(?:\.\w+)*)\s*\?\?=\s*(.+)")
def _transform_nca_line(line: str) -> str:
    regions = _find_line_regions(line)
    new_line = line
    for m in reversed(list(_NCA_RE.finditer(line))):
        if _outside_line_regions(regions, m.start()):
            var = m.group(1)
            value = m.group(2).rstrip()
            replacement = f"if ($null -eq {var}) {{ {var} = {value} }}"
            new_line = new_line[: m.start()] + replacement + new_line[m.end() :]
    return new_line

def _nc_rewrite_line(line: str, op_pos: int) -> str | None:
    left_end = op_pos
    while left_end > 0 and line[left_end - 1] == " ":
        left_end -= 1
    left_start = _find_expr_start(line, left_end)
    right_start = op_pos + 2
    while right_start < len(line) and line[right_start] == " ":
        right_start += 1
    right_end = _find_expr_end(line, right_start)
    left_expr = line[left_start:left_end].strip()
    right_expr = line[right_start:right_end].strip()
    if not left_expr or not right_expr:
        return None
    before = line[:left_start].rstrip()
    assign = _match_assignment(before)
    if assign:
        prefix, var_name = assign
        replacement = f"{prefix}{var_name} = if ($null -ne {left_expr}) {{ {left_expr} }} else {{ {right_expr} }}"
    else:
        replacement = f"{line[:left_start]}if ($null -ne {left_expr}) {{ {left_expr} }} else {{ {right_expr} }}"
    return replacement + line[right_end:]

def _transform_nc_line(line: str) -> str:
    while True:
        regions = _find_line_regions(line)
        rewritten = False
        pos = 0
        while pos < len(line):
            idx = line.find("??", pos)
            if idx == -1:
                break
            if _outside_line_regions(regions, idx):
                replacement = _nc_rewrite_line(line, idx)
                if replacement is not None:
                    line = replacement
                    rewritten = True
                    break
            pos = idx + 2
        if not rewritten:
            break
    return line

def _transform_ternary_line(line: str) -> str:
    regions = _find_line_regions(line)
    depth_arr = _compute_depth_array(line)
    pos = 0
    while pos < len(line):
        if (
            line[pos] == "?"
            and _outside_line_regions(regions, pos)
            and _depth_at(line, pos, depth_arr) == 0
            and not (pos > 0 and line[pos - 1] == "$")
        ):
            colon_pos = _find_matching_colon(line, pos + 1, depth_arr)
            if colon_pos != -1:
                cond_end = pos
                while cond_end > 0 and line[cond_end - 1] == " ":
                    cond_end -= 1
                cond_start = _find_expr_start(line, cond_end)
                condition = line[cond_start:cond_end].strip()
                true_start = pos + 1
                while true_start < len(line) and line[true_start] == " ":
                    true_start += 1
                true_end = colon_pos
                while true_end > true_start and line[true_end - 1] == " ":
                    true_end -= 1
                true_expr = line[true_start:true_end].strip()
                false_start = colon_pos + 1
                while false_start < len(line) and line[false_start] == " ":
                    false_start += 1
                false_end = _find_expr_end(line, false_start)
                false_expr = line[false_start:false_end].strip()
                before = line[:cond_start].rstrip()
                assign = _match_assignment(before)
                if assign:
                    prefix, var = assign
                    replacement = f"{prefix}{var} = if ({condition}) {{ {true_expr} }} else {{ {false_expr} }}"
                else:
                    replacement = f"{line[:cond_start]}if ({condition}) {{ {true_expr} }} else {{ {false_expr} }}"
                suffix = line[false_end:]
                line = replacement + suffix
                regions = _find_line_regions(line)
                depth_arr = _compute_depth_array(line)
                pos = len(replacement)
                continue
        pos += 1
    return line

def _find_matching_colon(line: str, start: int, depth_arr: list[int] | None = None) -> int:
    regions = _find_line_regions(line)
    for i in range(start, len(line)):
        if line[i] == ":" and _depth_at(line, i, depth_arr) == 0 and _outside_line_regions(regions, i):
            if i > 0 and line[i - 1] == ":":
                continue
            if i + 1 < len(line) and line[i + 1] == ":":
                continue
            return i
    return -1

def _find_expr_start(line: str, end: int) -> int:
    regions = _find_line_regions(line)
    depth = 0
    for i in range(end - 1, -1, -1):
        c = line[i]
        if c in ")]}":
            depth += 1
        elif c in "([{":
            depth -= 1
            if depth < 0:
                return i + 1
        elif c in "=;|&," and _outside_line_regions(regions, i):
            if c == "=":
                if i > 0 and line[i - 1] in "=!<>+-*/.":
                    continue
            return i + 1
    return 0

def _find_expr_end(line: str, start: int) -> int:
    regions = _find_line_regions(line)
    depth = 0
    for i in range(start, len(line)):
        c = line[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return i
        elif depth >= 0 and c in ";|&," and _outside_line_regions(regions, i):
            return i
        elif c == "#" and _outside_line_regions(regions, i):
            if i > 0 and line[i - 1] == "<":
                continue
            return i
    return len(line)

def _transform_chain_line(line: str) -> str:
    while True:
        regions = _find_line_regions(line)
        depth_arr = _compute_depth_array(line)
        best_pos = -1
        best_op = ""
        best_len = 0
        for op in ("&&", "||"):
            pos = 0
            op_len = len(op)
            while True:
                idx = line.find(op, pos)
                if idx == -1:
                    break
                if _depth_at(line, idx, depth_arr) == 0 and _outside_line_regions(regions, idx):
                    if idx > best_pos:
                        best_pos = idx
                        best_op = op
                        best_len = op_len
                pos = idx + op_len
        if best_pos == -1:
            break
        condition = "$?" if best_op == "&&" else "-not $?"
        left = line[:best_pos].strip()
        right = line[best_pos + best_len :].strip()
        line = f"{left}; if ({condition}) {{ {right} }}"
    return line

def _transform_null_conditional_dot_line(line: str) -> str:
    while True:
        regions = _find_line_regions(line)
        depth_arr = _compute_depth_array(line)
        pos = 0
        matched = False
        while True:
            idx = line.find("?.", pos)
            if idx == -1:
                break
            if _depth_at(line, idx, depth_arr) != 0 or not _outside_line_regions(regions, idx):
                pos = idx + 2; continue
            expr_end = idx
            while expr_end > 0 and line[expr_end - 1] == " ":
                expr_end -= 1
            expr_start = _find_expr_start(line, expr_end)
            base = line[expr_start:expr_end].strip()
            if not base:
                pos = idx + 2; continue
            chain: list[tuple[str, str, int]] = []
            pos = idx
            while pos < len(line) and line[pos : pos + 2] == "?.":
                ms = pos + 2
                while ms < len(line) and line[ms] == " ":
                    ms += 1
                me = ms
                while me < len(line) and (line[me].isalnum() or line[me] == "_"):
                    me += 1
                if me == ms:
                    break
                mem = line[ms:me]
                args = ""
                j = me
                while j < len(line) and line[j] == " ":
                    j += 1
                if j < len(line) and line[j] == "(":
                    d = 1
                    k = j + 1
                    while k < len(line) and d > 0:
                        if _outside_line_regions(regions, k):
                            if line[k] == "(":
                                d += 1
                            elif line[k] == ")":
                                d -= 1
                        k += 1
                    args = line[j:k]
                    me = k
                chain.append((mem, args, me))
                pos = me
            if not chain:
                pos = idx + 2; continue
            paths = [base]
            for mem, args, _ in chain:
                paths.append(f"{paths[-1]}.{mem}{args}")
            inner = paths[-1]
            for p in reversed(paths[:-1]):
                inner = f"if ($null -ne {p}) {{ {inner} }}"
            before = line[:expr_start]
            after = line[chain[-1][2] :]
            assign = _match_assignment(before)
            repl = (f"{assign[0]}{assign[1]} = " if assign else before) + inner
            line = repl + after
            depth_arr = _compute_depth_array(line)
            matched = True
            break
        if not matched:
            break
    return line

def _transform_null_conditional_bracket_line(line: str) -> str:
    while True:
        regions = _find_line_regions(line)
        depth_arr = _compute_depth_array(line)
        pos = 0
        matched = False
        while True:
            idx = line.find("?[", pos)
            if idx == -1:
                break
            if _depth_at(line, idx, depth_arr) != 0 or not _outside_line_regions(regions, idx):
                pos = idx + 2; continue
            expr_end = idx
            while expr_end > 0 and line[expr_end - 1] == " ":
                expr_end -= 1
            expr_start = _find_expr_start(line, expr_end)
            expr = line[expr_start:expr_end].strip()
            if not expr:
                pos = idx + 2; continue
            bracket_depth = 1
            bracket_end = idx + 2
            while bracket_end < len(line) and bracket_depth > 0:
                c = line[bracket_end]
                if _outside_line_regions(regions, bracket_end):
                    if c == "[":
                        bracket_depth += 1
                    elif c == "]":
                        bracket_depth -= 1
                bracket_end += 1
            index_expr = line[idx + 2 : bracket_end - 1]
            before = line[:expr_start]
            after = line[bracket_end:]
            assign = _match_assignment(before)
            if assign:
                prefix, target_var = assign
                repl = f"{prefix}{target_var} = if ($null -ne {expr}) {{ {expr}[{index_expr}] }}"
            else:
                repl = f"{before}if ($null -ne {expr}) {{ {expr}[{index_expr}] }}"
            line = repl + after
            depth_arr = _compute_depth_array(line)
            matched = True
            break
        if not matched:
            break
    return line

def _has_chain_operators(code: str) -> bool:
    """Check if code contains && or || outside string/comment regions."""
    regions = _find_string_regions(code)
    for op in ("&&", "||"):
        pos = 0
        while True:
            idx = code.find(op, pos)
            if idx == -1:
                break
            if not any(start <= idx < end for start, end in regions):
                return True
            pos = idx + 2
    return False



def pwsh_transform(code: str, *, warn_chain: bool = False) -> tuple[str, str]:
    """Transform PowerShell 7.x syntax to PowerShell 5.1 compatible syntax."""
    has_chain = _has_chain_operators(code) if warn_chain else False
    code = _join_continuation_lines(code)
    lines = code.split("\n")
    regions = _find_string_regions(code)
    offs = [0]
    for line in lines[:-1]:
        offs.append(offs[-1] + len(line) + 1)
    multi = {i for s, e in regions if "\n" in code[s:e] for i, o in enumerate(offs) if o < e and o + len(lines[i]) > s}
    result: list[str] = []
    for i, line in enumerate(lines):
        if i in multi:
            result.append(line)
            continue
        line = _transform_nca_line(line)
        line = _transform_nc_line(line)
        line = _transform_ternary_line(line)
        line = _transform_chain_line(line)
        line = _transform_null_conditional_dot_line(line)
        line = _transform_null_conditional_bracket_line(line)
        result.append(line)
    result_code = "\n".join(result)
    warning = ''
    if has_chain:
        warning = (
            "WARNING: PowerShell `&&` and `||` check the `$?` automatic variable "
            "(success of last native command), NOT the raw exit code like bash. "
        )
    return result_code, warning

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read()
    print(pwsh_transform(text))
