"""Unified syntax lint tool that dispatches based on file extension."""

import asyncio
import subprocess
import sys
from pathlib import Path

from kimi_agent_sdk import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field

from kimix.tools.common import _maybe_export_output_async


class Params(BaseModel):
    """Parameters for unified syntax lint."""

    file_path: str = Field(description="Path to the file to validate.")
    project_root: str = Field(
        default=".",
        description="Root directory of the project (default: current directory).",
    )
    clangd_path: str = Field(
        default="clangd",
        description="Path to the clangd executable for C++ files (default: 'clangd').",
    )
    verbose: bool = Field(
        default=False,
        description="Include verbose output.",
    )


_SCRIPT_DIR = Path(__file__).resolve().parents[4] / "scripts"

_EXTENSION_MAP = {
    ".py": ("py_lint.py", []),
    ".cpp": ("cpp_lint.py", ["--clangd-path", "{clangd_path}"]),
    ".cc": ("cpp_lint.py", ["--clangd-path", "{clangd_path}"]),
    ".cxx": ("cpp_lint.py", ["--clangd-path", "{clangd_path}"]),
    ".h": ("cpp_lint.py", ["--clangd-path", "{clangd_path}"]),
    ".hpp": ("cpp_lint.py", ["--clangd-path", "{clangd_path}"]),
    ".hxx": ("cpp_lint.py", ["--clangd-path", "{clangd_path}"]),
    ".hh": ("cpp_lint.py", ["--clangd-path", "{clangd_path}"]),
    ".c++": ("cpp_lint.py", ["--clangd-path", "{clangd_path}"]),
}


class SyntaxLint(CallableTool2):  # type: ignore[type-arg]
    """Check file syntax using the appropriate linter based on file extension."""

    name: str = "SyntaxLint"
    description: str = "Validate Python or C++ file syntax."
    params: type[Params] = Params

    async def __call__(self, params: Params) -> ToolReturnValue:
        file_path = Path(params.file_path)

        if not await asyncio.to_thread(file_path.exists):
            return ToolError(
                output="",
                message=f"File not found: {file_path}",
                brief="File not found",
            )

        ext = file_path.suffix.lower()
        mapping = _EXTENSION_MAP.get(ext)

        if mapping is None:
            supported = ", ".join(sorted(set(_EXTENSION_MAP.keys())))
            msg = f"Unsupported file extension: {ext}. Supported: {supported}"
            return ToolError(
                output=msg,
                message=msg,
                brief="Unsupported file type",
            )

        script_name, extra_template = mapping
        script_path = _SCRIPT_DIR / script_name

        cmd = [
            sys.executable,
            str(script_path),
            str(file_path),
            "--project-root",
            params.project_root,
        ]
        if params.verbose:
            cmd.append("--verbose")

        for arg in extra_template:
            cmd.append(arg.format(clangd_path=params.clangd_path))

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await proc.communicate()
        except Exception as e:
            return ToolError(
                output="",
                message=f"Lint tool failed: {e}",
                brief="Lint failed",
            )

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        output = stdout
        if stderr:
            output += f"\n{stderr}"

        output = await _maybe_export_output_async(output)

        if proc.returncode == 0:
            return ToolOk(output=output)
        elif proc.returncode == 1:
            return ToolError(
                output=output,
                message=output.strip().splitlines()[-1] if output.strip() else "Lint found issues",
                brief="Lint found issues",
            )
        else:
            return ToolError(
                output=output,
                message=output.strip().splitlines()[-1] if output.strip() else "Lint tool failed",
                brief="Lint failed",
            )
