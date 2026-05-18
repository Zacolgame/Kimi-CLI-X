"""grep tool - print lines that match patterns."""

from kimi_agent_sdk import CallableTool2, ToolError, ToolReturnValue
from .params import Params



class Grep(CallableTool2[Params]):
    name: str = "Grep"
    description: str = "Print lines that match patterns."
    params: type[Params] = Params

    async def __call__(self, params: Params) -> ToolReturnValue:
        return ToolError(
            message="grep command is not available. use the Grep tool instead.",
            output="",
            brief="use Grep tool",
        )

