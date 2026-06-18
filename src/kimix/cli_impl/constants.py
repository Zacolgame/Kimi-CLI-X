from typing import Any
from pathlib import Path
import os

curr_dir = Path(os.curdir)
if not curr_dir.is_absolute():
    curr_dir = curr_dir.absolute()

LSP_JSON = None
frame = 0
COMMENT = """Always write ' Start writen by AGENT ' comment before code, Always write ' End writen by AGENT ' comment after code"""

# Start writen by AGENT
OPTIONS_HELP_STR = '''Command line options:
  -c, --clean         - Delete cache file after quit
  --no_think          - Disable thinking mode
  --no_yolo           - Disable YOLO mode
  --no_color          - Disable colorful print
  --manually-cot      - Enable manually CoT mode
  --ralph             - Enable Ralph mode or set iterations
  -s, --skill-dir     - Specify custom skill directory
  --config            - Specify a JSON config file (see config_example.json for example)
'''
HELP_STR = OPTIONS_HELP_STR
# End writen by AGENT

CLEAN_MODE: bool | None = None
globals_dict: dict[str, Any] = {}
locals_dict: dict[str, Any] = {}
