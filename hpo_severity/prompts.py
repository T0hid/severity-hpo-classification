"""Loads the prompt files in prompts/ and fills their placeholders.

Every prompt sent to a language model is stored as a file under prompts/. Placeholders are written
as ${name}. Text files end with one newline that is not part of the prompt. In a JSON prompt, a
string that is exactly "${name}" is replaced by the value itself (a list, a dict or a string);
other strings that contain ${name} are filled as text.
"""
import json
import string
from pathlib import Path
from typing import Any, Dict

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def text(name: str) -> str:
    """The prompt file as sent, without the file's final newline."""
    content = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return content[:-1] if content.endswith("\n") else content


def render_text(name: str, **values: Any) -> str:
    return string.Template(text(name)).substitute({k: str(v) for k, v in values.items()})


def load_json(name: str) -> Any:
    return json.loads((PROMPT_DIR / name).read_text(encoding="utf-8"))


def fill(obj: Any, values: Dict[str, Any]) -> Any:
    """Replaces the ${name} placeholders in a loaded JSON prompt."""
    if isinstance(obj, dict):
        return {k: fill(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill(v, values) for v in obj]
    if isinstance(obj, str) and "${" in obj:
        if obj.startswith("${") and obj.endswith("}") and obj[2:-1] in values:
            return values[obj[2:-1]]
        return string.Template(obj).substitute({k: str(v) for k, v in values.items()})
    return obj
