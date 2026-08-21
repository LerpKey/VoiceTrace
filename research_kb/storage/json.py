"""Deterministic and atomic JSON persistence for Pydantic models."""

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel


def render_json_model(model: BaseModel) -> str:
    """Render a model as stable UTF-8 JSON."""
    payload = model.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_json_model(model: BaseModel, output_path: Path) -> bool:
    """Atomically write a model, returning False when content is unchanged."""
    target = output_path.expanduser().resolve()
    rendered = render_json_model(model)
    if target.is_file() and target.read_text(encoding="utf-8") == rendered:
        return False
    if target.exists() and not target.is_file():
        raise IsADirectoryError(f"JSON output path is not a file: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as temporary:
            temporary.write(rendered)
        os.replace(temporary_path, target)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return True
