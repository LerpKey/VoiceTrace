"""Atomic, idempotent UTF-8 text persistence."""

import os
import tempfile
from pathlib import Path


def write_text_atomic(content: str, output_path: Path) -> bool:
    """Atomically write UTF-8 text, returning False when unchanged."""
    target = output_path.expanduser().resolve()
    if target.is_file() and target.read_text(encoding="utf-8") == content:
        return False
    if target.exists() and not target.is_file():
        raise IsADirectoryError(f"Text output path is not a file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True
