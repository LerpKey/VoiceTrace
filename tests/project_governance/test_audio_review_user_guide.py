"""Keep the audio-review manual aligned with the live launcher and UI."""

import re
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from research_kb.cli import app

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs/reference/audio-review-user-guide.md"
WINDOWS_LAUNCHER = ROOT / "start-voicetrace-workspace.bat"
LONG_OPTION = re.compile(r"--[a-z][a-z-]+")
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ALLOWED_CJK_PATHS = {"audio-review-ui/app/locale.tsx"}


def test_audio_review_manual_covers_live_launcher_and_primary_ui() -> None:
    manual = MANUAL.read_text(encoding="utf-8")
    help_result = CliRunner().invoke(app, ["audio-review", "--help"])

    assert help_result.exit_code == 0, help_result.output
    assert "uv run voice-trace audio-review" in manual
    for option in set(LONG_OPTION.findall(help_result.stdout)):
        assert option in manual, f"audio-review lacks documented {option}"

    workspace = (ROOT / "audio-review-ui/app/page.tsx").read_text(encoding="utf-8")
    favorites = (ROOT / "audio-review-ui/app/favorites/page.tsx").read_text(encoding="utf-8")
    for key, label in (
        ("manageRecordings", "Manage recordings"),
        ("manageSpeakers", "Manage speakers"),
        ("calibrateStart", "Calibrate start time"),
        ("favoritesSummary", "Favorites summary"),
        ("addRecording", "Add recording"),
    ):
        assert f't("{key}")' in workspace
        assert label in manual
    for key, label in (("refreshFavorites", "Refresh favorites"), ("saveNote", "Save note")):
        assert f't("{key}")' in favorites
        assert label in manual


def test_windows_launcher_starts_the_workspace_from_its_own_directory() -> None:
    launcher_bytes = WINDOWS_LAUNCHER.read_bytes()
    launcher = launcher_bytes.decode("ascii")
    manual = MANUAL.read_text(encoding="utf-8")

    assert launcher_bytes.isascii()
    assert 'cd /d "%~dp0"' in launcher
    assert "uv run voice-trace audio-review" in launcher
    assert '--data-dir "%~dp0data"' in launcher
    assert '--frontend-dir "%~dp0audio-review-ui"' in launcher
    assert "--open-browser" in launcher
    assert "http://localhost:3000/" in launcher
    assert "http://127.0.0.1:8765/api/health" in launcher
    assert WINDOWS_LAUNCHER.name in manual


def test_tracked_cjk_is_limited_to_chinese_docs_and_locale_resource() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    offenders: list[str] = []
    for relative in tracked:
        if relative.endswith(".zh-CN.md") or relative in ALLOWED_CJK_PATHS:
            continue
        path = ROOT / relative
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CJK.search(line):
                offenders.append(f"{relative}:{line_number}:{line.strip()}")
    assert not offenders, "CJK text escaped the bilingual allowlist:\n" + "\n".join(offenders)
