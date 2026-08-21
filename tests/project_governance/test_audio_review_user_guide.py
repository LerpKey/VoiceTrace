"""Keep the audio-review manual aligned with the live launcher and UI."""

import re
from pathlib import Path

from typer.testing import CliRunner

from research_kb.cli import app

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "docs/reference/audio-review-user-guide.md"
WINDOWS_LAUNCHER = ROOT / "启动录音文本工作台.bat"
LONG_OPTION = re.compile(r"--[a-z][a-z-]+")


def test_audio_review_manual_covers_live_launcher_and_primary_ui() -> None:
    manual = MANUAL.read_text(encoding="utf-8")
    help_result = CliRunner().invoke(app, ["audio-review", "--help"])

    assert help_result.exit_code == 0, help_result.output
    assert "uv run file-assistant audio-review" in manual
    for option in set(LONG_OPTION.findall(help_result.stdout)):
        assert option in manual, f"audio-review lacks documented {option}"

    workspace = (ROOT / "audio-review-ui/app/page.tsx").read_text(encoding="utf-8")
    favorites = (ROOT / "audio-review-ui/app/favorites/page.tsx").read_text(encoding="utf-8")
    for label in ("管理录音", "管理说话人", "校准开始时间", "收藏汇总", "添加录音"):
        assert label in workspace
        assert label in manual
    for label in ("刷新收藏", "保存备注"):
        assert label in favorites
        assert label in manual


def test_windows_launcher_starts_the_workspace_from_its_own_directory() -> None:
    launcher_bytes = WINDOWS_LAUNCHER.read_bytes()
    launcher = launcher_bytes.decode("ascii")
    manual = MANUAL.read_text(encoding="utf-8")

    assert launcher_bytes.isascii()
    assert 'cd /d "%~dp0"' in launcher
    assert "uv run file-assistant audio-review" in launcher
    assert '--data-dir "%~dp0data"' in launcher
    assert '--frontend-dir "%~dp0audio-review-ui"' in launcher
    assert "--open-browser" in launcher
    assert "http://localhost:3000/" in launcher
    assert "http://127.0.0.1:8765/api/health" in launcher
    assert WINDOWS_LAUNCHER.name in manual

