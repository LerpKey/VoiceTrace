@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem If both local services are already healthy, only open the workspace again.
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $ui = (Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:3000/' -TimeoutSec 5).StatusCode; $api = (Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 5).StatusCode; if ($ui -eq 200 -and $api -eq 200) { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
  echo Audio workspace is already running. Opening the browser...
  start "" "http://localhost:3000/"
  exit /b 0
)

where uv >nul 2>nul
if errorlevel 1 (
  echo [Startup failed] uv was not found. Install uv or add it to PATH.
  echo.
  pause
  exit /b 1
)

echo Starting the audio workspace. Keep this window open...
echo The page will open automatically: http://localhost:3000/
echo.

uv run voice-trace audio-review --data-dir "%~dp0data" --frontend-dir "%~dp0audio-review-ui" --open-browser
set "APP_EXIT_CODE=%errorlevel%"

if not "%APP_EXIT_CODE%"=="0" (
  echo.
  echo [Workspace stopped] Exit code: %APP_EXIT_CODE%.
  echo Keep the error details in this window for troubleshooting.
  pause
)

exit /b %APP_EXIT_CODE%
