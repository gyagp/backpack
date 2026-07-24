@echo off
setlocal

cd /d "%~dp0"
set "DASHBOARD_URL=http://localhost:8787/status"

rem Reuse the control plane when it is already running.
powershell.exe -NoProfile -Command ^
  "try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 'http://localhost:8787/api/status'; if ($r.StatusCode -ne 200) { exit 1 } } catch { exit 1 }"
if errorlevel 1 (
  if not exist "gitignore\evolution" mkdir "gitignore\evolution"

  echo Starting Backpack dashboard...
  powershell.exe -NoProfile -Command ^
    "Start-Process -FilePath 'python.exe' -ArgumentList @('-m','evolution.server','--host','0.0.0.0','--port','8787') -WorkingDirectory '%CD%' -WindowStyle Hidden -RedirectStandardOutput '%CD%\gitignore\evolution\server.log' -RedirectStandardError '%CD%\gitignore\evolution\server-error.log'"

  rem Wait until the HTTP server is ready before opening the browser.
  powershell.exe -NoProfile -Command ^
    "$ready = $false; for ($i = 0; $i -lt 30; $i++) { try { $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 'http://localhost:8787/api/status'; if ($r.StatusCode -eq 200) { $ready = $true; break } } catch {}; Start-Sleep -Milliseconds 500 }; if (-not $ready) { exit 1 }"
  if errorlevel 1 (
    echo Failed to start the dashboard. See gitignore\evolution\server-error.log.
    pause
    exit /b 1
  )
)

start "" "%DASHBOARD_URL%"
endlocal
