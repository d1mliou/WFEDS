@echo off
rem WFEDS web server - double-click to start, then open http://localhost:8000
rem Needs: Docker Desktop installed, the wfeds:phase2 image built once
rem (docker build -f docker/Dockerfile -t wfeds:phase2 . from the repo root),
rem and an LLM key in the repo's .env (GEMINI_API_KEY for the default preset).

cd /d "%~dp0.."

rem --- 1. make sure the Docker daemon is up (start Docker Desktop if not) ---
docker info >nul 2>&1
if errorlevel 1 (
    echo Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    :waitdocker
    timeout /t 5 /nobreak >nul
    docker info >nul 2>&1
    if errorlevel 1 goto waitdocker
)

rem --- 2. resolve the data folder machine-independently (same logic as _paths.py) ---
set "DATA_DIR=%OneDriveCommercial%\Working progress\Scripts\Data"
if not exist "%DATA_DIR%" set "DATA_DIR=%OneDrive%\Working progress\Scripts\Data"
if not exist "%DATA_DIR%" (
    echo ERROR: data folder not found via OneDriveCommercial/OneDrive env vars.
    pause
    exit /b 1
)

rem --- 3. (re)start the server container ---
docker rm -f wfeds_web >nul 2>&1
docker run -d --name wfeds_web -p 8000:8000 --env-file .env ^
    -v "%DATA_DIR%:/data" -e WFEDS_DATA_DIR=/data wfeds:phase2
if errorlevel 1 (
    echo ERROR: container failed to start. Is the wfeds:phase2 image built?
    pause
    exit /b 1
)

echo Server running at http://localhost:8000  (stop it with: docker stop wfeds_web)
start "" http://localhost:8000
pause
