@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first, or use the WSL commands in README.md.
  exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn src.api:app --host 127.0.0.1 --port 8000
