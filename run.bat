@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first. For an existing Linux environment, use the WSL commands in README.md.
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
