@echo off
cd /d "%~dp0"
echo [1/4] Creating Python environment...
py -m venv .venv || goto :failed
echo [2/4] Installing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed
echo [3/4] Preparing recommendations from UCI retail data...
".venv\Scripts\python.exe" -m src.build || goto :failed
echo [4/4] Running project checks...
".venv\Scripts\python.exe" -m pytest -q || goto :failed
echo.
echo Setup complete. Run run.bat to open the dashboard.
exit /b 0
:failed
echo.
echo Setup stopped with an error. Read the message above; if UCI download failed, manually download Online Retail.xlsx into data\ and rerun setup.bat.
exit /b 1
