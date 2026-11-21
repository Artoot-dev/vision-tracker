@echo off
REM One-shot setup for a Windows box. Double-click or run from cmd.
REM Creates .venv next to this file and installs every dependency.
setlocal enabledelayedexpansion
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo [x] The Python launcher "py" was not found.
  echo     Install 64-bit Python 3.11 from https://www.python.org/downloads/windows/
  echo     and tick "Add python.exe to PATH" during setup, then run this again.
  pause
  exit /b 1
)

echo [1/4] creating .venv
py -3.11 -m venv .venv 2>nul
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo [x] venv creation failed. Is Python 3.11 or newer installed?
  pause
  exit /b 1
)

echo [2/4] upgrading pip
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip

echo [3/4] installing core requirements
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo [4/4] installing the ultralytics backend (YOLO + BoT-SORT)
".venv\Scripts\python.exe" -m pip install "ultralytics>=8.2,<9"
if errorlevel 1 goto fail

echo.
echo [check] importing everything
".venv\Scripts\python.exe" -c "import numpy,cv2,yaml,onnxruntime,torch,ultralytics;print('numpy',numpy.__version__);print('cv2',cv2.__version__);print('torch',torch.__version__);print('ultralytics',ultralytics.__version__);print('ALL OK')"
if errorlevel 1 goto fail

echo.
echo Done. To run:
echo     .venv\Scripts\python.exe run.py YOURCLIP.mp4
echo or activate the environment first:
echo     .venv\Scripts\activate.bat
pause
exit /b 0

:fail
echo.
echo [x] Install failed. Scroll up for the pip error.
pause
exit /b 1
