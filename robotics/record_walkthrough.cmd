@echo off
REM Handheld Orbbec walkthrough recorder (color + depth, 30 s).
REM Double-click, or run: record_walkthrough.cmd --serial CP2G8530008H --seconds 30

echo.
echo ============================================================
echo   WALKTHROUGH RECORDER - one movable Orbbec
echo ============================================================
echo   1. A window shows cameras 0, 1, 2 left to right.
echo      Move the one you can carry and press its number.
echo   2. SPACE starts a 3 s countdown, then 30 s of recording.
echo   3. Walk slowly. Q stops early.
echo   Saves to lab_scan\walkthrough_^<time^>
echo ============================================================
echo.

set "ORBBEC_CAMERA_ROOT=%~dp0"
for %%I in ("%ORBBEC_CAMERA_ROOT%") do set "ORBBEC_CAMERA_ROOT=%%~fI"

powershell -NoProfile -File "%~dp0camera_busy.ps1"
if errorlevel 1 (
  echo ERROR: a program still has the Orbbec cameras open.
  echo Stop ROS / Orbbec Viewer / the camera publishers, then run this again.
  pause
  exit /b 1
)

set "VIAL_PYTHON="
for %%P in (
  "%USERPROFILE%\anaconda3\python.exe"
  "C:\Users\Ceder-ALAB\anaconda3\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
  "%USERPROFILE%\miniconda3\python.exe"
  "%USERPROFILE%\AppData\Local\anaconda3\python.exe"
  "%USERPROFILE%\AppData\Local\miniconda3\python.exe"
  "C:\ProgramData\anaconda3\python.exe"
) do if not defined VIAL_PYTHON if exist %%~P set "VIAL_PYTHON=%%~P"
if not defined VIAL_PYTHON (
  for /f "delims=" %%I in ('where python 2^>nul') do (
    echo %%I | findstr /I /C:"WindowsApps" >nul
    if errorlevel 1 if not defined VIAL_PYTHON set "VIAL_PYTHON=%%I"
  )
)
if not defined VIAL_PYTHON (
  echo ERROR: Python not found. Install Anaconda/Python 3.10+.
  pause
  exit /b 1
)

cd /d "%ORBBEC_CAMERA_ROOT%"
"%VIAL_PYTHON%" record_walkthrough.py %*
echo.
pause
exit /b %ERRORLEVEL%
