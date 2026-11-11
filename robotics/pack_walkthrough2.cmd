@echo off
REM Zip walkthrough_20261106_110839 (color + depth + intrinsics + README, no preview video)
REM to the Desktop, then open Google Drive so you can drag it in.
setlocal
set "SRC=%~dp0lab_scan\walkthrough_20261106_110839"
set "DST=%USERPROFILE%\Desktop\alab_walkthrough2.zip"
if not exist "%SRC%\color" (
  echo Could not find %SRC%
  pause
  exit /b 1
)
echo Zipping %SRC%
echo to %DST%   (about 750 MB, takes a few minutes)
powershell -NoProfile -Command "Compress-Archive -Force -Path '%SRC%\color','%SRC%\depth','%SRC%\intrinsics.json','%SRC%\README.md' -DestinationPath '%DST%'"
if errorlevel 1 (
  echo Zipping failed. Check free disk space on C: - it needs about 800 MB.
  pause
  exit /b 1
)
echo Done: %DST%
start "" "https://drive.google.com/drive/my-drive"
explorer /select,"%DST%"
echo Drag alab_walkthrough2.zip into Google Drive, then Share - Anyone with the link - Copy link.
pause
