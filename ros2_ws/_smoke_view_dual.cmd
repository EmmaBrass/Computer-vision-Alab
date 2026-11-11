@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call view_dual_cameras.cmd --help 2>nul
REM instead just validate env portion
call _ros_env.cmd
if errorlevel 1 (echo FAIL ros_env & exit /b 1)
call install\setup.bat
where ros2
if errorlevel 1 (echo FAIL ros2 & exit /b 1)
echo OK view_dual_cameras prerequisites
