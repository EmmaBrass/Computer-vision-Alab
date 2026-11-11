@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
if errorlevel 1 exit /b 1
call install\setup.bat
echo Starting orbbec_cam1_pub...
ros2 run orbbec_dual_sync orbbec_camera_publisher --ros-args -r __node:=orbbec_cam1_pub --params-file D:\Jessica\Jessica\ros2_ws\install\share\orbbec_dual_sync\config\dual_sync.yaml
echo cam1_pub exited %ERRORLEVEL%
exit /b %ERRORLEVEL%
