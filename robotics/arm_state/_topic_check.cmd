@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
ros2 topic list
echo ---
ros2 topic hz /orbbec/cam1/color/image_raw --window 5
