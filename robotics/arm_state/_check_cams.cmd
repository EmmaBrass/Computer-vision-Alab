@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
echo === topics ===
ros2 topic list 2>nul
echo === cam1 hz (3s) ===
ros2 topic info /orbbec/cam1/color/image_raw 2>nul
echo === cam0 info ===
ros2 topic info /orbbec/cam0/color/image_raw 2>nul
echo === nodes ===
ros2 node list 2>nul
