@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
ros2 topic info /orbbec/cam1/color/image_raw
