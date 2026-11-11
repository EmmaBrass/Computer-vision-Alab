@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
python D:\Jessica\Jessica\robotics\arm_state\_hsv_probe.py
