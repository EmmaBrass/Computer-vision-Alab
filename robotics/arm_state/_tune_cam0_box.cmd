@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
set KMP_DUPLICATE_LIB_OK=TRUE
python D:\Jessica\Jessica\robotics\arm_state\_tune_cam0_box.py
