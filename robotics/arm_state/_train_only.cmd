@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=4
python D:\Jessica\Jessica\robotics\arm_agents.py train --epochs 35 --imgsz 640
python D:\Jessica\Jessica\robotics\arm_agents.py status
