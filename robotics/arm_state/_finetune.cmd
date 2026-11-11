@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
set KMP_DUPLICATE_LIB_OK=TRUE
python D:\Jessica\Jessica\robotics\arm_agents.py train --epochs 8 --imgsz 640 --model "D:\Jessica\Jessica\robotics\arm_state\runs\arm_detect\weights\best.pt"
python D:\Jessica\Jessica\robotics\arm_agents.py status
