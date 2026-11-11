@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
python D:\Jessica\Jessica\robotics\arm_agents.py capture --seconds 180 --interval 0.15 --source ros
