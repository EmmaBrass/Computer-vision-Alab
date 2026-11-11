@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
python D:\Jessica\Jessica\robotics\arm_state\_divcheck.py
echo ---LABEL---
python D:\Jessica\Jessica\robotics\arm_agents.py label
