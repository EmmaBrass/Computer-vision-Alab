@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
echo === LABEL ===
python D:\Jessica\Jessica\robotics\arm_agents.py label --method color --stride 2
echo === TRAIN ===
python D:\Jessica\Jessica\robotics\arm_agents.py train --epochs 35 --imgsz 640
python D:\Jessica\Jessica\robotics\arm_agents.py status
