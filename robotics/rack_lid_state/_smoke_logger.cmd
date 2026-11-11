@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
python -c "import rack_state_logger; print('ok', rack_state_logger.TOPIC_STATE, rack_state_logger.LOG_TXT)"
