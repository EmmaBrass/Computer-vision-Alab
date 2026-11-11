@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
python D:\Jessica\Jessica\robotics\agent_temp\build_arm_cv_report_pdf.py
