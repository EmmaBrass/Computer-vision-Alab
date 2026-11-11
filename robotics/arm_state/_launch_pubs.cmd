@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
echo Starting publishers only (no dual_view windows)...
ros2 launch orbbec_dual_sync dual_sync_detect.launch.py enable_detect:=false enable_rack_overlay:=false enable_dual_view:=false
