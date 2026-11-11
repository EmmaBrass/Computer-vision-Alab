@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
python -c "import arm_agents as aa, cv2; print('has refine', hasattr(aa,'refine_arm_xyxy')); img=cv2.imread(r'D:\Jessica\Jessica\robotics\arm_state\live_cam0.png'); yolo=[78.87,108.07,439.78,353.51]; print(aa.refine_arm_xyxy(img,yolo,cam='cam0'))"
