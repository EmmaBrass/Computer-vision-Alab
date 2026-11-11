@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
set KMP_DUPLICATE_LIB_OK=TRUE
python -c "from orbbec_dual_sync.arm_overlay_draw import get_worker, default_arm_weights; from pathlib import Path; import cv2; w=default_arm_weights(Path(r'D:\Jessica\Jessica\robotics')); print('weights', w); img=cv2.imread(r'D:\Jessica\Jessica\robotics\furnace_state\live_for_slots.png'); boxes=get_worker(w, conf=0.25).predict(img); print('n', len(boxes)); print(boxes[:2])"
