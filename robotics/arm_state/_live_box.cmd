@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
python -c "import cv2,arm_agents as aa; from pathlib import Path; img=cv2.imread(r'D:\Jessica\Jessica\robotics\furnace_state\live_for_slots.png'); b=aa._arm_color_box(img,pad=48,cam='cam0'); print('color', b); frames=aa.grab_frames_ros(2, topic='/orbbec/cam0/color/image_raw', timeout_s=12, interval_s=0.3); live=frames[-1] if frames else None; print('live', None if live is None else live.shape); 
import sys
if live is not None:
 cv2.imwrite(r'D:\Jessica\Jessica\robotics\arm_state\live_cam0.png', live); b2=aa._arm_color_box(live,pad=48,cam='cam0'); print('live color', b2); from orbbec_dual_sync.arm_overlay_draw import get_worker, default_arm_weights; w=default_arm_weights(Path(r'D:\Jessica\Jessica\robotics')); boxes=get_worker(w,0.2).predict(live,cam='cam0',refine=True); print('live yolo+ref', boxes); ov=live.copy();
 for xyxy,c in boxes:
  x1,y1,x2,y2=[int(v) for v in xyxy]; cv2.rectangle(ov,(x1,y1),(x2,y2),(255,220,0),2)
 if b2: x,y,ww,hh=b2; cv2.rectangle(ov,(x,y),(x+ww,y+hh),(0,255,0),2)
 cv2.imwrite(r'D:\Jessica\Jessica\robotics\arm_state\preview_cam0_live.png', ov); print('wrote live preview')"
