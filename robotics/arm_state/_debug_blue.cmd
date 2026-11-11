@echo off
cd /d D:\Jessica\Jessica\ros2_ws
call _ros_env.cmd
call install\setup.bat
set PYTHONPATH=D:\Jessica\Jessica\robotics;%PYTHONPATH%
python -c "import cv2,numpy as np; img=cv2.imread(r'D:\Jessica\Jessica\robotics\arm_state\live_cam0.png'); hs,ws=img.shape[:2]; x1,y1,x2,y2=78.87,108.07,439.78,353.51; hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); blue=cv2.inRange(hsv,(88,50,60),(115,255,255)); blue=cv2.morphologyEx(blue,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); blue=cv2.morphologyEx(blue,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)); cnts,_=cv2.findContours(blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); blobs=[]; 
for c in cnts:
 x,y,w,h=cv2.boundingRect(c); a=int(cv2.contourArea(c));
 if a>=30: blobs.append((x,y,w,h,a));
print('nblobs',len(blobs),'blue_px',cv2.countNonZero(blue));
seed=[b for b in blobs if not (b[0]+b[2]<x1-28 or b[1]+b[3]<y1-28 or b[0]>x2+28 or b[1]>y2+28)];
print('seed',len(seed), seed[:5]);
cluster=list(seed); used={id(b) for b in seed}; changed=True
while changed:
 changed=False
 for a in list(cluster):
  ax,ay=a[0]+a[2]/2,a[1]+a[3]/2
  for b in blobs:
   if id(b) in used: continue
   bx,by=b[0]+b[2]/2,b[1]+b[3]/2
   if ((bx-ax)**2+(by-ay)**2)**0.5<300:
    cluster.append(b); used.add(id(b)); changed=True
rx1=min(b[0] for b in cluster); ry1=min(b[1] for b in cluster); rx2=max(b[0]+b[2] for b in cluster); ry2=max(b[1]+b[3] for b in cluster);
print('blue cluster',rx1,ry1,rx2,ry2,'area',(rx2-rx1)*(ry2-ry1))"
