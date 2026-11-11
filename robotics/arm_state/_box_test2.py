import cv2, sys, importlib
from pathlib import Path
sys.path.insert(0, r"D:\Jessica\Jessica\robotics")
import arm_agents
importlib.reload(arm_agents)
from arm_agents import _arm_color_box
dest = Path(r"C:\Users\Ceder-ALAB\Documents\GitHub\alab_one\agent_temp\arm_overlays")
for cam in ("cam0","cam1"):
    paths = sorted(Path(rf"D:\Jessica\Jessica\robotics\arm_state\captures\{cam}").glob("*.png"))
    hits=0
    for i in (0,100,500,800,1100):
        img=cv2.imread(str(paths[i]))
        box=_arm_color_box(img, cam=cam)
        print(cam,i,box)
        if box:
            hits+=1
            x,y,w,h=box
            vis=img.copy(); cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.putText(vis,"arm",(x,max(16,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.imwrite(str(dest/f"{cam}_v2_{i}.png"), vis)
    n=0; h2=0
    for p in paths[::5]:
        n+=1
        if _arm_color_box(cv2.imread(str(p)), cam=cam): h2+=1
    print(cam,"rate",h2,"/",n)
