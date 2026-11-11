import cv2, sys
from pathlib import Path
sys.path.insert(0, r"D:\Jessica\Jessica\robotics")
import importlib, arm_agents
importlib.reload(arm_agents)
from arm_agents import _arm_color_box
dest = Path(r"C:\Users\Ceder-ALAB\Documents\GitHub\alab_one\agent_temp\arm_overlays")
dest.mkdir(parents=True, exist_ok=True)
for cam in ("cam0","cam1"):
    paths = sorted(Path(rf"D:\Jessica\Jessica\robotics\arm_state\captures\{cam}").glob("*.png"))
    ok = 0
    for i in (0, 100, 500, 800, 1100):
        img = cv2.imread(str(paths[i]))
        box = _arm_color_box(img)
        print(cam, i, box)
        if box:
            ok += 1
            x,y,w,h = box
            vis = img.copy()
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.putText(vis,"arm",(x,max(16,y-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.imwrite(str(dest / f"{cam}_box_{i}.png"), vis)
    # count success rate over stride 10
    hits = 0
    for p in paths[::10]:
        if _arm_color_box(cv2.imread(str(p))) is not None:
            hits += 1
    print(cam, "hit_rate", hits, "/", len(paths[::10]))
