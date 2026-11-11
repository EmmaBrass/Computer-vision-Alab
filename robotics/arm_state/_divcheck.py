import cv2, numpy as np
from pathlib import Path
p = Path(r"D:\Jessica\Jessica\robotics\arm_state\captures\cam1")
paths = sorted(p.glob("cam1_*.png"))
print("n", len(paths))
if len(paths) >= 3:
    imgs = [cv2.imread(str(paths[i]), cv2.IMREAD_GRAYSCALE) for i in (0, len(paths)//2, -1)]
    for a,b,lab in [(0,1,"0-mid"),(1,2,"mid-end"),(0,2,"0-end")]:
        d = cv2.absdiff(imgs[a], imgs[b]).astype(np.float32)
        print(lab, "mean", float(d.mean()), "std", float(d.std()), "pct>12", float((d>12).mean()))
