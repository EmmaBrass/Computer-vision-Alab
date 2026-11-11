import cv2, numpy as np
from pathlib import Path
p = Path(r"D:\Jessica\Jessica\robotics\arm_state\captures\cam0")
paths = sorted(p.glob("cam0_*.png"))
# compare last stamp batch vs first of previous
stamps = {}
for x in paths:
    # cam0_YYYYMMDD_HHMMSS_seqNNNNN.png
    parts = x.stem.split("_")
    stamp = parts[1]+"_"+parts[2]
    stamps.setdefault(stamp, []).append(x)
print("batches", {k: len(v) for k,v in stamps.items()})
keys = sorted(stamps)
if len(keys)>=2:
    a = cv2.imread(str(stamps[keys[0]][len(stamps[keys[0]])//2]),0)
    b = cv2.imread(str(stamps[keys[-1]][len(stamps[keys[-1]])//2]),0)
    if a is not None and b is not None and a.shape==b.shape:
        d=cv2.absdiff(a,b).astype(np.float32)
        print("cross-batch mean", float(d.mean()), "pct>12", float((d>12).mean()))
    # within latest batch
    latest=stamps[keys[-1]]
    imgs=[cv2.imread(str(latest[i]),0) for i in (0,len(latest)//2,-1)]
    for i,j,lab in [(0,1,"0-mid"),(1,2,"mid-end")]:
        d=cv2.absdiff(imgs[i],imgs[j]).astype(np.float32)
        print("latest", lab, float(d.mean()), float((d>12).mean()))
