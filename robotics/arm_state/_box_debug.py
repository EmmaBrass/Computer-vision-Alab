import cv2, numpy as np, sys
sys.path.insert(0, r"D:\Jessica\Jessica\robotics")
from arm_agents import _arm_color_box
from pathlib import Path

for cam in ("cam0","cam1"):
    p = sorted(Path(rf"D:\Jessica\Jessica\robotics\arm_state\captures\{cam}").glob("*.png"))[500]
    img = cv2.imread(str(p))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (85, 35, 70), (115, 255, 255))
    print(cam, "blue", cv2.countNonZero(blue), "box", _arm_color_box(img))
    # try tighter: higher S
    for smin in (40,50,60,70,80):
        b = cv2.inRange(hsv, (90, smin, 80), (112, 255, 255))
        print("  S>=", smin, "count", cv2.countNonZero(b))
        ys,xs = np.where(b>0)
        if len(xs)>100:
            print("   bbox", xs.min(), ys.min(), xs.max()-xs.min(), ys.max()-ys.min())
