import cv2
import numpy as np
from pathlib import Path

def probe(path):
    img = cv2.imread(str(path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    print("===", path.name, "shape", img.shape)
    # Sample grids and find bluish pixels by hue
    # Collect pixels with moderate-high S and H in broad blue range
    mask = cv2.inRange(hsv, (70, 20, 40), (130, 255, 255))
    n = cv2.countNonZero(mask)
    print("broad blue-ish pixels", n, "pct", round(100*n/(h*w),2))
    if n > 0:
        ys, xs = np.where(mask > 0)
        # sample up to 5000
        idx = np.linspace(0, len(xs)-1, min(5000, len(xs))).astype(int)
        vals = hsv[ys[idx], xs[idx]]
        print("H mean/std/min/max", vals[:,0].mean(), vals[:,0].std(), vals[:,0].min(), vals[:,0].max())
        print("S mean/std/min/max", vals[:,1].mean(), vals[:,1].std(), vals[:,1].min(), vals[:,1].max())
        print("V mean/std/min/max", vals[:,2].mean(), vals[:,2].std(), vals[:,2].min(), vals[:,2].max())
        # densest region
        heat = cv2.GaussianBlur(mask.astype(np.float32), (51,51), 0)
        cy, cx = np.unravel_index(int(np.argmax(heat)), heat.shape)
        print("blue heat peak at", cx, cy, "hsv", hsv[cy,cx].tolist(), "bgr", img[cy,cx].tolist())
        # show top-left of bounding of mask
        x,y,ww,hh = cv2.boundingRect(mask)
        print("blue bbox", x,y,ww,hh)

    # Try several HSV ranges and report counts
    ranges = [
        ("cyan85-115", (85,35,70), (115,255,255)),
        ("blue90-130", (90,20,40), (130,255,255)),
        ("teal70-100", (70,25,50), (100,255,255)),
        ("blue_lowS", (85,15,50), (120,255,255)),
        ("any_blue70-140_S15", (70,15,40), (140,255,255)),
    ]
    for name, lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        print(name, cv2.countNonZero(m))

    # Manual click-ish: for cam0 arm is center-right; sample known regions from description
    # Sample center-right patches and print mean HSV of saturated pixels
    patches = [
        ("mid", w//2, h//2, 80),
        ("right", int(w*0.65), int(h*0.45), 100),
        ("ur_elbowish", int(w*0.55), int(h*0.4), 60),
        ("cam1_right", int(w*0.75), int(h*0.45), 80),
    ]
    for name, cx, cy, r in patches:
        x0,x1 = max(0,cx-r), min(w,cx+r)
        y0,y1 = max(0,cy-r), min(h,cy+r)
        patch = hsv[y0:y1, x0:x1]
        sat = patch[:,:,1] > 30
        if sat.any():
            pv = patch[sat]
            print(f"patch {name} sat pixels {sat.sum()} H={pv[:,0].mean():.1f} S={pv[:,1].mean():.1f} V={pv[:,2].mean():.1f}")
        else:
            print(f"patch {name} no sat")

cap0 = sorted(Path(r"D:\Jessica\Jessica\robotics\arm_state\captures\cam0").glob("*.png"))[500]
cap1 = sorted(Path(r"D:\Jessica\Jessica\robotics\arm_state\captures\cam1").glob("*.png"))[500]
probe(cap0)
probe(cap1)

# Save a visualization of broad blue mask
for cam, p in [("cam0", cap0), ("cam1", cap1)]:
    img = cv2.imread(str(p))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (70, 15, 40), (140, 255, 255))
    out = img.copy()
    out[m>0] = (0, 255, 255)
    blend = cv2.addWeighted(img, 0.55, out, 0.45, 0)
    dest = Path(r"C:\Users\Ceder-ALAB\Documents\GitHub\alab_one\agent_temp\arm_overlays")
    dest.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest / f"{cam}_blue_probe.png"), blend)
    print("wrote", dest / f"{cam}_blue_probe.png")
