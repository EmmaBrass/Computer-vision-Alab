import cv2
import numpy as np

img = cv2.imread(
    r"D:\Jessica\Jessica\robotics\arm_state\live_cam0.png"
)
hs, ws = img.shape[:2]
x1, y1, x2, y2 = 78.87, 108.07, 439.78, 353.51
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
blue = cv2.inRange(hsv, (88, 50, 60), (115, 255, 255))
blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
blobs = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    a = int(cv2.contourArea(c))
    if a >= 30:
        blobs.append((x, y, w, h, a))
print("nblobs", len(blobs), "blue_px", cv2.countNonZero(blue))
seed = [
    b
    for b in blobs
    if not (
        b[0] + b[2] < x1 - 28
        or b[1] + b[3] < y1 - 28
        or b[0] > x2 + 28
        or b[1] > y2 + 28
    )
]
print("seed", len(seed), seed[:8])
cluster = list(seed)
used = {id(b) for b in seed}
changed = True
while changed:
    changed = False
    for a in list(cluster):
        ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
        for b in blobs:
            if id(b) in used:
                continue
            bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
            if ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 < 300:
                cluster.append(b)
                used.add(id(b))
                changed = True
rx1 = min(b[0] for b in cluster)
ry1 = min(b[1] for b in cluster)
rx2 = max(b[0] + b[2] for b in cluster)
ry2 = max(b[1] + b[3] for b in cluster)
print("blue cluster", rx1, ry1, rx2, ry2, "wh", rx2 - rx1, ry2 - ry1)
print("frame area frac", (rx2 - rx1) * (ry2 - ry1) / (ws * hs))

# Simulate grey expand + pad + left clamp like refine
grayish = cv2.inRange(hsv, (0, 0, 45), (179, 60, 215))
joint = np.zeros((hs, ws), np.uint8)
cv2.rectangle(joint, (rx1, ry1), (rx2, ry2), 255, -1)
near = cv2.dilate(joint, np.ones((35, 35), np.uint8), iterations=3)
grey_near = cv2.bitwise_and(grayish, near)
gcnts, _ = cv2.findContours(grey_near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
slack = 50
for c in gcnts:
    gx, gy, gw, gh = cv2.boundingRect(c)
    if gw * gh < 500:
        continue
    if (gx + gw) < 0.22 * ws and gx < 0.12 * ws:
        continue
    if (
        gx + gw < rx1 - slack
        or gy + gh < ry1 - slack
        or gx > rx2 + slack
        or gy > ry2 + slack
    ):
        continue
    rx1 = min(rx1, gx)
    ry1 = min(ry1, gy)
    rx2 = max(rx2, gx + gw)
    ry2 = max(ry2, gy + gh)
print("after grey", rx1, ry1, rx2, ry2, "frac", (rx2 - rx1) * (ry2 - ry1) / (ws * hs))
pad = 28
rx1 = max(0, rx1 - 8)
ry2 = min(hs - 1, ry2 + 36)
rx2 = min(ws - 1, rx2 + 24)
if rx1 < 0.15 * ws:
    blue_left = int(cv2.countNonZero(blue[:, : int(0.18 * ws)]))
    print("blue_left", blue_left)
    if blue_left < 80:
        rx1 = max(rx1, int(0.18 * ws))
out = [
    float(max(0, rx1 - pad)),
    float(max(0, ry1 - pad)),
    float(min(ws - 1, rx2 + pad)),
    float(min(hs - 1, ry2 + pad)),
]
ow, oh = out[2] - out[0], out[3] - out[1]
print("out", out, "frac", ow * oh / (ws * hs), "limit", 0.75)
