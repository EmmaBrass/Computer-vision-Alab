"""Multi-agent robot-arm dataset + YOLO train (moving target).

The arm is NOT fixed in the frame — collect many poses from cam0 and cam1,
auto-propose boxes (UR blue joints by default), then train a small YOLO detector.

Agents:
  CaptureAgent — high-rate ROS (or USB) grabs -> arm_state/captures/{cam0,cam1}/
  LabelAgent   — color (UR cyan joints) or motion boxes -> YOLO labels + keeps/
  TrainAgent   — ultralytics YOLOv8n fine-tune -> arm_state/runs/arm_detect/

Examples (use Pixi/ROS python so rclpy works for --source ros):
  cd D:\\Jessica\\Jessica\\ros2_ws
  call _ros_env.cmd && call install\\setup.bat
  set PYTHONPATH=..\\robotics;%PYTHONPATH%
  set KMP_DUPLICATE_LIB_OK=TRUE
  python ..\\robotics\\arm_agents.py capture --seconds 180 --source ros
  python ..\\robotics\\arm_agents.py label --method color
  python ..\\robotics\\arm_agents.py train --epochs 40
  python ..\\robotics\\arm_agents.py status
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "arm_state"
CAPTURES_DIR = STATE_DIR / "captures"
KEEPS_DIR = STATE_DIR / "keeps"
LABELS_DIR = STATE_DIR / "labels"
DATASET_DIR = STATE_DIR / "dataset"
RUNS_DIR = STATE_DIR / "runs"
PID_FILE = STATE_DIR / "agents.pids"
RUN_LOG = STATE_DIR / "agents.log"
CONFIG_PATH = STATE_DIR / "arm.json"

CAM_TOPICS = {
    "cam0": "/orbbec/cam0/color/image_raw",
    "cam1": "/orbbec/cam1/color/image_raw",
}
CLASS_NAME = "arm"
CLASS_ID = 0


def _agent_log(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(
            f"%(asctime)s [{name}] %(levelname)s %(message)s", "%H:%M:%S"
        )
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(RUN_LOG, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dirs() -> None:
    for d in (
        STATE_DIR,
        CAPTURES_DIR / "cam0",
        CAPTURES_DIR / "cam1",
        KEEPS_DIR,
        LABELS_DIR,
        DATASET_DIR,
        RUNS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.is_file():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    return {
        "class_name": CLASS_NAME,
        "motion_min_area": 2500,
        "motion_thresh": 25,
        "min_box_side": 40,
        "label_stride": 2,
        "yolo_model": "yolov8n.pt",
        "imgsz": 640,
        "conf": 0.25,
    }


def save_config(data: dict[str, Any]) -> None:
    ensure_dirs()
    CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------


def _ros_paths() -> list[Path]:
    ws = ROOT.parent / "ros2_ws"
    return [
        ws / "install" / "Lib" / "site-packages",
        ws / "src" / "orbbec_dual_sync",
    ]


def grab_dual_ros(
    *,
    seconds: float,
    interval_s: float = 0.25,
    timeout_startup_s: float = 20.0,
) -> dict[str, int]:
    """Subscribe to cam0+cam1; save every interval. Returns counts per cam."""
    for p in _ros_paths():
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import Image

    from orbbec_dual_sync.image_codec import imgmsg_to_numpy

    ensure_dirs()
    log = _agent_log("CaptureAgent")
    counts = {"cam0": 0, "cam1": 0}
    latest: dict[str, Optional[np.ndarray]] = {"cam0": None, "cam1": None}
    last_save = {"cam0": 0.0, "cam1": 0.0}
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if not rclpy.ok():
        rclpy.init()
    node = Node("arm_capture_dual")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    def make_cb(cam: str):
        def cb(msg: Image) -> None:
            try:
                img = imgmsg_to_numpy(msg)
                if img is not None and img.ndim == 3:
                    latest[cam] = img
            except Exception:
                pass

        return cb

    for cam, topic in CAM_TOPICS.items():
        node.create_subscription(Image, topic, make_cb(cam), qos)
        log.info("subscribed %s", topic)

    t0 = time.time()
    # Wait briefly for first frames
    while time.time() - t0 < timeout_startup_s and (
        latest["cam0"] is None and latest["cam1"] is None
    ):
        rclpy.spin_once(node, timeout_sec=0.05)

    end_t = time.time() + max(1.0, float(seconds))
    log.info("capturing for %.0fs interval=%.2fs", seconds, interval_s)
    while time.time() < end_t:
        rclpy.spin_once(node, timeout_sec=0.05)
        now = time.time()
        for cam in ("cam0", "cam1"):
            frame = latest[cam]
            if frame is None:
                continue
            if now - last_save[cam] < interval_s:
                continue
            last_save[cam] = now
            counts[cam] += 1
            seq = counts[cam]
            path = CAPTURES_DIR / cam / f"{cam}_{stamp}_seq{seq:05d}.png"
            cv2.imwrite(str(path), frame)
            if seq % 25 == 0:
                log.info("%s saved=%s", cam, seq)
        time.sleep(0.01)

    node.destroy_node()
    log.info("done cam0=%s cam1=%s", counts["cam0"], counts["cam1"])
    return counts


def grab_dual_usb(*, seconds: float, interval_s: float = 0.35) -> dict[str, int]:
    """Fallback USB capture (exclusive — stop ROS publishers first)."""
    ensure_dirs()
    log = _agent_log("CaptureAgent")
    counts = {"cam0": 0, "cam1": 0}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    caps = {}
    try:
        for idx, cam in enumerate(("cam0", "cam1")):
            cap = cv2.VideoCapture(idx, cv2.CAP_OBSENSOR)
            if not cap.isOpened():
                log.warning("USB %s index %s failed", cam, idx)
                continue
            for _ in range(40):
                cap.grab()
            caps[cam] = cap
            time.sleep(1.2)
        end_t = time.time() + max(1.0, float(seconds))
        next_t = {c: 0.0 for c in caps}
        while time.time() < end_t:
            for cam, cap in caps.items():
                if not cap.grab():
                    continue
                ok, frame = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
                if not ok or frame is None:
                    continue
                now = time.time()
                if now < next_t[cam]:
                    continue
                next_t[cam] = now + interval_s
                counts[cam] += 1
                path = CAPTURES_DIR / cam / f"{cam}_{stamp}_usb{counts[cam]:05d}.png"
                cv2.imwrite(str(path), frame)
            time.sleep(0.01)
    finally:
        for cap in caps.values():
            cap.release()
    log.info("USB done cam0=%s cam1=%s", counts["cam0"], counts["cam1"])
    return counts


# ---------------------------------------------------------------------------
# Color labeling (UR-style blue joints) — works even when arm is still
# ---------------------------------------------------------------------------


def _pose_grey_band(width: int, height: int, cam: str) -> Optional[np.ndarray]:
    """Pixels near the robot's 3D arm, projected into this camera.

    None when the controller pose is missing or this camera has no solved pose.
    The band replaces the old image-space chain that grew grey off blue blobs.
    """
    try:
        from cell_frames import load_cell_frames
        from ur_skeleton import robot_arm_points_mm
        from cam_geometry import Intrinsics, cam_to_pixel
    except ImportError:
        return None
    posed = robot_arm_points_mm()
    if posed is None:
        return None
    pts_mm, _source = posed
    frames = load_cell_frames()
    pose = frames.pose("cam1" if cam == "cam1" else "cam0")
    if not bool(pose.ready):
        return None
    fx = float(frames.fx_px) if frames.fx_px > 1 else 0.5 * float(width)
    k = Intrinsics(fx=fx, fy=fx, cx=0.5 * width, cy=0.5 * height)
    rotation = np.asarray(pose.R, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(pose.t_mm, dtype=np.float64).reshape(3)
    band = np.zeros((height, width), np.uint8)
    pixels: list[tuple[int, int]] = []
    for point in np.asarray(pts_mm, dtype=np.float64).reshape(-1, 3):
        p_cam = rotation.T @ (point - translation)
        uv = cam_to_pixel(p_cam, k)
        if uv is None:
            pixels.append((-1, -1))
            continue
        pixels.append((int(round(uv[0])), int(round(uv[1]))))
    drawn = 0
    for start, end in zip(pixels, pixels[1:]):
        if start[0] < 0 or end[0] < 0:
            continue
        cv2.line(band, start, end, 255, thickness=max(24, int(0.035 * min(width, height))), lineType=cv2.LINE_AA)
        drawn += 1
    if drawn == 0:
        return None
    radius = 21 if cam == "cam0" else 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    return cv2.dilate(band, kernel, iterations=1)


def _box_from_grey(grey: np.ndarray, *, min_side: int, pad: int) -> Optional[tuple[int, int, int, int]]:
    height, width = grey.shape[:2]
    contours, _ = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x1, y1, x2, y2 = width, height, 0, 0
    kept = 0
    for contour in contours:
        if cv2.contourArea(contour) < 400:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        x1, y1 = min(x1, x), min(y1, y)
        x2, y2 = max(x2, x + w), max(y2, y + h)
        kept += 1
    if kept == 0 or x2 <= x1 or y2 <= y1:
        return None
    x = max(0, x1 - pad)
    y = max(0, y1 - pad)
    w = min(width - x, (x2 - x1) + 2 * pad)
    h = min(height - y, (y2 - y1) + 2 * pad)
    if w < min_side or h < min_side or w * h > 0.62 * width * height:
        return None
    return int(x), int(y), int(w), int(h)


def _arm_color_box(
    bgr: np.ndarray,
    *,
    min_blue_area: int = 80,
    min_side: int = 48,
    pad: int = 36,
    cam: str = "",
) -> Optional[tuple[int, int, int, int]]:
    """Propose an arm box from grey metal on the robot's projected 3D pose.

    When the controller pose and a solved camera are both available, grey
    pixels are kept only along that projected arm. The image-space blue
    chain remains only for frames with no robot pose.
    """
    hs, ws = bgr.shape[:2]
    # cam0 needs a fuller silhouette (base + links); cam1 is tighter on the right.
    cluster_r = 280 if cam == "cam0" else 200
    grey_dilate = 31 if cam == "cam0" else 21
    grey_iters = 3 if cam == "cam0" else 2
    pad_use = pad + (16 if cam == "cam0" else 0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    grayish = cv2.inRange(hsv, (0, 0, 50), (179, 55, 210))
    grayish = cv2.morphologyEx(grayish, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pose_band = _pose_grey_band(ws, hs, cam)
    if pose_band is not None:
        return _box_from_grey(
            cv2.bitwise_and(grayish, pose_band),
            min_side=min_side,
            pad=pad_use,
        )

    # Tight cyan — lab has lots of low-S blue-ish clutter (pipes, plastics).
    blue = cv2.inRange(hsv, (90, 60, 70), (112, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    blue_area = int(cv2.countNonZero(blue))
    if blue_area < min_blue_area:
        blue = cv2.inRange(hsv, (88, 50, 60), (115, 255, 255))
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        blue_area = int(cv2.countNonZero(blue))
        if blue_area < min_blue_area:
            return None

    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    blobs: list[tuple[int, int, int, int, int]] = []  # x,y,w,h,area
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = int(cv2.contourArea(c))
        if area < max(25, min_blue_area // 4):
            continue
        blobs.append((x, y, w, h, area))
    if not blobs:
        return None
    blobs.sort(key=lambda b: b[4], reverse=True)

    def _cluster_from(seed: tuple[int, int, int, int, int]) -> list[tuple[int, int, int, int, int]]:
        # Grow cluster by chaining nearby blue joints (covers long UR reach on cam0).
        cluster = [seed]
        used = {id(seed)}
        changed = True
        while changed:
            changed = False
            for a in list(cluster):
                ax = a[0] + a[2] / 2
                ay = a[1] + a[3] / 2
                for b in blobs:
                    if id(b) in used:
                        continue
                    cx = b[0] + b[2] / 2
                    cy = b[1] + b[3] / 2
                    dist = ((cx - ax) ** 2 + (cy - ay) ** 2) ** 0.5
                    if dist < cluster_r:
                        cluster.append(b)
                        used.add(id(b))
                        changed = True
        return cluster

    def _score_cluster(cluster: list[tuple[int, int, int, int, int]]) -> tuple[float, tuple[int, int, int, int]]:
        x1 = min(b[0] for b in cluster)
        y1 = min(b[1] for b in cluster)
        x2 = max(b[0] + b[2] for b in cluster)
        y2 = max(b[1] + b[3] for b in cluster)
        # Expand with grey metal near joints (links + base plate on cam0).
        joint = np.zeros((hs, ws), np.uint8)
        cv2.rectangle(joint, (x1, y1), (x2, y2), 255, -1)
        near = cv2.dilate(
            joint, np.ones((grey_dilate, grey_dilate), np.uint8), iterations=grey_iters
        )
        grey_near = cv2.bitwise_and(grayish, near)
        gcnts, _ = cv2.findContours(grey_near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grey_px = int(cv2.countNonZero(grey_near))
        slack = 40 if cam == "cam0" else 20
        for c in gcnts:
            gx, gy, gw, gh = cv2.boundingRect(c)
            if gw * gh < 400:
                continue
            if (
                gx + gw < x1 - slack
                or gy + gh < y1 - slack
                or gx > x2 + slack
                or gy > y2 + slack
            ):
                continue
            x1 = min(x1, gx)
            y1 = min(y1, gy)
            x2 = max(x2, gx + gw)
            y2 = max(y2, gy + gh)
        # cam0: nudge box down/left a bit so base + shoulder stay inside
        if cam == "cam0":
            x1 = max(0, x1 - 12)
            y2 = min(hs, y2 + 28)
        bx = max(0, x1 - pad_use)
        by = max(0, y1 - pad_use)
        bw = min(ws - bx, (x2 - x1) + 2 * pad_use)
        bh = min(hs - by, (y2 - y1) + 2 * pad_use)
        blue_px = sum(b[4] for b in cluster)
        cx = bx + bw / 2
        # Arm-ness: blue joints + adjacent grey tubes; penalize huge / top-only clutter.
        score = float(blue_px) + 0.35 * float(grey_px)
        if bw * bh > 0.62 * ws * hs:
            score *= 0.05
        if grey_px < 800:
            score *= 0.25  # blue plastic bins without metal tubes
        # cam1 arm lives on the right; cam0 arm is center-right / lower half
        if cam == "cam1":
            score *= 0.4 + 0.6 * (cx / max(1.0, float(ws)))
        else:
            score *= 0.55 + 0.45 * (cx / max(1.0, float(ws)))
            cy = by + bh / 2
            score *= 0.7 + 0.3 * (cy / max(1.0, float(hs)))
        # Mild penalty for hugging the top edge only (ceiling clutter)
        if by < 8 and bh < 0.4 * hs:
            score *= 0.35
        return score, (int(bx), int(by), int(bw), int(bh))

    best_score = -1.0
    best_box: Optional[tuple[int, int, int, int]] = None
    for seed in blobs[:8]:
        cluster = _cluster_from(seed)
        score, box = _score_cluster(cluster)
        if score > best_score:
            best_score = score
            best_box = box
    if best_box is None:
        return None
    x, y, w, h = best_box
    if w < min_side or h < min_side:
        return None
    if w * h > 0.62 * ws * hs:
        return None
    if best_score < 40:
        return None
    return (int(x), int(y), int(w), int(h))


def refine_arm_xyxy(
    bgr: np.ndarray,
    xyxy: list[float] | tuple[float, ...],
    *,
    cam: str = "cam0",
) -> list[float]:
    """Grow a YOLO arm box toward gripper/base using nearby UR blue joints.

    Growth is limited to a band around the YOLO box so left-side cyan clutter
    (monitors / X-ray) cannot pull the box across the whole frame.
    """
    hs, ws = bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    if cam != "cam0":
        pad = 12
        return [
            max(0.0, x1 - pad),
            max(0.0, y1 - pad),
            min(float(ws - 1), x2 + pad),
            min(float(hs - 1), y2 + pad),
        ]

    # Search band: allow down/right for base+gripper; little left growth.
    band_x1 = max(0.0, x1 - 24)
    band_y1 = max(0.0, y1 - 36)
    band_x2 = min(float(ws - 1), x2 + 100)
    band_y2 = min(float(hs - 1), y2 + 110)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90, 55, 70), (112, 255, 255))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blues: list[tuple[float, float, float, float]] = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if int(cv2.contourArea(c)) < 40:
            continue
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        if not (band_x1 <= cx <= band_x2 and band_y1 <= cy <= band_y2):
            continue
        # Ignore far-left cyan (monitor / X-ray clutter on cam0)
        if cx < 0.22 * ws:
            continue
        blues.append((float(bx), float(by), float(bx + bw), float(by + bh)))

    if blues:
        # Left from blue joints (shrinks X-ray margin); right/bottom grow for gripper/base
        rx1 = min(b[0] for b in blues)
        ry1 = min(min(b[1] for b in blues), y1)
        rx2 = max(max(b[2] for b in blues), x2)
        ry2 = max(max(b[3] for b in blues), y2)
        pad_l, pad_t, pad_r, pad_b = 22, 20, 40, 44
        out = [
            float(max(0, rx1 - pad_l)),
            float(max(0, ry1 - pad_t)),
            float(min(ws - 1, rx2 + pad_r)),
            float(min(hs - 1, ry2 + pad_b)),
        ]
        ow, oh = out[2] - out[0], out[3] - out[1]
        if 60 <= ow and 60 <= oh and ow * oh <= 0.55 * ws * hs:
            return out

    # Fallback: keep YOLO, nudge right/down for gripper + base, trim a bit of left
    return [
        max(0.0, x1 + 20),
        max(0.0, y1 - 12),
        min(float(ws - 1), x2 + 56),
        min(float(hs - 1), y2 + 72),
    ]


def _write_label(
    cam: str,
    path: Path,
    img: np.ndarray,
    box: tuple[int, int, int, int],
) -> None:
    x, y, w, h = box
    fh, fw = img.shape[:2]
    yolo = xywh_to_yolo(x, y, w, h, fw, fh)
    keep_name = f"{cam}_{path.stem}.png"
    keep_img = KEEPS_DIR / keep_name
    keep_lbl = LABELS_DIR / f"{keep_name[:-4]}.txt"
    if not keep_img.is_file():
        shutil.copy2(path, keep_img)
    keep_lbl.write_text(
        f"{CLASS_ID} {yolo[0]:.6f} {yolo[1]:.6f} {yolo[2]:.6f} {yolo[3]:.6f}\n",
        encoding="utf-8",
    )
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(
        vis,
        "arm",
        (x, max(16, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(STATE_DIR / "overlays" / keep_name), vis)


def label_color(*, max_per_cam: int = 0, stride: int = 3) -> dict[str, int]:
    """Label arm using blue-joint color prior (pose-independent)."""
    cfg = load_config()
    log = _agent_log("LabelAgent")
    ensure_dirs()
    labeled = {"cam0": 0, "cam1": 0, "skipped": 0}
    (STATE_DIR / "overlays").mkdir(parents=True, exist_ok=True)
    min_blue = int(cfg.get("min_blue_area", 120))
    min_side = int(cfg.get("min_box_side", 48))
    pad = int(cfg.get("color_pad", 48))
    stride = max(1, int(cfg.get("label_stride", stride)))

    for cam in ("cam0", "cam1"):
        paths = sorted((CAPTURES_DIR / cam).glob(f"{cam}_*.png"))
        if max_per_cam > 0:
            paths = paths[-max_per_cam:]
        for i, path in enumerate(paths):
            if stride > 1 and (i % stride) != 0:
                continue
            img = cv2.imread(str(path))
            if img is None:
                labeled["skipped"] += 1
                continue
            box = _arm_color_box(
                img,
                min_blue_area=min_blue,
                min_side=min_side,
                pad=pad,
                cam=cam,
            )
            if box is None:
                continue
            _write_label(cam, path, img, box)
            labeled[cam] += 1
        log.info("%s color-labeled=%s of %s", cam, labeled[cam], len(paths))
    save_config(
        {
            **cfg,
            "last_label": labeled,
            "label_method": "color",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    return labeled


# ---------------------------------------------------------------------------
# Motion labeling
# ---------------------------------------------------------------------------


def _motion_box(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    *,
    thresh: int,
    min_area: int,
    min_side: int,
) -> Optional[tuple[int, int, int, int]]:
    diff = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(diff, int(thresh), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # Union of large motion blobs (arm + gripper often fragmented).
    hs, ws = gray.shape[:2]
    union = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area or w < min_side or h < min_side:
            continue
        box = (x, y, w, h)
        if union is None:
            union = box
        else:
            x1 = min(union[0], box[0])
            y1 = min(union[1], box[1])
            x2 = max(union[0] + union[2], box[0] + box[2])
            y2 = max(union[1] + union[3], box[1] + box[3])
            union = (x1, y1, x2 - x1, y2 - y1)
    if union is None:
        return None
    x, y, w, h = union
    # Clamp / pad slightly
    pad = 8
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(ws - x, w + 2 * pad)
    h = min(hs - y, h + 2 * pad)
    if w < min_side or h < min_side:
        return None
    # Reject near-full-frame (lighting flicker)
    if w * h > 0.85 * ws * hs:
        return None
    return (int(x), int(y), int(w), int(h))


def xywh_to_yolo(
    x: int, y: int, w: int, h: int, fw: int, fh: int
) -> tuple[float, float, float, float]:
    cx = (x + w / 2.0) / float(fw)
    cy = (y + h / 2.0) / float(fh)
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(1e-6, min(1.0, w / float(fw))),
        max(1e-6, min(1.0, h / float(fh))),
    )


def _build_background(paths: list[Path], *, sample: int = 40) -> Optional[np.ndarray]:
    """Median gray background from evenly spaced frames (static scene)."""
    if not paths:
        return None
    step = max(1, len(paths) // max(1, sample))
    stack: list[np.ndarray] = []
    for path in paths[::step][:sample]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stack.append(cv2.GaussianBlur(g, (5, 5), 0))
    if len(stack) < 3:
        return None
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)


def label_motion(*, max_per_cam: int = 0) -> dict[str, int]:
    """Label arm via background subtraction + frame-gap motion."""
    cfg = load_config()
    log = _agent_log("LabelAgent")
    ensure_dirs()
    labeled = {"cam0": 0, "cam1": 0, "skipped": 0}
    (STATE_DIR / "overlays").mkdir(parents=True, exist_ok=True)
    # Looser defaults — consecutive frames at 5 Hz often look static.
    thresh = int(cfg.get("motion_thresh", 18))
    min_area = int(cfg.get("motion_min_area", 1800))
    min_side = int(cfg.get("min_box_side", 32))
    stride = int(cfg.get("label_stride", 2))  # score every Nth frame vs background

    for cam in ("cam0", "cam1"):
        paths = sorted((CAPTURES_DIR / cam).glob(f"{cam}_*.png"))
        if max_per_cam > 0:
            paths = paths[-max_per_cam:]
        bg = _build_background(paths)
        if bg is None:
            log.warning("%s: no background (n=%s)", cam, len(paths))
            continue
        prev_gray = None
        for i, path in enumerate(paths):
            if stride > 1 and (i % stride) != 0:
                continue
            img = cv2.imread(str(path))
            if img is None:
                labeled["skipped"] += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            box_bg = _motion_box(
                bg, gray, thresh=thresh, min_area=min_area, min_side=min_side
            )
            box_prev = None
            if prev_gray is not None:
                box_prev = _motion_box(
                    prev_gray,
                    gray,
                    thresh=max(12, thresh - 4),
                    min_area=min_area,
                    min_side=min_side,
                )
            prev_gray = gray
            box = box_bg or box_prev
            if box is None:
                continue
            # Prefer larger of the two if both exist
            if box_bg and box_prev:
                a = box_bg[2] * box_bg[3]
                b = box_prev[2] * box_prev[3]
                box = box_bg if a >= b else box_prev
            _write_label(cam, path, img, box)
            labeled[cam] += 1
        log.info("%s motion-labeled=%s of %s", cam, labeled[cam], len(paths))
    save_config(
        {
            **cfg,
            "last_label": labeled,
            "label_method": "motion",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    return labeled


def build_yolo_dataset(*, val_frac: float = 0.15) -> Path:
    """Assemble ultralytics dataset YAML from keeps/ + labels/."""
    ensure_dirs()
    log = _agent_log("TrainAgent")
    images = sorted(KEEPS_DIR.glob("*.png"))
    pairs: list[tuple[Path, Path]] = []
    for img in images:
        lbl = LABELS_DIR / f"{img.stem}.txt"
        if lbl.is_file() and lbl.stat().st_size > 0:
            pairs.append((img, lbl))
    if len(pairs) < 8:
        raise RuntimeError(
            f"Need >=8 labeled images to train; have {len(pairs)}. "
            "Run capture then label (with arm moving)."
        )
    random.seed(42)
    random.shuffle(pairs)
    n_val = max(2, int(round(len(pairs) * val_frac)))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    for split, group in (("train", train_pairs), ("val", val_pairs)):
        idir = DATASET_DIR / "images" / split
        ldir = DATASET_DIR / "labels" / split
        if idir.exists():
            shutil.rmtree(idir)
        if ldir.exists():
            shutil.rmtree(ldir)
        idir.mkdir(parents=True)
        ldir.mkdir(parents=True)
        for img, lbl in group:
            shutil.copy2(img, idir / img.name)
            shutil.copy2(lbl, ldir / lbl.name)
    yaml_path = DATASET_DIR / "arm.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {DATASET_DIR.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log.info(
        "dataset train=%s val=%s yaml=%s",
        len(train_pairs),
        len(val_pairs),
        yaml_path,
    )
    return yaml_path


def train_yolo(*, epochs: int = 40, model: str = "yolov8n.pt", imgsz: int = 640) -> Path:
    import os

    # Windows often loads two OpenMP runtimes (torch + MKL); allow and continue.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from ultralytics import YOLO

    log = _agent_log("TrainAgent")
    yaml_path = build_yolo_dataset()
    cfg = load_config()
    model_name = str(cfg.get("yolo_model") or model)
    log.info("training %s epochs=%s imgsz=%s", model_name, epochs, imgsz)
    yolo = YOLO(model_name)
    results = yolo.train(
        data=str(yaml_path),
        epochs=int(epochs),
        imgsz=int(imgsz),
        project=str(RUNS_DIR),
        name="arm_detect",
        exist_ok=True,
        patience=15,
        batch=8,
        workers=0,  # Windows-friendly
        verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    cfg["weights"] = str(best)
    cfg["last_train"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg["n_train_images"] = len(list((DATASET_DIR / "images" / "train").glob("*.png")))
    save_config(cfg)
    log.info("weights -> %s", best)
    # Quick smoke on a few keeps
    if best.is_file():
        det = YOLO(str(best))
        samples = sorted(KEEPS_DIR.glob("*.png"))[:6]
        smoke_dir = STATE_DIR / "overlays" / "train_smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        for p in samples:
            r = det.predict(str(p), conf=float(cfg.get("conf", 0.25)), verbose=False)[0]
            plotted = r.plot()
            cv2.imwrite(str(smoke_dir / p.name), plotted)
        log.info("smoke overlays -> %s", smoke_dir)
    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_capture(args: argparse.Namespace) -> int:
    ensure_dirs()
    save_config(load_config())
    if args.source == "usb":
        counts = grab_dual_usb(seconds=float(args.seconds), interval_s=float(args.interval))
    else:
        try:
            counts = grab_dual_ros(
                seconds=float(args.seconds), interval_s=float(args.interval)
            )
        except Exception as exc:
            print(f"ROS capture failed ({exc}); trying USB")
            counts = grab_dual_usb(
                seconds=float(args.seconds), interval_s=float(args.interval)
            )
    total = counts["cam0"] + counts["cam1"]
    print(f"CaptureAgent: cam0={counts['cam0']} cam1={counts['cam1']} total={total}")
    print(f"Dir: {CAPTURES_DIR}")
    return 0 if total > 0 else 1


def cmd_label(args: argparse.Namespace) -> int:
    ensure_dirs()
    (STATE_DIR / "overlays").mkdir(parents=True, exist_ok=True)
    method = str(getattr(args, "method", "color") or "color").lower()
    if method == "motion":
        labeled = label_motion(max_per_cam=int(args.max_per_cam))
    else:
        labeled = label_color(
            max_per_cam=int(args.max_per_cam),
            stride=int(getattr(args, "stride", 3) or 3),
        )
    print(
        f"LabelAgent[{method}]: cam0={labeled['cam0']} cam1={labeled['cam1']} "
        f"skipped={labeled['skipped']}"
    )
    print(f"Keeps: {KEEPS_DIR}  Labels: {LABELS_DIR}")
    return 0 if (labeled["cam0"] + labeled["cam1"]) > 0 else 1


def cmd_train(args: argparse.Namespace) -> int:
    ensure_dirs()
    # Auto-label if no labels yet
    n_lbl = len(list(LABELS_DIR.glob("*.txt")))
    if n_lbl < 8:
        print("Few labels — running LabelAgent (color) first…")
        label_color()
    best = train_yolo(
        epochs=int(args.epochs),
        model=str(args.model),
        imgsz=int(args.imgsz),
    )
    print(f"TrainAgent: best weights -> {best}")
    return 0 if best.is_file() else 1


def cmd_run(args: argparse.Namespace) -> int:
    """Capture long burst, label, train — one shot pipeline."""
    ensure_dirs()
    log = _agent_log("SupervisorAgent")
    log.info(
        "run: capture %.0fs source=%s then label+train epochs=%s",
        args.seconds,
        args.source,
        args.epochs,
    )
    rc = cmd_capture(args)
    if rc != 0:
        return rc
    rc = cmd_label(argparse.Namespace(max_per_cam=0, method="color", stride=3))
    if rc != 0:
        print("WARN: color label found little arm — check lighting / blue joints visible.")
    return cmd_train(
        argparse.Namespace(
            epochs=args.epochs, model=args.model, imgsz=args.imgsz
        )
    )


def cmd_status(args: argparse.Namespace) -> int:
    ensure_dirs()
    c0 = len(list((CAPTURES_DIR / "cam0").glob("*.png")))
    c1 = len(list((CAPTURES_DIR / "cam1").glob("*.png")))
    keeps = len(list(KEEPS_DIR.glob("*.png")))
    labels = len(list(LABELS_DIR.glob("*.txt")))
    cfg = load_config()
    print("=== arm CV status ===")
    print(f"captures cam0={c0} cam1={c1}  keeps={keeps} labels={labels}")
    print(f"weights: {cfg.get('weights', '(none)')}")
    print(f"config: {CONFIG_PATH}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robot arm capture + YOLO train (moving)")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="CaptureAgent: many frames from cam0+cam1")
    c.add_argument("--seconds", type=float, default=180.0)
    c.add_argument("--interval", type=float, default=0.25, help="Seconds between saves/cam")
    c.add_argument("--source", choices=("ros", "usb"), default="ros")
    c.set_defaults(func=cmd_capture)

    l = sub.add_parser("label", help="LabelAgent: color (default) or motion -> YOLO labels")
    l.add_argument("--max-per-cam", type=int, default=0, help="0=all captures")
    l.add_argument(
        "--method",
        choices=("color", "motion"),
        default="color",
        help="color=UR blue joints (works when still); motion=needs moving arm",
    )
    l.add_argument("--stride", type=int, default=3, help="Label every Nth frame (color)")
    l.set_defaults(func=cmd_label)

    t = sub.add_parser("train", help="TrainAgent: YOLOv8n on labeled keeps")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--model", type=str, default="yolov8n.pt")
    t.add_argument("--imgsz", type=int, default=640)
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("run", help="Capture + label + train")
    r.add_argument("--seconds", type=float, default=180.0)
    r.add_argument("--interval", type=float, default=0.25)
    r.add_argument("--source", choices=("ros", "usb"), default="ros")
    r.add_argument("--epochs", type=int, default=40)
    r.add_argument("--model", type=str, default="yolov8n.pt")
    r.add_argument("--imgsz", type=int, default=640)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="Counts + weights path")
    s.set_defaults(func=cmd_status)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    mp.freeze_support()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
