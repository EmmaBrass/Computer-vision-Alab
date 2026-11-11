"""Overhead lab scan from the two cameras that face each other (cam1, cam2).

cam0 is the side view and is not used. Side faces (surfaces that do not
point back at a downward-looking camera) are dropped. The two clouds are
locked by the uneven 9x6 checkerboard on the bench: the same corners in
both pictures, read with each camera's own depth. Nothing is filled in
where a camera has no depth.

cell_frames.json is not used. cam2's saved translation is still the origin.

    cd D:\\Jessica\\Jessica\\ros2_ws; . .\\_ros_env.ps1; Enter-Ros2Base; Enter-WorkspaceOverlay; python ..\\robotics\\scan_lab_cloud.py
"""

from __future__ import annotations

import json
import math
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ROS_WS = ROOT.parent / "ros2_ws"
for p in (
    ROS_WS / "install" / "Lib" / "site-packages",
    ROS_WS / "src" / "orbbec_dual_sync",
    ROOT,
):
    s = str(p)
    if p.is_dir() and s not in sys.path:
        sys.path.insert(0, s)

CAMS = ("cam1", "cam2")
CAM_INDEX = {"cam1": 1, "cam2": 2}
# Both mounts look down. 50° from horizontal is only the old tops guess.
TILT_FROM_HORIZONTAL_DEG = 50.0
# Uneven checkerboard: 9 by 6 inner corners. The long side makes the corner
# order the same physical corner in both cameras.
BOARD_COLS = 9
BOARD_ROWS = 6
# A top faces the camera (which is above it) when the toward-camera normal
# is within this many degrees of the optical axis.
TOP_MAX_DEG = 42.0
# Distinct colors so the three clouds stay separable while extrinsics are wrong.
CAM_RGB = {
    "cam0": (220, 70, 70),
    "cam1": (70, 180, 90),
    "cam2": (70, 140, 230),
}
# Orbbec millimeters. Drop zeros and the far noise floor. Nothing is invented.
Z_MIN_MM = 80.0
Z_MAX_MM = 4500.0
STRIDE = 2
OUT_DIR = ROOT / "lab_scan"


def _matvec_rows(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """p_base = R @ p_cam + t, without np.matmul / @ / np.dot."""
    P = np.asarray(pts, dtype=np.float64)
    A = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tt = np.asarray(t, dtype=np.float64).reshape(3)
    out = np.empty_like(P)
    for i in range(3):
        out[:, i] = (
            float(A[i, 0]) * P[:, 0]
            + float(A[i, 1]) * P[:, 1]
            + float(A[i, 2]) * P[:, 2]
            + float(tt[i])
        )
    return out


def depth_intrinsics(info: dict, depth_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    """Scale color-camera K into the depth grid. Not a depth-to-color align."""
    dh, dw = int(depth_hw[0]), int(depth_hw[1])
    ch = int(info.get("height") or 0)
    cw = int(info.get("width") or 0)
    fx, fy = float(info["fx"]), float(info["fy"])
    cx, cy = float(info["cx"]), float(info["cy"])
    if ch > 0 and cw > 0 and (ch, cw) != (dh, dw):
        sx, sy = dw / float(cw), dh / float(ch)
        return fx * sx, fy * sy, cx * sx, cy * sy
    return fx, fy, cx, cy


def backproject(
    depth_mm: np.ndarray, info: dict, stride: int, color: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Camera-frame XYZ millimeters. Drops invalid and zero depth.

    Returns photo RGB only when color and depth are the same size, which is
    this stack's signal that the frames share a pixel grid. Different sizes
    stay uncolored.
    """
    z_img = np.asarray(depth_mm, dtype=np.float64)
    if z_img.ndim != 2:
        raise ValueError(f"depth must be HxW, got {z_img.shape}")
    dh, dw = z_img.shape
    fx, fy, cx, cy = depth_intrinsics(info, (dh, dw))
    if fx <= 1.0 or fy <= 1.0:
        raise ValueError(f"bad intrinsics fx={fx} fy={fy}")
    vs = np.arange(0, dh, stride, dtype=np.int32)
    us = np.arange(0, dw, stride, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    z = z_img[vv, uu]
    ok = np.isfinite(z) & (z >= Z_MIN_MM) & (z <= Z_MAX_MM)
    uu_i = uu[ok].astype(np.int32)
    vv_i = vv[ok].astype(np.int32)
    z = z[ok]
    x = (uu_i.astype(np.float64) - cx) * z / fx
    y = (vv_i.astype(np.float64) - cy) * z / fy
    rgb = None
    if color is not None and getattr(color, "ndim", 0) == 3 and color.shape[:2] == z_img.shape[:2]:
        bgr = np.asarray(color)[vv_i, uu_i]
        rgb = np.stack([bgr[:, 2], bgr[:, 1], bgr[:, 0]], axis=1).astype(np.uint8)
    return np.stack([x, y, z], axis=1), rgb


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(xyz.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"comment {comment}\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    rec = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("r", "u1"),
            ("g", "u1"),
            ("b", "u1"),
        ],
    )
    rec["x"] = xyz[:, 0]
    rec["y"] = xyz[:, 1]
    rec["z"] = xyz[:, 2]
    rec["r"] = rgb[:, 0]
    rec["g"] = rgb[:, 1]
    rec["b"] = rgb[:, 2]
    with path.open("wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def _stamp_ns(msg) -> int:
    st = msg.header.stamp
    return int(st.sec) * 1_000_000_000 + int(st.nanosec)


def grab_usb() -> dict:
    """One color+depth frame per camera index. Used when ROS topics are down."""
    import depthcamera as dc

    got: dict[str, dict] = {}
    for cam in CAMS:
        device = dc.OrbbecDepthCamera(camera_index=CAM_INDEX[cam])
        try:
            device.open(require_frame=True, retries=2, warm_tries=30)
            color, depth = device.read_images()
            if color is None or depth is None:
                raise RuntimeError(f"{cam}: USB opened but returned no frame")
            k = None
            if device.capture is not None:
                k = dc.intrinsics_from_capture(device.capture, int(color.shape[1]), int(color.shape[0]))
            if k is None:
                raise RuntimeError(f"{cam}: camera did not report intrinsics")
            fx, fy, cx, cy = k
            got[cam] = {
                "depth": depth,
                "color": color,
                "color_hw": (int(color.shape[0]), int(color.shape[1])),
                "depth_stamp_ns": 0,
                "info": {
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "width": int(color.shape[1]),
                    "height": int(color.shape[0]),
                },
                "source": "usb",
            }
        finally:
            device.close()
            time.sleep(0.4)
    return got


def grab(timeout_s: float = 8.0) -> dict:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    from orbbec_dual_sync.image_codec import imgmsg_to_numpy

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    got: dict[str, dict] = {c: {} for c in CAMS}
    rclpy.init()
    node = Node("lab_scan_once")

    def on_depth(cam):
        def cb(msg: Image) -> None:
            try:
                got[cam]["depth"] = imgmsg_to_numpy(msg)
                got[cam]["depth_stamp_ns"] = _stamp_ns(msg)
            except Exception as exc:
                got[cam]["depth_error"] = str(exc)

        return cb

    def on_color(cam):
        def cb(msg: Image) -> None:
            try:
                got[cam]["color"] = imgmsg_to_numpy(msg)
                got[cam]["color_hw"] = (int(msg.height), int(msg.width))
            except Exception as exc:
                got[cam]["color_error"] = str(exc)

        return cb

    def on_info(cam):
        def cb(msg: CameraInfo) -> None:
            k = list(msg.k)
            got[cam]["info"] = {
                "fx": float(k[0]),
                "fy": float(k[4]),
                "cx": float(k[2]),
                "cy": float(k[5]),
                "width": int(msg.width),
                "height": int(msg.height),
            }

        return cb

    for cam in CAMS:
        idx = CAM_INDEX[cam]
        node.create_subscription(Image, f"/orbbec/cam{idx}/depth/image_raw", on_depth(cam), qos)
        node.create_subscription(Image, f"/orbbec/cam{idx}/color/image_raw", on_color(cam), qos)
        node.create_subscription(CameraInfo, f"/orbbec/cam{idx}/color/camera_info", on_info(cam), qos)

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.05)
        if all("depth" in got[c] and "info" in got[c] for c in CAMS):
            break
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass
    missing = [c for c in CAMS if "depth" not in got[c] or "info" not in got[c]]
    if missing:
        raise TimeoutError(
            "No depth+camera_info from "
            + ", ".join(missing)
            + ". Are the Orbbec publishers running?"
        )
    return got


def _mul3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out[i, j] = (
                float(a[i, 0]) * float(b[0, j])
                + float(a[i, 1]) * float(b[1, j])
                + float(a[i, 2]) * float(b[2, j])
            )
    return out


def _det3(a: np.ndarray) -> float:
    return float(
        a[0, 0] * (a[1, 1] * a[2, 2] - a[1, 2] * a[2, 1])
        - a[0, 1] * (a[1, 0] * a[2, 2] - a[1, 2] * a[2, 0])
        + a[0, 2] * (a[1, 0] * a[2, 1] - a[1, 1] * a[2, 0])
    )


def _pose_R(tilt_deg: float, *, opposite: bool) -> np.ndarray:
    """Camera-to-world rotation. World Z is up. Opposite cameras look toward -Y."""
    th = math.radians(tilt_deg)
    c, s = math.cos(th), math.sin(th)
    pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]],
        dtype=np.float64,
    )
    if opposite:
        level = np.array(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
    else:
        level = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            dtype=np.float64,
        )
    return _mul3(level, pitch)


def upward_tops(
    depth_mm: np.ndarray, info: dict, stride: int, color: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Camera-frame points whose surface faces back at the camera.

    A downward-looking camera sees floor and benchtops that way. Vertical
    side faces do not, so they are dropped. Returns (xyz, rgb, n_valid).
    """
    z_img = np.asarray(depth_mm, dtype=np.float64)
    dh, dw = z_img.shape
    fx, fy, cx, cy = depth_intrinsics(info, (dh, dw))
    vs = np.arange(0, dh, stride, dtype=np.int32)
    us = np.arange(0, dw, stride, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    z = z_img[vv, uu]
    valid = np.isfinite(z) & (z >= Z_MIN_MM) & (z <= Z_MAX_MM)
    n_valid = int(np.count_nonzero(valid))
    x = (uu.astype(np.float64) - cx) * z / fx
    y = (vv.astype(np.float64) - cy) * z / fy
    xyz = np.stack([x, y, z], axis=2)
    xyz[~valid] = np.nan
    # right × down. +Z component points into the scene, away from the camera.
    right = xyz[:, 2:, :] - xyz[:, :-2, :]
    down = xyz[2:, 1:-1, :] - xyz[:-2, 1:-1, :]
    right = right[1:-1, :, :]
    nx = right[:, :, 1] * down[:, :, 2] - right[:, :, 2] * down[:, :, 1]
    ny = right[:, :, 2] * down[:, :, 0] - right[:, :, 0] * down[:, :, 2]
    nz = right[:, :, 0] * down[:, :, 1] - right[:, :, 1] * down[:, :, 0]
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    ok_n = np.isfinite(norm) & (norm > 1e-6)
    nz_unit = np.full(nz.shape, np.nan)
    nz_unit[ok_n] = nz[ok_n] / norm[ok_n]
    # Toward the camera is -Z. Tops seen from above are close to that axis.
    cos_lim = math.cos(math.radians(TOP_MAX_DEG))
    top = ok_n & (nz_unit > cos_lim)
    sel = np.zeros(valid.shape, dtype=bool)
    sel[1:-1, 1:-1] = top
    sel &= valid
    pts = xyz[sel]
    if color is not None and getattr(color, "ndim", 0) == 3 and color.shape[:2] == z_img.shape[:2]:
        bgr = np.asarray(color)[vv[sel], uu[sel]]
        rgb = np.stack([bgr[:, 2], bgr[:, 1], bgr[:, 0]], axis=1).astype(np.uint8)
    else:
        rgb = np.tile(np.array(CAM_RGB["cam1"], dtype=np.uint8), (pts.shape[0], 1))
    return pts, rgb, n_valid


def _centroid(pts: np.ndarray) -> np.ndarray:
    c = np.zeros(3, dtype=np.float64)
    n = float(pts.shape[0])
    for i in range(3):
        c[i] = float(np.sum(pts[:, i])) / n
    return c


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid map of src onto dst. 3x3 only, no large GEMM."""
    cs, cd = _centroid(src), _centroid(dst)
    hs = src - cs
    hd = dst - cd
    h = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            h[i, j] = float(np.sum(hs[:, i] * hd[:, j]))
    u, _s, vt = np.linalg.svd(h)
    v = vt.T
    r = _mul3(v, u.T)
    if _det3(r) < 0.0:
        v[:, 2] *= -1.0
        r = _mul3(v, u.T)
    t = cd - _matvec_rows(cs.reshape(1, 3), r, np.zeros(3)).reshape(3)
    return r, t


def _nn_gate(ref: np.ndarray, qry: np.ndarray, voxel: float, gate: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices (qry_i, ref_i) whose nearest ref point is inside ``gate`` mm."""
    buckets: dict[tuple[int, int, int], list[int]] = {}
    keys = np.floor(ref / voxel).astype(np.int32)
    for i in range(ref.shape[0]):
        key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        buckets.setdefault(key, []).append(i)
    qi: list[int] = []
    ri: list[int] = []
    qkeys = np.floor(qry / voxel).astype(np.int32)
    gate2 = gate * gate
    for i in range(qry.shape[0]):
        qx, qy, qz = int(qkeys[i, 0]), int(qkeys[i, 1]), int(qkeys[i, 2])
        best = gate2
        best_j = -1
        px, py, pz = float(qry[i, 0]), float(qry[i, 1]), float(qry[i, 2])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    hits = buckets.get((qx + dx, qy + dy, qz + dz))
                    if not hits:
                        continue
                    for j in hits:
                        ddx = px - float(ref[j, 0])
                        ddy = py - float(ref[j, 1])
                        ddz = pz - float(ref[j, 2])
                        d2 = ddx * ddx + ddy * ddy + ddz * ddz
                        if d2 < best:
                            best = d2
                            best_j = j
        if best_j >= 0:
            qi.append(i)
            ri.append(best_j)
    if not qi:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return np.asarray(qi, dtype=np.int32), np.asarray(ri, dtype=np.int32)


def _find_checkerboard(color: np.ndarray) -> np.ndarray | None:
    """Inner corners of the 9x6 board, in original pixels. None if it is not a real grid."""
    import cv2

    bgr = np.asarray(color)
    if bgr.ndim != 3:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pattern = (BOARD_COLS, BOARD_ROWS)

    def consider(corners, scale: float, origin: tuple[float, float]) -> None:
        nonlocal best, best_span
        if corners is None:
            return
        pts = corners.reshape(-1, 2).astype(np.float64) / scale
        if pts.shape[0] != BOARD_COLS * BOARD_ROWS:
            return
        pts[:, 0] += origin[0]
        pts[:, 1] += origin[1]
        if not _grid_is_even(pts):
            return
        short = min(
            float(np.max(pts[:, 0]) - np.min(pts[:, 0])),
            float(np.max(pts[:, 1]) - np.min(pts[:, 1])),
        )
        # Cam1 sees the board only about 20 px tall. Smaller than that is table noise.
        if short < 16.0:
            return
        span = _corner_span(pts)
        if span > best_span:
            best_span = span
            best = pts

    best = None
    best_span = 0.0
    for scale in (3.0, 4.0):
        up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ok, corners = cv2.findChessboardCornersSB(
            up, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if ok:
            consider(corners, scale, (0.0, 0.0))
    if best is not None:
        return best
    # The board is small in cam1, so a full-frame search misses it. Look again
    # in table-sized windows, enlarged.
    dh, dw = gray.shape
    for y0 in range(100, max(101, dh - 180), 40):
        for x0 in range(40, max(41, dw - 200), 40):
            crop = gray[y0 : y0 + 180, x0 : x0 + 200]
            if crop.shape[0] < 180 or crop.shape[1] < 200:
                continue
            up = cv2.resize(crop, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC)
            ok, corners = cv2.findChessboardCornersSB(
                up, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE
            )
            if ok:
                consider(corners, 8.0, (float(x0), float(y0)))
                if best is not None:
                    return best
    return best


def _corner_span(pts: np.ndarray) -> float:
    w = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
    h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
    return math.hypot(w, h)


def _grid_is_even(pts: np.ndarray) -> bool:
    """Neighboring corners of a real board are about the same distance apart."""
    g = pts.reshape(BOARD_ROWS, BOARD_COLS, 2)
    dists: list[float] = []
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS - 1):
            dx = float(g[r, c + 1, 0] - g[r, c, 0])
            dy = float(g[r, c + 1, 1] - g[r, c, 1])
            dists.append(math.hypot(dx, dy))
    med = float(np.median(np.asarray(dists, dtype=np.float64)))
    if med < 2.0:
        return False
    dev = [abs(d - med) for d in dists]
    return float(np.median(np.asarray(dev))) / med < 0.35


def _corners_to_xyz(depth_mm: np.ndarray, info: dict, corners: np.ndarray) -> np.ndarray:
    """Depth at each corner. Invalid corners stay NaN so the pair can be dropped."""
    z_img = np.asarray(depth_mm, dtype=np.float64)
    dh, dw = z_img.shape
    fx, fy, cx, cy = depth_intrinsics(info, (dh, dw))
    out = np.full((corners.shape[0], 3), np.nan, dtype=np.float64)
    for i in range(corners.shape[0]):
        u = float(corners[i, 0])
        v = float(corners[i, 1])
        ui = int(round(u))
        vi = int(round(v))
        y0, y1 = max(0, vi - 3), min(dh, vi + 4)
        x0, x1 = max(0, ui - 3), min(dw, ui + 4)
        patch = z_img[y0:y1, x0:x1]
        good = patch[np.isfinite(patch) & (patch >= Z_MIN_MM) & (patch <= Z_MAX_MM)]
        if good.size < 4:
            continue
        z = float(np.median(good))
        out[i, 0] = (u - cx) * z / fx
        out[i, 1] = (v - cy) * z / fy
        out[i, 2] = z
    return out


def _save_board_overlay(name: str, color: np.ndarray, corners: np.ndarray | None) -> None:
    import cv2

    vis = np.asarray(color).copy()
    if corners is not None:
        for i in range(corners.shape[0]):
            u = int(round(float(corners[i, 0])))
            v = int(round(float(corners[i, 1])))
            cv2.circle(vis, (u, v), 3, (0, 0, 255), -1)
            if i % BOARD_COLS == 0:
                cv2.circle(vis, (u, v), 6, (0, 255, 255), 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / f"board_lock_{name}.jpg"), vis)


def _flip180(pts: np.ndarray) -> np.ndarray:
    g = pts.reshape(BOARD_ROWS, BOARD_COLS, 2)
    return g[::-1, ::-1, :].reshape(-1, 2).copy()


def align_by_board(
    depth1: np.ndarray,
    info1: dict,
    color1: np.ndarray,
    depth2: np.ndarray,
    info2: dict,
    color2: np.ndarray,
) -> dict:
    """Put cam2 into cam1's camera frame using the shared checkerboard."""
    c1 = _find_checkerboard(color1)
    c2 = _find_checkerboard(color2)
    _save_board_overlay("cam1", color1, c1)
    _save_board_overlay("cam2", color2, c2)
    if c1 is None or c2 is None:
        missing = [name for name, pts in (("cam1", c1), ("cam2", c2)) if pts is None]
        raise RuntimeError(
            "Checkerboard not found in " + " and ".join(missing) + ". Leave the 9x6 board where both cameras see the whole pattern."
        )
    p1 = _corners_to_xyz(depth1, info1, c1)
    # Same corner index is the same square. Try the 180° flip and keep the tighter fit.
    best = None
    for label, corners in (("as_found", c2), ("flipped", _flip180(c2))):
        p2 = _corners_to_xyz(depth2, info2, corners)
        ok = np.isfinite(p1[:, 2]) & np.isfinite(p2[:, 2])
        n = int(np.count_nonzero(ok))
        if n < 15:
            continue
        r, t = _kabsch(p2[ok], p1[ok])
        moved = _matvec_rows(p2[ok], r, t)
        err = moved - p1[ok]
        mean_mm = float(np.mean(np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2 + err[:, 2] ** 2)))
        if best is None or mean_mm < best["mean_mm"]:
            best = {
                "R": r,
                "t": t,
                "mean_mm": mean_mm,
                "corners": n,
                "order": label,
                "span1_px": _corner_span(c1),
                "span2_px": _corner_span(corners),
            }
    if best is None:
        raise RuntimeError("Both cameras saw the board, but depth is missing on too many corners.")
    return best


def align_tops(cam1: np.ndarray, cam2: np.ndarray, tilt_deg: float) -> dict:
    """Park both center-rays on one point, then fit cam2 tops onto cam1 tops."""
    r1 = _pose_R(tilt_deg, opposite=False)
    r2 = _pose_R(tilt_deg, opposite=True)
    c1 = _centroid(cam1)
    c2 = _centroid(cam2)
    t1 = -_matvec_rows(c1.reshape(1, 3), r1, np.zeros(3)).reshape(3)
    t2 = -_matvec_rows(c2.reshape(1, 3), r2, np.zeros(3)).reshape(3)
    p1 = _matvec_rows(cam1, r1, t1)
    p2 = _matvec_rows(cam2, r2, t2)
    # Camera origins in this frame. Their separation is the starting baseline.
    origin_gap = t2 - t1
    baseline = math.sqrt(float(origin_gap[0] ** 2 + origin_gap[1] ** 2 + origin_gap[2] ** 2))
    step = max(1, p1.shape[0] // 3500)
    ref = p1[::step]
    moving = p2
    r_acc = np.eye(3, dtype=np.float64)
    t_acc = np.zeros(3, dtype=np.float64)
    mean_mm = float("nan")
    n_in = 0
    for _ in range(12):
        sample = moving[::step]
        qi, ri = _nn_gate(ref, sample, voxel=50.0, gate=180.0)
        n_in = int(qi.shape[0])
        if n_in < 80:
            break
        r, t = _kabsch(sample[qi], ref[ri])
        moving = _matvec_rows(moving, r, t)
        r_acc = _mul3(r, r_acc)
        t_acc = _matvec_rows(t_acc.reshape(1, 3), r, t).reshape(3)
        d = sample[qi]
        d = _matvec_rows(d, r, t)
        err = d - ref[ri]
        mean_mm = float(np.mean(np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2 + err[:, 2] ** 2)))
        if mean_mm < 25.0:
            break
    return {
        "cam1_world": p1,
        "cam2_world": moving,
        "tilt_deg": tilt_deg,
        "baseline_mm": baseline,
        "top_agreement_mm": mean_mm,
        "inliers": n_in,
    }


def write_viewer(
    path: Path,
    clouds: list[tuple[str, np.ndarray, np.ndarray]],
    note: str,
    default_show: str = "cam1",
) -> None:
    """Orbit viewer. Drag to turn, click a point to read its XYZ."""
    chunks = []
    for name, xyz, rgb in clouds:
        step = max(1, int(xyz.shape[0] / 12000))
        sel = xyz[::step]
        col = rgb[::step]
        chunks.append(
            {
                "name": name,
                "xyz": np.round(sel, 1).astype(np.float64).tolist(),
                "rgb": col.astype(np.int32).tolist(),
            }
        )
    payload = json.dumps(chunks)
    banner = note
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Lab scan</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #16181c; color: #e8eaed; font-family: Segoe UI, sans-serif; }}
  header {{ padding: 10px 14px 0; }}
  h1 {{ font-size: 18px; margin: 0; font-weight: 600; }}
  p {{ margin: 4px 0 0; color: #c4ccd6; font-size: 13px; }}
  .vis-row {{ display: flex; gap: 8px; margin-top: 8px; }}
  button {{ background: #2a3140; color: #e8eaed; border: 1px solid #445; border-radius: 6px; padding: 6px 12px; cursor: pointer; }}
  button.on {{ background: #3d5a40; border-color: #8c8; }}
  #wrap {{ position: absolute; inset: 108px 0 0 0; }}
  canvas {{ width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }}
</style>
</head>
<body>
<header>
  <h1>Overhead floor plan</h1>
  <p>{banner}</p>
  <div class="vis-row">
    <button id="b1" type="button" class="{"on" if default_show == "cam1" else ""}">Cam 1</button>
    <button id="b2" type="button" class="{"on" if default_show == "cam2" else ""}">Cam 2</button>
    <button id="both" type="button" class="{"on" if default_show == "both" else ""}">Both</button>
  </div>
  <p id="status">Looking straight down. Drag sideways to turn the plan, drag up to tilt, scroll to zoom.</p>
</header>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
const clouds = {payload};
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
let yaw = 0, pitch = 1.35, zoom = 1, drag = null, show = "{default_show}";
const all = [];
clouds.forEach(function (cl) {{
  cl.xyz.forEach(function (p, i) {{ all.push({{ p: p, rgb: cl.rgb[i], name: cl.name }}); }});
}});
function shown() {{
  if (show === "both") return all;
  const key = show === "cam1" ? "cam1" : "cam2";
  return all.filter(function (it) {{ return it.name.indexOf(key) === 0; }});
}}
function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}
function flat(p) {{
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x1 = cy * p[0] - sy * p[1];
  const y1 = sy * p[0] + cy * p[1];
  const horiz = x1;
  const vert = Math.sin(pitch) * y1 + Math.cos(pitch) * p[2];
  return [horiz, vert, p[2]];
}}
function frame(items) {{
  let minH = 1e9, maxH = -1e9, minV = 1e9, maxV = -1e9;
  items.forEach(function (it) {{
    const q = flat(it.p);
    if (q[0] < minH) minH = q[0];
    if (q[0] > maxH) maxH = q[0];
    if (q[1] < minV) minV = q[1];
    if (q[1] > maxV) maxV = q[1];
  }});
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const spanH = Math.max(1, maxH - minH), spanV = Math.max(1, maxV - minV);
  const scale = 0.88 * Math.min(w / spanH, h / spanV) * zoom;
  return {{ minH: minH, maxV: maxV, scale: scale, w: w, h: h }};
}}
function project(p, fr) {{
  const q = flat(p);
  return [
    0.06 * fr.w + (q[0] - fr.minH) * fr.scale,
    0.06 * fr.h + (fr.maxV - q[1]) * fr.scale,
    q[2]
  ];
}}
function draw() {{
  const items = shown();
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#16181c";
  ctx.fillRect(0, 0, w, h);
  if (!items.length) return;
  const fr = frame(items);
  const drawn = items.map(function (it) {{
    return {{ q: project(it.p, fr), it: it }};
  }});
  drawn.sort(function (a, b) {{ return a.q[2] - b.q[2]; }});
  const dot = Math.max(2, Math.min(7, fr.scale * 18));
  drawn.forEach(function (d) {{
    ctx.fillStyle = "rgb(" + d.it.rgb[0] + "," + d.it.rgb[1] + "," + d.it.rgb[2] + ")";
    ctx.fillRect(d.q[0], d.q[1], dot, dot);
  }});
}}
function setShow(which) {{
  show = which;
  ["b1", "b2", "both"].forEach(function (id) {{
    document.getElementById(id).className = "";
  }});
  document.getElementById(which === "cam1" ? "b1" : which === "cam2" ? "b2" : "both").className = "on";
  const label = which === "both" ? "Both cameras. Overlap is where they see the same tops." : (which === "cam1" ? "Cam 1 only, looking down." : "Cam 2 only, looking down.");
  status.textContent = label + " Drag sideways to turn, drag up to tilt.";
  draw();
}}
document.getElementById("b1").addEventListener("click", function () {{ setShow("cam1"); }});
document.getElementById("b2").addEventListener("click", function () {{ setShow("cam2"); }});
document.getElementById("both").addEventListener("click", function () {{ setShow("both"); }});
canvas.addEventListener("pointerdown", function (e) {{
  drag = {{ x: e.clientX, y: e.clientY, yaw: yaw, pitch: pitch, moved: false }};
  canvas.setPointerCapture(e.pointerId);
}});
canvas.addEventListener("pointermove", function (e) {{
  if (!drag) return;
  if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) drag.moved = true;
  yaw = drag.yaw + (e.clientX - drag.x) * 0.005;
  pitch = Math.max(0.15, Math.min(1.57, drag.pitch - (e.clientY - drag.y) * 0.005));
  draw();
}});
function end(e) {{
  if (drag && !drag.moved) pick(e.clientX, e.clientY);
  drag = null;
}}
canvas.addEventListener("pointerup", end);
canvas.addEventListener("pointercancel", function () {{ drag = null; }});
canvas.addEventListener("wheel", function (e) {{
  e.preventDefault();
  zoom = Math.max(0.4, Math.min(6, zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
  draw();
}}, {{ passive: false }});
function pick(cx, cy) {{
  const rect = canvas.getBoundingClientRect();
  const x = cx - rect.left, y = cy - rect.top;
  const items = shown();
  const fr = frame(items);
  let best = null, bestD = 18 * 18;
  items.forEach(function (it) {{
    const q = project(it.p, fr);
    const d = (q[0] - x) * (q[0] - x) + (q[1] - y) * (q[1] - y);
    if (d < bestD) {{ bestD = d; best = it; }}
  }});
  if (!best) return;
  const p = best.p;
  status.textContent = best.name + "  x " + p[0].toFixed(0) + "  y " + p[1].toFixed(0) + "  z " + p[2].toFixed(0) + " mm";
}}
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    source = "ros"
    try:
        frames = grab()
    except TimeoutError as exc:
        print(f"{exc} Falling back to direct USB, one camera at a time.")
        source = "usb"
        frames = grab_usb()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tops: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    report_cams: dict[str, dict] = {}
    for cam in CAMS:
        info = frames[cam]["info"]
        depth = frames[cam]["depth"]
        pts, rgb, n_valid = upward_tops(depth, info, STRIDE, frames[cam].get("color"))
        if cam == "cam2" and rgb.shape[0] and not (
            frames[cam].get("color") is not None
            and frames[cam]["color"].shape[:2] == depth.shape[:2]
        ):
            rgb[:] = CAM_RGB["cam2"]
        tops[cam] = (pts, rgb)
        report_cams[cam] = {
            "depth_hw": [int(depth.shape[0]), int(depth.shape[1])],
            "valid_depth_px": n_valid,
            "top_points": int(pts.shape[0]),
            "side_faces_dropped": int(n_valid - pts.shape[0]),
            "photo_colored": bool(
                frames[cam].get("color") is not None
                and np.asarray(frames[cam]["color"]).shape[:2] == depth.shape[:2]
            ),
        }
        print(
            f"{cam}: valid depth {n_valid}  tops kept {pts.shape[0]}  "
            f"(side faces dropped)"
        )

    board = align_by_board(
        frames["cam1"]["depth"],
        frames["cam1"]["info"],
        frames["cam1"]["color"],
        frames["cam2"]["depth"],
        frames["cam2"]["info"],
        frames["cam2"]["color"],
    )
    cam1_pts = tops["cam1"][0]
    cam2_pts = _matvec_rows(tops["cam2"][0], board["R"], board["t"])
    agree = board["mean_mm"]
    note = (
        f"Merged on the 9x6 checkerboard in cam1's frame. "
        f"The same corners agree within {agree:.0f} mm "
        f"({board['corners']} corners, {board['order']}). "
        f"Side faces and cam0 are not in this cloud."
    )
    print(note)
    viewer = [
        ("cam1 tops", cam1_pts, tops["cam1"][1]),
        ("cam2 tops", cam2_pts, tops["cam2"][1]),
    ]
    xyz_m = np.concatenate([cam1_pts, cam2_pts], axis=0)
    rgb_m = np.concatenate([tops["cam1"][1], tops["cam2"][1]], axis=0)
    merged_path = OUT_DIR / "overhead_tops.ply"
    write_ply(merged_path, xyz_m, rgb_m, "cam1+cam2 tops locked by the 9x6 checkerboard")
    view_path = OUT_DIR / "lab_scan.html"
    write_viewer(view_path, viewer, note, default_show="both")
    report = {
        "source": source,
        "cameras_used": list(CAMS),
        "dropped": "cam0 side view, and surfaces that do not face the camera",
        "alignment": "9x6 checkerboard corners, depth at each corner, cam2 moved into cam1",
        "board_agreement_mm": agree,
        "board_corners": board["corners"],
        "board_corner_order": board["order"],
        "board_span_px": {"cam1": board["span1_px"], "cam2": board["span2_px"]},
        "cameras": report_cams,
        "merged_ply": str(merged_path),
        "viewer": str(view_path),
        "points_merged": int(xyz_m.shape[0]),
    }
    (OUT_DIR / "overhead_tops.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"tops {xyz_m.shape[0]} points -> {merged_path}")
    webbrowser.open(view_path.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
