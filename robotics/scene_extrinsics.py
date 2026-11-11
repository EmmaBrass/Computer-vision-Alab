"""ArUco-free dual-camera extrinsics from shared scene + purple tape.

Pipeline:
  1. Match ORB features across cam0/cam1 RGB (static structure).
  2. Lift matches with Orbbec depth → 3D point pairs.
  3. RANSAC + Umeyama rigid solve: X1 = R @ X0 + t.
  4. Purple tape centroid = shared origin; long axis locks yaw.

    python scene_extrinsics_calib.py status
    python scene_extrinsics_calib.py solve --live
    python scene_extrinsics_calib.py solve --color0 a.jpg --depth0 a.npy ...
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from cell_frames import (
    CELL_FRAMES_PATH,
    CellFrames,
    apply_scene_stereo_to_cell_frames,
    load_cell_frames,
    save_cell_frames,
)
from cam_geometry import (
    EXTRINSICS_PATH,
    Intrinsics,
    default_intrinsics,
    pixel_to_cam,
)

ROOT = Path(__file__).resolve().parent

# Magenta / purple tape on the pedestal (same band as dual_view baby_blue_draw).
DEFAULT_PURPLE_HSV_LO = (150, 45, 80)
DEFAULT_PURPLE_HSV_HI = (168, 255, 255)

# Soft blue arm mask — exclude from ORB so moving links do not poison the solve.
DEFAULT_BLUE_HSV_LO = (90, 40, 40)
DEFAULT_BLUE_HSV_HI = (130, 255, 255)

DEFAULT_RANSAC_ITERS = 400
DEFAULT_INLIER_MM = 35.0
DEFAULT_MIN_INLIERS = 12
DEFAULT_MAX_DEPTH_MM = 3500.0
DEFAULT_MIN_DEPTH_MM = 80.0
TAPE_AXIS_WEIGHT = 8  # synthetic point pairs along tape axis (orientation prior)
TAPE_CENTER_WEIGHT = 12  # repeated centroid pairs


@dataclass
class PurpleTapeObservation:
    """Centroid + long-axis direction for the purple tape strip."""

    box: tuple[int, int, int, int]  # x,y,w,h
    centroid_uv: tuple[float, float]
    axis_uv: tuple[float, float]  # unit direction in image (long axis)
    length_px: float
    area: int
    p_cam_mm: Optional[np.ndarray] = None  # 3D centroid when depth ok
    axis_cam: Optional[np.ndarray] = None  # 3D unit direction when depth ok


@dataclass
class SceneExtrinsicsResult:
    R: np.ndarray
    t_mm: np.ndarray
    n_matches: int
    n_inliers: int
    inlier_rmse_mm: float
    tape0: Optional[PurpleTapeObservation] = None
    tape1: Optional[PurpleTapeObservation] = None
    used_tape_axis: bool = False
    notes: str = ""


def purple_tape_mask(
    bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int] = DEFAULT_PURPLE_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_PURPLE_HSV_HI,
) -> np.ndarray:
    """Mask for the purple tape strip (shared cell origin)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)
    )
    b, g, r = cv2.split(bgr)
    bi = b.astype(np.int16)
    gi = g.astype(np.int16)
    ri = r.astype(np.int16)
    # Magenta: green darkest; reject strong pure-red edges.
    magenta = (gi < bi) & (gi < ri) & ((ri - bi) < 90)
    mask = cv2.bitwise_and(mask, magenta.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def blue_arm_mask(
    bgr: np.ndarray,
    *,
    hsv_lo: tuple[int, int, int] = DEFAULT_BLUE_HSV_LO,
    hsv_hi: tuple[int, int, int] = DEFAULT_BLUE_HSV_HI,
) -> np.ndarray:
    """Coarse mask of blue arm links to exclude from scene feature matching."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8)
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.dilate(mask, k, iterations=2)


def find_purple_tape(
    bgr: np.ndarray,
    *,
    depth_mm: Optional[np.ndarray] = None,
    K: Optional[Intrinsics] = None,
    min_area: int = 20,
    near_xy: Optional[tuple[float, float]] = None,
) -> Optional[PurpleTapeObservation]:
    """Largest purple-tape blob; centroid + long axis (minAreaRect / PCA)."""
    mask = purple_tape_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    fh, fw = bgr.shape[:2]
    best = None
    best_score = -1.0
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + 0.5 * w, y + 0.5 * h
        score = area
        if near_xy is not None:
            score -= 0.05 * ((cx - near_xy[0]) ** 2 + (cy - near_xy[1]) ** 2)
        else:
            # Prefer lower half (pedestal) when no prior.
            if cy < 0.35 * fh:
                score *= 0.35
        if score > best_score:
            best_score = score
            best = (c, (x, y, w, h), (cx, cy), area)
    if best is None:
        return None
    c, box, (cx, cy), area = best
    rect = cv2.minAreaRect(c)
    (rw, rh) = rect[1]
    angle = rect[2]
    if rw < rh:
        length_px = float(rh)
        # OpenCV: angle of the width edge; long axis is +90 when h>w.
        ang_rad = math.radians(angle + 90.0)
    else:
        length_px = float(rw)
        ang_rad = math.radians(angle)
    axis_uv = (math.cos(ang_rad), math.sin(ang_rad))
    # PCA fallback when the rect is nearly square.
    if max(rw, rh) < 1.25 * max(1.0, min(rw, rh)):
        pts = c.reshape(-1, 2).astype(np.float64)
        mean, eigenvectors = cv2.PCACompute(pts, mean=None, maxComponents=2)
        axis_uv = (float(eigenvectors[0, 0]), float(eigenvectors[0, 1]))
        length_px = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        _ = mean

    obs = PurpleTapeObservation(
        box=box,
        centroid_uv=(float(cx), float(cy)),
        axis_uv=axis_uv,
        length_px=max(1.0, length_px),
        area=int(area),
    )
    if depth_mm is not None and K is not None:
        z = _median_depth_at(depth_mm, cx, cy, radius=4)
        if z is not None:
            obs.p_cam_mm = np.asarray(
                pixel_to_cam(cx, cy, z, K), dtype=np.float64
            )
            # Two points along image long axis → 3D direction.
            half = 0.35 * obs.length_px
            u0 = cx - half * axis_uv[0]
            v0 = cy - half * axis_uv[1]
            u1 = cx + half * axis_uv[0]
            v1 = cy + half * axis_uv[1]
            z0 = _median_depth_at(depth_mm, u0, v0, radius=3) or z
            z1 = _median_depth_at(depth_mm, u1, v1, radius=3) or z
            p_a = np.asarray(pixel_to_cam(u0, v0, z0, K), dtype=np.float64)
            p_b = np.asarray(pixel_to_cam(u1, v1, z1, K), dtype=np.float64)
            d = p_b - p_a
            n = float(np.linalg.norm(d))
            if n > 5.0:
                obs.axis_cam = d / n
    return obs


def _median_depth_at(
    depth_mm: np.ndarray, u: float, v: float, *, radius: int = 3
) -> Optional[float]:
    h, w = depth_mm.shape[:2]
    x0 = max(0, int(round(u)) - radius)
    x1 = min(w, int(round(u)) + radius + 1)
    y0 = max(0, int(round(v)) - radius)
    y1 = min(h, int(round(v)) + radius + 1)
    patch = depth_mm[y0:y1, x0:x1].astype(np.float64)
    valid = patch[(patch > DEFAULT_MIN_DEPTH_MM) & (patch < DEFAULT_MAX_DEPTH_MM)]
    if valid.size < 3:
        return None
    return float(np.median(valid))


def unproject_depth(
    u: float,
    v: float,
    depth_mm: np.ndarray,
    K: Intrinsics,
) -> Optional[np.ndarray]:
    z = _median_depth_at(depth_mm, u, v, radius=2)
    if z is None:
        return None
    return np.asarray(pixel_to_cam(u, v, z, K), dtype=np.float64)


def match_scene_points(
    color0: np.ndarray,
    color1: np.ndarray,
    depth0: np.ndarray,
    depth1: np.ndarray,
    K0: Intrinsics,
    K1: Intrinsics,
    *,
    mask_arm: bool = True,
    max_points: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    """ORB match + depth lift → (N,3) point clouds in each camera frame."""
    gray0 = cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
    mask0 = None
    mask1 = None
    if mask_arm:
        # Keep features where arm is NOT (invert blue mask).
        m0 = blue_arm_mask(color0)
        m1 = blue_arm_mask(color1)
        mask0 = cv2.bitwise_not(m0)
        mask1 = cv2.bitwise_not(m1)
        # Also drop invalid depth.
        bad0 = ((depth0 <= DEFAULT_MIN_DEPTH_MM) | (depth0 >= DEFAULT_MAX_DEPTH_MM)).astype(
            np.uint8
        ) * 255
        bad1 = ((depth1 <= DEFAULT_MIN_DEPTH_MM) | (depth1 >= DEFAULT_MAX_DEPTH_MM)).astype(
            np.uint8
        ) * 255
        if bad0.shape[:2] == mask0.shape[:2]:
            mask0 = cv2.bitwise_and(mask0, cv2.bitwise_not(bad0))
        if bad1.shape[:2] == mask1.shape[:2]:
            mask1 = cv2.bitwise_and(mask1, cv2.bitwise_not(bad1))

    orb = cv2.ORB_create(nfeatures=max(1000, max_points * 2))
    kp0, des0 = orb.detectAndCompute(gray0, mask0)
    kp1, des1 = orb.detectAndCompute(gray1, mask1)
    if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
        return np.zeros((0, 3)), np.zeros((0, 3))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des0, des1, k=2)
    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good.sort(key=lambda m: m.distance)
    good = good[:max_points]

    pts0: list[np.ndarray] = []
    pts1: list[np.ndarray] = []
    for m in good:
        u0, v0 = kp0[m.queryIdx].pt
        u1, v1 = kp1[m.trainIdx].pt
        p0 = unproject_depth(u0, v0, depth0, K0)
        p1 = unproject_depth(u1, v1, depth1, K1)
        if p0 is None or p1 is None:
            continue
        pts0.append(p0)
        pts1.append(p1)
    if not pts0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.asarray(pts0, dtype=np.float64), np.asarray(pts1, dtype=np.float64)


def umeyama_rigid(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (no scale) Umeyama: dst ≈ R @ src + t. src/dst shaped (N,3)."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 3:
        raise ValueError("need >=3 point pairs")
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    H = (sc.T @ dc) / float(src.shape[0])
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt = Vt.copy()
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def ransac_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    iters: int = DEFAULT_RANSAC_ITERS,
    inlier_mm: float = DEFAULT_INLIER_MM,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return R, t, inlier_mask, rmse_mm."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    n = src.shape[0]
    if n < 3:
        raise ValueError("need >=3 point pairs for RANSAC")
    rng = rng or np.random.default_rng(0)
    best_inl = None
    best_count = -1
    best_R = np.eye(3)
    best_t = np.zeros(3)
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        try:
            R, t = umeyama_rigid(src[idx], dst[idx])
        except Exception:
            continue
        pred = (R @ src.T).T + t
        err = np.linalg.norm(pred - dst, axis=1)
        inl = err <= inlier_mm
        count = int(inl.sum())
        if count > best_count:
            best_count = count
            best_inl = inl
            best_R, best_t = R, t
    if best_inl is None or best_count < min(min_inliers, max(3, n // 5)):
        # Fallback: all points.
        R, t = umeyama_rigid(src, dst)
        pred = (R @ src.T).T + t
        err = np.linalg.norm(pred - dst, axis=1)
        inl = err <= inlier_mm * 1.5
        if int(inl.sum()) >= 3:
            R, t = umeyama_rigid(src[inl], dst[inl])
            pred = (R @ src.T).T + t
            err = np.linalg.norm(pred - dst, axis=1)
            inl = err <= inlier_mm * 1.5
        rmse = float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("inf")
        return R, t, inl, rmse
    R, t = umeyama_rigid(src[best_inl], dst[best_inl])
    pred = (R @ src.T).T + t
    err = np.linalg.norm(pred - dst, axis=1)
    inl = err <= inlier_mm
    if int(inl.sum()) >= 3:
        R, t = umeyama_rigid(src[inl], dst[inl])
        pred = (R @ src.T).T + t
        err = np.linalg.norm(pred - dst, axis=1)
        inl = err <= inlier_mm
    rmse = float(np.sqrt(np.mean(err[inl] ** 2))) if inl.any() else float("inf")
    return R, t, inl, rmse


def _append_weighted(
    src_list: list[np.ndarray],
    dst_list: list[np.ndarray],
    p0: np.ndarray,
    p1: np.ndarray,
    weight: int,
) -> None:
    for _ in range(max(1, int(weight))):
        src_list.append(p0)
        dst_list.append(p1)


def align_yaw_with_tape_axis(
    R: np.ndarray,
    axis0: np.ndarray,
    axis1: np.ndarray,
) -> np.ndarray:
    """Rotate R about its translation-free yaw so R@axis0 ≈ ±axis1 (best sign)."""
    a0 = np.asarray(axis0, dtype=np.float64).reshape(3)
    a1 = np.asarray(axis1, dtype=np.float64).reshape(3)
    a0 = a0 / (np.linalg.norm(a0) + 1e-9)
    a1 = a1 / (np.linalg.norm(a1) + 1e-9)
    v = R @ a0
    # Choose sign of axis1 (tape has 180° ambiguity).
    if float(np.dot(v, a1)) < float(np.dot(v, -a1)):
        a1 = -a1
    # Rotation that takes v → a1 (Rodrigues).
    cross = np.cross(v, a1)
    s = float(np.linalg.norm(cross))
    c = float(np.clip(np.dot(v, a1), -1.0, 1.0))
    if s < 1e-8:
        return R
    k = cross / s
    Kx = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )
    R_delta = np.eye(3) + Kx + Kx @ Kx * ((1.0 - c) / (s * s))
    return R_delta @ R


def solve_scene_extrinsics(
    color0: np.ndarray,
    color1: np.ndarray,
    depth0: np.ndarray,
    depth1: np.ndarray,
    *,
    K0: Optional[Intrinsics] = None,
    K1: Optional[Intrinsics] = None,
    mask_arm: bool = True,
    use_tape: bool = True,
    ransac_iters: int = DEFAULT_RANSAC_ITERS,
    inlier_mm: float = DEFAULT_INLIER_MM,
    min_inliers: int = DEFAULT_MIN_INLIERS,
) -> SceneExtrinsicsResult:
    """Solve X_cam1 = R @ X_cam0 + t from shared scene + optional purple tape."""
    h0, w0 = color0.shape[:2]
    h1, w1 = color1.shape[:2]
    K0 = K0 or default_intrinsics(w0, h0)
    K1 = K1 or default_intrinsics(w1, h1)

    src, dst = match_scene_points(
        color0, color1, depth0, depth1, K0, K1, mask_arm=mask_arm
    )
    tape0 = tape1 = None
    used_axis = False
    if use_tape:
        tape0 = find_purple_tape(color0, depth_mm=depth0, K=K0)
        tape1 = find_purple_tape(color1, depth_mm=depth1, K=K1)

    src_list = [p for p in src]
    dst_list = [p for p in dst]
    if (
        tape0 is not None
        and tape1 is not None
        and tape0.p_cam_mm is not None
        and tape1.p_cam_mm is not None
    ):
        _append_weighted(
            src_list, dst_list, tape0.p_cam_mm, tape1.p_cam_mm, TAPE_CENTER_WEIGHT
        )
        if tape0.axis_cam is not None and tape1.axis_cam is not None:
            # Two synthetic points along ±axis at ~40 mm for orientation prior.
            for sign in (-1.0, 1.0):
                p0 = tape0.p_cam_mm + sign * 40.0 * tape0.axis_cam
                p1 = tape1.p_cam_mm + sign * 40.0 * tape1.axis_cam
                # Ambiguity: try both axis1 signs later via align_yaw; here use as-is.
                _append_weighted(src_list, dst_list, p0, p1, TAPE_AXIS_WEIGHT // 2)
            used_axis = True

    if len(src_list) < 3:
        raise RuntimeError(
            f"Need >=3 depth-lifted correspondences; got {len(src_list)}. "
            "Ensure both cameras see static structure (and purple tape if enabled) "
            "with valid depth; park or mask the blue arm."
        )

    src_a = np.asarray(src_list, dtype=np.float64)
    dst_a = np.asarray(dst_list, dtype=np.float64)
    R, t, inl, rmse = ransac_umeyama(
        src_a,
        dst_a,
        iters=ransac_iters,
        inlier_mm=inlier_mm,
        min_inliers=min_inliers,
    )
    if (
        used_axis
        and tape0 is not None
        and tape1 is not None
        and tape0.axis_cam is not None
        and tape1.axis_cam is not None
    ):
        R = align_yaw_with_tape_axis(R, tape0.axis_cam, tape1.axis_cam)
        # Recompute t from inliers with refined R (fix translation only).
        if inl.any():
            src_i = src_a[inl]
            dst_i = dst_a[inl]
            t = dst_i.mean(axis=0) - (R @ src_i.mean(axis=0))

    notes = (
        f"scene ORB+depth Umeyama; matches={src.shape[0]} inliers={int(inl.sum())} "
        f"rmse={rmse:.1f}mm tape_axis={used_axis}"
    )
    return SceneExtrinsicsResult(
        R=R,
        t_mm=t,
        n_matches=int(src.shape[0]),
        n_inliers=int(inl.sum()),
        inlier_rmse_mm=rmse,
        tape0=tape0,
        tape1=tape1,
        used_tape_axis=used_axis,
        notes=notes,
    )


def write_stereo_extrinsics(
    result: SceneExtrinsicsResult,
    *,
    K0: Intrinsics,
    K1: Intrinsics,
    path: Path | None = None,
) -> Path:
    path = path or EXTRINSICS_PATH
    payload = {
        "from_camera": 0,
        "to_camera": 1,
        "R": result.R.tolist(),
        "t_mm": result.t_mm.reshape(3).tolist(),
        "K0": [K0.fx, K0.fy, K0.cx, K0.cy],
        "K1": [K1.fx, K1.fy, K1.cx, K1.cy],
        "notes": (
            f"scene_tape calib (no ArUco). {result.notes}. "
            f"Updated {time.strftime('%Y-%m-%dT%H:%M:%S')}"
        ),
        "n_matches": result.n_matches,
        "n_inliers": result.n_inliers,
        "inlier_rmse_mm": result.inlier_rmse_mm,
        "used_tape_axis": result.used_tape_axis,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def apply_result_to_cell_frames(
    result: SceneExtrinsicsResult,
    *,
    path: Path | None = None,
) -> CellFrames:
    cf = load_cell_frames(path)
    origin0 = result.tape0.p_cam_mm if result.tape0 is not None else None
    origin1 = result.tape1.p_cam_mm if result.tape1 is not None else None
    apply_scene_stereo_to_cell_frames(
        cf,
        result.R,
        result.t_mm,
        origin_cam0_mm=origin0,
        origin_cam1_mm=origin1,
    )
    save_cell_frames(cf)
    return cf


def draw_debug(
    color0: np.ndarray,
    color1: np.ndarray,
    result: SceneExtrinsicsResult,
) -> np.ndarray:
    """Side-by-side overlay: tape box/axis + status text."""
    a = color0.copy()
    b = color1.copy()
    for img, tape, label in ((a, result.tape0, "cam0"), (b, result.tape1, "cam1")):
        if tape is None:
            cv2.putText(
                img,
                f"{label}: no purple tape",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            continue
        x, y, w, h = tape.box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cx, cy = tape.centroid_uv
        cv2.circle(img, (int(cx), int(cy)), 5, (255, 255, 255), -1)
        ax, ay = tape.axis_uv
        half = 0.5 * tape.length_px
        p0 = (int(cx - half * ax), int(cy - half * ay))
        p1 = (int(cx + half * ax), int(cy + half * ay))
        cv2.line(img, p0, p1, (0, 255, 255), 2)
        cv2.putText(
            img,
            f"{label} tape L={tape.length_px:.0f}px",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2,
        )
    # Match heights for hstack.
    h = max(a.shape[0], b.shape[0])
    if a.shape[0] != h:
        a = cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    if b.shape[0] != h:
        b = cv2.copyMakeBorder(b, 0, h - b.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    panel = np.hstack([a, b])
    cv2.putText(
        panel,
        f"inliers={result.n_inliers}/{result.n_matches} rmse={result.inlier_rmse_mm:.1f}mm "
        f"|t|={np.linalg.norm(result.t_mm):.0f}mm",
        (12, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    return panel
