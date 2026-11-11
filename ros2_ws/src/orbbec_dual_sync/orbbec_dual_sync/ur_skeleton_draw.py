"""Draw UR FK stick figure, confirm vs vision blues, FK rack keep-out alerts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def _ensure_robotics(robotics: Optional[Path] = None) -> Optional[Path]:
    if robotics is not None and (robotics / "ur_skeleton.py").is_file():
        if str(robotics) not in sys.path:
            sys.path.insert(0, str(robotics))
        return robotics
    import os

    env = os.environ.get("ORBBEC_CAMERA_ROOT", "").strip()
    for root in (
        Path(env) if env else None,
        Path(__file__).resolve().parents[4] / "robotics",
        Path(__file__).resolve().parents[3] / "robotics",
    ):
        if root is not None and (root / "ur_skeleton.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    return None


def base_to_cam(p_base_mm: np.ndarray, pose) -> np.ndarray:
    """Inverse of cell_frames.cam_to_base: p_cam = R.T @ (p_base - t)."""
    p = np.asarray(p_base_mm, dtype=np.float64).reshape(3)
    R = np.asarray(pose.R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(pose.t_mm, dtype=np.float64).reshape(3)
    return R.T @ (p - t)


def project_cam_to_pixel(
    p_cam_mm: np.ndarray, fx: float, fy: float, cx: float, cy: float
) -> Optional[tuple[int, int]]:
    x, y, z = [float(v) for v in np.asarray(p_cam_mm, dtype=np.float64).reshape(3)]
    if z <= 1e-3:
        return None
    u = int(round(fx * x / z + cx))
    v = int(round(fy * y / z + cy))
    return u, v


def load_ur_state(robotics: Optional[Path] = None):
    root = _ensure_robotics(robotics)
    if root is None:
        return None
    from ur_skeleton import (  # noqa: WPS433
        fk_joint_positions_mm,
        fk_rack_proximity,
        load_fk_config,
        load_joints,
        teachpad_fk_residual_mm,
        teachpad_tcp_mm,
        ur_base_to_vision_mm,
        write_example_files,
    )

    write_example_files()
    cfg = load_fk_config()
    q, model, source = load_joints(cfg)
    pts_ur = fk_joint_positions_mm(q, model=model)
    pts_vis = ur_base_to_vision_mm(pts_ur, cfg)
    tcp = teachpad_tcp_mm(cfg)
    tcp_err = None
    if tcp is not None:
        try:
            tcp_err = float(teachpad_fk_residual_mm(q, tcp, model=model))
        except Exception:
            tcp_err = None
    return {
        "cfg": cfg,
        "q": q,
        "model": model,
        "source": source,
        "pts_vision_mm": pts_vis,
        "pts_ur_mm": pts_ur,
        "teachpad_tcp_mm": tcp,
        "teachpad_tcp_err_mm": tcp_err,
        "rack_hits": fk_rack_proximity(pts_vis, cfg),
    }


def confirm_vision_vs_fk(
    views: list,
    pts_vision_mm: np.ndarray,
    *,
    gate_mm: float = 80.0,
) -> list[tuple[int, int, float]]:
    """Match vision TrackView (with p_base) to nearest FK joint.

    Returns list of (track_id, fk_index, dist_mm) for matches within gate.
    """
    matches: list[tuple[int, int, float]] = []
    pts = np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 3)
    for v in views:
        pb = getattr(v, "p_base", None)
        if pb is None:
            continue
        p = np.asarray(pb, dtype=np.float64).reshape(3)
        # Skip base FK index 0 for moving joints; allow for is_base tracks.
        start = 0 if getattr(v, "is_base", False) else 1
        best_i, best_d = -1, 1e9
        for i in range(start, pts.shape[0]):
            d = float(np.linalg.norm(p - pts[i]))
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= float(gate_mm):
            matches.append((int(v.track_id), best_i, best_d))
    return matches


def draw_ur_skeleton(
    bgr: np.ndarray,
    pts_vision_mm: np.ndarray,
    pose,
    *,
    fx: float = 520.0,
    matches: Optional[list[tuple[int, int, float]]] = None,
    rack_hits: Optional[list[tuple[int, float]]] = None,
    source: str = "",
    model: str = "",
    pts_upright_mm: Optional[np.ndarray] = None,
    check_ok: Optional[bool] = None,
    check_mm: Optional[float] = None,
    check_unit: str = "mm",
    base_anchor_uv: Optional[tuple[float, float]] = None,
    img_deg: float = 0.0,
    img_scale: float = 1.0,
    blue_uv: Optional[np.ndarray] = None,
    has_tool: bool = False,
    tag_uv: Optional[list] = None,
    pts_fk_mm: Optional[np.ndarray] = None,
    cam: str = "",
    calibrated: bool = False,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    K=None,
    dist=None,
    teachpad_tcp_err_mm: Optional[float] = None,
) -> np.ndarray:
    """Project FK joints into the camera image and draw a stick figure."""
    out = bgr
    h, w = out.shape[:2]
    fy = float(fx)
    cx, cy = 0.5 * w, 0.5 * h
    pts = np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 3)
    pix: list[Optional[tuple[int, int]]] = []
    for p in pts:
        p_cam = base_to_cam(p, pose)
        pix.append(project_cam_to_pixel(p_cam, fx, fy, cx, cy))

    # Bones between consecutive joints
    for i in range(len(pix) - 1):
        a, b = pix[i], pix[i + 1]
        if a is None or b is None:
            continue
        if not (0 <= a[0] < w and 0 <= a[1] < h) and not (
            0 <= b[0] < w and 0 <= b[1] < h
        ):
            # still draw if either end is near frame
            pass
        color = (80, 220, 80)
        if rack_hits and any(hi == i + 1 for hi, _ in rack_hits):
            color = (0, 0, 255)
        cv2.line(out, a, b, color, 2, cv2.LINE_AA)

    agree_fk = {m[1] for m in (matches or [])}
    for i, uv in enumerate(pix):
        if uv is None:
            continue
        u, v = uv
        if u < -50 or v < -50 or u > w + 50 or v > h + 50:
            continue
        ok = i in agree_fk
        hit = rack_hits and any(hi == i for hi, _ in rack_hits)
        col = (0, 0, 255) if hit else ((0, 255, 0) if ok else (0, 200, 255))
        cv2.circle(out, (u, v), 6, col, -1, cv2.LINE_AA)
        cv2.putText(
            out,
            f"FK{i}",
            (u + 6, v - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            col,
            1,
            cv2.LINE_AA,
        )

    # Vision confirm ticks on matched tracks
    if matches:
        for tid, fki, dist in matches:
            cv2.putText(
                out,
                f"J{tid}->FK{fki} {dist:.0f}mm",
                (12, 40 + 14 * tid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    if check_unit == "tag":
        hud = f"KIN on {int(check_mm or 0)} tags · UR7e"
        hud_col = (80, 220, 80)
    elif check_ok is True:
        unit = "px" if check_unit == "px" else "mm"
        hud = (
            f"KIN CHECK ok {check_mm:.0f}{unit}"
            if check_mm is not None
            else "KIN CHECK ok"
        )
        hud_col = (80, 220, 80)
    elif check_ok is False:
        unit = "px" if check_unit == "px" else "mm"
        extra = f" {check_mm:.0f}{unit}" if check_mm is not None else ""
        hud = f"KIN CHECK off{extra} — blue disagrees"
        hud_col = (0, 140, 255)
    else:
        hud = f"UR {model} kinematics (no blue check)"
        hud_col = (200, 180, 0)
    if source:
        token = source.split(":")[0].split()[0]
        if token.startswith("teachpad-live") or token.startswith("rtde-live"):
            token = "TEACHPAD"
        elif token.startswith("teachpad-hold") or token.startswith("rtde-hold"):
            token = "HOLD"
        elif token.startswith("photo"):
            token = "PHOTO"
        hud = f"{hud} [{token}]"
    if teachpad_tcp_err_mm is not None and math.isfinite(float(teachpad_tcp_err_mm)):
        hud += f"  TCPΔ{float(teachpad_tcp_err_mm):.1f}mm"
    if rack_hits:
        worst = min(rack_hits, key=lambda x: x[1])
        hud += f"  RACK FK{worst[0]} gap={worst[1]:.0f}mm"
    cv2.putText(
        out,
        hud,
        (12, h - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255) if rack_hits else (80, 220, 80),
        1,
        cv2.LINE_AA,
    )
    return out


def fk_alerts_as_proximity(
    rack_hits: list[tuple[int, float]],
) -> list[tuple[int, int, int, int, float, str]]:
    """Synthetic proximity alert tuples for draw_proximity_error compatibility.

    Uses a dummy box in the corner; gap_mm and label carry the meaning.
    """
    alerts = []
    for idx, gap in rack_hits:
        alerts.append((20, 20 + 30 * idx, 40, 24, float(gap), f"fk_rack_{idx}"))
    return alerts
