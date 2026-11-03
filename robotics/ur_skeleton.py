"""UR forward kinematics + joint-angle sources for stick-figure / crash assist.

Joint sources (first that works):
  1. Optional RTDE (``ur_rtde`` or ``rtde_receive``) when ``host`` is set
  2. JSON file ``ur_joints.json`` (teach-pendant / external writer)
  3. Mock zeros (draw at home) for offline overlay tests

    python -c "from ur_skeleton import fk_joint_positions_mm, load_joints; print(fk_joint_positions_mm(load_joints()[0]))"
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
JOINTS_PATH = ROOT / "ur_joints.json"
CONFIG_PATH = ROOT / "ur_fk_config.json"

# DH (meters) — Universal Robots published parameters (approx by model family).
UR_MODELS_M = {
    "ur5": {
        "d": [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823],
        "a": [0.0, -0.42500, -0.39225, 0.0, 0.0, 0.0],
    },
    "ur5e": {
        "d": [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996],
        "a": [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0],
    },
    "ur10e": {
        "d": [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655],
        "a": [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0],
    },
}
UR_ALPHAS = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]


@dataclass
class UrFkConfig:
    model: str = "ur5e"
    host: str = ""  # RTDE host; empty = file/mock only
    joints_path: Path = JOINTS_PATH
    # Offset: p_vision_base = R @ p_ur_base + t_mm (align pendant base to cell ur_base)
    R_ur_to_vision: np.ndarray | None = None
    t_ur_to_vision_mm: np.ndarray | None = None
    rack_keepout_min_mm: tuple[float, float, float] = (200.0, -400.0, 0.0)
    rack_keepout_max_mm: tuple[float, float, float] = (700.0, 400.0, 500.0)
    confirm_gate_mm: float = 80.0
    proximity_mm: float = 100.0


def load_fk_config(path: Path | None = None) -> UrFkConfig:
    path = path or CONFIG_PATH
    cfg = UrFkConfig()
    if not path.is_file():
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return cfg
    cfg.model = str(data.get("model", cfg.model))
    cfg.host = str(data.get("host", cfg.host) or "")
    jp = data.get("joints_path")
    if jp:
        cfg.joints_path = Path(jp)
        if not cfg.joints_path.is_absolute():
            cfg.joints_path = ROOT / cfg.joints_path
    if data.get("R_ur_to_vision") is not None:
        cfg.R_ur_to_vision = np.asarray(data["R_ur_to_vision"], dtype=np.float64).reshape(3, 3)
    if data.get("t_ur_to_vision_mm") is not None:
        cfg.t_ur_to_vision_mm = np.asarray(
            data["t_ur_to_vision_mm"], dtype=np.float64
        ).reshape(3)
    if data.get("rack_keepout_min_mm") is not None:
        cfg.rack_keepout_min_mm = tuple(float(v) for v in data["rack_keepout_min_mm"])  # type: ignore
    if data.get("rack_keepout_max_mm") is not None:
        cfg.rack_keepout_max_mm = tuple(float(v) for v in data["rack_keepout_max_mm"])  # type: ignore
    cfg.confirm_gate_mm = float(data.get("confirm_gate_mm", cfg.confirm_gate_mm))
    cfg.proximity_mm = float(data.get("proximity_mm", cfg.proximity_mm))
    return cfg


def write_example_files() -> None:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "model": "ur5e",
                    "host": "",
                    "joints_path": "ur_joints.json",
                    "R_ur_to_vision": np.eye(3).tolist(),
                    "t_ur_to_vision_mm": [0.0, 0.0, 0.0],
                    "rack_keepout_min_mm": [200.0, -400.0, 0.0],
                    "rack_keepout_max_mm": [700.0, 400.0, 500.0],
                    "confirm_gate_mm": 80.0,
                    "proximity_mm": 100.0,
                    "notes": (
                        "Write q_rad into ur_joints.json from teach pendant / script, "
                        "or set host to robot IP for RTDE when ur_rtde is installed."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not JOINTS_PATH.exists():
        JOINTS_PATH.write_text(
            json.dumps(
                {
                    "q_rad": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
                    "model": "ur5e",
                    "source": "example",
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fk_transforms(q_rad: np.ndarray, model: str = "ur5e") -> list[np.ndarray]:
    """Return T_0_i (4x4) for i=0..6 (base + 6 joints) in UR base meters."""
    q = np.asarray(q_rad, dtype=np.float64).reshape(6)
    key = model if model in UR_MODELS_M else "ur5e"
    d = UR_MODELS_M[key]["d"]
    a = UR_MODELS_M[key]["a"]
    T = np.eye(4, dtype=np.float64)
    out = [T.copy()]
    for i in range(6):
        T = T @ _dh(float(q[i]), float(d[i]), float(a[i]), float(UR_ALPHAS[i]))
        out.append(T.copy())
    return out


def fk_joint_positions_mm(q_rad: np.ndarray, model: str = "ur5e") -> np.ndarray:
    """(7,3) positions in UR base: base origin + 6 joint frames, millimeters."""
    Ts = fk_transforms(q_rad, model=model)
    pts = np.zeros((7, 3), dtype=np.float64)
    for i, T in enumerate(Ts):
        pts[i] = T[:3, 3] * 1000.0
    return pts


def ur_base_to_vision_mm(
    pts_ur_mm: np.ndarray, cfg: UrFkConfig
) -> np.ndarray:
    """Map UR-base points into vision ur_base (mm)."""
    P = np.asarray(pts_ur_mm, dtype=np.float64).reshape(-1, 3)
    R = cfg.R_ur_to_vision if cfg.R_ur_to_vision is not None else np.eye(3)
    t = (
        cfg.t_ur_to_vision_mm
        if cfg.t_ur_to_vision_mm is not None
        else np.zeros(3)
    )
    return (R @ P.T).T + t.reshape(3)


def _read_joints_file(path: Path) -> Optional[tuple[np.ndarray, str]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    q = data.get("q_rad") or data.get("q")
    if q is None or len(q) < 6:
        return None
    model = str(data.get("model", "ur5e"))
    return np.asarray(q[:6], dtype=np.float64), model


def _rtde_receive(host: str):
    """Open one RTDE reader. Returns the interface, or None."""
    if not host:
        return None
    try:
        from ur_rtde import RTDEReceiveInterface  # type: ignore

        return RTDEReceiveInterface(host)
    except Exception:
        pass
    try:
        import rtde_receive  # type: ignore

        return rtde_receive.RTDEReceiveInterface(host)
    except Exception:
        return None


def _close_rtde(rtde) -> None:
    disconnect = getattr(rtde, "disconnect", None)
    if callable(disconnect):
        try:
            disconnect()
        except Exception:
            pass


def _read_joints_rtde(host: str) -> Optional[tuple[np.ndarray, str]]:
    rtde = _rtde_receive(host)
    if rtde is None:
        return None
    try:
        q = np.asarray(rtde.getActualQ(), dtype=np.float64).reshape(6)
    except Exception:
        _close_rtde(rtde)
        return None
    _close_rtde(rtde)
    return q, "ur5e"


def read_robot_tcp_mm(host: str) -> Optional[np.ndarray]:
    """Tool position in the robot base, millimeters, from the controller.

    This is ``getActualTCPPose`` (x, y, z). It is the pendant's Cartesian
    number, not a pose recomputed here.
    """
    rtde = _rtde_receive(host)
    if rtde is None:
        return None
    try:
        pose = np.asarray(rtde.getActualTCPPose(), dtype=np.float64).reshape(-1)
    except Exception:
        _close_rtde(rtde)
        return None
    _close_rtde(rtde)
    if pose.size < 3 or not np.isfinite(pose[:3]).all():
        return None
    return pose[:3] * 1000.0


def robot_arm_points_mm(cfg: UrFkConfig | None = None) -> Optional[tuple[np.ndarray, str]]:
    """Arm points in the vision base from the controller or ``ur_joints.json``.

    Returns None for the offline mock pose so a mask is not drawn on a
    pretend arm. The tool tip is the controller Cartesian pose when RTDE
    answers. The other points are the reported joint angles placed on the
    UR link model, which is only there so the mask can follow each tube.
    """
    cfg = cfg or load_fk_config()
    q: Optional[np.ndarray] = None
    source = ""
    tcp_mm = None
    if cfg.host:
        got = _read_joints_rtde(cfg.host)
        if got is not None:
            q = got[0]
            source = f"rtde:{cfg.host}"
            tcp_mm = read_robot_tcp_mm(cfg.host)
    if q is None:
        filed = _read_joints_file(cfg.joints_path)
        if filed is None:
            return None
        q, model = filed
        source = f"file:{cfg.joints_path.name}"
    else:
        model = cfg.model or "ur5e"
    pts = ur_base_to_vision_mm(fk_joint_positions_mm(q, model), cfg)
    if tcp_mm is not None:
        pts = np.array(pts, dtype=np.float64, copy=True)
        pts[-1] = ur_base_to_vision_mm(tcp_mm.reshape(1, 3), cfg)[0]
    return pts, source


def load_joints(
    cfg: UrFkConfig | None = None,
) -> tuple[np.ndarray, str, str]:
    """Return (q_rad, model, source_name)."""
    cfg = cfg or load_fk_config()
    if cfg.host:
        got = _read_joints_rtde(cfg.host)
        if got is not None:
            return got[0], cfg.model or got[1], f"rtde:{cfg.host}"
    got = _read_joints_file(cfg.joints_path)
    if got is not None:
        return got[0], str(got[1] or cfg.model), f"file:{cfg.joints_path.name}"
    # Mock “elbow up” pose so the stick figure is visible offline.
    q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)
    return q, cfg.model, "mock"


def point_to_aabb_gap_mm(
    p_mm: np.ndarray,
    mn: tuple[float, float, float],
    mx: tuple[float, float, float],
) -> float:
    """0 if inside AABB; else Euclidean distance to box surface (mm)."""
    p = np.asarray(p_mm, dtype=np.float64).reshape(3)
    mn_a = np.asarray(mn, dtype=np.float64)
    mx_a = np.asarray(mx, dtype=np.float64)
    closest = np.minimum(np.maximum(p, mn_a), mx_a)
    if np.all(p >= mn_a) and np.all(p <= mx_a):
        return 0.0
    return float(np.linalg.norm(p - closest))


def fk_rack_proximity(
    pts_vision_mm: np.ndarray,
    cfg: UrFkConfig,
) -> list[tuple[int, float]]:
    """Return list of (joint_index, gap_mm) within cfg.proximity_mm of rack keep-out.

    joint_index: 0=base ... 6=tool.
    """
    hits: list[tuple[int, float]] = []
    for i, p in enumerate(np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 3)):
        if i == 0:
            continue  # skip base origin
        gap = point_to_aabb_gap_mm(p, cfg.rack_keepout_min_mm, cfg.rack_keepout_max_mm)
        if gap <= float(cfg.proximity_mm):
            hits.append((i, gap))
    return hits
