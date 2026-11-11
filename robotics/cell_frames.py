"""Cell frame graph: ur_base (0,0,0) and fixed camera poses.

p_base = R @ p_cam + t_mm

Physical layout (Jessica cell):
  - cam1 and cam2 face each other (~180 deg), mounted high, looking down onto the arm
  - cam0 is a separate close / side view — not part of that opposite pair

Default prior ``opposite_180``: cam1 R = I, cam2 R = 180 deg yaw about Y.
cam0 keeps its own R (not forced relative to cam1). Absolute tilt (high-down)
is refined by OnlineCamPose from tags/blues. t is bootstrapped so the blue
pedestal maps to the origin when enabled.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# cam2 optical frame relative to cam1 when mounts face each other (level / high):
# yaw 180 deg about Y (OpenCV: X right, Y down, Z forward).
OPPOSITE_R_CAM2 = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)
# Legacy: cam0⊥cam1 at 90 deg (wrong for this cell; kept for old JSON only).
PERPENDICULAR_R_CAM1 = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
LAYOUT_OPPOSITE_180 = "opposite_180"
LAYOUT_PERPENDICULAR = "perpendicular_90"
LAYOUT_SCENE_TAPE = "scene_tape"
LAYOUT_ASYMMETRIC_BOARD = "asymmetric_board"

# Pair used for opposite high-down localization. cam0 is independent.
OPPOSITE_PAIR = ("cam1", "cam2")


@dataclass
class CamPose:
    frame_id: str
    R: np.ndarray  # 3x3
    t_mm: np.ndarray  # 3,
    ready: bool = False


@dataclass
class CellFrames:
    path: Path
    bootstrap_from_blue_base: bool
    fx_px: float
    rack_height_mm: float
    cam0: CamPose
    cam1: CamPose
    cam2: CamPose
    layout: str = LAYOUT_OPPOSITE_180

    def pose(self, cam: str) -> CamPose:
        if cam == "cam0":
            return self.cam0
        if cam == "cam2":
            return self.cam2
        return self.cam1


def default_cell_frames_path(robotics_root: Optional[Path] = None) -> Path:
    if robotics_root is not None:
        return Path(robotics_root) / "cell_frames.json"
    import os

    env = os.environ.get("ORBBEC_CAMERA_ROOT", "").strip()
    if env:
        return Path(env) / "cell_frames.json"
    return Path(__file__).resolve().parents[4] / "robotics" / "cell_frames.json"


def _pose_from_dict(d: dict, default_id: str) -> CamPose:
    R = np.asarray(d.get("R", np.eye(3)), dtype=np.float64).reshape(3, 3)
    t = np.asarray(d.get("t_mm", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return CamPose(
        frame_id=str(d.get("frame_id", default_id)),
        R=R,
        t_mm=t,
        ready=bool(d.get("ready", False)),
    )


def apply_opposite_180_layout(cf: CellFrames) -> CellFrames:
    """cam1↔cam2 face each other (180 deg). cam0 left alone (separate view)."""
    cf.layout = LAYOUT_OPPOSITE_180
    # Keep cam1 as the reference heading for the opposite pair.
    if cf.cam1.R is None or not np.isfinite(cf.cam1.R).all():
        cf.cam1.R = np.eye(3, dtype=np.float64)
    # cam2 = 180 yaw about Y relative to cam1's current R.
    R1 = np.asarray(cf.cam1.R, dtype=np.float64).reshape(3, 3)
    out = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += float(OPPOSITE_R_CAM2[i, k]) * float(R1[k, j])
            out[i, j] = s
    cf.cam2.R = out
    return cf


def apply_perpendicular_layout(cf: CellFrames) -> CellFrames:
    """Legacy cam0⊥cam1 90 deg. Prefer ``apply_opposite_180_layout`` for this cell."""
    cf.layout = LAYOUT_PERPENDICULAR
    cf.cam0.R = np.eye(3, dtype=np.float64)
    cf.cam1.R = PERPENDICULAR_R_CAM1.copy()
    return cf


def load_cell_frames(path: Optional[Path] = None) -> CellFrames:
    path = Path(path) if path is not None else default_cell_frames_path()
    eye = np.eye(3, dtype=np.float64)
    z = np.zeros(3, dtype=np.float64)
    if not path.is_file():
        return CellFrames(
            path=path,
            bootstrap_from_blue_base=True,
            fx_px=520.0,
            rack_height_mm=400.0,
            cam0=CamPose("orbbec_cam0", eye.copy(), z.copy(), False),
            cam1=CamPose("orbbec_cam1", eye.copy(), z.copy(), False),
            cam2=CamPose("orbbec_cam2", OPPOSITE_R_CAM2.copy(), z.copy(), False),
            layout=LAYOUT_OPPOSITE_180,
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    cams = data.get("cameras") or {}
    layout = str(data.get("layout") or LAYOUT_OPPOSITE_180)
    # Migrate old perpendicular default when JSON still says that.
    if layout == LAYOUT_PERPENDICULAR and "cam2" not in cams:
        layout = LAYOUT_OPPOSITE_180
    cf = CellFrames(
        path=path,
        bootstrap_from_blue_base=bool(data.get("bootstrap_from_blue_base", True)),
        fx_px=float(data.get("fx_px", 520.0)),
        rack_height_mm=float(data.get("rack_height_mm", 400.0)),
        cam0=_pose_from_dict(cams.get("cam0") or {}, "orbbec_cam0"),
        cam1=_pose_from_dict(cams.get("cam1") or {}, "orbbec_cam1"),
        cam2=_pose_from_dict(
            cams.get("cam2") or {"R": OPPOSITE_R_CAM2.tolist()},
            "orbbec_cam2",
        ),
        layout=layout,
    )
    if layout == LAYOUT_OPPOSITE_180:
        apply_opposite_180_layout(cf)
    elif layout == LAYOUT_PERPENDICULAR:
        apply_perpendicular_layout(cf)
    return cf


def save_cell_frames(cf: CellFrames) -> None:
    payload = {
        "world_frame": "ur_base",
        "origin_mm": [0.0, 0.0, 0.0],
        "layout": cf.layout,
        "notes": (
            "ur_base is (0,0,0) at the blue pedestal / purple-tape origin. "
            "p_base = R @ p_cam + t_mm. "
            "layout=opposite_180: cam1↔cam2 face each other (180 deg about Y), "
            "high looking down onto the arm; cam0 is a separate side/close view. "
            "layout=perpendicular_90: legacy cam0⊥cam1 (not this cell). "
            "layout=scene_tape / asymmetric_board / arm_tags: measured R,t. "
            "bootstrap updates t so each camera maps the pedestal/tape to the origin."
        ),
        "bootstrap_from_blue_base": bool(cf.bootstrap_from_blue_base),
        "fx_px": float(cf.fx_px),
        "rack_height_mm": float(cf.rack_height_mm),
        "cameras": {
            "cam0": {
                "frame_id": cf.cam0.frame_id,
                "R": cf.cam0.R.tolist(),
                "t_mm": cf.cam0.t_mm.reshape(3).tolist(),
                "ready": bool(cf.cam0.ready),
                "role": "side_close_independent",
            },
            "cam1": {
                "frame_id": cf.cam1.frame_id,
                "R": cf.cam1.R.tolist(),
                "t_mm": cf.cam1.t_mm.reshape(3).tolist(),
                "ready": bool(cf.cam1.ready),
                "role": "high_down_opposite_a",
            },
            "cam2": {
                "frame_id": cf.cam2.frame_id,
                "R": cf.cam2.R.tolist(),
                "t_mm": cf.cam2.t_mm.reshape(3).tolist(),
                "ready": bool(cf.cam2.ready),
                "role": "high_down_opposite_b",
            },
        },
        "stereo_extrinsics": "stereo_extrinsics.json",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    cf.path.parent.mkdir(parents=True, exist_ok=True)
    cf.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cam_to_base(p_cam_mm: np.ndarray, pose: CamPose) -> np.ndarray:
    p = np.asarray(p_cam_mm, dtype=np.float64).reshape(3)
    return pose.R @ p + pose.t_mm.reshape(3)


def bootstrap_pose_from_base_landmark(
    p_base_in_cam_mm: np.ndarray,
    *,
    R: Optional[np.ndarray] = None,
    frame_id: str = "orbbec_cam",
) -> CamPose:
    """Set translation so the seen pedestal maps to ur_base origin (R default I)."""
    R_use = np.eye(3, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
    p = np.asarray(p_base_in_cam_mm, dtype=np.float64).reshape(3)
    t = -R_use @ p
    return CamPose(frame_id=frame_id, R=R_use, t_mm=t, ready=True)


def format_base_retreat(p_joint_base_mm: np.ndarray) -> str:
    """Instruction to move a joint at p_joint_base toward ur_base (0,0,0)."""
    p = np.asarray(p_joint_base_mm, dtype=np.float64).reshape(3)
    v = -p
    dist = float(np.linalg.norm(v))
    if dist < 20.0:
        return (
            "FUSED RETREAT -> ur_base (0,0,0): already near BASE "
            f"(~{dist / 10.0:.0f}cm) - pull gently to home"
        )
    parts: list[str] = []
    axes = (
        ("+X" if v[0] > 0 else "-X", abs(v[0])),
        ("+Y" if v[1] > 0 else "-Y", abs(v[1])),
        ("+Z" if v[2] > 0 else "-Z", abs(v[2])),
    )
    for name, mag in axes:
        if mag >= 30.0:
            parts.append(f"{name} ~{mag / 10.0:.0f}cm")
    if not parts:
        parts.append(f"toward origin ~{dist / 10.0:.0f}cm")
    return (
        "FUSED RETREAT -> move arm toward ROBOT BASE ur_base (0,0,0): "
        + ", ".join(parts)
        + f"  |  remaining ~{dist / 10.0:.0f}cm"
    )
