"""Camera intrinsics and stereo extrinsics used by scene texture matching.

The vial mesh and size prior used to live here. Scene matching and the arm
mask only need the pinhole helpers and the stereo JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXTRINSICS_PATH = ROOT / "stereo_extrinsics.json"
DEFAULT_HFOV_DEG = 90.0
DEFAULT_IMAGE_WIDTH = 1280
DEFAULT_IMAGE_HEIGHT = 800


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.fx, self.fy, self.cx, self.cy

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class StereoExtrinsics:
    """X_to = R @ X_from + t, in millimeters."""

    R: np.ndarray
    t_mm: np.ndarray
    from_camera: int = 0
    to_camera: int = 1
    K0: Intrinsics | None = None
    K1: Intrinsics | None = None

    def transform_point(self, xyz_mm: np.ndarray) -> np.ndarray:
        return self.R @ np.asarray(xyz_mm, dtype=np.float64).reshape(3) + self.t_mm


def default_intrinsics(
    width: int = DEFAULT_IMAGE_WIDTH, height: int = DEFAULT_IMAGE_HEIGHT
) -> Intrinsics:
    fx = width / (2.0 * math.tan(math.radians(DEFAULT_HFOV_DEG) / 2.0))
    return Intrinsics(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0)


def intrinsics_from_tuple(values) -> Intrinsics | None:
    if values is None:
        return None
    fx, fy, cx, cy = values
    if fx <= 1 or fy <= 1:
        return None
    return Intrinsics(float(fx), float(fy), float(cx), float(cy))


def pixel_to_cam(u: float, v: float, z_mm: float, K: Intrinsics) -> tuple[float, float, float]:
    return (u - K.cx) * z_mm / K.fx, (v - K.cy) * z_mm / K.fy, float(z_mm)


def cam_to_pixel(xyz_mm: np.ndarray, K: Intrinsics) -> tuple[float, float] | None:
    x, y, z = [float(v) for v in xyz_mm]
    if z <= 1e-3:
        return None
    return K.fx * x / z + K.cx, K.fy * y / z + K.cy


def is_extrinsics_placeholder(path: Path | None = None) -> bool:
    """True when the stereo file is missing, marked placeholder, or still identity."""
    path = path or EXTRINSICS_PATH
    if not path.exists():
        return True
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if "PLACEHOLDER" in str(data.get("notes", "")).upper():
        return True
    rotation = np.asarray(data.get("R", []), dtype=np.float64).reshape(-1)
    if rotation.size == 9 and float(np.linalg.norm(rotation - np.eye(3).reshape(-1))) < 1e-6:
        return True
    return False


def extrinsics_reproject_state(path: Path | None = None) -> str:
    path = path or EXTRINSICS_PATH
    if not path.exists():
        return "missing"
    if is_extrinsics_placeholder(path):
        return "placeholder"
    return "ready"


def load_extrinsics(path: Path | None = None) -> StereoExtrinsics | None:
    path = path or EXTRINSICS_PATH
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    k0 = Intrinsics(*[float(v) for v in data["K0"]]) if data.get("K0") is not None else None
    k1 = Intrinsics(*[float(v) for v in data["K1"]]) if data.get("K1") is not None else None
    return StereoExtrinsics(
        R=np.asarray(data["R"], dtype=np.float64).reshape(3, 3),
        t_mm=np.asarray(data["t_mm"], dtype=np.float64).reshape(3),
        from_camera=int(data.get("from_camera", 0)),
        to_camera=int(data.get("to_camera", 1)),
        K0=k0,
        K1=k1,
    )


def write_extrinsics_template(path: Path | None = None) -> Path:
    """Identity stub so the stereo file format is discoverable before calibration."""
    path = path or EXTRINSICS_PATH
    if path.exists():
        return path
    k = default_intrinsics()
    payload = {
        "from_camera": 0,
        "to_camera": 1,
        "R": np.eye(3).tolist(),
        "t_mm": [120.0, 0.0, 0.0],
        "K0": [k.fx, k.fy, k.cx, k.cy],
        "K1": [k.fx, k.fy, k.cx, k.cy],
        "notes": (
            "PLACEHOLDER — replace R,t after checkerboard stereoCalibrate. "
            "t_mm is a rough baseline guess along +X; identity R is wrong for angled mounts."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
