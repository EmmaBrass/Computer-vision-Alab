"""Small rigid-transform helpers. No numpy matrix products (this PC aborts on them)."""

from __future__ import annotations

import math

import numpy as np


def matvec(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R @ v for a 3-vector."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    out = np.empty(3, dtype=np.float64)
    out[0] = float(R[0, 0]) * x + float(R[0, 1]) * y + float(R[0, 2]) * z
    out[1] = float(R[1, 0]) * x + float(R[1, 1]) * y + float(R[1, 2]) * z
    out[2] = float(R[2, 0]) * x + float(R[2, 1]) * y + float(R[2, 2]) * z
    return out


def matvec_T(R: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R.T @ v for a 3-vector."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    out = np.empty(3, dtype=np.float64)
    out[0] = float(R[0, 0]) * x + float(R[1, 0]) * y + float(R[2, 0]) * z
    out[1] = float(R[0, 1]) * x + float(R[1, 1]) * y + float(R[2, 1]) * z
    out[2] = float(R[0, 2]) * x + float(R[1, 2]) * y + float(R[2, 2]) * z
    return out


def apply_R(R: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """R @ each row of an (N, 3) array."""
    P = np.asarray(pts, dtype=np.float64)
    out = np.empty_like(P)
    for i in range(3):
        out[:, i] = (
            float(R[i, 0]) * P[:, 0] + float(R[i, 1]) * P[:, 1] + float(R[i, 2]) * P[:, 2]
        )
    return out


def apply_RT(R: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """R.T @ each row."""
    P = np.asarray(pts, dtype=np.float64)
    out = np.empty_like(P)
    for i in range(3):
        out[:, i] = (
            float(R[0, i]) * P[:, 0] + float(R[1, i]) * P[:, 1] + float(R[2, i]) * P[:, 2]
        )
    return out


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """COLMAP center: -R.T @ t."""
    return -matvec_T(R, t)


def world_to_cam(R: np.ndarray, t: np.ndarray, pts_world: np.ndarray) -> np.ndarray:
    """X_cam = R @ X_world + t."""
    return apply_R(R, pts_world) + np.asarray(t, dtype=np.float64).reshape(1, 3)


def cam_to_world(R: np.ndarray, t: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    """X_world = R.T @ (X_cam - t)."""
    shifted = np.asarray(pts_cam, dtype=np.float64) - np.asarray(t, dtype=np.float64).reshape(1, 3)
    return apply_RT(R, shifted)


def backproject(u: float, v: float, z_m: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """OpenCV camera coordinates, meters. x right, y down, z forward."""
    return np.array(
        [(u - cx) * z_m / fx, (v - cy) * z_m / fy, z_m],
        dtype=np.float64,
    )


def qvec_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) to a world-to-camera rotation."""
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle of R_a @ R_b.T, in degrees."""
    trace = 0.0
    for i in range(3):
        trace += (
            float(R_a[i, 0]) * float(R_b[i, 0])
            + float(R_a[i, 1]) * float(R_b[i, 1])
            + float(R_a[i, 2]) * float(R_b[i, 2])
        )
    cos_th = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cos_th))


def camera_forward(R: np.ndarray) -> np.ndarray:
    """Camera +z axis expressed in the world. R is world-to-camera."""
    return np.array([float(R[2, 0]), float(R[2, 1]), float(R[2, 2])], dtype=np.float64)


def ray_plane_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    normal: np.ndarray,
    plane_d: float,
) -> np.ndarray | None:
    """Intersect origin + s * direction with n·X + d = 0. None if parallel or behind."""
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    denom = float(n[0] * d[0] + n[1] * d[1] + n[2] * d[2])
    if abs(denom) < 1e-9:
        return None
    nd = float(n[0] * o[0] + n[1] * o[1] + n[2] * o[2])
    s = -(nd + float(plane_d)) / denom
    if s <= 1e-6:
        return None
    return o + s * d
