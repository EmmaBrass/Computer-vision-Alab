"""Read a COLMAP sparse model. Poses are whatever unit the files were written in."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geometry import camera_center, qvec_to_R


@dataclass
class MapImage:
    image_id: int
    name: str
    camera_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    R: np.ndarray
    center: np.ndarray


def read_images(path: Path) -> dict[str, MapImage]:
    """Poses keyed by the COLMAP image file name (for example 000123.jpg)."""
    images: dict[str, MapImage] = {}
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            props = struct.unpack("<idddddddi", f.read(64))
            image_id = int(props[0])
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = int(props[8])
            name_b = bytearray()
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name_b += ch
            n2 = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * n2, 1)
            R = qvec_to_R(float(qvec[0]), float(qvec[1]), float(qvec[2]), float(qvec[3]))
            name = name_b.decode("utf-8")
            images[name] = MapImage(
                image_id=image_id,
                name=name,
                camera_id=camera_id,
                qvec=qvec,
                tvec=tvec,
                R=R,
                center=camera_center(R, tvec),
            )
    return images


def read_points_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Map points as (N, 3) xyz and (N, 3) uint8 rgb. Tracks are skipped."""
    xyz = []
    rgb = []
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            f.read(8)
            xyz.append(struct.unpack("<ddd", f.read(24)))
            rgb.append(struct.unpack("<BBB", f.read(3)))
            f.read(8)
            track_n = struct.unpack("<Q", f.read(8))[0]
            f.seek(8 * track_n, 1)
    return np.asarray(xyz, dtype=np.float64), np.asarray(rgb, dtype=np.uint8)
