"""Turn per-camera detections into one track in the COLMAP map frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment

from geometry import backproject, cam_to_world, camera_forward, matvec_T, ray_plane_intersection


class FusionError(RuntimeError):
    pass


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pose(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data["R"] = np.asarray(data["R"], dtype=np.float64)
    data["t"] = np.asarray(data["t"], dtype=np.float64).reshape(3)
    data["camera_center_m"] = np.asarray(data["camera_center_m"], dtype=np.float64).reshape(3)
    return data


def _pixel(det: dict) -> tuple[float, float]:
    if "u" in det and "v" in det:
        return float(det["u"]), float(det["v"])
    box = det.get("bbox")
    if not box or len(box) != 4:
        raise FusionError("A detection needs u,v or bbox [x, y, w, h].")
    x, y, w, h = (float(v) for v in box)
    return x + 0.5 * w, y + 0.5 * h


def _center_depth_m(depth_mm: np.ndarray, det: dict) -> float | None:
    """Median depth in meters over the middle of the bbox, or a 5x5 patch at u,v."""
    h, w = depth_mm.shape[:2]
    if "bbox" in det and det["bbox"] is not None:
        x, y, bw, bh = (float(v) for v in det["bbox"])
        x0 = int(round(x + 0.25 * bw))
        x1 = int(round(x + 0.75 * bw))
        y0 = int(round(y + 0.25 * bh))
        y1 = int(round(y + 0.75 * bh))
    else:
        u, v = _pixel(det)
        x0, x1 = int(round(u)) - 2, int(round(u)) + 3
        y0, y1 = int(round(v)) - 2, int(round(v)) + 3
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    window = depth_mm[y0:y1, x0:x1].astype(np.float64)
    good = window[(window > 0) & np.isfinite(window)]
    if good.size < 5:
        return None
    return float(np.median(good)) / 1000.0


def detection_to_world(
    det: dict,
    pose: dict,
    cam_cfg: dict,
    plane: dict,
    depth_mm: np.ndarray | None,
) -> np.ndarray:
    """Map one detection to COLMAP world meters."""
    u, v = _pixel(det)
    fx, fy = float(cam_cfg["fx"]), float(cam_cfg["fy"])
    cx, cy = float(cam_cfg["cx"]), float(cam_cfg["cy"])
    z_m = det.get("depth_m")
    if z_m is None and depth_mm is not None and cam_cfg.get("has_depth"):
        z_m = _center_depth_m(depth_mm, det)
    if z_m is not None:
        x_cam = backproject(u, v, float(z_m), fx, fy, cx, cy)
        return cam_to_world(pose["R"], pose["t"], x_cam.reshape(1, 3))[0]
    if not plane or not plane.get("enabled"):
        raise FusionError(
            "Detection has no depth, the camera has no depth image, and the support plane is disabled."
        )
    direction_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    direction_world = matvec_T(pose["R"], direction_cam)
    hit = ray_plane_intersection(
        pose["camera_center_m"],
        direction_world,
        np.asarray(plane["normal"], dtype=np.float64),
        float(plane["d"]),
    )
    if hit is None:
        raise FusionError("The detection ray does not meet the support plane in front of the camera.")
    return hit


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    d0 = float(a[0] - b[0])
    d1 = float(a[1] - b[1])
    d2 = float(a[2] - b[2])
    return (d0 * d0 + d1 * d1 + d2 * d2) ** 0.5


def associate(points: list[dict], gate_m: float) -> list[dict]:
    """Merge same-label points from different cameras that sit within gate_m."""
    if not points:
        return []
    labels = sorted({p["label"] for p in points})
    fused: list[dict] = []
    for label in labels:
        group = [p for p in points if p["label"] == label]
        n = len(group)
        if n == 1:
            fused.append(_fuse_group([group[0]]))
            continue
        cost = np.full((n, n), 1e6, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                if group[i]["camera"] == group[j]["camera"]:
                    continue
                dist = _distance(group[i]["xyz"], group[j]["xyz"])
                if dist <= gate_m:
                    cost[i, j] = dist
                    cost[j, i] = dist
        # Each point is its own cluster unless a cross-camera pair claims it.
        used = set()
        row, col = linear_sum_assignment(cost)
        pairs: list[tuple[int, int]] = []
        for i, j in zip(row, col):
            if i < j and cost[i, j] <= gate_m:
                pairs.append((int(i), int(j)))
                used.add(int(i))
                used.add(int(j))
        claimed = set()
        for i, j in pairs:
            if i in claimed or j in claimed:
                continue
            fused.append(_fuse_group([group[i], group[j]]))
            claimed.add(i)
            claimed.add(j)
        for i, point in enumerate(group):
            if i not in claimed:
                fused.append(_fuse_group([point]))
    return fused


def _fuse_group(group: list[dict]) -> dict:
    weights = np.asarray([max(float(p["score"]), 1e-6) for p in group], dtype=np.float64)
    acc = np.zeros(3, dtype=np.float64)
    for p, w in zip(group, weights):
        acc += w * p["xyz"]
    xyz = acc / float(np.sum(weights))
    cameras = []
    for p in group:
        if p["camera"] not in cameras:
            cameras.append(p["camera"])
    return {
        "label": group[0]["label"],
        "xyz_m": xyz,
        "score": float(np.sum(weights)),
        "contributing_cameras": cameras,
    }


class TrackBook:
    """Nearest-neighbor tracks. Same label, within track_gate_m, keeps its id."""

    def __init__(self, track_gate_m: float):
        self.track_gate_m = float(track_gate_m)
        self.tracks: list[dict] = []
        self._next = 1

    def update(self, fused: list[dict]) -> list[dict]:
        claimed: set[int] = set()
        out = []
        for item in fused:
            best_i = None
            best_d = self.track_gate_m
            for i, track in enumerate(self.tracks):
                if i in claimed or track["label"] != item["label"]:
                    continue
                dist = _distance(track["xyz_m"], item["xyz_m"])
                if dist < best_d:
                    best_d = dist
                    best_i = i
            if best_i is None:
                track = {
                    "id": self._next,
                    "label": item["label"],
                    "xyz_m": np.asarray(item["xyz_m"], dtype=np.float64).copy(),
                    "contributing_cameras": list(item["contributing_cameras"]),
                }
                self._next += 1
                self.tracks.append(track)
            else:
                claimed.add(best_i)
                track = self.tracks[best_i]
                track["xyz_m"] = np.asarray(item["xyz_m"], dtype=np.float64).copy()
                track["contributing_cameras"] = list(item["contributing_cameras"])
            out.append(_public(track))
        return out


def _public(track: dict) -> dict:
    xyz = track["xyz_m"]
    return {
        "id": int(track["id"]),
        "label": track["label"],
        "xyz_m": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
        "contributing_cameras": list(track["contributing_cameras"]),
    }


def observations_to_points(
    detections_by_camera: dict,
    poses: dict[str, dict],
    cameras_cfg: dict,
    plane: dict,
    depth_images: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    points = []
    depth_images = depth_images or {}
    for camera, dets in detections_by_camera.items():
        if camera not in poses:
            raise FusionError(f"No pose loaded for {camera}.")
        if camera not in cameras_cfg:
            raise FusionError(f"No intrinsics for {camera} in the config.")
        for det in dets:
            xyz = detection_to_world(
                det,
                poses[camera],
                cameras_cfg[camera],
                plane,
                depth_images.get(camera),
            )
            points.append(
                {
                    "camera": camera,
                    "label": str(det["label"]),
                    "score": float(det.get("score", 1.0)),
                    "xyz": np.asarray(xyz, dtype=np.float64),
                }
            )
    return points


def step(
    detections_by_camera: dict,
    poses: dict[str, dict],
    cameras_cfg: dict,
    config: dict,
    book: TrackBook,
    depth_images: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    points = observations_to_points(
        detections_by_camera,
        poses,
        cameras_cfg,
        config.get("plane") or {},
        depth_images,
    )
    fused = associate(points, float(config["fusion"]["gate_m"]))
    return book.update(fused)


def save_topdown(path: Path, poses: dict[str, dict], items: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    for name, pose in poses.items():
        c = pose["camera_center_m"]
        forward = camera_forward(pose["R"])
        ax.scatter([c[0]], [c[1]], s=40)
        ax.arrow(c[0], c[1], forward[0] * 0.4, forward[1] * 0.4, head_width=0.05, length_includes_head=True)
        ax.text(c[0], c[1], " " + name)
    for item in items:
        x, y, _z = item["xyz_m"]
        ax.scatter([x], [y], marker="x")
        ax.text(x, y, f" {item['id']}:{item['label']}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.set_title("Top-down map frame (COLMAP x, y)")
    ax.grid(True, linewidth=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fuse cam1 and cam2 detections in the lab map.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True, help="JSON: {time, cameras: {cam1: [...], cam2: [...]}}")
    parser.add_argument("--poses", type=Path, required=True, help="Directory containing camN_pose.json files.")
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    payload = json.loads(args.detections.read_text(encoding="utf-8"))
    cameras = payload["cameras"]
    poses = {}
    for name in cameras:
        pose_path = args.poses / f"{name}_pose.json"
        if not pose_path.is_file():
            print(f"Missing {pose_path}", file=sys.stderr)
            return 2
        poses[name] = load_pose(pose_path)
    book = TrackBook(float(config["fusion"]["track_gate_m"]))
    items = step(cameras, poses, config["cameras"], config, book)
    out_dir = Path(config["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fused.json"
    out_path.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    plot_path = args.plot or (out_dir / "fused_topdown.png")
    save_topdown(plot_path, poses, items)
    print(f"{len(items)} tracks -> {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FusionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
