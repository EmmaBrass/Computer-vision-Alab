"""Place one still image in the COLMAP lab map.

The map frame is the scaled COLMAP world, in meters. Walkthrough image names
match the depth files (000123.jpg <-> 000123.png). Nerfstudio's transforms.json
is not used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from colmap_io import read_images, read_points_xyz_rgb
from geometry import backproject, cam_to_world, camera_center, world_to_cam


class LocalizationError(RuntimeError):
    """The still did not lock to the map tightly enough to trust."""


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _intrinsics_ready(cam: dict) -> bool:
    return all(cam.get(k) is not None for k in ("fx", "fy", "cx", "cy", "width", "height"))


def _K(cam: dict) -> np.ndarray:
    return np.array(
        [
            [float(cam["fx"]), 0.0, float(cam["cx"])],
            [0.0, float(cam["fy"]), float(cam["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _dist(cam: dict) -> np.ndarray:
    raw = cam.get("dist") or [0.0, 0.0, 0.0, 0.0, 0.0]
    return np.asarray(raw, dtype=np.float64).reshape(-1, 1)


def _sift(nfeatures: int) -> cv2.SIFT:
    return cv2.SIFT_create(nfeatures=int(nfeatures))


def _gray_resized(bgr: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = gray.shape[:2]
    if w <= max_width:
        return gray, 1.0
    scale = max_width / float(w)
    small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


def _ratio_count(des_q: np.ndarray, des_db: np.ndarray, ratio: float) -> int:
    if des_q is None or des_db is None or len(des_q) < 2 or len(des_db) < 2:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    try:
        pairs = matcher.knnMatch(des_q, des_db, k=2)
    except cv2.error:
        return 0
    count = 0
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            count += 1
    return count


def _mutual_matches(des_a: np.ndarray, des_b: np.ndarray, ratio: float) -> list[tuple[int, int]]:
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def one_way(src: np.ndarray, dst: np.ndarray) -> dict[int, int]:
        try:
            pairs = matcher.knnMatch(src, dst, k=2)
        except cv2.error:
            return {}
        kept: dict[int, int] = {}
        for pair in pairs:
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < ratio * second.distance:
                kept[best.queryIdx] = best.trainIdx
        return kept

    forward = one_way(des_a, des_b)
    backward = one_way(des_b, des_a)
    return [(i, j) for i, j in forward.items() if backward.get(j) == i]


def _ratio_matches(des_a: np.ndarray, des_b: np.ndarray, ratio: float) -> list[tuple[int, int]]:
    """One-way Lowe ratio matches, one query keypoint per frame keypoint (best distance wins)."""
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return []
    try:
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_a, des_b, k=2)
    except cv2.error:
        return []
    best_for_train: dict[int, tuple[float, int]] = {}
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < ratio * second.distance:
            prev = best_for_train.get(best.trainIdx)
            if prev is None or best.distance < prev[0]:
                best_for_train[best.trainIdx] = (best.distance, best.queryIdx)
    return [(qi, ti) for ti, (_d, qi) in best_for_train.items()]


def _verify_pairs(kp_q, kp_f, pairs: list[tuple[int, int]], thresh_px: float) -> list[tuple[int, int]]:
    """Keep pairs consistent with one fundamental matrix between the two images."""
    if len(pairs) < 12:
        return []
    pq = np.float64([kp_q[i].pt for i, _ in pairs])
    pf = np.float64([kp_f[j].pt for _, j in pairs])
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    try:
        F, mask = cv2.findFundamentalMat(pq, pf, method, float(thresh_px), 0.999, 10000)
    except cv2.error:
        return []
    if F is None or mask is None:
        return []
    keep = mask.reshape(-1).astype(bool)
    return [p for p, k in zip(pairs, keep) if k]


def _depth_at(depth_mm: np.ndarray, u: float, v: float, spread_m: float, max_m: float) -> float | None:
    h, w = depth_mm.shape
    ui = int(round(u))
    vi = int(round(v))
    if ui < 1 or vi < 1 or ui >= w - 1 or vi >= h - 1:
        return None
    window = depth_mm[vi - 1 : vi + 2, ui - 1 : ui + 2].astype(np.float64)
    good = window[(window > 0) & np.isfinite(window)]
    if good.size < 5:
        return None
    z_mm = float(np.median(good))
    if (float(good.max()) - float(good.min())) > spread_m * 1000.0:
        return None
    z_m = z_mm / 1000.0
    if z_m <= 0.0 or z_m > max_m:
        return None
    return z_m


def _lift_frame(
    query_kp,
    frame_kp,
    pairs: list[tuple[int, int]],
    depth_mm: np.ndarray,
    frame_pose,
    walk: dict,
    spread_m: float,
    max_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    obj = []
    img = []
    fx, fy = float(walk["fx"]), float(walk["fy"])
    cx, cy = float(walk["cx"]), float(walk["cy"])
    for qi, fi in pairs:
        u_f, v_f = frame_kp[fi].pt
        z_m = _depth_at(depth_mm, u_f, v_f, spread_m, max_m)
        if z_m is None:
            continue
        x_cam = backproject(u_f, v_f, z_m, fx, fy, cx, cy)
        x_world = cam_to_world(frame_pose.R, frame_pose.tvec, x_cam.reshape(1, 3))[0]
        u_q, v_q = query_kp[qi].pt
        obj.append(x_world)
        img.append((u_q, v_q))
    if not obj:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=np.float64)
    return np.asarray(obj, dtype=np.float64), np.asarray(img, dtype=np.float64)


def _median_reproj(
    obj: np.ndarray, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    delta = proj.reshape(-1, 2) - img.reshape(-1, 2)
    err = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
    return float(np.median(err))


def solve_pnp(obj: np.ndarray, img: np.ndarray, cam: dict, reproj_px: float, iterations: int = 10000) -> dict:
    if obj.shape[0] < 12:
        raise LocalizationError(f"Only {obj.shape[0]} map correspondences. Need at least 12 before PnP.")
    flags = cv2.SOLVEPNP_SQPNP if hasattr(cv2, "SOLVEPNP_SQPNP") else cv2.SOLVEPNP_EPNP
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        _K(cam),
        _dist(cam),
        iterationsCount=int(iterations),
        reprojectionError=float(reproj_px),
        confidence=0.999,
        flags=flags,
    )
    if not ok or inliers is None or len(inliers) < 6:
        raise LocalizationError("solvePnPRansac did not find a pose.")
    inl = inliers.reshape(-1)
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], _K(cam), _dist(cam), rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    median_px = _median_reproj(obj[inl], img[inl], rvec, tvec, _K(cam), _dist(cam))
    return {
        "R": R,
        "t": t,
        "camera_center_m": camera_center(R, t),
        "inliers": int(inl.size),
        "median_reproj_px": median_px,
    }


def _rank_walkthrough(query_bgr: np.ndarray, color_dir: Path, names: list[str], cfg: dict, exclude: set[str]) -> list[tuple[str, int]]:
    loc = cfg["localize"]
    small, _scale = _gray_resized(query_bgr, int(loc["retrieve_max_width"]))
    sift = _sift(int(loc["sift_features_retrieve"]))
    _kp_q, des_q = sift.detectAndCompute(small, None)
    scored: list[tuple[str, int]] = []
    max_w = int(loc["retrieve_max_width"])
    ratio = float(loc["ratio"])
    for i, name in enumerate(names):
        if name in exclude or Path(name).name in exclude:
            continue
        frame = cv2.imread(str(color_dir / name), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        gray, _ = _gray_resized(frame, max_w)
        _kp, des = sift.detectAndCompute(gray, None)
        scored.append((name, _ratio_count(des_q, des, ratio)))
        if (i + 1) % 50 == 0:
            print(f"retrieval {i + 1}/{len(names)}", flush=True)
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: int(loc.get("shortlist_k", loc["top_k"]))]


def localize_still(
    query_bgr: np.ndarray,
    intrinsics: dict,
    config: dict,
    *,
    exclude_names: set[str] | None = None,
    overlay_path: Path | None = None,
) -> dict:
    """Return the query's world-to-camera pose in the COLMAP map frame."""
    if not _intrinsics_ready(intrinsics):
        raise LocalizationError(
            "Camera intrinsics are missing. Fill fx, fy, cx, cy, width, and height in config.yaml."
        )
    h, w = query_bgr.shape[:2]
    if int(w) != int(intrinsics["width"]) or int(h) != int(intrinsics["height"]):
        raise LocalizationError(
            f"Query is {w}x{h}, config says {intrinsics['width']}x{intrinsics['height']}."
        )
    exclude = exclude_names or set()
    color_dir = Path(config["paths"]["color_dir"])
    depth_dir = Path(config["paths"]["depth_dir"])
    images = read_images(Path(config["paths"]["images_bin"]))
    names = sorted(images)
    print(f"map images {len(names)}", flush=True)
    top = _rank_walkthrough(query_bgr, color_dir, names, config, exclude)
    if not top or top[0][1] < 8:
        raise LocalizationError("Retrieval found no walkthrough frame with enough SIFT matches.")
    print("top frames: " + ", ".join(f"{name}:{count}" for name, count in top), flush=True)

    loc = config["localize"]
    sift = _sift(int(loc["sift_features_match"]))
    gray_q = cv2.cvtColor(query_bgr, cv2.COLOR_BGR2GRAY) if query_bgr.ndim == 3 else query_bgr
    kp_q, des_q = sift.detectAndCompute(gray_q, None)
    mutual = bool(loc.get("mutual_matches", False))
    verify_px = float(loc.get("verify_fundamental_px", 3.0))
    candidates = []
    for name, _count in top:
        frame = cv2.imread(str(color_dir / name), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_dir / (Path(name).stem + ".png")), cv2.IMREAD_UNCHANGED)
        if frame is None or depth is None:
            continue
        if depth.shape[:2] != frame.shape[:2]:
            raise LocalizationError(f"{name} depth is {depth.shape} and color is {frame.shape}.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_f, des_f = sift.detectAndCompute(gray, None)
        raw = _mutual_matches(des_q, des_f, float(loc["ratio"])) if mutual else _ratio_matches(des_q, des_f, float(loc["ratio"]))
        pairs = _verify_pairs(kp_q, kp_f, raw, verify_px)
        candidates.append((len(pairs), name, kp_f, pairs, depth, len(raw)))
    candidates.sort(key=lambda c: c[0], reverse=True)
    obj_parts = []
    img_parts = []
    for n_ver, name, kp_f, pairs, depth, n_raw in candidates[: int(loc["top_k"])]:
        if n_ver == 0:
            continue
        obj, img = _lift_frame(
            kp_q,
            kp_f,
            pairs,
            depth,
            images[name],
            config["walkthrough"],
            float(loc["depth_window_spread_m"]),
            float(loc["depth_max_m"]),
        )
        print(f"  {name} matches {n_raw} verified {n_ver} lifted {obj.shape[0]}", flush=True)
        if obj.shape[0]:
            obj_parts.append(obj)
            img_parts.append(img)
    if not obj_parts:
        raise LocalizationError("None of the retrieved frames produced a depth-backed 3D point.")
    obj = np.concatenate(obj_parts, axis=0)
    img = np.concatenate(img_parts, axis=0)
    solved = solve_pnp(obj, img, intrinsics, float(loc["ransac_reproj_px"]), int(loc.get("pnp_iterations", 10000)))
    min_inliers = int(loc["min_inliers"])
    max_med = float(loc["max_median_reproj_px"])
    if solved["inliers"] < min_inliers or solved["median_reproj_px"] > max_med:
        raise LocalizationError(
            f"Pose rejected: {solved['inliers']} inliers, "
            f"median reprojection {solved['median_reproj_px']:.2f} px "
            f"(need >= {min_inliers} inliers and <= {max_med:.1f} px)."
        )
    result = {
        "R": solved["R"].tolist(),
        "t": solved["t"].tolist(),
        "camera_center_m": solved["camera_center_m"].tolist(),
        "inliers": solved["inliers"],
        "median_reproj_px": solved["median_reproj_px"],
        "retrieved_frames": [c[1] for c in candidates[: int(loc["top_k"])]],
    }
    if overlay_path is not None:
        xyz, rgb = read_points_xyz_rgb(Path(config["paths"]["points3d_bin"]))
        _write_overlay(query_bgr, solved["R"], solved["t"], intrinsics, xyz, rgb, Path(overlay_path))
    return result


def _write_overlay(
    query_bgr: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cam: dict,
    xyz: np.ndarray,
    rgb: np.ndarray,
    path: Path,
) -> None:
    x_cam = world_to_cam(R, t, xyz)
    z = x_cam[:, 2]
    keep = z > 0.05
    if not np.any(keep):
        raise LocalizationError("Solved pose puts every map point behind the camera.")
    obj = xyz[keep]
    colors = rgb[keep]
    proj, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t.reshape(3, 1), _K(cam), _dist(cam))
    vis = query_bgr.copy()
    h, w = vis.shape[:2]
    pts = proj.reshape(-1, 2)
    step = max(1, pts.shape[0] // 8000)
    for i in range(0, pts.shape[0], step):
        u = int(round(float(pts[i, 0])))
        v = int(round(float(pts[i, 1])))
        if u < 0 or v < 0 or u >= w or v >= h:
            continue
        color = (int(colors[i, 2]), int(colors[i, 1]), int(colors[i, 0]))
        cv2.circle(vis, (u, v), 2, color, -1, lineType=cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), vis)


def write_pose(path: Path, pose: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pose, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Localize one fixed camera in the lab map.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera", required=True, help="Key under cameras: in the config, for example cam1.")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    cameras = config.get("cameras") or {}
    if args.camera not in cameras:
        print(f"No camera '{args.camera}' in the config.", file=sys.stderr)
        return 2
    intrinsics = cameras[args.camera]
    if not _intrinsics_ready(intrinsics):
        print(
            f"{args.camera} intrinsics are empty. Put fx, fy, cx, cy, width, height, and dist "
            "in config.yaml. These are not the walkthrough camera's numbers.",
            file=sys.stderr,
        )
        return 2
    query = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if query is None:
        print(f"Could not read {args.image}", file=sys.stderr)
        return 2
    out_dir = Path(config["paths"]["output_dir"])
    pose = localize_still(
        query,
        intrinsics,
        config,
        overlay_path=out_dir / f"{args.camera}_overlay.png",
    )
    write_pose(out_dir / f"{args.camera}_pose.json", pose)
    center = pose["camera_center_m"]
    print(
        f"{args.camera} center {center[0]:.3f} {center[1]:.3f} {center[2]:.3f} m, "
        f"inliers {pose['inliers']}, median reproj {pose['median_reproj_px']:.2f} px"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LocalizationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
