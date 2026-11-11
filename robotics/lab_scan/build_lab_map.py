"""Install walkthrough #2 into lab_map/.

Walkthrough2 ships with COLMAP already in meters. splat.ply stays in the
nerfstudio dataparser frame; lab_map_frame.json holds the transform back to
COLMAP meters. Do not re-scale the sparse model.
"""

from __future__ import annotations

import json
import math
import re
import struct
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:\Jessica\Jessica\robotics\lab_scan")
SRC = ROOT / "drive_train" / "walkthrough2"
DEPTH_DIR = ROOT / "walkthrough_20261106_110839" / "depth"
SPARSE = SRC / "colmap" / "sparse" / "0"
OUT = ROOT / "lab_map"
ZIP_PATH = ROOT / "lab_map.zip"

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def qvec_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    c = np.zeros(3, dtype=np.float64)
    for i in range(3):
        c[i] = -(R[0, i] * t[0] + R[1, i] * t[1] + R[2, i] * t[2])
    return c


def read_cameras(path: Path) -> dict[int, dict]:
    cams: dict[int, dict] = {}
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            name, nparam = CAMERA_MODELS[model_id]
            params = struct.unpack("<" + "d" * nparam, f.read(8 * nparam))
            cams[cam_id] = {
                "model": name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cams


def read_images(path: Path) -> dict[int, dict]:
    images: dict[int, dict] = {}
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
            raw = np.fromfile(f, dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]), count=n2)
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name_b.decode("utf-8"),
                "xy": np.stack([raw["x"], raw["y"]], axis=1) if n2 else np.zeros((0, 2)),
                "point3d_ids": raw["id"].astype(np.int64) if n2 else np.zeros(0, dtype=np.int64),
            }
    return images


def read_points(path: Path) -> list[dict]:
    points = []
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            track_n = struct.unpack("<Q", f.read(8))[0]
            track = np.fromfile(f, dtype=np.dtype([("im", "<i4"), ("idx", "<i4")]), count=track_n)
            points.append(
                {
                    "id": int(pid),
                    "xyz": np.array(xyz, dtype=np.float64),
                    "rgb": rgb,
                    "error": float(err),
                    "track": track,
                }
            )
    return points


def frame_number(name: str) -> int | None:
    m = re.search(r"(\d+)", Path(name).stem)
    if not m:
        return None
    return int(m.group(1))


def depth_path_for(name: str) -> Path | None:
    n = frame_number(name)
    if n is None:
        return None
    for cand in (DEPTH_DIR / f"{n:06d}.png", DEPTH_DIR / f"{n - 1:06d}.png"):
        if cand.is_file():
            return cand
    return None


def choose_depth_shift(images: dict[int, dict]) -> int:
    """0 if frame_000004 maps to 000004.png, 1 if the names are 1-based."""
    nums = [frame_number(im["name"]) for im in images.values()]
    nums = [n for n in nums if n is not None]
    lo, hi = min(nums), max(nums)
    for shift in (0, 1):
        if (DEPTH_DIR / f"{lo - shift:06d}.png").is_file() and (
            DEPTH_DIR / f"{hi - shift:06d}.png"
        ).is_file():
            return shift
    raise RuntimeError(f"depth pngs do not match frame numbers {lo}..{hi}")


def fx_fy_cx_cy(cam: dict) -> tuple[float, float, float, float]:
    p = cam["params"]
    if cam["model"] in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
        return float(p[0]), float(p[1]), float(p[2]), float(p[3])
    if cam["model"] in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "FOV"):
        return float(p[0]), float(p[0]), float(p[1]), float(p[2])
    raise RuntimeError(f"unsupported camera model {cam['model']}")


def metric_scale(images: dict[int, dict], points: list[dict], cameras: dict[int, dict], shift: int):
    by_image: dict[int, list[tuple[float, float, float]]] = {i: [] for i in images}
    for pt in points:
        x, y, z = (float(pt["xyz"][0]), float(pt["xyz"][1]), float(pt["xyz"][2]))
        for im_id, idx in pt["track"]:
            im = images.get(int(im_id))
            if im is None or idx < 0 or idx >= im["xy"].shape[0]:
                continue
            R = im["R"]
            t = im["tvec"]
            zc = R[2, 0] * x + R[2, 1] * y + R[2, 2] * z + float(t[2])
            if not math.isfinite(zc) or zc <= 1e-8:
                continue
            u = float(im["xy"][idx, 0])
            v = float(im["xy"][idx, 1])
            by_image[int(im_id)].append((u, v, zc))

    frame_medians = []
    all_ratios = []
    used_frames = 0
    for im_id, obs in by_image.items():
        if len(obs) < 30:
            continue
        im = images[im_id]
        n = frame_number(im["name"])
        if n is None:
            continue
        depth_file = DEPTH_DIR / f"{(n - shift):06d}.png"
        if not depth_file.is_file():
            continue
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            continue
        h, w = depth.shape
        ratios = []
        for u, v, zc in obs:
            ui = int(round(u))
            vi = int(round(v))
            if ui < 0 or vi < 0 or ui >= w or vi >= h:
                continue
            d_mm = int(depth[vi, ui])
            if d_mm <= 0:
                continue
            ratios.append(d_mm / zc)
        if len(ratios) < 30:
            continue
        med = float(np.median(np.asarray(ratios, dtype=np.float64)))
        frame_medians.append(med)
        all_ratios.extend(ratios)
        used_frames += 1
        print(f"  {im['name']} depth pairs {len(ratios)} median mm/unit {med:.4f}", flush=True)

    if not frame_medians:
        raise RuntimeError("No frame had enough depth pairs to scale the map.")
    fm = np.asarray(frame_medians, dtype=np.float64)
    ar = np.asarray(all_ratios, dtype=np.float64)
    return {
        "frames_used": used_frames,
        "observations": int(ar.size),
        "median_mm_per_unit_across_frames": float(np.median(fm)),
        "p16_mm_per_unit": float(np.percentile(fm, 16)),
        "p84_mm_per_unit": float(np.percentile(fm, 84)),
        "global_median_mm_per_unit": float(np.median(ar)),
    }


def write_cameras_bin(path: Path, cameras: dict[int, dict]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam_id, cam in cameras.items():
            model_id = next(i for i, (name, _) in CAMERA_MODELS.items() if name == cam["model"])
            f.write(struct.pack("<iiQQ", cam_id, model_id, cam["width"], cam["height"]))
            f.write(struct.pack("<" + "d" * len(cam["params"]), *cam["params"]))


def write_images_bin(path: Path, images: dict[int, dict]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, im in images.items():
            q = im["qvec"]
            t = im["tvec"]
            f.write(
                struct.pack(
                    "<idddddddi",
                    image_id,
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float(q[3]),
                    float(t[0]),
                    float(t[1]),
                    float(t[2]),
                    im["camera_id"],
                )
            )
            f.write(im["name"].encode("utf-8") + b"\x00")
            n2 = int(im["xy"].shape[0])
            f.write(struct.pack("<Q", n2))
            rec = np.empty(n2, dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
            if n2:
                rec["x"] = im["xy"][:, 0]
                rec["y"] = im["xy"][:, 1]
                rec["id"] = im["point3d_ids"]
            f.write(rec.tobytes())


def write_points_bin(path: Path, points: list[dict]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pt in points:
            f.write(struct.pack("<Q", pt["id"]))
            f.write(struct.pack("<ddd", float(pt["xyz"][0]), float(pt["xyz"][1]), float(pt["xyz"][2])))
            f.write(struct.pack("<BBB", *pt["rgb"]))
            f.write(struct.pack("<d", pt["error"]))
            track = pt["track"]
            f.write(struct.pack("<Q", int(track.shape[0])))
            f.write(track.tobytes())


def write_cameras_txt(path: Path, cameras: dict[int, dict]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for cam_id, cam in cameras.items():
        params = " ".join(f"{p:.12g}" for p in cam["params"])
        lines.append(f"{cam_id} {cam['model']} {cam['width']} {cam['height']} {params}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images_txt(path: Path, images: dict[int, dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for image_id, im in images.items():
            q = im["qvec"]
            t = im["tvec"]
            f.write(
                f"{image_id} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} {im['camera_id']} {im['name']}\n"
            )
            parts = []
            xy = im["xy"]
            ids = im["point3d_ids"]
            for i in range(xy.shape[0]):
                parts.append(f"{xy[i, 0]:.4f} {xy[i, 1]:.4f} {int(ids[i])}")
            f.write(" ".join(parts) + "\n")


def write_points_txt(path: Path, points: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for pt in points:
            x, y, z = pt["xyz"]
            r, g, b = pt["rgb"]
            tr = " ".join(f"{int(a)} {int(c)}" for a, c in pt["track"])
            f.write(f"{pt['id']} {x:.8g} {y:.8g} {z:.8g} {r} {g} {b} {pt['error']:.8g} {tr}\n")


def scale_splat(src: Path, dst: Path, meters_per_unit: float) -> dict:
    with src.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("splat ply has no end_header")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        payload = np.fromfile(f, dtype="<f4")
    n = 0
    for line in header_lines:
        if line.startswith(b"element vertex"):
            n = int(line.split()[-1])
    if payload.size != n * 62:
        raise RuntimeError(f"splat vertex size {payload.size} is not {n}*62")
    raw = payload.reshape(n, 62).copy()
    finite = np.isfinite(raw[:, 0]) & np.isfinite(raw[:, 1]) & np.isfinite(raw[:, 2])
    huge = finite & (
        (np.abs(raw[:, 0]) > 1e6) | (np.abs(raw[:, 1]) > 1e6) | (np.abs(raw[:, 2]) > 1e6)
    )
    raw[finite, 0:3] *= np.float32(meters_per_unit)
    log_s = np.float32(math.log(meters_per_unit))
    raw[finite, 55:58] += log_s
    out_header = []
    for line in header_lines:
        if line.startswith(b"comment "):
            out_header.append(line)
        elif line.startswith(b"element "):
            out_header.append(b"comment Scaled to meters. Gaussian scales are log-space.\n")
            out_header.append(line)
        else:
            out_header.append(line)
    with dst.open("wb") as f:
        f.writelines(out_header)
        f.write(raw.astype("<f4").tobytes())
    return {"vertices": n, "finite": int(finite.sum()), "runaway_gt_1e6_before_scale": int(huge.sum())}


def install_walkthrough2() -> None:
    """Copy the pre-scaled walkthrough2 package into lab_map/."""
    import shutil

    if not (SPARSE / "images.bin").exists():
        raise SystemExit(f"Missing {SPARSE / 'images.bin'}. Unpack the Drive zip into {SRC}.")
    OUT.mkdir(parents=True, exist_ok=True)
    sparse_out = OUT / "colmap" / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        shutil.copy2(SPARSE / name, sparse_out / name)
    for name in ("splat.ply", "transforms.json", "scale.json", "NOTES.md", "dataparser_transforms.json"):
        shutil.copy2(SRC / name, OUT / name)
    dp = json.loads((SRC / "dataparser_transforms.json").read_text(encoding="utf-8"))
    frame = {
        "note": "splat.ply is in the nerfstudio dataparser frame. COLMAP sparse/0 is already meters.",
        "source": "walkthrough2 / alab_walkthrough2_splat_20261106.zip",
        "scale": float(dp["scale"]),
        "transform_colmap_m_to_splat": dp["transform"],
        "meters_from_splat_ply": "p_colmap_m = R.T @ (p_splat / scale - t), with R=transform[:,:3], t=transform[:,3]",
        "splat_from_colmap_m": "p_splat = scale * (R @ p_colmap_m + t)",
        "do_not_also_apply": "transforms.json applied_transform (already folded into dataparser transform)",
        "raw_colmap_to_meters": "already applied in the shipped sparse model (see scale.json)",
    }
    (OUT / "lab_map_frame.json").write_text(json.dumps(frame, indent=2) + "\n", encoding="utf-8")
    print(f"Installed walkthrough2 into {OUT}", flush=True)
    print(f"  images.bin {(sparse_out / 'images.bin').stat().st_size} bytes", flush=True)
    print(f"  splat.ply {(OUT / 'splat.ply').stat().st_size} bytes", flush=True)


def legacy_scale_unscaled_colmap() -> None:
    print("reading COLMAP", flush=True)
    cameras = read_cameras(SPARSE / "cameras.bin")
    images = read_images(SPARSE / "images.bin")
    points = read_points(SPARSE / "points3D.bin")
    for im in images.values():
        q = im["qvec"]
        im["R"] = qvec_to_R(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    registered = len(images)
    names = {im["name"] for im in images.values()}
    print(f"registered images {registered}", flush=True)
    print(f"example name {next(iter(images.values()))['name']}", flush=True)
    print(f"points {len(points)} cameras {len(cameras)}", flush=True)

    weight = 0.0
    acc = 0.0
    for pt in points:
        ntr = float(pt["track"].shape[0])
        acc += pt["error"] * ntr
        weight += ntr
    mean_error = acc / weight if weight else float("nan")
    print(f"mean reprojection error px {mean_error:.4f}", flush=True)

    shift = choose_depth_shift(images)
    print(f"depth filename shift {shift}", flush=True)
    print("scaling from depth", flush=True)
    stats = metric_scale(images, points, cameras, shift)
    mm_per = stats["median_mm_per_unit_across_frames"]
    meters_per = mm_per / 1000.0
    print(
        f"median {mm_per:.4f} mm/unit  "
        f"p16 {stats['p16_mm_per_unit']:.4f}  p84 {stats['p84_mm_per_unit']:.4f}  "
        f"-> {meters_per:.6f} m/unit",
        flush=True,
    )

    # Confirm Nerfstudio only swapped axes, so one factor scales the splat too.
    transforms = json.loads((SRC / "transforms.json").read_text(encoding="utf-8"))
    by_id = {int(fr["colmap_im_id"]): fr for fr in transforms["frames"]}
    center_err = []
    for image_id, im in images.items():
        fr = by_id.get(image_id)
        if fr is None:
            continue
        c = camera_center(im["R"], im["tvec"])
        pred = np.array([c[0], c[2], -c[1]], dtype=np.float64)
        got = np.array([fr["transform_matrix"][i][3] for i in range(3)], dtype=np.float64)
        center_err.append(float(np.max(np.abs(pred - got))))
    center_err_a = np.asarray(center_err, dtype=np.float64)
    print(
        f"nerfstudio vs colmap center max-abs median {float(np.median(center_err_a)):.6g} "
        f"max {float(np.max(center_err_a)):.6g}",
        flush=True,
    )

    for im in images.values():
        im["tvec"] = im["tvec"] * meters_per
    for pt in points:
        pt["xyz"] = pt["xyz"] * meters_per

    for fr in transforms["frames"]:
        for row in range(3):
            fr["transform_matrix"][row][3] *= meters_per

    missing = []
    registered_idx = set()
    for im in images.values():
        n = frame_number(im["name"])
        if n is None:
            missing.append(im["name"])
            continue
        registered_idx.add(n - shift)
    for i in range(374):
        if i not in registered_idx:
            missing.append(f"{i:06d}")

    OUT.mkdir(parents=True, exist_ok=True)
    sparse_out = OUT / "colmap" / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)
    print("writing scaled sparse model", flush=True)
    write_cameras_bin(sparse_out / "cameras.bin", cameras)
    write_images_bin(sparse_out / "images.bin", images)
    write_points_bin(sparse_out / "points3D.bin", points)
    write_cameras_txt(sparse_out / "cameras.txt", cameras)
    write_images_txt(sparse_out / "images.txt", images)
    write_points_txt(sparse_out / "points3D.txt", points)
    (sparse_out / "project.ini").write_text((SPARSE / "project.ini").read_text(encoding="utf-8"), encoding="utf-8")

    (OUT / "transforms.json").write_text(json.dumps(transforms, indent=4) + "\n", encoding="utf-8")
    print("scaling splat", flush=True)
    splat_info = scale_splat(SRC / "splat_raw.ply", OUT / "splat.ply", meters_per)

    scale_doc = {
        "units": "meters",
        "meters_per_original_unit": meters_per,
        "mm_per_original_unit": mm_per,
        "definition": (
            "For each registered frame, COLMAP camera-z of a 3D point divided into "
            "the sensor depth (mm) at that keypoint. 0 depth is ignored. "
            "The factor is the median of the per-frame medians. "
            "Multiply original COLMAP xyz, tvec, splat xyz, and nerfstudio translations by meters_per_original_unit."
        ),
        "spread_mm_per_unit_p16": stats["p16_mm_per_unit"],
        "spread_mm_per_unit_p84": stats["p84_mm_per_unit"],
        "global_median_mm_per_unit": stats["global_median_mm_per_unit"],
        "frames_used_for_scale": stats["frames_used"],
        "depth_observations": stats["observations"],
        "depth_index_shift": shift,
        "nerfstudio_center_agreement_median_original_units": float(np.median(center_err_a)),
    }
    (OUT / "scale.json").write_text(json.dumps(scale_doc, indent=2) + "\n", encoding="utf-8")

    cam = next(iter(cameras.values()))
    fx, fy, cx, cy = fx_fy_cx_cy(cam)
    missing_txt = ", ".join(missing) if missing else "none"
    notes = f"""# Lab walkthrough map

Units are meters. The original COLMAP reconstruction and the Nerfstudio splat shared one arbitrary unit. `scale.json` is the median sensor-depth ratio that converts that unit to meters.

## Coordinates

COLMAP sparse model (`colmap/sparse/0`): camera x right, y down, z forward. World axes are the original reconstruction axes, only scaled. A camera pose is `X_cam = R(qvec) X + tvec`.

`splat.ply` and `transforms.json`: Nerfstudio frame. `applied_transform` maps a COLMAP world point `(x, y, z)` to `(x, z, -y)`, and the splat comment says vertical axis is +z. `transform_matrix` is camera-to-world in that frame. The camera axes in that matrix follow Nerfstudio (x right, y up, z backward).

Gaussian `scale_0/1/2` are log-lengths and were shifted by ln(scale). Rotations were left as trained.

## Registration

Registered frames: {registered} of 374.
Mean reprojection error: {mean_error:.4f} px (track-length weighted mean of the per-point COLMAP errors).
Frames with no pose: {missing_txt}.

Walkthrough camera: Orbbec Gemini 335L, {cam['width']}x{cam['height']}, model {cam['model']}, fx {fx:.3f}, fy {fy:.3f}, cx {cx:.3f}, cy {cy:.3f}.

## Scale

Median {mm_per:.2f} mm per original unit ({meters_per:.6f} m/unit), from {stats['frames_used']} frames and {stats['observations']} depth pairs.
Per-frame medians sit between {stats['p16_mm_per_unit']:.2f} and {stats['p84_mm_per_unit']:.2f} mm/unit (16th to 84th percentile).

Depth PNGs are the local walkthrough (`walkthrough_20261105_182629/depth`), uint16 millimeters, 0 = no reading, aligned to color. Filename match uses the number in the COLMAP image name minus {shift}.

## What is not in this package

`database.db` was not in the Drive folder, so SIFT descriptors are not included. `images.bin` still has every keypoint pixel and its 3D point id. Registering cam1 or cam2 needs a COLMAP database built from the walkthrough images, then `image_registrator` with this sparse model and that camera's own intrinsics (they are not the walkthrough fx/fy/cx/cy).

Splat vertices: {splat_info['vertices']}. Finite: {splat_info['finite']}. Vertices whose original coordinate exceeded 1e6: {splat_info['runaway_gt_1e6_before_scale']} (left in place).
"""
    (OUT / "NOTES.md").write_text(notes, encoding="utf-8")

    print("zipping", flush=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in OUT.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(OUT).as_posix())
    print(f"zip {ZIP_PATH} bytes {ZIP_PATH.stat().st_size}", flush=True)


def main() -> None:
    install_walkthrough2()


if __name__ == "__main__":
    main()
