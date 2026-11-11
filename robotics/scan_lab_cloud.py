"""Live lab scan from the fixed cameras, placed in the walkthrough map.

Each fixed camera's depth frame becomes a colored surface (not dots) and is
placed with the pose that maploc/localize.py solved against the walkthrough map.
No checkerboard is needed. The photoreal Gaussian splat of the lab is drawn
behind the live surfaces, in the same frame (COLMAP map frame, meters).

Before the first run, and again whenever a camera is bumped:
    cd lab_scan\\maploc
    python localize.py --config config.yaml --camera cam0 --image <still from cam0>
    python localize.py --config config.yaml --camera cam1 --image <still from cam1>

Then:
    python scan_lab_cloud.py            (Orbbec SDK, one camera at a time; stop ROS / Orbbec Viewer first)
    python scan_lab_cloud.py --ros      (read the ROS topics instead, if the publishers are running)
    python scan_lab_cloud.py --frames saved.npz   (offline, from a saved grab)
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import warnings
import webbrowser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ROS_WS = ROOT.parent / "ros2_ws"
for p in (
    ROS_WS / "install" / "Lib" / "site-packages",
    ROS_WS / "src" / "orbbec_dual_sync",
    ROOT,
):
    s = str(p)
    if p.is_dir() and s not in sys.path:
        sys.path.insert(0, s)

# Fixed cameras that have a map pose. Index 2 is the handheld walkthrough camera.
CAMS = ("cam0", "cam1")
CAM_INDEX = {"cam0": 0, "cam1": 1}
CAM_RGB = {"cam0": (230, 120, 60), "cam1": (60, 170, 220)}

OUT_DIR = ROOT / "lab_scan"
MAPLOC_DIR = OUT_DIR / "maploc"
MAPLOC_CONFIG = MAPLOC_DIR / "config.yaml"
SPLAT_PATH = OUT_DIR / "lab_map" / "lab_colmap.splat"
SPLAT_FRAME = OUT_DIR / "lab_map" / "lab_colmap_frame.json"

Z_MIN_MM = 150.0
Z_MAX_MM = 4500.0
STRIDE = 2
# Two neighboring depth pixels belong to one surface when their depths differ
# by less than this fraction of the nearer one (or EDGE_MIN_M).
EDGE_FRAC = 0.04
EDGE_MIN_M = 0.03


def _stamp_ns(msg) -> int:
    st = msg.header.stamp
    return int(st.sec) * 1_000_000_000 + int(st.nanosec)


def grab_usb() -> dict:
    """One color+depth frame per camera index. Used when ROS topics are down."""
    import depthcamera as dc

    got: dict[str, dict] = {}
    for cam in CAMS:
        device = dc.OrbbecDepthCamera(camera_index=CAM_INDEX[cam])
        try:
            device.open(require_frame=True, retries=2, warm_tries=30)
            color, depth = device.read_images()
            if color is None or depth is None:
                raise RuntimeError(f"{cam}: USB opened but returned no frame")
            k = None
            if device.capture is not None:
                k = dc.intrinsics_from_capture(device.capture, int(color.shape[1]), int(color.shape[0]))
            if k is None:
                raise RuntimeError(f"{cam}: camera did not report intrinsics")
            fx, fy, cx, cy = k
            got[cam] = {
                "depth": depth,
                "color": color,
                "color_hw": (int(color.shape[0]), int(color.shape[1])),
                "depth_stamp_ns": 0,
                "info": {
                    "fx": float(fx),
                    "fy": float(fy),
                    "cx": float(cx),
                    "cy": float(cy),
                    "width": int(color.shape[1]),
                    "height": int(color.shape[0]),
                },
                "source": "usb",
            }
        finally:
            device.close()
            time.sleep(0.4)
    return got


def grab(timeout_s: float = 8.0) -> dict:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    from orbbec_dual_sync.image_codec import imgmsg_to_numpy

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    got: dict[str, dict] = {c: {} for c in CAMS}
    rclpy.init()
    node = Node("lab_scan_once")

    def on_depth(cam):
        def cb(msg: Image) -> None:
            try:
                got[cam]["depth"] = imgmsg_to_numpy(msg)
                got[cam]["depth_stamp_ns"] = _stamp_ns(msg)
            except Exception as exc:
                got[cam]["depth_error"] = str(exc)

        return cb

    def on_color(cam):
        def cb(msg: Image) -> None:
            try:
                got[cam]["color"] = imgmsg_to_numpy(msg)
                got[cam]["color_hw"] = (int(msg.height), int(msg.width))
            except Exception as exc:
                got[cam]["color_error"] = str(exc)

        return cb

    def on_info(cam):
        def cb(msg: CameraInfo) -> None:
            k = list(msg.k)
            got[cam]["info"] = {
                "fx": float(k[0]),
                "fy": float(k[4]),
                "cx": float(k[2]),
                "cy": float(k[5]),
                "width": int(msg.width),
                "height": int(msg.height),
            }

        return cb

    for cam in CAMS:
        idx = CAM_INDEX[cam]
        node.create_subscription(Image, f"/orbbec/cam{idx}/depth/image_raw", on_depth(cam), qos)
        node.create_subscription(Image, f"/orbbec/cam{idx}/color/image_raw", on_color(cam), qos)
        node.create_subscription(CameraInfo, f"/orbbec/cam{idx}/color/camera_info", on_info(cam), qos)

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.05)
        if all("depth" in got[c] and "info" in got[c] for c in CAMS):
            break
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass
    missing = [c for c in CAMS if "depth" not in got[c] or "info" not in got[c]]
    if missing:
        raise TimeoutError(
            "No depth+camera_info from "
            + ", ".join(missing)
            + ". Are the Orbbec publishers running?"
        )
    return got


# ---------------------------------------------------------------- Orbbec SDK grab (Gemini 335L)
# OpenCV CAP_OBSENSOR enumerates the Gemini 335L cameras but never delivers frames
# on this PC, so the default path uses pyorbbecsdk2 (same approach as record_walkthrough.py).
#
# In --watch mode each camera is opened once and kept streaming. Starting and stopping the cameras
# every few seconds is what produced "setXu failed" / "Error stopping pipeline" and made a camera drop
# off USB. A camera that errors is closed and reopened on a later round. A camera is only ever used
# when its serial number matches; it is never swapped for whatever is plugged in at that index.

SDK_COLOR_W, SDK_COLOR_H = 640, 480   # the map poses were solved on 640x480 stills
DEPTH_AVG_FRAMES = 5   # depth = median of this many consecutive frames (~1/3 s at 15 fps)
SDK_FLUSH_FRAMES = 8   # frames to drop at the start of each read (they piled up while processing)
SDK_REOPEN_S = 20      # after a camera fails, wait this long before trying it again
_SDK: dict = {"ctx": None, "open": {}, "failed_at": {}}


def _camera_serials() -> dict[str, str]:
    try:
        import yaml

        cfg = yaml.safe_load(MAPLOC_CONFIG.read_text(encoding="utf-8")) or {}
        return {k: str(v.get("serial")) for k, v in (cfg.get("cameras") or {}).items() if v.get("serial")}
    except Exception:
        return {}


def _sdk_devices():
    from pyorbbecsdk import Context

    if _SDK["ctx"] is None:
        _SDK["ctx"] = Context()
    dev_list = _SDK["ctx"].query_devices()
    out = []
    for i in range(dev_list.get_count()):
        try:
            d = dev_list.get_device_by_index(i)
            out.append((i, d, d.get_device_info().get_serial_number()))
        except Exception as exc:
            print(f"Orbbec device {i}: could not read it ({exc})")
    return out


def _sdk_open(cam: str, device, serial: str) -> dict:
    from pyorbbecsdk import AlignFilter, Config, OBFormat, OBSensorType, OBStreamType, Pipeline

    try:
        from pyorbbecsdk import OBFrameAggregateOutputMode
    except ImportError:
        OBFrameAggregateOutputMode = None
    pipe = Pipeline(device)
    cfg = Config()
    cprof = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = None
    for fmt in (OBFormat.MJPG, OBFormat.RGB):
        for fps in (15, 30, 10, 5):
            try:
                color_profile = cprof.get_video_stream_profile(SDK_COLOR_W, SDK_COLOR_H, fmt, fps)
                break
            except Exception:
                continue
        if color_profile is not None:
            break
    if color_profile is None:
        color_profile = cprof.get_default_video_stream_profile()
    cfg.enable_stream(color_profile)
    cfg.enable_stream(pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile())
    if OBFrameAggregateOutputMode is not None:
        cfg.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    try:
        pipe.enable_frame_sync()
    except Exception:
        pass
    pipe.start(cfg)
    st = {"pipe": pipe, "align": AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM), "serial": serial, "info": None}
    print(f"{cam}: opened serial {serial}")
    return st


def _sdk_close(cam: str) -> None:
    st = _SDK["open"].pop(cam, None)
    if st:
        try:
            st["pipe"].stop()
        except Exception:
            pass


def sdk_close_all() -> None:
    for cam in list(_SDK["open"]):
        _sdk_close(cam)


def _sdk_read(cam: str, st: dict, warm_frames: int, timeout_s: float) -> dict:
    import cv2
    from pyorbbecsdk import OBFormat

    pipe, align = st["pipe"], st["align"]
    color = depth = None
    recent: list[np.ndarray] = []
    n_ok = 0
    t0 = time.time()
    # Frames queued while the scan was busy are old: throw away whatever is already waiting.
    for _ in range(SDK_FLUSH_FRAMES):
        try:
            if not pipe.wait_for_frames(1):
                break
        except Exception:
            break
    while time.time() - t0 < timeout_s:
        frames = pipe.wait_for_frames(200)
        if not frames:
            continue
        frames = align.process(frames)
        if not frames:
            continue
        if hasattr(frames, "as_frame_set"):
            frames = frames.as_frame_set()
        cf, df = frames.get_color_frame(), frames.get_depth_frame()
        if cf is None or df is None:
            continue
        data = np.frombuffer(cf.get_data(), dtype=np.uint8)
        fmt = cf.get_format()
        if fmt == OBFormat.MJPG:
            c = cv2.imdecode(data, cv2.IMREAD_COLOR)
        elif fmt == OBFormat.RGB:
            c = cv2.cvtColor(data.reshape(cf.get_height(), cf.get_width(), 3), cv2.COLOR_RGB2BGR)
        else:
            c = data.reshape(cf.get_height(), cf.get_width(), 3)
        if c is None:
            continue
        raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(df.get_height(), df.get_width())
        scale = float(df.get_depth_scale()) or 1.0
        color, depth = c, raw.astype(np.float32) * scale
        n_ok += 1
        if n_ok > warm_frames - DEPTH_AVG_FRAMES:
            recent.append(depth)
        if n_ok >= warm_frames:
            break
    if color is None or depth is None:
        raise RuntimeError(f"no color+depth in {timeout_s:.0f} s")
    if len(recent) >= 3:
        # Median of the last few depth frames: single frames are noisy by a few cm at 2-3 m.
        stk = np.stack(recent).astype(np.float32)
        stk[stk <= 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(stk, axis=0)
            valid_n = np.isfinite(stk).sum(axis=0)
        depth = np.where(valid_n * 2 > len(recent), med, 0.0).astype(np.float32)
    if st["info"] is None or (st["info"]["width"], st["info"]["height"]) != (color.shape[1], color.shape[0]):
        param = pipe.get_camera_param()
        rgb = param.rgb_intrinsic
        fx, fy, cx, cy = float(rgb.fx), float(rgb.fy), float(rgb.cx), float(rgb.cy)
        rw, rh = int(getattr(rgb, "width", 0) or 0), int(getattr(rgb, "height", 0) or 0)
        if rw and rh and (rw, rh) != (color.shape[1], color.shape[0]):
            sx, sy = color.shape[1] / rw, color.shape[0] / rh
            fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        dist = [0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            d = param.rgb_distortion
            dist = [float(d.k1), float(d.k2), float(d.p1), float(d.p2), float(d.k3)]
        except Exception:
            pass
        st["info"] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "dist": dist, "from_sdk": True,
                      "width": int(color.shape[1]), "height": int(color.shape[0])}
        print(f"{cam}: lens fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f}")
    return {"depth": depth, "color": color, "info": dict(st["info"]), "color_hw": color.shape[:2],
            "depth_stamp_ns": 0, "source": "sdk", "serial": st["serial"]}


def grab_sdk(warm_frames: int = 20, timeout_s: float = 8.0, keep_open: bool = False) -> dict:
    """One color frame + depth aligned to it, per fixed camera.

    keep_open=True (watch mode): cameras stay streaming between captures, so later captures only need a
    few fresh frames and nothing is started/stopped every round."""
    serials = _camera_serials()
    got: dict[str, dict] = {}
    need_open = [c for c in CAMS if c not in _SDK["open"]]
    devices = _sdk_devices() if need_open else []
    if need_open:
        print("Orbbec devices: " + (", ".join(f"{i}={s}" for i, _d, s in devices) or "none"))
    for cam in CAMS:
        want = serials.get(cam)
        if cam not in _SDK["open"]:
            if time.time() - _SDK["failed_at"].get(cam, 0) < SDK_REOPEN_S:
                print(f"{cam}: waiting before retrying (it failed {time.time() - _SDK['failed_at'][cam]:.0f} s ago)")
                continue
            match = next((x for x in devices if want and x[2] == want), None)
            if match is None and not want:
                match = next((x for x in devices if x[0] == CAM_INDEX[cam]), None)
            if match is None:
                print(f"{cam}: serial {want} is not connected; skipped this round "
                      "(check its USB cable; a camera is never swapped for another one)")
                continue
            try:
                _SDK["open"][cam] = _sdk_open(cam, match[1], match[2])
                first = True
            except Exception as exc:
                print(f"{cam}: could not open ({exc}); will retry")
                _SDK["failed_at"][cam] = time.time()
                continue
        else:
            first = False
        st = _SDK["open"][cam]
        try:
            # once streaming, only a short run of fresh frames is needed (enough for the depth median)
            n = warm_frames if first else DEPTH_AVG_FRAMES + 2
            got[cam] = _sdk_read(cam, st, n, timeout_s)
            fr = got[cam]
            if first:
                print(f"{cam}: serial {fr['serial']}, color {fr['color'].shape[1]}x{fr['color'].shape[0]}, "
                      f"depth aligned, {int((fr['depth'] > 0).mean() * 100)}% valid")
        except Exception as exc:
            print(f"{cam}: read failed ({exc}); closing it and retrying in {SDK_REOPEN_S} s")
            _sdk_close(cam)
            _SDK["failed_at"][cam] = time.time()
            continue
        if not keep_open:
            _sdk_close(cam)
            time.sleep(0.5)
    if not got:
        raise RuntimeError("No fixed camera returned frames through the Orbbec SDK.")
    return got


# ---------------------------------------------------------------- poses / intrinsics


def load_pose(cam: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """World-to-camera pose from maploc (X_cam = R X_map + t), meters."""
    path = MAPLOC_DIR / "out" / f"{cam}_pose.json"
    if not path.is_file():
        raise RuntimeError(
            f"No map pose for {cam} ({path}). Run maploc/localize.py for {cam} with a still from that camera first."
        )
    d = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(d["R"], dtype=np.float64), np.asarray(d["t"], dtype=np.float64), d


def load_lens(cam: str, info: dict, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """K and distortion. The SDK's values for this exact frame win; else maploc config; else camera_info."""
    fx, fy, cx, cy = float(info["fx"]), float(info["fy"]), float(info["cx"]), float(info["cy"])
    dist = np.zeros(5, dtype=np.float64)
    if info.get("from_sdk"):
        if info.get("dist"):
            dist = np.asarray(info["dist"], dtype=np.float64)
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64), dist
    try:
        import yaml

        cfg = yaml.safe_load(MAPLOC_CONFIG.read_text(encoding="utf-8")) or {}
        c = (cfg.get("cameras") or {}).get(cam) or {}
        if c.get("fx") and int(c.get("width", 0)) == hw[1] and int(c.get("height", 0)) == hw[0]:
            fx, fy, cx, cy = float(c["fx"]), float(c["fy"]), float(c["cx"]), float(c["cy"])
            if c.get("dist"):
                dist = np.asarray(c["dist"], dtype=np.float64)
    except Exception as exc:  # config is optional here
        print(f"{cam}: using stream intrinsics ({exc})")
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, dist


def locate_camera(cam: str, frame: dict, K: np.ndarray, dist: np.ndarray, relocalize: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pose of this camera in the map, from this grab's own color frame.

    The saved pose is reused only when it was solved for the same camera serial, image size and
    lens. Otherwise (or with --relocalize) the fresh color frame is matched against the
    walkthrough map with maploc/localize.py, so the pose always fits the frame being drawn.
    """
    path = MAPLOC_DIR / "out" / f"{cam}_pose.json"
    color = frame.get("color")
    h, w = (np.asarray(color).shape[:2] if color is not None else (0, 0))
    serial = frame.get("serial")
    if path.is_file() and not relocalize:
        d = json.loads(path.read_text(encoding="utf-8"))
        same = (
            d.get("serial") == serial and serial is not None
            and d.get("size") == [int(w), int(h)]
            and abs(float(d.get("fx", 0)) - float(K[0, 0])) < 0.01 * float(K[0, 0])
        )
        if same:
            print(f"{cam}: reusing map pose solved {d.get('solved_at', '?')} for serial {serial}")
            return np.asarray(d["R"], dtype=np.float64), np.asarray(d["t"], dtype=np.float64), d
    if color is None:
        return load_pose(cam)
    print(f"{cam}: locating this camera in the map from the frame just captured (about a minute)...")
    try:
        if str(MAPLOC_DIR) not in sys.path:
            sys.path.insert(0, str(MAPLOC_DIR))
        import localize  # maploc/localize.py

        cfg = localize.load_config(MAPLOC_CONFIG)
        intr = {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                "width": int(w), "height": int(h), "dist": [float(x) for x in np.ravel(dist)]}
        out_dir = MAPLOC_DIR / "out"
        res = localize.localize_still(np.asarray(color), intr, cfg, overlay_path=out_dir / f"{cam}_overlay.png")
    except Exception as exc:
        print(f"{cam}: could not locate from this frame ({exc}).")
        if path.is_file():
            print(f"{cam}: falling back to the saved pose. It may be off if it came from a different camera or lens.")
            return load_pose(cam)
        raise RuntimeError(f"{cam}: no usable map pose.") from exc
    res.update({"serial": serial, "size": [int(w), int(h)], "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                "cx": float(K[0, 2]), "cy": float(K[1, 2]), "dist": [float(x) for x in np.ravel(dist)],
                "solved_at": time.strftime("%Y-%m-%d %H:%M"), "solved_from": "scan_lab_cloud live frame"})
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    res["_fresh"] = True
    cv2_still = out_dir / f"{cam}_still.png"
    try:
        import cv2

        cv2.imwrite(str(cv2_still), np.asarray(color))
    except Exception:
        pass
    print(f"{cam}: located with {res['inliers']} matched points, {res['median_reproj_px']:.2f} px, "
          f"camera at {np.round(res['camera_center_m'], 3)} m")
    return np.asarray(res["R"], dtype=np.float64), np.asarray(res["t"], dtype=np.float64), res


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """-R^T t, written out (this PC aborts on numpy matrix products)."""
    return -np.array([R[0, i] * t[0] + R[1, i] * t[1] + R[2, i] * t[2] for i in range(3)], dtype=np.float64)


def cam_to_map(R: np.ndarray, t: np.ndarray, P: np.ndarray) -> np.ndarray:
    """X_map = R^T (X_cam - t) for (N, 3) rows, without matrix products."""
    d0, d1, d2 = P[:, 0] - t[0], P[:, 1] - t[1], P[:, 2] - t[2]
    out = np.empty_like(P)
    for i in range(3):
        out[:, i] = R[0, i] * d0 + R[1, i] * d1 + R[2, i] * d2
    return out


# ---------------------------------------------------------------- depth -> surface


GRAZING_COS = 0.25   # keep triangles facing the camera within ~75 degrees


def changed_mask(depth_mm: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray,
                 splat_bytes: bytes, cell: int = 4, strict: list | None = None) -> np.ndarray:
    """Pixels where the live depth disagrees with the splat: things that moved or appeared (the arm, a
    rack that was carried in). Everything else is better shown by the photoreal splat itself."""
    import cv2

    h, w = depth_mm.shape
    roi = (0, 0, w - w % cell, h - h % cell)
    zs = _splat_depth_roi(R, t, K, dist, _splat_points(splat_bytes), roi, cell)
    zl = _live_depth_roi(depth_mm, roi, cell)
    gh, gw = zl.shape
    pad = np.pad(zs, 1, constant_values=np.nan)
    best = np.full((gh, gw), np.inf)
    anyv = np.zeros((gh, gw), dtype=bool)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            nb = pad[dy:dy + gh, dx:dx + gw]
            ok = np.isfinite(nb)
            anyv |= ok
            best = np.where(ok, np.minimum(best, np.abs(zl - np.where(ok, nb, 0))), best)
    z = np.nan_to_num(zl)
    tol = CHANGE_TOL_M + CHANGE_TOL_Z2 * z * z          # depth noise grows with distance squared
    ch = np.isfinite(zl) & (z < CHANGE_MAX_M) & anyv & (best > tol)
    ch8 = ch.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ch8, connectivity=8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < CHANGE_MIN_CELLS:
            ch8[lab == i] = 0
    if strict is not None:
        # Per-pixel version for tracking: a pixel counts only if its own depth differs from every nearby
        # splat surface. (The cell mask above is blocky and padded for display; around the arm that
        # padding is bench, which made the bench count as "arm" and every clearance look like ~5 cm.)
        cells = cv2.dilate(ch8, np.ones((3, 3), np.uint8)).astype(bool)
        Dp = np.asarray(depth_mm, dtype=np.float64)[: gh * cell, : gw * cell] / 1000.0
        bestp = np.full(Dp.shape, np.inf)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                nb = np.repeat(np.repeat(pad[dy:dy + gh, dx:dx + gw], cell, 0), cell, 1)
                ok = np.isfinite(nb)
                bestp = np.where(ok, np.minimum(bestp, np.abs(Dp - np.where(ok, nb, 0))), bestp)
        tolp = CHANGE_TOL_M + CHANGE_TOL_Z2 * Dp * Dp
        sp = np.repeat(np.repeat(cells, cell, 0), cell, 1) & (Dp > 0) & np.isfinite(bestp) & (bestp > tolp)
        sfull = np.zeros((h, w), dtype=bool)
        sfull[: gh * cell, : gw * cell] = sp
        strict.append(sfull)
    ch8 = cv2.dilate(ch8, np.ones((3, 3), np.uint8))
    full = np.zeros((h, w), dtype=bool)
    full[: gh * cell, : gw * cell] = np.repeat(np.repeat(ch8.astype(bool), cell, 0), cell, 1)
    return full


CHANGE_TOL_M = 0.05
CHANGE_TOL_Z2 = 0.012
CHANGE_MAX_M = 3.0     # beyond this the splat is shown; live depth there is too noisy to say what changed
CHANGE_MIN_CELLS = 25


def depth_surface(
    depth_mm: np.ndarray,
    color_bgr: np.ndarray | None,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    stride: int = STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangle surface in the map frame: (verts float32 N×3, colors uint8 N×3, faces uint32 M×3).

    Neighboring pixels are joined only when their depths agree, so object edges
    stay open instead of being smeared into curtains. Nothing is filled in where
    the camera has no depth.
    """
    import cv2

    z_img = np.asarray(depth_mm, dtype=np.float64)
    if z_img.ndim == 3:
        z_img = z_img[:, :, 0]
    h, w = z_img.shape
    vs = np.arange(0, h, stride)
    us = np.arange(0, w, stride)
    uu, vv = np.meshgrid(us, vs)
    gh, gw = uu.shape
    pix = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pix, K, dist).reshape(gh, gw, 2)
    z = z_img[vv, uu] / 1000.0
    valid = np.isfinite(z) & (z >= Z_MIN_MM / 1000.0) & (z <= Z_MAX_MM / 1000.0)

    P = np.stack([norm[:, :, 0] * z, norm[:, :, 1] * z, z], axis=2).reshape(-1, 3)
    if color_bgr is not None and np.asarray(color_bgr).shape[:2] == (h, w):
        bgr = np.asarray(color_bgr)[vv, uu].reshape(-1, 3)
        rgb = bgr[:, ::-1].astype(np.uint8)
    else:
        rgb = np.full((P.shape[0], 3), 150, dtype=np.uint8)

    idx = np.arange(gh * gw).reshape(gh, gw)
    zc = np.where(valid, z, np.nan)

    def joined(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        near = np.fmin(a, b)
        return np.abs(a - b) < np.maximum(EDGE_MIN_M, EDGE_FRAC * near)

    z00, z01, z10, z11 = zc[:-1, :-1], zc[:-1, 1:], zc[1:, :-1], zc[1:, 1:]
    i00, i01, i10, i11 = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    ok_a = joined(z00, z10) & joined(z10, z01) & joined(z00, z01)
    ok_b = joined(z01, z10) & joined(z10, z11) & joined(z01, z11)
    faces = np.concatenate(
        [
            np.stack([i00[ok_a], i10[ok_a], i01[ok_a]], axis=1),
            np.stack([i01[ok_b], i10[ok_b], i11[ok_b]], axis=1),
        ]
    )
    # Drop triangles seen almost edge-on: they are the long smeared "curtains" that make the surface look
    # blurry from any angle other than the camera's own.
    if faces.size:
        a, b, cc = P[faces[:, 0]], P[faces[:, 1]], P[faces[:, 2]]
        e1, e2 = b - a, cc - a
        n = np.stack([e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1],
                      e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2],
                      e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]], axis=1)
        ray = (a + b + cc) / 3.0
        cosang = np.abs((n * ray).sum(axis=1)) / (np.sqrt((n * n).sum(axis=1) * (ray * ray).sum(axis=1)) + 1e-12)
        faces = faces[cosang > GRAZING_COS]
    if faces.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros((0, 3), np.uint32)
    used = np.unique(faces)
    remap = np.full(gh * gw, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    verts = cam_to_map(R, t, P[used]).astype(np.float32)
    return verts, rgb[used], remap[faces].astype(np.uint32)


def write_mesh_ply(path: Path, parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]], comment: str) -> None:
    verts, cols, faces, off = [], [], [], 0
    for v, c, f in parts:
        verts.append(v)
        cols.append(c)
        faces.append(f.astype(np.int64) + off)
        off += v.shape[0]
    V = np.concatenate(verts) if verts else np.zeros((0, 3), np.float32)
    C = np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8)
    Fc = np.concatenate(faces) if faces else np.zeros((0, 3), np.int64)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"comment {comment}\n"
        f"element vertex {V.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {Fc.shape[0]}\n"
        "property list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    vrec = np.empty(V.shape[0], dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    vrec["p"], vrec["c"] = V, C
    frec = np.empty(Fc.shape[0], dtype=[("n", "u1"), ("i", "<i4", 3)])
    frec["n"], frec["i"] = 3, Fc
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(vrec.tobytes())
        f.write(frec.tobytes())



# ---------------------------------------------------------------- live carving of the background splat
# The splat is a snapshot from the walkthrough, so the robot arms (and anything else that has
# moved since) are baked into it. Each fixed camera's depth says the space between the camera
# and the measured surface is empty *now*. Splat blobs sitting in that empty space are stale
# and are dropped, so each arm shows once: live, where it is now.

SPLAT_DTYPE = np.dtype([("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
CARVE_MARGIN_M = 0.05   # keep blobs within this distance in front of the live surface
CARVE_MARGIN_FRAC = 0.03   # plus this fraction of the measured depth (depth noise grows with range)


def carve_splat(splat_bytes: bytes, views: list[dict]) -> tuple[bytes, int]:
    """Drop splat blobs that a fixed camera currently sees as empty space. Returns (bytes, dropped)."""
    drop = carve_mask(splat_bytes, views)
    rec = np.frombuffer(splat_bytes, dtype=SPLAT_DTYPE)
    return rec[~drop].tobytes(), int(drop.sum())


def carve_mask(splat_bytes: bytes, views: list[dict]) -> np.ndarray:
    """True for every splat blob some camera currently sees straight through (free space now)."""
    import cv2

    rec = np.frombuffer(splat_bytes, dtype=SPLAT_DTYPE)
    P = rec["p"].astype(np.float64)
    drop = np.zeros(P.shape[0], dtype=bool)
    for v in views:
        R, t, K, dist, depth = v["R"], v["t"], v["K"], v["dist"], v["depth_m"]
        h, w = depth.shape
        # X_cam = R X + t, written out (no numpy matrix products on this PC)
        xc = R[0, 0] * P[:, 0] + R[0, 1] * P[:, 1] + R[0, 2] * P[:, 2] + t[0]
        yc = R[1, 0] * P[:, 0] + R[1, 1] * P[:, 1] + R[1, 2] * P[:, 2] + t[1]
        zc = R[2, 0] * P[:, 0] + R[2, 1] * P[:, 1] + R[2, 2] * P[:, 2] + t[2]
        front = zc > 0.1
        if not np.any(front):
            continue
        cam_pts = np.stack([xc[front], yc[front], zc[front]], axis=1)
        uv, _ = cv2.projectPoints(cam_pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
        uv = uv.reshape(-1, 2)
        u = np.round(uv[:, 0]).astype(np.int64)
        vv = np.round(uv[:, 1]).astype(np.int64)
        inside = (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        idx = np.flatnonzero(front)[inside]
        d = depth[vv[inside], u[inside]]
        z = zc[idx]
        stale = (d > 0.15) & (z < d - (CARVE_MARGIN_M + CARVE_MARGIN_FRAC * d))
        drop[idx[stale]] = True
    return drop


# ---------------------------------------------------------------- live depth scale vs the splat
# The camera's map pose comes from image matching and is accurate. What can still disagree is distance:
# each Orbbec's depth reads a few percent long or short relative to the map (whose scale came from the
# walkthrough camera's depth). That shows up as live surfaces sitting in front of or behind the splat,
# e.g. the white instrument boxes appearing twice. Measure the ratio per camera and correct the depth.


def robust_zbuf(cell_idx: np.ndarray, z: np.ndarray, n: int, q: float = 0.3, min_count: int = 2) -> np.ndarray:
    """Splat surface depth per cell. The front-most blob is often a floater hanging in mid-air, which made the
    splat look nearer than it is (that is what pulled cam1's depth scale to 0.88 and put the live black box
    18 cm in front of the splat's box). Take a low quantile of the blobs in each cell instead; nan if too few."""
    out = np.full(n, np.nan)
    if z.size == 0:
        return out
    order = np.lexsort((z, cell_idx))
    ci, zz = cell_idx[order], z[order]
    starts = np.flatnonzero(np.r_[True, ci[1:] != ci[:-1]])
    counts = np.diff(np.r_[starts, ci.size])
    good = counts >= min_count
    pick = starts + np.floor(q * (counts - 1)).astype(np.int64)
    out[ci[starts[good]]] = zz[pick[good]]
    return out


def depth_scale_vs_splat(R: np.ndarray, t: np.ndarray, depth_mm: np.ndarray, K: np.ndarray, dist: np.ndarray,
                         splat_bytes: bytes, cell: int = 4) -> dict:
    """Median of (splat depth / live depth) over pixels where both exist. 1.0 means they agree."""
    import cv2

    rec = np.frombuffer(splat_bytes, dtype=SPLAT_DTYPE)
    keep = (rec["c"][:, 3] > 128) & (rec["s"].max(axis=1) < 0.05)
    P = rec["p"][keep].astype(np.float64)
    xc = R[0, 0] * P[:, 0] + R[0, 1] * P[:, 1] + R[0, 2] * P[:, 2] + t[0]
    yc = R[1, 0] * P[:, 0] + R[1, 1] * P[:, 1] + R[1, 2] * P[:, 2] + t[1]
    zc = R[2, 0] * P[:, 0] + R[2, 1] * P[:, 1] + R[2, 2] * P[:, 2] + t[2]
    front = (zc > 0.3) & (zc < 4.0)
    cam = np.stack([xc[front], yc[front], zc[front]], axis=1)
    uv, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    h, w = depth_mm.shape
    gu = (uv[:, 0] // cell).astype(np.int64)
    gv = (uv[:, 1] // cell).astype(np.int64)
    gw, gh = w // cell, h // cell
    inside = (gu >= 0) & (gu < gw) & (gv >= 0) & (gv < gh)
    zbuf = robust_zbuf(gv[inside] * gw + gu[inside], cam[inside, 2], gh * gw)   # splat surface per cell
    live = np.asarray(depth_mm, dtype=np.float64)[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell)
    live = np.where(live > 0, live, np.nan)
    with np.errstate(all="ignore"):
        live = np.nanmedian(live, axis=(1, 3)).ravel() / 1000.0
    both = np.isfinite(zbuf) & np.isfinite(live) & (live > 0.3)
    if both.sum() < 200:
        return {"scale": 1.0, "cells": int(both.sum()), "ok": False}
    ratio = zbuf[both] / live[both]
    ratio = ratio[(ratio > 0.75) & (ratio < 1.33)]
    if ratio.size < 200:
        return {"scale": 1.0, "cells": int(ratio.size), "ok": False}
    med = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - med)))
    return {"scale": round(med, 4), "spread": round(mad, 4), "cells": int(ratio.size), "ok": True}


# ---------------------------------------------------------------- robot arm tracking in lab coordinates
# Goal: know where the arm is relative to the lab, from the cameras alone. The camera's pose in the
# lab map (image matching) plus its depth turns every arm pixel into a 3D point in lab coordinates.
#   1. Find the arm in the color picture: the trained YOLO detector (arm_state/runs/arm_detect),
#      or a box given with --arm-box for testing.
#   2. Inside the box, the arm stands above the bench, so arm pixels are the ones clearly nearer to the
#      camera than the far surface in that box (the bench / floor behind it).
#   3. Back-project those pixels with the corrected depth and the camera pose -> arm points in the lab.
#   4. Summarise (centre, highest point, extent), show it in the page and log it.

ARM_CAMS_DEFAULT = ("cam1",)
ARM_CFG = ROOT / "arm_state" / "arm.json"
ARM_LOG = OUT_DIR / "arm_log.csv"
ARM_TRACK = OUT_DIR / "arm_track.json"
ARM_MIN_GAP_M = 0.06       # arm must be at least this much nearer than the surface behind it
ARM_GAP_FRAC = 0.04        # ...or this fraction of that distance, whichever is larger
ARM_STRIDE = 2


def _yolo():
    """Load the trained arm detector once. None if ultralytics or the weights are missing."""
    if hasattr(_yolo, "model"):
        return _yolo.model
    _yolo.model = None
    try:
        cfg = json.loads(ARM_CFG.read_text(encoding="utf-8")) if ARM_CFG.is_file() else {}
        w = Path(cfg.get("weights") or cfg.get("yolo_model") or (ROOT / "arm_state" / "runs" / "arm_detect" / "weights" / "best.pt"))
        if not w.is_file():
            print(f"arm: detector weights not found ({w})")
            return None
        from ultralytics import YOLO

        _yolo.model = YOLO(str(w))
        _yolo.conf = float(cfg.get("conf", 0.25))
        _yolo.imgsz = int(cfg.get("imgsz", 640))
        print(f"arm: detector loaded ({w.name})")
    except Exception as exc:
        print(f"arm: detector unavailable ({exc})")
    return _yolo.model


def detect_arm_boxes(color_bgr: np.ndarray, forced: list[float] | None = None) -> list[list[float]]:
    """Arm boxes [x, y, w, h, conf] in pixels."""
    if forced:
        x, y, w, h = forced
        return [[float(x), float(y), float(w), float(h), 1.0]]
    model = _yolo()
    if model is None or color_bgr is None:
        return []
    res = model.predict(np.asarray(color_bgr), imgsz=_yolo.imgsz, conf=_yolo.conf, verbose=False)
    out = []
    for r in res:
        if r.boxes is None:
            continue
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x0, y0, x1, y1 = (float(v) for v in b)
            out.append([x0, y0, x1 - x0, y1 - y0, float(c)])
    out.sort(key=lambda b: -b[4])
    return out


class ArmBackground:
    """Per-camera learned 'empty' depth: a high percentile of recent depth per pixel.

    Fixed things (bench, racks, instruments) always show the same depth. The arm moves, so over time
    every pixel it covers also shows the farther surface behind it. Saved between runs.
    """

    def __init__(self, cam: str, hw: tuple[int, int], keep: int = 40):
        self.path = OUT_DIR / f"arm_background_{cam}.npz"
        self.hw = (hw[0] // ARM_STRIDE, hw[1] // ARM_STRIDE)
        self.keep = keep
        self.stack = []
        if self.path.is_file():
            try:
                z = np.load(self.path)
                if tuple(z["shape"]) == self.hw:
                    self.stack = [a for a in z["stack"]]
            except Exception:
                self.stack = []

    def add(self, depth_m_grid: np.ndarray) -> None:
        self.stack.append(np.where(np.isfinite(depth_m_grid), depth_m_grid, 0).astype(np.float16))
        del self.stack[:-self.keep]
        try:
            np.savez_compressed(self.path, stack=np.stack(self.stack), shape=np.array(self.hw))
        except Exception:
            pass

    def ready(self) -> bool:
        return len(self.stack) >= 6

    def background(self) -> np.ndarray:
        a = np.stack(self.stack).astype(np.float32)
        a[a <= 0] = np.nan
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.nanpercentile(a, 90, axis=0)


_ARM_BG: dict[str, ArmBackground] = {}


def _bench_plane_mask(Pm: np.ndarray, up: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fallback: points standing more than ARM_MIN_GAP_M above the dominant horizontal surface."""
    up = up / np.linalg.norm(up)
    h = Pm[..., 0] * up[0] + Pm[..., 1] * up[1] + Pm[..., 2] * up[2]
    hv = h[valid]
    if hv.size < 50:
        return np.zeros_like(valid)
    # the bench is the most common height in the box (histogram mode, 1 cm bins)
    hist, edges = np.histogram(hv, bins=np.arange(hv.min(), hv.max() + 0.02, 0.01))
    bench = edges[int(np.argmax(hist))] + 0.005
    return valid & (h > bench + ARM_MIN_GAP_M)


def arm_points(cam: str, depth_mm: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray,
               box: list[float] | None, up: np.ndarray, learn: bool = True) -> tuple[np.ndarray, np.ndarray, str]:
    """Arm pixels -> (points in lab coordinates N×3, pixel coords N×2, method used)."""
    import cv2
    from scipy import ndimage

    z_full = np.asarray(depth_mm, dtype=np.float64) / 1000.0
    h, w = z_full.shape
    vs, us = np.mgrid[0:h - h % ARM_STRIDE:ARM_STRIDE, 0:w - w % ARM_STRIDE:ARM_STRIDE]
    z = z_full[vs, us]
    valid = (z > 0.2) & (z < 6.0)
    zg = np.where(valid, z, np.nan)

    bg = _ARM_BG.get(cam)
    if bg is None or bg.hw != zg.shape:
        bg = _ARM_BG[cam] = ArmBackground(cam, (h, w))
    use_bg = bg.ready()
    if use_bg:
        back = bg.background()
        gap = np.maximum(ARM_MIN_GAP_M, ARM_GAP_FRAC * np.nan_to_num(back, nan=0))
        cand = valid & np.isfinite(back) & (z < back - gap)
        method = f"learned background ({len(bg.stack)} captures)"
    else:
        nrm_all = cv2.undistortPoints(np.stack([us.ravel(), vs.ravel()], 1).astype(np.float64).reshape(-1, 1, 2), K, dist).reshape(zg.shape + (2,))
        Pc = np.stack([nrm_all[..., 0] * np.nan_to_num(zg), nrm_all[..., 1] * np.nan_to_num(zg), np.nan_to_num(zg)], axis=-1)
        Pm = cam_to_map(R, t, Pc.reshape(-1, 3)).reshape(Pc.shape)
        inbox = np.zeros_like(valid)
        if box is not None:
            x, y, bw, bh = box[:4]
            inbox = (us >= x) & (us < x + bw) & (vs >= y) & (vs < y + bh)
        cand = _bench_plane_mask(Pm, up, valid & inbox) if box is not None else np.zeros_like(valid)
        method = f"height above bench (learning background: {len(bg.stack)}/6 captures)"
    if learn:
        bg.add(zg)
    if box is not None:
        x, y, bw, bh = box[:4]
        pad = 0.08
        cand &= (us >= x - pad * bw) & (us < x + (1 + pad) * bw) & (vs >= y - pad * bh) & (vs < y + (1 + pad) * bh)
    cand = ndimage.binary_opening(cand, structure=np.ones((3, 3)))
    lab, n = ndimage.label(cand)
    if n == 0:
        return np.zeros((0, 3)), np.zeros((0, 2)), method
    sizes = ndimage.sum(cand, lab, index=np.arange(1, n + 1))
    keep_ids = np.flatnonzero(sizes >= max(25, 0.2 * sizes.max())) + 1
    mask = np.isin(lab, keep_ids)
    u = us[mask].astype(np.float64)
    v = vs[mask].astype(np.float64)
    zz = z[mask]
    nrm = cv2.undistortPoints(np.stack([u, v], 1).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    Pc = np.stack([nrm[:, 0] * zz, nrm[:, 1] * zz, zz], axis=1)
    return cam_to_map(R, t, Pc), np.stack([u, v], 1), method


def summarize_arm(P: np.ndarray, up: np.ndarray) -> dict:
    """Centre, highest point and extent of the arm points, lab coordinates (m)."""
    if P.shape[0] == 0:
        return {"seen": False}
    up = up / np.linalg.norm(up)
    height = P[:, 0] * up[0] + P[:, 1] * up[1] + P[:, 2] * up[2]
    lo = np.percentile(P, 2, axis=0)
    hi = np.percentile(P, 98, axis=0)
    top = P[int(np.argmax(height))]
    return {
        "seen": True,
        "points": int(P.shape[0]),
        "center_m": [round(float(x), 4) for x in np.median(P, axis=0)],
        "top_m": [round(float(x), 4) for x in top],
        "bbox_min_m": [round(float(x), 4) for x in lo],
        "bbox_max_m": [round(float(x), 4) for x in hi],
        "height_span_m": round(float(np.percentile(height, 98) - np.percentile(height, 2)), 3),
    }


def track_arm(cam_data: dict, arm_cams: tuple, forced_box: list[float] | None, up: list[float]) -> dict:
    """Run detection + 3D on the chosen cameras; write arm_track.json and append arm_log.csv."""
    all_pts = []
    per_cam = {}
    for cam in arm_cams:
        d = cam_data.get(cam)
        if d is None:
            per_cam[cam] = {"seen": False, "reason": "no frame or no pose"}
            continue
        boxes = detect_arm_boxes(d["color"], forced_box)
        box = boxes[0] if boxes else None
        if box is None and not (_ARM_BG.get(cam) and _ARM_BG[cam].ready()):
            # still learn the background so the next captures can find the arm without the detector
            arm_points(cam, d["depth_mm"], d["K"], d["dist"], d["R"], d["t"], None, np.asarray(up, float))
            per_cam[cam] = {"seen": False, "reason": "no arm box from the detector yet; learning the background"}
            continue
        P, uv, method = arm_points(cam, d["depth_mm"], d["K"], d["dist"], d["R"], d["t"], box, np.asarray(up, float))
        per_cam[cam] = {"box": [round(b, 1) for b in box] if box else None, "method": method,
                        **summarize_arm(P, np.asarray(up, float))}
        if P.shape[0]:
            all_pts.append(P)
            d["arm_uv"] = uv
    P = np.concatenate(all_pts) if all_pts else np.zeros((0, 3))
    track = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "frame": "lab map (COLMAP), meters",
             "cameras": list(arm_cams), **summarize_arm(P, np.asarray(up, float)), "per_camera": per_cam}
    ARM_TRACK.write_text(json.dumps(track, indent=2) + "\n", encoding="utf-8")
    new = not ARM_LOG.is_file()
    with ARM_LOG.open("a", encoding="utf-8") as f:
        if new:
            f.write("time,seen,points,center_x,center_y,center_z,top_x,top_y,top_z,cams\n")
        if track.get("seen"):
            c, tp = track["center_m"], track["top_m"]
            f.write(f"{track['time']},1,{track['points']},{c[0]},{c[1]},{c[2]},{tp[0]},{tp[1]},{tp[2]},{'+'.join(arm_cams)}\n")
        else:
            f.write(f"{track['time']},0,0,,,,,,,{'+'.join(arm_cams)}\n")
    # points for the page (thinned)
    step = max(1, P.shape[0] // 6000)
    track["_points"] = P[::step]
    if track.get("seen"):
        c = track["center_m"]
        print(f"arm: {track['points']} points, centre ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) m in the lab, "
              f"seen by {', '.join(k for k, v in per_cam.items() if v.get('seen'))}")
    else:
        print("arm: not found this capture (" + "; ".join(f"{k}: {v.get('reason', '')}" for k, v in per_cam.items()) + ")")
    return track


# ---------------------------------------------------------------- arm tracking from "what changed", connected to the base
# The splat is the lab without the robot's walkthrough pose. Each capture, both cameras already mark which
# of their pixels differ from it (people, moved things, the arm). The arm is the part of that change that is
# physically connected to the robot's base: change points are dropped into 4 cm cells and grown outward from
# the base, within the arm's reach. People standing nearby are not connected to the base and are ignored.
# No detector or training needed. Base position and reach live in lab_scan/robot.json.

ROBOT_FILE = OUT_DIR / "robot.json"
ROBOT_DEFAULT = {"name": "Jeeves (UR5e)", "base_m": [0.24, 0.41, 1.04], "reach_m": 1.0, "seed_radius_m": 0.22,
                 "ignore_base_m": 0.30,
                 "note": "base_m = top of the arm's base in the lab map, measured from cam0 and cam1 (they agree within ~8 cm)"}
ARM_VOXEL_M = 0.04
ARM_TRACK_LOG = OUT_DIR / "arm_track_log.csv"


def _robot() -> dict:
    if not ROBOT_FILE.is_file():
        ROBOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        ROBOT_FILE.write_text(json.dumps(ROBOT_DEFAULT, indent=2) + "\n", encoding="utf-8")
    try:
        return {**ROBOT_DEFAULT, **json.loads(ROBOT_FILE.read_text(encoding="utf-8"))}
    except Exception:
        return ROBOT_DEFAULT


def _changed_points(d: dict, step: int = 2, everything: bool = False) -> np.ndarray:
    import cv2

    D = np.asarray(d["depth_mm"], dtype=np.float64)
    ch = np.ones(D.shape, dtype=bool) if everything else d.get("changed")
    if ch is None:
        return np.zeros((0, 3))
    # Drop "flying pixels" on depth edges: where the arm passes in front of the bench, edge pixels get
    # depths in between the two and form a false bridge from the arm down to whatever is behind it.
    Df = D.astype(np.float32)
    hi = cv2.dilate(Df, np.ones((3, 3), np.uint8))
    Dz = np.where(Df > 0, Df, 1e6).astype(np.float32)
    lo = cv2.erode(Dz, np.ones((3, 3), np.uint8))
    edge = (hi - lo) > np.maximum(60.0, 0.04 * Df)
    vv, uu = np.nonzero(ch & (D > 0) & ~edge)
    vv, uu = vv[::step], uu[::step]
    if vv.size == 0:
        return np.zeros((0, 3))
    z = D[vv, uu] / 1000.0
    nrm = cv2.undistortPoints(np.stack([uu, vv], 1).astype(np.float64).reshape(-1, 1, 2), d["K"], d["dist"]).reshape(-1, 2)
    return cam_to_map(d["R"], d["t"], np.stack([nrm[:, 0] * z, nrm[:, 1] * z, z], 1))


def _seen_through(P: np.ndarray, cam_data: dict, margin: float = 0.06) -> np.ndarray:
    """Drop points that some camera sees straight through (it measures something farther behind them):
    they are noise floating in mid-air, not a solid part of the robot."""
    import cv2

    if P.shape[0] == 0:
        return P
    bad = np.zeros(P.shape[0], dtype=bool)
    for d in cam_data.values():
        R, t, K, dist = d["R"], d["t"], d["K"], d["dist"]
        D = np.asarray(d["depth_mm"], dtype=np.float32) / 1000.0
        # nearest depth in a 9x9 neighbourhood: a thin arm seen slightly off (calibration) still counts as hit
        D = cv2.erode(np.where(D > 0, D, 99.0).astype(np.float32), np.ones((9, 9), np.uint8))
        D = np.where(D < 90, D, 0.0).astype(np.float64)
        h, w = D.shape
        xc = R[0, 0] * P[:, 0] + R[0, 1] * P[:, 1] + R[0, 2] * P[:, 2] + t[0]
        yc = R[1, 0] * P[:, 0] + R[1, 1] * P[:, 1] + R[1, 2] * P[:, 2] + t[1]
        zc = R[2, 0] * P[:, 0] + R[2, 1] * P[:, 1] + R[2, 2] * P[:, 2] + t[2]
        f = zc > 0.2
        if not f.any():
            continue
        uv, _ = cv2.projectPoints(np.stack([xc[f], yc[f], zc[f]], 1).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
        uv = uv.reshape(-1, 2)
        u, v = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
        ins = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.flatnonzero(f)[ins]
        dd = D[v[ins], u[ins]]
        bad[idx[(dd > 0) & (dd > zc[idx] + margin + 0.02 * dd)]] = True
    return P[~bad]


def _grow_from_base(P: np.ndarray, base: np.ndarray, reach: float, seed_r: float) -> np.ndarray:
    """Mask of points whose 4 cm cell connects (26-neighbour) to a cell near the base, within reach."""
    from collections import deque

    d = P - base
    near = np.sqrt((d * d).sum(axis=1)) < reach
    idx = np.flatnonzero(near)
    if idx.size == 0:
        return np.zeros(P.shape[0], dtype=bool)
    keys = np.floor(P[idx] / ARM_VOXEL_M).astype(np.int64)
    cells: dict = {}
    for i, k in zip(idx, map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    seeds = []
    for k in cells:
        c = (np.array(k) + 0.5) * ARM_VOXEL_M - base
        if np.sqrt((c * c).sum()) < seed_r:
            seeds.append(k)
    seen = set(seeds)
    q = deque(seeds)
    nb = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    while q:
        k = q.popleft()
        for a, b, c in nb:
            n = (k[0] + a, k[1] + b, k[2] + c)
            if n in cells and n not in seen:
                seen.add(n)
                q.append(n)
    out = np.zeros(P.shape[0], dtype=bool)
    for k in seen:
        out[cells[k]] = True
    return out


NEAR_M = 0.10            # warn when any part of the robot is within this of anything that isn't empty space
NEAR_MIN_POINTS = 12     # need this many scene points that close (one stray splat blob doesn't count)
NEAR_ALERTS = OUT_DIR / "robot_alerts.csv"


BASE_MOUNT_M = 0.25      # ignore the bench/mount right around the base (the robot is bolted to it)
BASE_IGNORE_M = 0.30     # robot parts this close to the base aren't checked (the base sits on the bench)


ARM_PX_MARGIN = 21   # pixels (about 5 cm at 2 m)


def _confirmed_by_camera(O: np.ndarray, A: np.ndarray, cam_data: dict, tol: float = 0.04) -> np.ndarray:
    """True for scene points some camera sees right now: the camera's depth at that pixel matches the point
    and the pixel isn't the robot. Splat blobs hidden behind or inside the live arm (bits of the walkthrough's
    arm that no camera can see past) or behind other things are not confirmed and don't count."""
    import cv2

    ok = np.zeros(O.shape[0], dtype=bool)
    for d in cam_data.values():
        R, t, K, dist = d["R"], d["t"], d["K"], d["dist"]
        D = np.asarray(d["depth_mm"], dtype=np.float64) / 1000.0
        h, w = D.shape

        def proj(X):
            xc = R[0, 0] * X[:, 0] + R[0, 1] * X[:, 1] + R[0, 2] * X[:, 2] + t[0]
            yc = R[1, 0] * X[:, 0] + R[1, 1] * X[:, 1] + R[1, 2] * X[:, 2] + t[1]
            zc = R[2, 0] * X[:, 0] + R[2, 1] * X[:, 1] + R[2, 2] * X[:, 2] + t[2]
            uv, _ = cv2.projectPoints(np.stack([xc, yc, np.maximum(zc, 1e-3)], 1).reshape(-1, 1, 3),
                                      np.zeros(3), np.zeros(3), K, dist)
            uv = uv.reshape(-1, 2)
            return np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64), zc
        armpx = np.zeros((h, w), np.uint8)
        if A.shape[0]:
            u, v, z = proj(A)
            m = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            armpx[v[m], u[m]] = 1
            # generous margin around the arm's outline: joints that sit where they were during the walkthrough
            # don't show up as "changed", but they are still robot, right next to the parts that did
            armpx = cv2.dilate(armpx, np.ones((ARM_PX_MARGIN, ARM_PX_MARGIN), np.uint8))
        u, v, z = proj(O)
        m = (z > 0.2) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.flatnonzero(m)
        dd = D[v[m], u[m]]
        seen = (dd > 0) & (np.abs(dd - z[m]) < tol + 0.015 * dd) & (armpx[v[m], u[m]] == 0)
        ok[idx[seen]] = True
    return ok


def robot_clearance(arm_pts: np.ndarray, base: np.ndarray, tip: np.ndarray, live_other: np.ndarray,
                    splat_bytes: bytes | None, cam_data: dict | None = None) -> dict:
    """How close any part of the robot is to the rest of the scene: the cleaned splat (benches, racks,
    machines) plus live points that changed but aren't the robot (a person, a rack someone put down)."""
    if arm_pts.shape[0] == 0:
        return {"near": False, "clearance_m": None}
    A = arm_pts[:: max(1, arm_pts.shape[0] // 4000)]
    # The bottom of the robot (base and the part just above it) is always close to the bench it's bolted
    # to; it is left out of the check. Set "ignore_base_m" in robot.json to change how much is left out.
    ign = float(_robot().get("ignore_base_m", BASE_IGNORE_M))
    Ac = A[np.sqrt(((A - base) ** 2).sum(axis=1)) > ign]
    if Ac.shape[0] == 0:
        return {"near": False, "clearance_m": None}
    lo, hi = Ac.min(axis=0) - NEAR_M * 1.5, Ac.max(axis=0) + NEAR_M * 1.5
    parts, kinds = [], []
    if splat_bytes:
        rec = np.frombuffer(splat_bytes, dtype=SPLAT_DTYPE)
        p = rec["p"]
        sel = np.all((p >= lo) & (p <= hi), axis=1) & (rec["c"][:, 3] > 120) & (rec["s"].max(axis=1) < 0.04)
        parts.append(p[sel].astype(np.float64))
        kinds.append(np.zeros(int(sel.sum()), dtype=np.int8))            # 0 = the lab (splat)
    if live_other.shape[0]:
        sel = np.all((live_other >= lo) & (live_other <= hi), axis=1)
        parts.append(live_other[sel])
        kinds.append(np.ones(int(sel.sum()), dtype=np.int8))             # 1 = something live
    # distance straight to the never-changing objects' boxes (XRD machine, black box)
    sd, sn = static_distance(Ac)
    j = int(np.argmin(sd)) if sd.size else -1
    static_hit = None
    if j >= 0 and np.isfinite(sd[j]):
        static_hit = {"clearance_m": float(sd[j]), "what": str(sn[j]), "at": Ac[j]}
    if not parts or sum(len(x) for x in parts) == 0:
        return _with_static({"near": False, "clearance_m": None}, static_hit, base, tip)
    O, Kd = np.concatenate(parts), np.concatenate(kinds)
    keep_out = ~in_static(O, 0.0)      # those are covered exactly by the box distance above
    O, Kd = O[keep_out], Kd[keep_out]
    # Not obstacles: the robot's own surface (same or next 2 cm cell as a robot point) and its mount.
    g = 0.025
    armc = set(map(tuple, np.floor(arm_pts / g).astype(np.int64)))
    oc = np.floor(O / g).astype(np.int64)
    on_arm = np.fromiter((tuple(k) in armc for k in oc), bool, O.shape[0])
    db = np.sqrt(((O - base) ** 2).sum(axis=1))
    keep = ~on_arm & (db > BASE_MOUNT_M)
    O, Kd = O[keep], Kd[keep]
    if cam_data and O.shape[0]:
        conf = _confirmed_by_camera(O, A, cam_data)
        O, Kd = O[conf | (Kd == 1)], Kd[conf | (Kd == 1)]
    if O.shape[0] == 0:
        return _with_static({"near": False, "clearance_m": None}, static_hit, base, tip)
    # Coarse: only scene points in a 10 cm cell next to a robot cell; then exact distances in chunks.
    c = NEAR_M
    rc = set(map(tuple, np.floor(Ac / c).astype(np.int64)))
    occ = np.floor(O / c).astype(np.int64)
    cand = np.zeros(O.shape[0], dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                sh = occ + np.array([dx, dy, dz])
                cand |= np.fromiter((tuple(k) in rc for k in sh), bool, O.shape[0])
    O, Kd = O[cand], Kd[cand]
    if O.shape[0] < NEAR_MIN_POINTS:
        return _with_static({"near": False, "clearance_m": None}, static_hit, base, tip)
    dmin = np.empty(O.shape[0])
    amin = np.empty(O.shape[0], dtype=np.int64)
    for i in range(0, O.shape[0], 400):
        blk = O[i:i + 400]
        d2 = ((blk[:, None, 0] - Ac[None, :, 0]) ** 2 + (blk[:, None, 1] - Ac[None, :, 1]) ** 2
              + (blk[:, None, 2] - Ac[None, :, 2]) ** 2)
        amin[i:i + 400] = np.argmin(d2, axis=1)
        dmin[i:i + 400] = np.sqrt(d2[np.arange(blk.shape[0]), amin[i:i + 400]])
    order = np.argsort(dmin)
    close = order[:NEAR_MIN_POINTS]
    clearance = float(dmin[close[-1]])
    at = np.median(Ac[amin[close]], axis=0)
    reach = float(np.sqrt(((tip - base) ** 2).sum())) or 1.0
    f = float(np.sqrt(((at - base) ** 2).sum())) / reach
    part = "gripper / wrist" if f > 0.75 else ("forearm / elbow" if f > 0.35 else "shoulder / upper arm")
    what = "something live (person or moved object)" if Kd[close].mean() > 0.5 else "the lab (bench / equipment)"
    return _with_static({"near": bool(clearance < NEAR_M), "clearance_m": round(clearance, 3), "part": part, "what": what,
                         "at_m": [round(float(x), 3) for x in at],
                         "toward_m": [round(float(x), 3) for x in (np.median(O[close], axis=0) - at)]}, static_hit, base, tip)


def _with_static(res: dict, hit: dict | None, base: np.ndarray, tip: np.ndarray) -> dict:
    """Use the distance to a never-changing object if it's closer than anything else."""
    if not hit or (res.get("clearance_m") is not None and res["clearance_m"] <= hit["clearance_m"]):
        return res
    at = hit["at"]
    reach = float(np.sqrt(((tip - base) ** 2).sum())) or 1.0
    f = float(np.sqrt(((at - base) ** 2).sum())) / reach
    part = "gripper / wrist" if f > 0.75 else ("forearm / elbow" if f > 0.35 else "shoulder / upper arm")
    return {"near": bool(hit["clearance_m"] < NEAR_M), "clearance_m": round(hit["clearance_m"], 3), "part": part,
            "what": f"the {hit['what']}", "at_m": [round(float(x), 3) for x in at], "toward_m": [0.0, 0.0, 0.0]}


def track_arm_changed(cam_data: dict, up: list[float], splat_bytes: bytes | None = None) -> dict:
    """Arm points in the lab from both cameras; writes arm_track.json and appends arm_track_log.csv."""
    rb = _robot()
    base = np.asarray(rb["base_m"], dtype=np.float64)
    per_cam, all_P = {}, []
    for cam, d in cam_data.items():
        P = _changed_points(d)
        all_P.append(P)
        per_cam[cam] = {"changed_points": int(P.shape[0])}
    P = np.concatenate(all_P) if all_P else np.zeros((0, 3))
    P = _seen_through(P, cam_data)
    arm = _grow_from_base(P, base, float(rb["reach_m"]), float(rb["seed_radius_m"])) if P.shape[0] else np.zeros(0, bool)
    A = P[arm]
    track = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "frame": "lab map (COLMAP), meters",
             "method": "change vs splat, connected to the base", "robot": rb["name"], "base_m": rb["base_m"],
             "cameras": [c for c in cam_data], **summarize_arm(A, np.asarray(up, float)), "per_camera": per_cam}
    if A.shape[0] >= 50:
        dist = np.sqrt(((A - base) ** 2).sum(axis=1))
        far = np.argsort(dist)[-max(5, A.shape[0] // 100):]          # farthest 1% = gripper end
        tip = np.median(A[far], axis=0)
        track["tip_m"] = [round(float(x), 4) for x in tip]
        track["tip_reach_m"] = round(float(np.sqrt(((tip - base) ** 2).sum())), 3)
        track["clearance"] = robot_clearance(A, base, tip, P[~arm], splat_bytes, cam_data)
    else:
        track["seen"] = False
    ARM_TRACK.write_text(json.dumps(track, indent=2) + "\n", encoding="utf-8")
    new = not ARM_TRACK_LOG.is_file()
    with ARM_TRACK_LOG.open("a", encoding="utf-8") as f:
        if new:
            f.write("time,seen,points,center_x,center_y,center_z,tip_x,tip_y,tip_z,tip_reach_m\n")
        if track.get("seen"):
            c, tp = track["center_m"], track["tip_m"]
            f.write(f"{track['time']},1,{track['points']},{c[0]},{c[1]},{c[2]},{tp[0]},{tp[1]},{tp[2]},{track['tip_reach_m']}\n")
        else:
            f.write(f"{track['time']},0,0,,,,,,,\n")
    step = max(1, A.shape[0] // 6000)
    track["_points"] = A[::step]
    if track.get("seen"):
        tp = track["tip_m"]
        print(f"arm: {track['points']} points connected to the base; gripper end at ({tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f}) m, "
              f"{track['tip_reach_m']:.2f} m from the base")
        g = track.get("clearance", {})
        if g.get("near"):
            print(f"\a*** ROBOT NEAR: {g['part']} is {g['clearance_m']*100:.0f} cm from {g['what']} ***")
            new_a = not NEAR_ALERTS.is_file()
            with NEAR_ALERTS.open("a", encoding="utf-8") as f:
                if new_a:
                    f.write("time,clearance_cm,part,what,at_x,at_y,at_z\n")
                at = g["at_m"]
                f.write(f"{track['time']},{g['clearance_m']*100:.1f},{g['part']},{g['what']},{at[0]},{at[1]},{at[2]}\n")
        elif g.get("clearance_m") is not None:
            print(f"arm: robot clear ({g['clearance_m']*100:.0f} cm from the nearest thing, at the {g['part']})")
    else:
        print(f"arm: nothing connected to the base this capture (check base_m in {ROBOT_FILE.name} if this persists)")
    return track

# ---------------------------------------------------------------- fixed objects and their states (the vial rack)
# Objects that never move (the vial rack) are stored as a 3D box in the lab map, not as a pixel crop.
# Because the box lives in lab coordinates it is found in every camera from that camera's pose, so a
# re-aimed camera no longer breaks it. (The old rack_lid_state crop now points at the wrong part of
# cam1's picture, because cam1 was moved after the crop was drawn.)
#
# State comes from depth, not from comparing pictures, so lighting doesn't matter:
#   1. Calibrate once with every tray pushed in (--calibrate-rack): the space in and around the rack's
#      box is cut into 3 cm cells and the occupied ones are remembered ("how the closed rack looks").
#   2. Every capture: cells that are newly occupied, are attached to the rack's box, and aren't the robot
#      arm = something sticking out of the rack, i.e. a tray pulled out.
#      OPEN  + which shelf (height) + how far out + which side;  CLOSED otherwise.
#   People walking past aren't attached to the rack, so they don't count.

OBJECTS_FILE = OUT_DIR / "objects.json"
OBJECTS_DEFAULT = {
    "rack": {
        "name": "vial rack",
        "box_version": 2,
        # Fitted 2026-11-10 to the rack's 4 top corners (cam1 depth) and the bench plane (splat).
        # axes (unit vectors, lab map): along the long top edge, along the short top edge, up (bench normal)
        "center_m": [0.9319, -0.0538, 1.1496],
        "axes": [[-0.0065, 0.8149, -0.5795], [0.9712, -0.1329, -0.1979], [-0.2383, -0.5641, -0.7906]],
        "min_m": [-0.202, -0.139, -0.185],
        "max_m": [0.202, 0.139, 0.185],
        "shelves": 6,
        "search_m": 0.40,          # look this far outside the box for pulled-out trays
        "open_min_cells": 12,      # attached new cells (3 cm) needed to call it open
        "note": "Box fitted to the rack's top corners and the bench. Run --calibrate-rack once with every tray pushed in.",
    }
}
# Things that never move or change (the XRD machine, the black instrument box). They are kept in the splat
# no matter what (never carved away), never drawn as "changed", and the robot's distance to them is
# measured straight to their box, so they count as obstacles even where no camera sees them.
STATIC_DEFAULT = {
        "black box": {"center_m": [-0.085, 1.053, 0.4183], "axes": [[-0.923, -0.381, 0.046], [-0.344, 0.767, -0.541], [0.171, -0.516, -0.84]], "min_m": [-0.258, -0.194, -0.306], "max_m": [0.258, 0.194, 0.306]},
        "XRD machine": {"center_m": [-0.4049, 0.0575, 1.1677], "axes": [[-0.923, -0.381, 0.046], [-0.344, 0.767, -0.541], [0.171, -0.516, -0.84]], "min_m": [-0.383, -0.372, -0.318], "max_m": [0.383, 0.372, 0.318]}
    }
OBJECT_CELL_M = 0.03
OBJECT_STATE_FILE = OUT_DIR / "object_states.json"
OBJECT_LOG = OUT_DIR / "object_states_log.csv"


def _objects() -> dict:
    if not OBJECTS_FILE.is_file():
        OBJECTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        OBJECTS_FILE.write_text(json.dumps({**OBJECTS_DEFAULT, "static": STATIC_DEFAULT}, indent=2) + "\n", encoding="utf-8")
    try:
        d = json.loads(OBJECTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {**OBJECTS_DEFAULT, "static": STATIC_DEFAULT}
    changed = False
    if "static" not in d:                        # older file: add the fixed objects once
        d["static"] = STATIC_DEFAULT
        changed = True
    r = d.get("rack", {})
    if r.get("box_version", 1) < OBJECTS_DEFAULT["rack"]["box_version"]:
        # the first rack box was tilted ~24 deg (it used the map's rough "up", not the bench); replace it
        for k in ("box_version", "center_m", "axes", "min_m", "max_m", "note"):
            r[k] = OBJECTS_DEFAULT["rack"][k]
        d["rack"] = {**OBJECTS_DEFAULT["rack"], **r}
        changed = True
        old_cal = _baseline_path("rack")
        if old_cal.is_file():
            old_cal.rename(old_cal.with_suffix(".old_box.npz"))
    if changed:
        OBJECTS_FILE.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return d


def static_boxes() -> dict:
    return _objects().get("static", {})


def in_static(P: np.ndarray, grow: float = 0.0) -> np.ndarray:
    """True for map points inside any never-changing object's box (grown by `grow` m)."""
    inside = np.zeros(P.shape[0], dtype=bool)
    for o in static_boxes().values():
        B = _to_box(P, o)
        inside |= np.all((B > np.asarray(o["min_m"]) - grow) & (B < np.asarray(o["max_m"]) + grow), axis=1)
    return inside


def static_distance(P: np.ndarray):
    """Distance from each point to the nearest never-changing object's box surface (0 inside), and its name."""
    best = np.full(P.shape[0], np.inf)
    name = np.full(P.shape[0], "", dtype=object)
    for k, o in static_boxes().items():
        B = _to_box(P, o)
        out = np.maximum(np.maximum(np.asarray(o["min_m"]) - B, B - np.asarray(o["max_m"])), 0.0)
        d = np.sqrt((out * out).sum(axis=1))
        better = d < best
        best[better], name[better] = d[better], k
    return best, name


def _all_points(cam_data: dict, step: int = 2) -> np.ndarray:
    pts = [_changed_points(d, step=step, everything=True) for c, d in cam_data.items()
           if not c.startswith("_") and isinstance(d, dict) and "depth_mm" in d]
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    return _seen_through(P, cam_data) if P.shape[0] else P


def _to_box(P: np.ndarray, o: dict) -> np.ndarray:
    """Coordinates along the box axes (a, b, up), relative to the box center."""
    c = np.asarray(o["center_m"], dtype=np.float64)
    A = np.asarray(o["axes"], dtype=np.float64)
    d = P - c
    return np.stack([d[:, 0] * A[i, 0] + d[:, 1] * A[i, 1] + d[:, 2] * A[i, 2] for i in range(3)], axis=1)


def _region_cells(cam_data: dict, o: dict, arm_pts: np.ndarray | None):
    """Occupied 3 cm cells (box coordinates) in the box grown by search_m sideways, at the rack's heights."""
    P = _all_points(cam_data)
    if arm_pts is not None and len(arm_pts):
        g = 0.03
        armc = set(map(tuple, np.floor(np.asarray(arm_pts) / g).astype(np.int64)))
        on_arm = np.fromiter((tuple(k) in armc for k in np.floor(P / g).astype(np.int64)), bool, P.shape[0])
        P = P[~on_arm]
    B = _to_box(P, o)
    mn, mx, s = np.asarray(o["min_m"], float), np.asarray(o["max_m"], float), float(o.get("search_m", 0.4))
    reg = ((B[:, 0] > mn[0] - s) & (B[:, 0] < mx[0] + s) & (B[:, 1] > mn[1] - s) & (B[:, 1] < mx[1] + s)
           & (B[:, 2] > mn[2] + 0.02) & (B[:, 2] < mx[2] + 0.05))
    B = B[reg]
    keys, counts = np.unique(np.floor(B / OBJECT_CELL_M).astype(np.int64), axis=0, return_counts=True)
    return keys[counts >= 2], B


def _baseline_path(name: str) -> Path:
    return OUT_DIR / f"{name}_closed_cells.npz"


def calibrate_rack(cam_data: dict, arm_pts=None) -> None:
    o = _objects()["rack"]
    keys, B = _region_cells(cam_data, o, arm_pts)
    if keys.shape[0] < 50:
        print(f"rack: only {keys.shape[0]} occupied cells near the rack; not calibrated (is it in view?)")
        return
    # merge with an existing calibration from earlier captures this run, so several views add up
    p = _baseline_path("rack")
    if getattr(calibrate_rack, "merging", False) and p.is_file():
        keys = np.unique(np.vstack([np.load(p)["keys"], keys]), axis=0)
    np.savez_compressed(p, keys=keys, at=time.strftime("%Y-%m-%d %H:%M"))
    calibrate_rack.merging = True
    print(f"rack: closed look saved ({keys.shape[0]} occupied cells around the rack) -> {p.name}")


def rack_state(cam_data: dict, arm_pts=None) -> dict:
    from collections import deque

    o = _objects()["rack"]
    res = {"name": o["name"], "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "box": {k: o[k] for k in ("center_m", "axes", "min_m", "max_m")}}
    p = _baseline_path("rack")
    if not p.is_file():
        res.update(state="NOT CALIBRATED", reason="run once with --calibrate-rack while every tray is pushed in")
        return res
    base = set(map(tuple, np.load(p)["keys"]))
    # tolerate 1 cell of noise around anything that was there when closed
    near_base = set()
    for k in base:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    near_base.add((k[0] + dx, k[1] + dy, k[2] + dz))
    keys, _B = _region_cells(cam_data, o, arm_pts)
    live = set(map(tuple, keys))
    new = live - near_base
    mn, mx = np.asarray(o["min_m"], float), np.asarray(o["max_m"], float)
    lo_c, hi_c = np.floor(mn / OBJECT_CELL_M).astype(int), np.floor(mx / OBJECT_CELL_M).astype(int)

    def touches_box(k):
        # within 3 cells (9 cm) of the box: the first cells right at the face are often already occupied
        # by the rack's own front when closed, so a pulled tray's new cells start a little way out
        return all(lo_c[i] - 3 <= k[i] <= hi_c[i] + 3 for i in range(3))
    seeds = [k for k in new if touches_box(k)]
    grown = set(seeds)
    q = deque(seeds)
    nb = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    while q:
        k = q.popleft()
        for a, b, c in nb:
            n = (k[0] + a, k[1] + b, k[2] + c)
            if n in new and n not in grown:
                grown.add(n)
                q.append(n)
    res.update(live_cells=len(live), new_cells=len(new), attached_new_cells=len(grown))
    if len(live) < 30:
        res.update(state="UNKNOWN", reason="rack hardly visible (blocked or out of view)")
        return res
    if len(grown) < int(o.get("open_min_cells", 12)):
        res.update(state="CLOSED", reason="nothing new sticking out of the rack")
        return res
    G = (np.array(list(grown), dtype=np.float64) + 0.5) * OBJECT_CELL_M
    # how far outside the box, and on which side
    outa = np.maximum(G[:, 0] - mx[0], mn[0] - G[:, 0])
    outb = np.maximum(G[:, 1] - mx[1], mn[1] - G[:, 1])
    out = np.maximum(np.maximum(outa, outb), 0.0)
    far = float(np.percentile(out, 90))
    side_votes = {"+a": int((G[:, 0] > mx[0]).sum()), "-a": int((G[:, 0] < mn[0]).sum()),
                  "+b": int((G[:, 1] > mx[1]).sum()), "-b": int((G[:, 1] < mn[1]).sum())}
    side = max(side_votes, key=side_votes.get)
    n = int(o.get("shelves", 6))
    h = np.clip(((G[:, 2] - mn[2]) / max(1e-6, mx[2] - mn[2]) * n).astype(int), 0, n - 1)
    cnt = np.bincount(h, minlength=n)
    shelves = [int(i) + 1 for i in np.flatnonzero(cnt >= max(3, 0.3 * cnt.max()))]
    res.update(state="OPEN", shelves_open=shelves, pulled_out_cm=round(far * 100, 1), side=side,
               reason=f"{len(grown)} new 3 cm cells attached to the rack, up to {far * 100:.0f} cm outside it")
    return res


def report_objects(cam_data: dict, arm: dict | None) -> dict:
    states = {}
    arm_pts = arm.get("_points") if arm else None
    try:
        left = int(getattr(build, "calibrate_rack", 0) or 0)
        if left > 0:                        # calibrate over a few captures so both views and noise add up
            calibrate_rack(cam_data, arm_pts)
            build.calibrate_rack = left - 1
            if left - 1 == 0:
                print("rack: calibration done; from now on trays sticking out are reported as OPEN")
        states["rack"] = rack_state(cam_data, arm_pts)
    except Exception as exc:
        states["rack"] = {"name": "vial rack", "state": "UNKNOWN", "reason": f"error: {exc}"}
    r = states["rack"]
    if r.get("state") == "OPEN":
        sh = ", ".join(str(x) for x in r.get("shelves_open", [])) or "?"
        print(f"rack: OPEN (shelf {sh} from the bottom, out about {r['pulled_out_cm']:.0f} cm, side {r['side']})")
    else:
        print(f"rack: {r.get('state')} ({r.get('reason', '')})")
    OBJECT_STATE_FILE.write_text(json.dumps(states, indent=2) + "\n", encoding="utf-8")
    new = not OBJECT_LOG.is_file()
    with OBJECT_LOG.open("a", encoding="utf-8") as f:
        if new:
            f.write("time,object,state,shelves_open,pulled_out_cm,side,attached_new_cells\n")
        f.write(f"{r.get('time', '')},rack,{r.get('state')},{' '.join(map(str, r.get('shelves_open', [])))},"
                f"{r.get('pulled_out_cm', '')},{r.get('side', '')},{r.get('attached_new_cells', '')}\n")
    return states

# ---------------------------------------------------------------- cam1 lock on a fixed landmark
# The fixed cameras should never move. A landmark that never moves (cam1: the black instrument box with
# the grey top) proves it every capture: its picture and depth are compared with a saved reference.
#   locked     shift < ANCHOR_MAX_SHIFT_PX and depth change < ANCHOR_MAX_DEPTH_M
#   moved      twice in a row -> the camera is located in the map again and a new reference is saved
# Anchor boxes live in lab_scan/anchors.json so they can be edited without touching the code.

ANCHORS_FILE = OUT_DIR / "anchors.json"
ANCHORS_DEFAULT = {"cam1": {"box": [415, 180, 140, 120], "size": [640, 480],
                            "what": "black instrument box with the grey top, right of the arm"}}
ANCHOR_MAX_SHIFT_PX = 2.0
ANCHOR_MAX_DEPTH_M = 0.02
_ANCHOR_MISSES: dict[str, int] = {}


def _anchors() -> dict:
    if not ANCHORS_FILE.is_file():
        ANCHORS_FILE.parent.mkdir(parents=True, exist_ok=True)
        ANCHORS_FILE.write_text(json.dumps(ANCHORS_DEFAULT, indent=2) + "\n", encoding="utf-8")
    try:
        return json.loads(ANCHORS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return ANCHORS_DEFAULT


def check_anchor(cam: str, color: np.ndarray | None, depth_mm: np.ndarray, fresh_pose: bool) -> dict:
    """Compare the landmark with its reference. Returns {'status': locked|moved|new|none, ...}."""
    import cv2

    a = _anchors().get(cam)
    if a is None or color is None:
        return {"status": "none"}
    h, w = depth_mm.shape
    sx, sy = w / a["size"][0], h / a["size"][1]
    x, y, bw, bh = a["box"]
    x0, y0, x1, y1 = int(x * sx), int(y * sy), int((x + bw) * sx), int((y + bh) * sy)
    gray = cv2.cvtColor(np.asarray(color), cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi = gray[y0:y1, x0:x1]
    dz = np.asarray(depth_mm, dtype=np.float64)[y0:y1, x0:x1]
    dz = dz[(dz > 0) & np.isfinite(dz)]
    dmed = float(np.median(dz)) / 1000.0 if dz.size > 50 else float("nan")
    ref_path = OUT_DIR / f"anchor_ref_{cam}.npz"
    if fresh_pose or not ref_path.is_file():
        np.savez_compressed(ref_path, roi=roi, depth=dmed)
        _ANCHOR_MISSES[cam] = 0
        return {"status": "new", "what": a.get("what", ""), "depth_m": round(dmed, 3)}
    ref = np.load(ref_path)
    rroi = ref["roi"]
    if rroi.shape != roi.shape:
        np.savez_compressed(ref_path, roi=roi, depth=dmed)
        return {"status": "new", "what": a.get("what", "")}
    win = cv2.createHanningWindow(roi.shape[::-1], cv2.CV_32F)
    (du, dv), resp = cv2.phaseCorrelate(rroi, roi, win)
    shift = float(np.hypot(du, dv))
    ddepth = abs(dmed - float(ref["depth"])) if np.isfinite(dmed) else float("nan")
    ok = shift < ANCHOR_MAX_SHIFT_PX and (not np.isfinite(ddepth) or ddepth < ANCHOR_MAX_DEPTH_M) and resp > 0.1
    _ANCHOR_MISSES[cam] = 0 if ok else _ANCHOR_MISSES.get(cam, 0) + 1
    return {"status": "locked" if ok else "moved", "shift_px": round(shift, 2), "depth_change_m": round(ddepth, 3),
            "match": round(float(resp), 2), "misses": _ANCHOR_MISSES[cam], "what": a.get("what", "")}


# ---------------------------------------------------------------- keep cam1 locked onto the black box, every capture
# The black instrument box never moves, so every capture cam1 is pinned to it:
#   1. distance: the live depth is scaled so the box's depth equals the splat box's depth
#      (the box alone sets the scale, not the whole frame, where floaters and moved things mislead it);
#   2. aim: the splat is rendered as a depth image around the box and slid a few pixels each way; the
#      slide where live box and splat box agree best becomes a small turn of the camera.
# Both are smoothed over the last captures so one noisy frame can't jerk the view.

BOX_MARGIN = 0.5            # compare this much around the box too (fraction of box size) so its edges count
BOX_SEARCH_PX = 16          # slide the splat picture up to this far each way
BOX_SCALE_RANGE = (0.88, 1.12)
BOX_HISTORY: dict[str, list] = {}


def _box_roi(cam: str, hw: tuple[int, int], margin: float):
    a = _anchors().get(cam)
    if a is None:
        return None
    h, w = hw
    sx, sy = w / a["size"][0], h / a["size"][1]
    x, y, bw, bh = a["box"]
    mx, my = bw * margin, bh * margin
    x0, y0 = max(0, int((x - mx) * sx)), max(0, int((y - my) * sy))
    x1, y1 = min(w, int((x + bw + mx) * sx)), min(h, int((y + bh + my) * sy))
    return x0, y0, x1, y1


def _splat_points(splat_bytes: bytes) -> np.ndarray:
    key = id(splat_bytes)
    if _SPLAT_PTS.get("key") != key:
        rec = np.frombuffer(splat_bytes, dtype=SPLAT_DTYPE)
        keep = (rec["c"][:, 3] > 100) & (rec["s"].max(axis=1) < 0.05)
        _SPLAT_PTS.update(key=key, P=rec["p"][keep].astype(np.float64))
    return _SPLAT_PTS["P"]


_SPLAT_PTS: dict = {}


def _splat_depth_roi(R, t, K, dist, P, roi, cell=2, pad=0):
    """Splat surface depth per cell over the ROI grown by `pad` cells on every side (meters, nan where empty)."""
    import cv2

    x0, y0, x1, y1 = roi
    x0, y0, x1, y1 = x0 - pad * cell, y0 - pad * cell, x1 + pad * cell, y1 + pad * cell
    xc = R[0, 0] * P[:, 0] + R[0, 1] * P[:, 1] + R[0, 2] * P[:, 2] + t[0]
    yc = R[1, 0] * P[:, 0] + R[1, 1] * P[:, 1] + R[1, 2] * P[:, 2] + t[1]
    zc = R[2, 0] * P[:, 0] + R[2, 1] * P[:, 1] + R[2, 2] * P[:, 2] + t[2]
    front = (zc > 0.2) & (zc < 6.0)
    cam = np.stack([xc[front], yc[front], zc[front]], axis=1)
    uv, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    gw, gh = (x1 - x0) // cell, (y1 - y0) // cell
    gu = np.floor((uv[:, 0] - x0) / cell).astype(np.int64)
    gv = np.floor((uv[:, 1] - y0) / cell).astype(np.int64)
    ins = (gu >= 0) & (gu < gw) & (gv >= 0) & (gv < gh)
    return robust_zbuf(gv[ins] * gw + gu[ins], cam[ins, 2], gh * gw).reshape(gh, gw)


def _live_depth_roi(depth_mm, roi, cell=2):
    x0, y0, x1, y1 = roi
    gw, gh = (x1 - x0) // cell, (y1 - y0) // cell
    d = np.asarray(depth_mm, dtype=np.float64)[y0:y0 + gh * cell, x0:x0 + gw * cell].reshape(gh, cell, gw, cell)
    d = np.where(d > 0, d, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmedian(d, axis=(1, 3)) / 1000.0


def _rot_xy(a: float, b: float) -> np.ndarray:
    """Small camera turn: a about the camera's y axis (moves the picture in u), b about x (moves it in v)."""
    import cv2

    Rc, _ = cv2.Rodrigues(np.array([b, a, 0.0]))
    return Rc


def _mul3(A, B):
    return np.array([[sum(A[i, k] * B[k, j] for k in range(3)) for j in range(3)] for i in range(3)])


def _mulv(A, v):
    return np.array([sum(A[i, k] * v[k] for k in range(3)) for i in range(3)])


def box_scale(cam, R, t, depth_mm, K, dist, splat_bytes):
    """Splat box depth / live box depth over the box itself."""
    roi = _box_roi(cam, depth_mm.shape[:2], 0.0)
    zs = _splat_depth_roi(R, t, K, dist, _splat_points(splat_bytes), roi, 4)
    zl = _live_depth_roi(depth_mm, roi, 4)
    b = np.isfinite(zs) & np.isfinite(zl) & (zl > 0.2)
    if b.sum() < 30:
        return None
    r = zs[b] / zl[b]
    r = r[(r > 0.75) & (r < 1.33)]
    return float(np.median(r)) if r.size >= 30 else None


def box_offset(cam, R, t, depth_mm, K, dist, splat_bytes, cell=2):
    """Best slide (du, dv) in pixels of the splat picture onto the live box, and how well each fits (meters)."""
    roi = _box_roi(cam, depth_mm.shape[:2], BOX_MARGIN)
    pad = BOX_SEARCH_PX // cell
    zs = _splat_depth_roi(R, t, K, dist, _splat_points(splat_bytes), roi, cell, pad)
    zl = _live_depth_roi(depth_mm, roi, cell)
    gh, gw = zl.shape

    def cost(i, j):   # live cell (y, x) compared with splat cell (y - i, x - j): splat slid by (j, i)
        s = zs[pad - i: pad - i + gh, pad - j: pad - j + gw]
        b = np.isfinite(s) & np.isfinite(zl)
        if b.sum() < 0.2 * gh * gw:
            return np.inf, 0
        return float(np.mean(np.minimum(np.abs(s[b] - zl[b]), 0.15))), int(b.sum())

    best, c0 = (0, 0), cost(0, 0)[0]
    bc = c0
    for i in range(-pad, pad + 1):
        for j in range(-pad, pad + 1):
            c, _ = cost(i, j)
            if c < bc - 1e-6:
                bc, best = c, (i, j)
    return {"du": best[1] * cell, "dv": best[0] * cell, "cost": bc, "cost0": c0}


def align_on_box(cam, R, t, depth_mm, K, dist, splat_bytes):
    """Returns (R, t, depth_mm scaled, info) with cam pinned to its landmark box."""
    if splat_bytes is None:
        return R, t, depth_mm, {"status": "none"}
    s = box_scale(cam, R, t, depth_mm, K, dist, splat_bytes)
    if s is None:
        return R, t, depth_mm, {"status": "box not visible"}
    s = min(BOX_SCALE_RANGE[1], max(BOX_SCALE_RANGE[0], s))
    d1 = depth_mm * s
    a = b = 0.0
    Rw, tw = R, t
    for _ in range(3):            # slide, turn, check again (a turn is not exactly a slide off-center)
        o = box_offset(cam, Rw, tw, d1, K, dist, splat_bytes)
        # The map pose's aim is already good to about a pixel; only turn when the slide clearly fits better.
        if (o["du"] == 0 and o["dv"] == 0) or o["cost"] > 0.85 * o["cost0"]:
            break
        a += o["du"] / K[0, 0]
        b += -o["dv"] / K[1, 1]
        Rc = _rot_xy(a, b)
        Rw, tw = _mul3(Rc, R), _mulv(Rc, t)
    hist = BOX_HISTORY.setdefault(cam, [])
    hist.append((a, b, s))
    del hist[:-5]
    ma, mb, ms = (float(np.median([h[i] for h in hist])) for i in range(3))
    Rc = _rot_xy(ma, mb)
    R1, t1, d1 = _mul3(Rc, R), _mulv(Rc, t), depth_mm * ms
    left = box_offset(cam, R1, t1, d1, K, dist, splat_bytes)
    roi = _box_roi(cam, depth_mm.shape[:2], 0.0)
    zs = _splat_depth_roi(R1, t1, K, dist, _splat_points(splat_bytes), roi, 4)
    zl = _live_depth_roi(d1, roi, 4)
    bb = np.isfinite(zs) & np.isfinite(zl)
    off = float(np.median(zl[bb] - zs[bb])) if bb.sum() > 10 else float("nan")
    info = {"status": "aligned", "turn_deg": round(float(np.degrees(np.hypot(ma, mb))), 3), "scale": round(ms, 4),
            "left_px": round(float(np.hypot(left["du"], left["dv"])), 1),
            "fit_cm": round(left["cost"] * 100, 1), "box_depth_off_cm": round(off * 100, 1)}
    return R1, t1, d1, info

# ---------------------------------------------------------------- viewer


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def recorded_views(step: int = 8) -> list[dict]:
    """Walkthrough camera poses (map frame, meters): the viewpoints the splat was trained from, so they look sharp."""
    path = OUT_DIR / "lab_map" / "transforms.json"
    if not path.is_file():
        return []
    try:
        t = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    frames = sorted(t.get("frames", []), key=lambda f: f.get("file_path", ""))
    fy = float(t.get("fl_y", 0) or 0)
    h = float(t.get("h", 0) or 0)
    fov = 2 * np.degrees(np.arctan(h / 2 / fy)) if fy and h else 60.0
    out = []
    for f in frames[::step]:
        M = np.asarray(f["transform_matrix"], dtype=np.float64)
        # transforms.json is in nerfstudio's applied frame: (x, y, z)_colmap -> (x, z, -y).
        # Undo it: colmap = (x, -z, y). Camera axes are OpenGL (x right, y up, z back).
        def un(v):
            return [float(v[0]), float(-v[2]), float(v[1])]
        out.append({"p": [round(x, 4) for x in un(M[:3, 3])],
                    "f": [round(-x, 4) for x in un(M[:3, 2])],
                    "u": [round(x, 4) for x in un(M[:3, 1])]})
    return [{"fov": round(float(fov), 2), **o} for o in out]


def write_viewer(path: Path, surfaces: list[dict], splat_bytes: bytes | None, up: list[float], note: str, arm: dict | None = None, page_version: int = 0) -> None:
    """Self-contained page: live surfaces + camera markers, splat background when available."""
    meta = []
    blobs = []
    for i, s in enumerate(surfaces):
        meta.append(
            {
                "name": s["name"],
                "rgb": list(s["marker_rgb"]),
                "center": [round(float(x), 4) for x in s["center"]],
                "R": np.asarray(s["R"]).round(6).tolist(),
                "K": np.asarray(s["K"]).round(4).tolist(),
                "size": [int(s["hw"][1]), int(s["hw"][0])],
                "inliers": s.get("inliers"),
                "reproj": s.get("reproj"),
                "verts": int(s["verts"].shape[0]),
                "faces": int(s["faces"].shape[0]),
                "still": s.get("still_b64", ""),
                "anchor": {k: s.get("anchor", {}).get(k) for k in ("status", "shift_px", "depth_change_m", "what")},
            }
        )
        blobs.append(f'<script type="application/octet-stream" id="v{i}">{_b64(s["verts"].astype(np.float32))}</script>')
        blobs.append(f'<script type="application/octet-stream" id="c{i}">{_b64(s["colors"].astype(np.uint8))}</script>')
        blobs.append(f'<script type="application/octet-stream" id="f{i}">{_b64(s["faces"].astype(np.uint32))}</script>')
    if splat_bytes:
        blobs.append(f'<script type="application/octet-stream" id="splat">{base64.b64encode(splat_bytes).decode("ascii")}</script>')
    payload = json.dumps({"surfaces": meta, "up": up, "note": note, "hasSplat": bool(splat_bytes), "views": recorded_views(), "arm": arm, "page": page_version})
    html = VIEWER_HTML.replace("__PAYLOAD__", payload).replace("__BLOBS__", "\n".join(blobs))
    path.write_text(html, encoding="utf-8")


VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>A-Lab Live Scan</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  /* Same look as the A-Lab walkthrough splat viewer: full-bleed stage, readout card top-left, control strip bottom-centre. */
  :root { --bg:#0e1319; --panel:rgba(250,251,252,0.9); --fg:#16202b; --muted:#5a6776; --line:rgba(22,32,43,0.14);
          --accent:#0f7a8a; --accent-fg:#ffffff;
          --display:"Archivo","Segoe UI","Helvetica Neue",Arial,sans-serif; --mono:"JetBrains Mono",Consolas,Menlo,monospace; }
  @media (prefers-color-scheme: dark) { :root { --panel:rgba(20,27,35,0.86); --fg:#e4eaf0; --muted:#8d9aa8;
          --line:rgba(228,234,240,0.14); --accent:#3cc3d4; --accent-fg:#06191c; color-scheme:dark; } }
  html,body { margin:0; height:100%; background:var(--bg); color:var(--fg); font-family:var(--display); overflow:hidden; }
  #stage { position:fixed; inset:0; } #stage canvas { display:block; touch-action:none; }
  .card { position:fixed; top:14px; left:16px; width:min(340px, calc(100vw - 32px)); background:var(--panel); border:1px solid var(--line);
          border-radius:10px; padding:14px 16px; display:grid; gap:9px; backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); z-index:2; }
  h1 { margin:0; font-size:17px; font-weight:700; }
  .note { margin:0; font-size:12.5px; color:var(--muted); line-height:1.45; }
  .row { display:grid; grid-template-columns:auto 1fr auto; gap:8px; align-items:center; font-size:12.5px; }
  .sw { width:10px; height:10px; border-radius:2px; }
  .stat { font-family:var(--mono); font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
  .stats { display:grid; grid-template-columns:repeat(3,auto); gap:4px 16px; font-family:var(--mono); font-size:11.5px; margin:0; }
  .stats dt { color:var(--muted); text-transform:uppercase; letter-spacing:.08em; font-size:9.5px; } .stats dd { margin:0; } .stats div { display:grid; gap:2px; }
  label { display:flex; gap:6px; align-items:center; cursor:pointer; }
  .bar { position:fixed; bottom:16px; left:50%; transform:translateX(-50%); display:flex; flex-wrap:wrap; gap:6px; justify-content:center;
         background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:6px; z-index:2; max-width:calc(100vw - 32px);
         backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
  .bar button { font:500 12.5px/1 var(--display); letter-spacing:.02em; color:var(--fg); background:transparent; border:1px solid transparent; border-radius:8px; padding:9px 12px; cursor:pointer; }
  .bar button:hover { border-color:var(--line); } .bar button[aria-pressed="true"] { background:var(--accent); color:var(--accent-fg); }
  .bar button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  #readout { font-family:var(--mono); font-size:11.5px; min-height:1.2em; }
  #status { position:fixed; inset:0; display:grid; place-items:center; font-family:var(--mono); font-size:12px; color:#8d9aa8; pointer-events:none; z-index:1; }
  #status[hidden] { display:none; }
  .armrow { display:grid; grid-template-columns:auto 1fr; gap:8px; align-items:start; font-family:var(--mono); font-size:11.5px; line-height:1.45; }
  .armrow .sw { margin-top:3px; }
  #pip { position:fixed; right:16px; top:14px; margin:0; width:min(320px, 40vw); background:var(--panel); border:1px solid var(--line);
         border-radius:10px; padding:6px; z-index:2; backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); }
  #pip[hidden] { display:none; } #pip img { display:block; width:100%; border-radius:6px; }
  #pip figcaption { font-family:var(--mono); font-size:10.5px; color:var(--muted); padding:5px 2px 1px; }
  @media (max-width:520px) { .note { display:none; } .bar button { padding:9px 9px; font-size:12px; } }
</style>
</head>
<body>
<div id="stage"></div>
<div id="status">Loading lab background…</div>
<div id="near-alert" hidden style="position:fixed;top:12px;left:50%;transform:translateX(-50%);z-index:20;background:#d62828;color:#fff;font:600 16px system-ui,sans-serif;padding:10px 18px;border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.4)"></div>
<section class="card">
  <h1>A-Lab live scan</h1>
  <p class="note" id="note"></p>
  <dl class="stats"><div><dt>Cameras</dt><dd id="s-cams">–</dd></div><div><dt>Triangles</dt><dd id="s-tris">–</dd></div><div><dt>Cam gap</dt><dd id="s-gap">–</dd></div></dl>
  <div id="rows"></div>
  <label class="stat"><input type="checkbox" id="t-splat" checked /> Lab background (walkthrough splat)</label>
  <div id="arm" class="armrow"><span class="sw" style="background:#ff6a2b"></span><span id="arm-text">Arm: waiting for the first capture</span></div>
  <div class="armrow"><span class="sw" style="background:#3fd27f"></span><span id="obj-text">Rack: waiting for the first capture</span></div>
  <label class="stat"><input type="checkbox" id="t-arm" checked /> Show tracked arm</label>
  <div id="live" class="stat"></div>
  <div id="readout"></div>
  <div class="stat">Drag to orbit · scroll to zoom. The splat is sharp near recorded views (◀ ▶) and smears far from them. "From cam" shows the splat from that camera next to its live picture</div>
</section>
<figure id="pip" hidden><img id="pip-img" alt="Live picture from the selected camera" /><figcaption id="pip-cap"></figcaption></figure>
<nav class="bar" id="bar"><button type="button" id="b-prev" aria-label="Previous recorded view">&#9664;</button><button type="button" id="b-over" aria-pressed="false">Overview</button><button type="button" id="b-next" aria-label="Next recorded view">&#9654;</button></nav>

__BLOBS__

<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/",
  "@mkkellogg/gaussian-splats-3d": "https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@0.4.7/build/gaussian-splats-3d.module.js"
} }
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
const DATA = __PAYLOAD__;
const status = document.getElementById("status");
document.getElementById("note").textContent = DATA.note;
function bytes(id) { const s = atob(document.getElementById(id).textContent.trim()); const u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return u; }

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
document.getElementById("stage").appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0f141a);
const camera = new THREE.PerspectiveCamera(60, 1, 0.02, 200);
const UP = new THREE.Vector3(...DATA.up).normalize();
camera.up.copy(UP);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;

const meshes = [], centers = [], box = new THREE.Box3();
DATA.surfaces.forEach((s, i) => {
  const v = new Float32Array(bytes("v" + i).buffer), c = bytes("c" + i), f = new Uint32Array(bytes("f" + i).buffer);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(v, 3));
  g.setAttribute("color", new THREE.BufferAttribute(c, 3, true));
  g.setIndex(new THREE.BufferAttribute(f, 1));
  g.computeBoundingBox(); box.union(g.boundingBox);
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  m.userData.name = s.name; m.visible = true; scene.add(m); meshes.push(m);
  // camera marker: frustum drawn from the solved pose
  const C = new THREE.Vector3(...s.center); centers.push(C);
  const R = s.R, K = s.K, [W, H] = s.size, d = 0.25;
  const corner = (u, v) => { const x = (u - K[0][2]) / K[0][0] * d, y = (v - K[1][2]) / K[1][1] * d, z = d;
    return new THREE.Vector3(R[0][0]*x + R[1][0]*y + R[2][0]*z, R[0][1]*x + R[1][1]*y + R[2][1]*z, R[0][2]*x + R[1][2]*y + R[2][2]*z).add(C); };
  const q = [corner(0,0), corner(W,0), corner(W,H), corner(0,H)];
  const pts = [C,q[0],C,q[1],C,q[2],C,q[3],q[0],q[1],q[1],q[2],q[2],q[3],q[3],q[0]];
  const col = new THREE.Color(s.rgb[0]/255, s.rgb[1]/255, s.rgb[2]/255);
  const fr = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: col }));
  scene.add(fr); m.userData.marker = fr;
  const row = document.createElement("div"); row.className = "row";
  const quality = s.inliers ? `${s.inliers} pts · ${(+s.reproj).toFixed(2)} px` : "";
  row.innerHTML = `<span class="sw" style="background:rgb(${s.rgb})"></span><label><input type="checkbox" id="t-${i}" checked/> ${s.name} live (what changed)</label><span class="stat">${quality}</span>`;
  document.getElementById("rows").appendChild(row);
  row.querySelector("input").onchange = (e) => { m.visible = e.target.checked; };
  const btn = document.createElement("button"); btn.type = "button"; btn.id = "b-cam" + i; btn.textContent = "From " + s.name; btn.setAttribute("aria-pressed", "false");
  btn.onclick = () => viewFrom(i); document.getElementById("bar").appendChild(btn);
});
const target = new THREE.Vector3(); box.getCenter(target);
document.getElementById("s-cams").textContent = DATA.surfaces.length;
document.getElementById("s-tris").textContent = Math.round(DATA.surfaces.reduce((a, s) => a + s.faces, 0) / 1000) + "k";
document.getElementById("s-gap").textContent = centers.length >= 2 ? centers[0].distanceTo(centers[1]).toFixed(2) + " m" : "–";

function press(id) { document.querySelectorAll(".bar button").forEach(b => b.setAttribute("aria-pressed", String(b === id))); }
let viewIdx = 0, currentPip = -1;
function recorded(i) {
  const V = DATA.views; if (!V || !V.length) return false;
  viewIdx = (i + V.length) % V.length; const r = V[viewIdx];
  document.getElementById("pip").hidden = true; currentPip = -1;
  camera.fov = r.fov; camera.updateProjectionMatrix();
  camera.position.set(...r.p); camera.up.set(...r.u);
  controls.target.copy(camera.position).add(new THREE.Vector3(...r.f).multiplyScalar(1.5)); controls.update();
  camera.up.copy(UP);
  document.getElementById("readout").textContent = `Recorded view ${viewIdx + 1} of ${V.length}`;
  return true;
}
function overview() {
  press(document.getElementById("b-over")); document.getElementById("pip").hidden = true; currentPip = -1;
  // Pick the recorded walkthrough view nearest the fixed cameras: the splat is sharp only from viewpoints it was trained on.
  const V = DATA.views || [];
  if (V.length) {
    const mid = centers.reduce((a, c) => a.add(c), new THREE.Vector3()).multiplyScalar(1 / Math.max(1, centers.length));
    let best = 0, bd = Infinity;
    V.forEach((r, i) => { const d = mid.distanceTo(new THREE.Vector3(...r.p)); if (d < bd) { bd = d; best = i; } });
    recorded(best); return;
  }
  camera.fov = 60; camera.updateProjectionMatrix();
  camera.position.copy(target).addScaledVector(UP, 1.0);
  controls.target.copy(target); controls.update();
}
function viewFrom(i, back = 0) {
  const s = DATA.surfaces[i], R = s.R, K = s.K;
  press(document.getElementById("b-cam" + i));
  const pip = document.getElementById("pip");
  currentPip = i;
  if (s.stillUrl || s.still) { document.getElementById("pip-img").src = s.stillUrl || ("data:image/jpeg;base64," + s.still);
    document.getElementById("pip-cap").textContent = s.name + " live picture. The view behind is the splat from the same pose.";
    pip.hidden = false; } else pip.hidden = true;
  camera.fov = 2 * Math.atan(s.size[1] / 2 / K[1][1]) * 180 / Math.PI; camera.updateProjectionMatrix();
  const fwd = new THREE.Vector3(R[2][0], R[2][1], R[2][2]);
  camera.position.copy(centers[i]).addScaledVector(fwd, -back).addScaledVector(UP, back * 0.4);
  controls.target.copy(centers[i]).addScaledVector(fwd, 1.0); controls.update();
}
document.getElementById("b-over").onclick = overview;
document.getElementById("b-prev").onclick = () => { press(null); recorded(viewIdx - 1); };
document.getElementById("b-next").onclick = () => { press(null); recorded(viewIdx + 1); };
if (!(DATA.views && DATA.views.length)) { document.getElementById("b-prev").hidden = true; document.getElementById("b-next").hidden = true; }

if (DATA.hasSplat) {
  import("@mkkellogg/gaussian-splats-3d").then(async (GS) => {
    const url = URL.createObjectURL(new Blob([bytes("splat")], { type: "application/octet-stream" }));
    const dv = new GS.DropInViewer({ gpuAcceleratedSort: false, sharedMemoryForWorkers: false, sphericalHarmonicsDegree: 0 });
    await dv.addSplatScenes([{ path: url, format: GS.SceneFormat.Splat, splatAlphaRemovalThreshold: 5 }], false);
    scene.add(dv); status.hidden = true;
    document.getElementById("t-splat").onchange = (e) => { dv.visible = e.target.checked; };
    dv.visible = document.getElementById("t-splat").checked;
  }).catch((e) => { console.error(e); status.hidden = true; document.getElementById("t-splat").parentElement.textContent = "Lab background could not load (needs internet for the splat renderer)."; });
} else {
  document.getElementById("t-splat").parentElement.textContent = "No lab background file found (lab_map/lab_colmap.splat).";
}


// ---- tracked arm (lab coordinates, from the arm cameras' depth)
const armGroup = new THREE.Group(); scene.add(armGroup);
function showArm(a) {
  armGroup.clear();
  const txt = document.getElementById("arm-text");
  if (!a) { txt.textContent = "Arm tracking off"; return; }
  if (!a.seen) { txt.textContent = `Arm: not found at ${a.time.slice(11)}` + (a.method ? " (nothing connected to the base)" : ` (${Object.entries(a.per_camera || {}).map(([k, v]) => k + ": " + (v.reason || "")).join("; ")})`); return; }
  const p = new Float32Array(a.pts);
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(p, 3));
  armGroup.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xff6a2b, size: 0.012, sizeAttenuation: true, depthTest: false, transparent: true, opacity: 0.85 })));
  const b = new THREE.Box3(new THREE.Vector3(...a.bbox_min_m), new THREE.Vector3(...a.bbox_max_m));
  armGroup.add(new THREE.Box3Helper(b, 0xff6a2b));
  const top = new THREE.Mesh(new THREE.SphereGeometry(0.02, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffd23f, depthTest: false }));
  const t = a.tip_m || a.top_m;
  const gr = a.clearance || {};
  top.position.set(...t); armGroup.add(top);
  if (gr.clearance_m != null && gr.at_m) {
    const ring = new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.03, gr.clearance_m), 24, 16), new THREE.MeshBasicMaterial({ color: gr.near ? 0xff2020 : 0x3fd27f, wireframe: true, transparent: true, opacity: 0.45, depthTest: false }));
    ring.position.set(...gr.at_m); armGroup.add(ring);
    if (gr.near) { const pa = armGroup.children[0]; if (pa && pa.material) pa.material.color.set(0xff2020); }
  }
  const al = document.getElementById("near-alert");
  if (al) { al.hidden = !gr.near; if (gr.near) al.textContent = `Robot ${gr.part} within ${Math.round(gr.clearance_m * 100)} cm of ${gr.what}`; }
  if (a.base_m) { const bs = new THREE.Mesh(new THREE.SphereGeometry(0.03, 16, 12), new THREE.MeshBasicMaterial({ color: 0x3fa7ff, depthTest: false })); bs.position.set(...a.base_m); armGroup.add(bs); }
  txt.textContent = a.tip_m
    ? `Arm at ${a.time.slice(11)}: gripper end ${t.map(v => v.toFixed(2)).join(", ")} m, ${a.tip_reach_m.toFixed(2)} m from the base (blue)` + (gr.clearance_m != null ? ` · closest ${Math.round(gr.clearance_m * 100)} cm (${gr.part})` : "") + ` · ${a.points} pts (${a.cameras.join("+")})`
    : `Arm at ${a.time.slice(11)}: centre ${a.center_m.map(v => v.toFixed(2)).join(", ")} m · top ${t.map(v => v.toFixed(2)).join(", ")} m · ${a.points} pts (${a.cameras.join("+")})`;
}
const objGroup = new THREE.Group(); scene.add(objGroup);
function showObjects(objs) {
  objGroup.clear();
  const el = document.getElementById("obj-text");
  if (!objs || !objs.rack) { if (el) el.textContent = ""; return; }
  const r = objs.rack, b = r.box;
  const col = r.state === "OPEN" ? 0xff9a2b : r.state === "CLOSED" ? 0x3fd27f : 0x9aa0a6;
  if (b) {
    const c = new THREE.Vector3(...b.center_m), A = b.axes.map(a => new THREE.Vector3(...a));
    const P = (i, j, k) => c.clone().addScaledVector(A[0], [b.min_m[0], b.max_m[0]][i]).addScaledVector(A[1], [b.min_m[1], b.max_m[1]][j]).addScaledVector(A[2], [b.min_m[2], b.max_m[2]][k]);
    const pts = [];
    for (const [a, bb] of [[[0,0,0],[1,0,0]],[[0,1,0],[1,1,0]],[[0,0,1],[1,0,1]],[[0,1,1],[1,1,1]],[[0,0,0],[0,1,0]],[[1,0,0],[1,1,0]],[[0,0,1],[0,1,1]],[[1,0,1],[1,1,1]],[[0,0,0],[0,0,1]],[[1,0,0],[1,0,1]],[[0,1,0],[0,1,1]],[[1,1,0],[1,1,1]]]) { pts.push(P(...a), P(...bb)); }
    objGroup.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: col, depthTest: false })));
  }
  if (el) el.textContent = r.state === "OPEN"
    ? `Rack: OPEN · shelf ${(r.shelves_open || []).join(", ") || "?"} from the bottom · out ~${Math.round(r.pulled_out_cm)} cm`
    : `Rack: ${r.state}` + (r.state === "NOT CALIBRATED" ? " (run with --calibrate-rack, all trays in)" : "");
}
showArm(DATA.arm); showObjects(DATA.arm && DATA.arm.objects);
document.getElementById("t-arm").onchange = (e) => { armGroup.visible = e.target.checked; };

// ---- live updates (python scan_lab_live.py --watch serves this page over http://localhost)
let liveVer = null;
async function pollLive() {
  if (!location.protocol.startsWith("http")) return;
  try {
    const m = await (await fetch("live/manifest.json?t=" + Date.now(), { cache: "no-store" })).json();
    if (m.page && DATA.page && m.page !== DATA.page) { saveView(); location.reload(); return; }
    if (m.version !== liveVer) {
      liveVer = m.version;
      for (const c of m.cams) {
        const i = DATA.surfaces.findIndex(s => s.name === c.name);
        if (i < 0) continue;
        const buf = await (await fetch("live/" + c.bin + "?v=" + m.version, { cache: "no-store" })).arrayBuffer();
        const v = new Float32Array(buf, 0, c.nv * 3), f = new Uint32Array(buf, c.nv * 12, c.nf * 3);
        const col = new Uint8Array(buf, c.nv * 12 + c.nf * 12, c.nv * 3);
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(v, 3));
        g.setAttribute("color", new THREE.BufferAttribute(col, 3, true));
        g.setIndex(new THREE.BufferAttribute(f, 1));
        meshes[i].geometry.dispose(); meshes[i].geometry = g;
        if (c.jpg) DATA.surfaces[i].stillUrl = "live/" + c.jpg + "?v=" + m.version;
      }
      if (currentPip >= 0 && DATA.surfaces[currentPip].stillUrl) document.getElementById("pip-img").src = DATA.surfaces[currentPip].stillUrl;
      if (m.arm) { try { const a = await (await fetch("live/arm.json?v=" + m.version, { cache: "no-store" })).json(); showArm(a); showObjects(a.objects); } catch (e) {} }
      const lk = Object.entries(m.anchors || {}).map(([k, s]) => k + (s === "locked" ? " locked" : s === "moved" ? " MOVED" : s === "new" ? " reference saved" : "")).filter(x => !x.endsWith(" ")).join(" · ");
      liveTime = m.time; liveLabel = lk ? " · " + lk : "";
      document.getElementById("live").textContent = "Live · last capture " + m.time + liveLabel;
    }
  } catch (e) { document.getElementById("live").textContent = "Live updates paused (is --watch still running?)"; }
  setTimeout(pollLive, 1000);
}
pollLive();
// How old is what's on screen? (capture time vs this computer's clock; same machine as the scan)
let liveTime = null, liveLabel = "";
setInterval(() => {
  if (!liveTime) return;
  const [h, mi, se] = liveTime.split(":").map(Number), n = new Date();
  let age = (n.getHours() * 3600 + n.getMinutes() * 60 + n.getSeconds()) - (h * 3600 + mi * 60 + se);
  if (age < 0) age += 86400;
  document.getElementById("live").textContent = `Live · captured ${age} s ago` + liveLabel;
}, 1000);
const ray = new THREE.Raycaster(), ndc = new THREE.Vector2(); let down = null;
renderer.domElement.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", (e) => {
  if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObjects(meshes.filter(m => m.visible), false)[0];
  if (!meshes.some(m => m.visible)) { document.getElementById("readout").textContent = "Tick a camera's live depth to click surfaces for XYZ."; return; }
  document.getElementById("readout").textContent = hit
    ? `${hit.object.userData.name}  x ${hit.point.x.toFixed(3)}  y ${hit.point.y.toFixed(3)}  z ${hit.point.z.toFixed(3)} m`
    : "No live surface under that point.";
});
function resize() { const w = innerWidth, h = innerHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize();
// Keep the view and the checkboxes when the page reloads itself (the splat was refreshed because the arm moved).
function saveView() {
  try {
    const boxes = {}; document.querySelectorAll("input[type=checkbox]").forEach(b => { if (b.id) boxes[b.id] = b.checked; });
    sessionStorage.setItem("labview", JSON.stringify({ p: camera.position.toArray(), t: controls.target.toArray(), fov: camera.fov, pip: currentPip, boxes }));
  } catch (e) {}
}
function restoreView() {
  try {
    const s = JSON.parse(sessionStorage.getItem("labview") || "null"); if (!s) return false;
    sessionStorage.removeItem("labview");
    camera.fov = s.fov; camera.updateProjectionMatrix();
    camera.position.fromArray(s.p); controls.target.fromArray(s.t); controls.update();
    for (const [id, on] of Object.entries(s.boxes || {})) { const b = document.getElementById(id); if (b && b.checked !== on) { b.checked = on; b.dispatchEvent(new Event("change")); } }
    return true;
  } catch (e) { return false; }
}
if (!restoreView()) { if (DATA.surfaces.length) viewFrom(DATA.surfaces.length - 1, 0.6); else overview(); }
if (!DATA.hasSplat) status.hidden = true;
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>
</body>
</html>
"""



# ---------------------------------------------------------------- main

_SCALE_HISTORY: dict[str, list[float]] = {}
RELOCALIZE_NEXT: set[str] = set()
PAGE_STATE = {"version": 0, "arm_removed": False}
SPLAT_REFRESH_BLOBS = 3000   # rebuild the page when this many splat blobs change visibility
SPLAT_REFRESH_S = 60         # but not more often than this
STICKY_VOTES = 2             # captures that must see through a blob before it is removed for good


_STATIC_SPLAT: dict = {}


def _static_splat_mask(splat_bytes: bytes) -> np.ndarray:
    """Splat blobs inside the never-changing objects (cached)."""
    key = (id(splat_bytes), OBJECTS_FILE.stat().st_mtime if OBJECTS_FILE.is_file() else 0)
    if _STATIC_SPLAT.get("key") != key:
        P = np.frombuffer(splat_bytes, dtype=SPLAT_DTYPE)["p"].astype(np.float64)
        _STATIC_SPLAT.update(key=key, mask=in_static(P, 0.02))
    return _STATIC_SPLAT["mask"]


def _static_pixels(depth_mm: np.ndarray, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray,
                   step: int = 2) -> np.ndarray:
    """Pixels whose surface lies inside a never-changing object: never drawn as 'changed'."""
    import cv2

    h, w = depth_mm.shape
    vv, uu = np.mgrid[0:h:step, 0:w:step]
    z = np.asarray(depth_mm, dtype=np.float64)[vv, uu] / 1000.0
    ok = z > 0
    out = np.zeros(vv.shape, dtype=bool)
    if ok.any():
        nrm = cv2.undistortPoints(np.stack([uu[ok], vv[ok]], 1).astype(np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        P = cam_to_map(R, t, np.stack([nrm[:, 0] * z[ok], nrm[:, 1] * z[ok], z[ok]], 1))
        out[ok] = in_static(P, -0.02)          # 2 cm inside the box: the arm right next to it still counts
    full = np.repeat(np.repeat(out, step, 0), step, 1)[:h, :w]
    return full


def background_views() -> list[dict]:
    """Learned 'empty lab' depth for the arm cameras, as full-size depth images, for removing the arm from the splat."""
    out = []
    for cam, bg in _ARM_BG.items():
        if not bg.ready() or cam not in _LAST_POSE:
            continue
        R, t, K, dist, hw = _LAST_POSE[cam]
        back = np.nan_to_num(bg.background(), nan=0.0).astype(np.float64)
        full = np.repeat(np.repeat(back, ARM_STRIDE, axis=0), ARM_STRIDE, axis=1)
        pad = np.zeros(hw)
        pad[: full.shape[0], : full.shape[1]] = full[: hw[0], : hw[1]]
        out.append({"R": R, "t": t, "K": K, "dist": dist, "depth_m": pad})
    return out


_LAST_POSE: dict[str, tuple] = {}


def _splat_bytes() -> bytes | None:
    if not hasattr(_splat_bytes, "cache"):
        _splat_bytes.cache = SPLAT_PATH.read_bytes() if SPLAT_PATH.is_file() else None
    return _splat_bytes.cache


def compute_surfaces(frames: dict, report: dict, cam_data: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Pose, depth-scale correction and live surface for every camera that has a frame."""
    import warnings

    surfaces, views = [], []
    splat = _splat_bytes()
    for cam in CAMS:
        if cam not in frames:
            print(f"{cam}: no frame, skipped")
            continue
        fr = frames[cam]
        depth = np.asarray(fr["depth"], dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        hw = depth.shape[:2]
        K, dist = load_lens(cam, fr["info"], hw)
        try:
            relocal = getattr(build, "relocalize", False) or cam in RELOCALIZE_NEXT
            R, t, pose = locate_camera(cam, fr, K, dist, relocal)
            RELOCALIZE_NEXT.discard(cam)
        except RuntimeError as exc:
            print(exc)
            report["cameras"][cam] = {"skipped": str(exc)}
            continue

        # Is the camera still where we think? Check the fixed landmark against its reference.
        anchor = check_anchor(cam, fr.get("color"), depth, bool(pose.get("_fresh")))
        if anchor["status"] == "locked":
            print(f"{cam}: locked on the {anchor['what'] or 'landmark'} (shift {anchor['shift_px']} px, "
                  f"depth change {anchor['depth_change_m']} m)")
        elif anchor["status"] == "moved":
            print(f"{cam}: landmark moved (shift {anchor['shift_px']} px, depth change {anchor['depth_change_m']} m)"
                  + ("; locating this camera again next capture" if anchor["misses"] >= 2 else "; checking again next capture"))
            if anchor["misses"] >= 2:
                RELOCALIZE_NEXT.add(cam)
        elif anchor["status"] == "new":
            print(f"{cam}: saved a new landmark reference")

        # cam1 (any camera with a landmark): pin it to the black box every capture. This sets both its depth
        # scale and a small aim correction from the box alone, so the box always lands on the splat's box.
        box = {"status": "none"}
        scale_info = {"scale": 1.0, "ok": False}
        if splat and not getattr(build, "no_box_align", False) and cam in _anchors():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                R2, t2, d2, box = align_on_box(cam, R, t, depth, K, dist, splat)
            if box["status"] == "aligned":
                R, t, depth = R2, t2, d2
                scale_info = {"scale": box["scale"], "ok": True, "applied": box["scale"], "from": "black box"}
                print(f"{cam}: pinned to the black box (depth x{box['scale']}, aim turned {box['turn_deg']} deg; "
                      f"box now off by {box['left_px']} px and {box['box_depth_off_cm']:+.1f} cm)")
            else:
                print(f"{cam}: black box check: {box['status']}")
        anchor["box_align"] = box

        # Other cameras: correct the depth scale against the splat over the whole frame.
        if box["status"] != "aligned" and splat and not getattr(build, "no_depth_fix", False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scale_info = depth_scale_vs_splat(R, t, depth, K, dist, splat)
            if scale_info.get("ok"):
                hist = _SCALE_HISTORY.setdefault(cam, [])
                hist.append(scale_info["scale"])
                del hist[:-7]
                s = float(np.median(hist))
                s = min(1.15, max(0.87, s))
                if abs(s - 1.0) > 0.015:
                    depth = depth * s
                scale_info["applied"] = round(s if abs(s - 1.0) > 0.015 else 1.0, 4)
                print(f"{cam}: live depth vs splat = {scale_info['scale']:.3f} "
                      f"(spread {scale_info['spread']:.3f}); applying x{scale_info['applied']}")

        color = fr.get("color")
        if color is not None and np.asarray(color).shape[:2] != hw:
            print(f"{cam}: color {np.asarray(color).shape[:2]} and depth {hw} differ, surface left uncolored")
            color = None
        live_depth = depth
        changed_frac = None
        if splat and not getattr(build, "live_all", False):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strict: list = []
                ch = changed_mask(depth, K, dist, R, t, PAGE_STATE.get("splat_shown") or splat, strict=strict)
            stat_px = _static_pixels(depth, K, dist, R, t)
            ch &= ~stat_px
            if strict:
                strict[0] &= ~stat_px
            changed_frac = float(ch.mean())
            live_depth = np.where(ch, depth, 0.0)
            if cam_data is not None:
                cam_data.setdefault("_changed", {})[cam] = strict[0] if strict else ch
            print(f"{cam}: {changed_frac*100:.0f}% of the view differs from the splat (shown live); the rest is the splat")
        verts, cols, faces = depth_surface(live_depth, color, K, dist, R, t)
        jpg_bytes = b""
        if color is not None:
            import cv2

            ok, jpg = cv2.imencode(".jpg", np.asarray(color), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                jpg_bytes = jpg.tobytes()
        views.append({"R": R, "t": t, "K": K, "dist": dist, "depth_m": depth / 1000.0})
        _LAST_POSE[cam] = (R, t, K, dist, hw)
        if cam_data is not None:
            cam_data[cam] = {"color": color, "depth_mm": depth, "K": K, "dist": dist, "R": R, "t": t}
        center = camera_center(R, t)
        print(f"{cam}: {verts.shape[0]} vertices, {faces.shape[0]} triangles, camera at {center.round(3)} m")
        surfaces.append(
            {
                "name": cam, "verts": verts, "colors": cols, "faces": faces, "R": R, "center": center, "K": K, "hw": hw,
                "marker_rgb": CAM_RGB.get(cam, (200, 200, 200)),
                "inliers": pose.get("inliers"), "reproj": pose.get("median_reproj_px"), "anchor": anchor,
                "still_b64": base64.b64encode(jpg_bytes).decode("ascii") if jpg_bytes else "", "jpg": jpg_bytes,
            }
        )
        report["cameras"][cam] = {
            "camera_center_m": center.tolist(), "vertices": int(verts.shape[0]), "triangles": int(faces.shape[0]),
            "pose_inliers": pose.get("inliers"), "pose_median_reproj_px": pose.get("median_reproj_px"),
            "depth_scale_vs_splat": scale_info, "anchor": anchor,
        }
    return surfaces, views


OVERLAP_PREFER = "cam1"     # in the area both cameras see, show this camera's surface (it is pinned to the black box)
OVERLAP_TOL_M = 0.10        # points within this of the preferred camera's surface count as the same surface


def dedupe_overlap(surfaces: list[dict], cam_data: dict, report: dict) -> None:
    """Where both cameras see the same surface, keep one copy (the preferred camera's) instead of two
    slightly offset layers. Points the preferred camera can't see, or sees something else in front of, stay."""
    import cv2

    best = cam_data.get(OVERLAP_PREFER)
    if best is None:
        return
    R, t, K, dist = best["R"], best["t"], best["K"], best["dist"]
    D = np.asarray(best["depth_mm"], dtype=np.float64) / 1000.0
    h, w = D.shape
    for srf in surfaces:
        if srf["name"] == OVERLAP_PREFER or srf["faces"].size == 0:
            continue
        V = srf["verts"].astype(np.float64)
        xc = R[0, 0] * V[:, 0] + R[0, 1] * V[:, 1] + R[0, 2] * V[:, 2] + t[0]
        yc = R[1, 0] * V[:, 0] + R[1, 1] * V[:, 1] + R[1, 2] * V[:, 2] + t[1]
        zc = R[2, 0] * V[:, 0] + R[2, 1] * V[:, 1] + R[2, 2] * V[:, 2] + t[2]
        same = np.zeros(V.shape[0], dtype=bool)
        front = zc > 0.2
        if front.any():
            uv, _ = cv2.projectPoints(np.stack([xc[front], yc[front], zc[front]], 1).reshape(-1, 1, 3),
                                      np.zeros(3), np.zeros(3), K, dist)
            uv = uv.reshape(-1, 2)
            u, v = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            d = np.zeros(u.size)
            d[inside] = D[v[inside], u[inside]]
            ok = (d > 0) & (np.abs(zc[front] - d) < OVERLAP_TOL_M + 0.02 * d)
            idx = np.flatnonzero(front)
            same[idx[ok]] = True
        f = srf["faces"]
        keep = ~(same[f[:, 0]] & same[f[:, 1]] & same[f[:, 2]])
        dropped = int((~keep).sum())
        srf["faces"] = f[keep]
        report["cameras"].setdefault(srf["name"], {})["overlap_faces_hidden"] = dropped
        print(f"{srf['name']}: hid {dropped} triangles that {OVERLAP_PREFER} also sees (one copy instead of two)")


def build(frames: dict, source: str, write_page: bool = True) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"source": source, "frame": "COLMAP walkthrough map, meters", "cameras": {}}
    cam_data: dict = {}
    surfaces, views = compute_surfaces(frames, report, cam_data)
    splat0 = _splat_bytes()
    if splat0 and "splat_shown" not in PAGE_STATE and not getattr(build, "keep_arm_in_splat", False) and views:
        # First capture: clean the splat with this capture first, then judge "what changed" against the
        # cleaned splat (otherwise the walkthrough's frozen arm counts as part of the scene).
        drop = carve_mask(splat0, views + background_views()) & ~_static_splat_mask(splat0)
        PAGE_STATE["votes"] = drop.astype(np.uint8) * STICKY_VOTES
        PAGE_STATE["drop"] = drop
        PAGE_STATE["splat_shown"] = np.frombuffer(splat0, dtype=SPLAT_DTYPE)[~drop].tobytes()
        print(f"splat: removed {int(drop.sum())} blobs the cameras see through (robot's old pose, moved things)")
        report = {"source": source, "frame": "COLMAP walkthrough map, meters", "cameras": {}}
        cam_data = {}
        surfaces, views = compute_surfaces(frames, report, cam_data)
    if not getattr(build, "keep_overlap", False):
        dedupe_overlap(surfaces, cam_data, report)
    if not surfaces:
        print("No camera had both a frame and a map pose. Nothing to show.")
        return 1
    if len(surfaces) >= 2:
        a, b = surfaces[0]["center"], surfaces[1]["center"]
        report["camera_distance_m"] = float(np.sqrt(((a - b) ** 2).sum()))

    up = [0.0, 0.0, 1.0]
    if SPLAT_FRAME.is_file():
        up = json.loads(SPLAT_FRAME.read_text(encoding="utf-8")).get("up", up)
    arm = None
    if not getattr(build, "no_arm", False):
        try:
            changed = cam_data.pop("_changed", {})
            if changed and not getattr(build, "arm_detector", False):
                for c, ch in changed.items():
                    if c in cam_data:
                        cam_data[c]["changed"] = ch
                arm = track_arm_changed({c: cam_data[c] for c in changed if c in cam_data}, up,
                                        PAGE_STATE.get("splat_shown") or _splat_bytes())
            else:
                arm = track_arm(cam_data, getattr(build, "arm_cams", ARM_CAMS_DEFAULT), getattr(build, "arm_box", None), up)
            report["arm"] = {k: v for k, v in arm.items() if not k.startswith("_")}
        except Exception as exc:
            print(f"arm: tracking failed ({exc})")
    try:
        objs = report_objects(cam_data, arm)
        report["objects"] = objs
        if arm is not None:
            arm["objects"] = objs
    except Exception as exc:
        print(f"objects: state check failed ({exc})")
    write_live(surfaces, arm)
    names = tuple(s["name"] for s in surfaces)
    if PAGE_STATE.get("cams") not in (None, names):
        print(f"cameras now {', '.join(names)} (was {', '.join(PAGE_STATE['cams'])}); rebuilding the page")
        write_page = True
    PAGE_STATE["cams"] = names

    # Take the robot (and anything else that moved) out of the splat. A blob that the cameras have seen
    # straight through in STICKY_VOTES captures stays removed, even when the arm later blocks the view of
    # that spot. (Before, blobs came back whenever the arm hid them, so the splat changed every time the arm
    # moved and the whole page had to reload: that was most of the lag.) The page is rebuilt only when the
    # removed set grows by a lot, at most once a minute, keeping your view.
    splat = _splat_bytes()
    if splat and not getattr(build, "keep_arm_in_splat", False):
        cviews = views + background_views()
        if cviews:
            now = carve_mask(splat, cviews) & ~_static_splat_mask(splat)
            votes = PAGE_STATE.get("votes")
            if votes is None or votes.shape != now.shape:
                votes = now.astype(np.uint8) * STICKY_VOTES
            votes = np.minimum(votes.astype(np.int16) + now, 250).astype(np.uint8)
            PAGE_STATE["votes"] = votes
            drop = votes >= STICKY_VOTES
            old = PAGE_STATE.get("drop")
            changed = int(drop.size if old is None else (drop & ~old).sum())
            due = time.time() - PAGE_STATE.get("written_at", 0) > SPLAT_REFRESH_S
            if write_page or (changed > SPLAT_REFRESH_BLOBS and due):
                if not write_page:
                    print(f"splat: {changed} more blobs confirmed stale since the page was built; refreshing the page")
                write_page = True
                PAGE_STATE["drop"] = drop
                splat = np.frombuffer(splat, dtype=SPLAT_DTYPE)[~drop].tobytes()
                PAGE_STATE["splat_shown"] = splat
                PAGE_STATE["arm_removed"] = True
                report["splat_blobs_removed"] = int(drop.sum())
            else:
                splat = PAGE_STATE.get("splat_shown", splat)
    if write_page:
        PAGE_STATE["written_at"] = time.time()
        mesh_path = OUT_DIR / "lab_scan_mesh.ply"
        write_mesh_ply(mesh_path, [(s["verts"], s["colors"], s["faces"]) for s in surfaces], "fixed-camera surfaces, map frame, meters")
        PAGE_STATE["version"] += 1
        stamp = time.strftime("%Y-%m-%d %H:%M")
        live = " Updating live." if getattr(build, "watching", False) else ""
        note = (
            f"Captured {stamp} ({source}).{live} The scene is the premade splat; cameras are placed with poses "
            "solved against the walkthrough map. Tick a camera's live depth to overlay what it measures now. Meters."
        )
        view_path = OUT_DIR / "lab_scan.html"
        write_viewer(view_path, surfaces, splat, up, note, arm_payload(arm), PAGE_STATE["version"])
        report.update({"mesh": str(mesh_path), "viewer": str(view_path), "splat_background": bool(splat)})
        print(f"viewer -> {view_path}")
    (OUT_DIR / "lab_scan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------- live updates for --watch

LIVE_DIR = OUT_DIR / "live"


def arm_payload(arm: dict | None) -> dict | None:
    if not arm:
        return None
    pts = arm.get("_points")
    out = {k: v for k, v in arm.items() if not k.startswith("_")}
    out["pts"] = np.round(pts, 3).ravel().tolist() if pts is not None and len(pts) else []
    return out


def write_live(surfaces: list[dict], arm: dict | None = None) -> None:
    """Small per-capture files the open page polls: geometry + still per camera, and a manifest."""
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    version = time.strftime("%Y%m%d-%H%M%S")
    cams = []
    for s in surfaces:
        name = s["name"]
        v = np.ascontiguousarray(s["verts"], dtype=np.float32).tobytes()
        f = np.ascontiguousarray(s["faces"], dtype=np.uint32).tobytes()
        c = np.ascontiguousarray(s["colors"], dtype=np.uint8).tobytes()
        tmp = LIVE_DIR / f"{name}.bin.tmp"
        tmp.write_bytes(v + f + c)            # floats, then indices, then colors (keeps 4-byte alignment)
        tmp.replace(LIVE_DIR / f"{name}.bin")
        if s.get("jpg"):
            tj = LIVE_DIR / f"{name}.jpg.tmp"
            tj.write_bytes(s["jpg"])
            tj.replace(LIVE_DIR / f"{name}.jpg")
        cams.append({"name": name, "nv": int(s["verts"].shape[0]), "nf": int(s["faces"].shape[0]),
                     "bin": f"{name}.bin", "jpg": f"{name}.jpg" if s.get("jpg") else ""})
    if arm is not None:
        ta = LIVE_DIR / "arm.json.tmp"
        ta.write_text(json.dumps(arm_payload(arm)), encoding="utf-8")
        ta.replace(LIVE_DIR / "arm.json")
    anchors = {s["name"]: s.get("anchor", {}).get("status") for s in surfaces}
    man = {"version": version, "time": time.strftime("%H:%M:%S"), "cams": cams, "arm": arm is not None,
           "page": PAGE_STATE["version"], "anchors": anchors}
    tm = LIVE_DIR / "manifest.json.tmp"
    tm.write_text(json.dumps(man), encoding="utf-8")
    tm.replace(LIVE_DIR / "manifest.json")


def serve_out_dir(port: int) -> int:
    """Serve lab_scan/ over http://localhost so the page can fetch live updates (file:// can't)."""
    import functools
    import http.server
    import socketserver
    import threading

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

    handler = functools.partial(Quiet, directory=str(OUT_DIR))
    for p in range(port, port + 20):
        try:
            srv = socketserver.ThreadingTCPServer(("127.0.0.1", p), handler)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("No free port for the live page server.")
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return p


def grab_frames(args) -> tuple[dict, str]:
    if args.frames:
        z = np.load(args.frames, allow_pickle=True)
        return z["frames"].item(), f"saved {args.frames.name}"
    if args.ros:
        return grab(), "ros"
    if args.usb:
        return grab_usb(), "usb (OpenCV)"
    try:
        return grab_sdk(keep_open=getattr(args, "watch", None) is not None), "Orbbec SDK"
    except ImportError:
        print("pyorbbecsdk is not installed in this Python. Trying ROS topics, then OpenCV USB.")
        try:
            return grab(), "ros"
        except (TimeoutError, ImportError) as exc:
            print(f"{exc} Falling back to OpenCV USB.")
            return grab_usb(), "usb (OpenCV)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Live fixed-camera scan placed in the lab map.")
    ap.add_argument("--ros", action="store_true", help="Read the ROS topics instead of the Orbbec SDK.")
    ap.add_argument("--usb", action="store_true", help="Old OpenCV USB path (does not work with Gemini 335L on this PC).")
    ap.add_argument("--frames", type=Path, help="Use a saved grab (.npz from --save) instead of the cameras.")
    ap.add_argument("--save", type=Path, help="Also save this grab to an .npz for offline reruns.")
    ap.add_argument("--no-open", action="store_true", help="Don't open the viewer in a browser.")
    ap.add_argument("--relocalize", action="store_true", help="Locate every camera again from this grab, even if a saved pose matches.")
    ap.add_argument("--carve", action="store_true", help="Remove splat blobs the live cameras see as empty space (only useful with live depth on).")
    ap.add_argument("--no-depth-fix", action="store_true", help="Don't rescale each camera's depth to agree with the splat.")
    ap.add_argument("--watch", nargs="?", const=15.0, type=float, metavar="SECONDS",
                    help="Keep capturing every SECONDS (default 15) and update the open page in place.")
    ap.add_argument("--no-arm", action="store_true", help="Don't track the robot arm.")
    ap.add_argument("--arm-detector", action="store_true", help="Old arm tracking (YOLO box + learned background) instead of change-connected-to-base.")
    ap.add_argument("--calibrate-rack", action="store_true", help="Save the rack's closed look (push every tray in first).")
    ap.add_argument("--keep-overlap", action="store_true", help="Show both cameras' surfaces where they overlap.")
    ap.add_argument("--live-all", action="store_true", help="Show each camera's whole live depth, not just what changed vs the splat.")
    ap.add_argument("--no-box-align", action="store_true", help="Don't lock cam1 onto the black box each capture.")
    ap.add_argument("--keep-arm-in-splat", action="store_true", help="Leave the walkthrough's frozen copy of the arm in the splat.")
    ap.add_argument("--arm-cams", default=",".join(ARM_CAMS_DEFAULT), help="Cameras used to track the arm (default cam1; e.g. cam0,cam1).")
    ap.add_argument("--arm-box", help="Testing: arm box x,y,w,h in pixels instead of the detector.")
    ap.add_argument("--port", type=int, default=8765, help="Local port for --watch (default 8765).")
    args = ap.parse_args(argv)

    build.carve = args.carve
    build.relocalize = args.relocalize
    build.no_depth_fix = args.no_depth_fix
    build.watching = args.watch is not None
    build.no_arm = args.no_arm
    build.keep_overlap = args.keep_overlap
    build.calibrate_rack = 3 if args.calibrate_rack else 0
    build.arm_detector = args.arm_detector
    build.live_all = args.live_all
    build.no_box_align = args.no_box_align
    build.keep_arm_in_splat = args.keep_arm_in_splat
    build.arm_cams = tuple(c.strip() for c in args.arm_cams.split(",") if c.strip())
    build.arm_box = [float(x) for x in args.arm_box.split(",")] if args.arm_box else None

    if args.watch is None:
        frames, source = grab_frames(args)
        if args.save:
            np.savez_compressed(args.save, frames=np.array(frames, dtype=object))
            print(f"saved grab -> {args.save}")
        code = build(frames, source)
        if code == 0 and not args.no_open:
            webbrowser.open((OUT_DIR / "lab_scan.html").as_uri())
        return code

    interval = max(3.0, float(args.watch))
    port = serve_out_dir(args.port)
    url = f"http://localhost:{port}/lab_scan.html"
    print(f"Watching: capturing every {interval:.0f} s. Page: {url}  (Ctrl+C to stop)")
    first = True
    try:
        return _watch_loop(args, url, interval, first)
    finally:
        try:
            sdk_close_all()
        except Exception:
            pass


def _watch_loop(args, url: str, interval: float, first: bool) -> int:
    while True:
        t0 = time.time()
        try:
            frames, source = grab_frames(args)
            need_page = first
            code = build(frames, source, write_page=need_page)
            if first and code == 0:
                build.relocalize = False          # locate once, then reuse
                if not args.no_open:
                    webbrowser.open(url)
                first = False
            print(f"[{time.strftime('%H:%M:%S')}] updated in {time.time() - t0:.1f} s")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[{time.strftime('%H:%M:%S')}] capture failed: {exc}. Trying again next round.")
        try:
            time.sleep(max(0.0, interval - (time.time() - t0)))
        except KeyboardInterrupt:
            break
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.")
        raise SystemExit(0)
