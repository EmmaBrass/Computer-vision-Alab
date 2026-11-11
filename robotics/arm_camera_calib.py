#!/usr/bin/env python3
"""Save each camera's place in the robot frame from the arm stickers.

The cameras stay screwed down. You move the arm, pause, and press SPACE.
Each pause records the stickers that camera can see and where the robot
says those joints are. S writes one pose per camera into cell_frames.json.
Later starts of the viewer read that file. Run this again only if a camera
is physically moved.

    cd D:\\Jessica\\Jessica\\robotics
    save_arm_camera.cmd

SPACE stores this pause. S saves. C clears the stored pauses. Q quits.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cell_frames import (  # noqa: E402
    LAYOUT_ARM_TAGS,
    load_cell_frames,
    save_cell_frames,
)

OBS_PATH = ROOT / "arm_camera_obs.json"
INTR_PATH = ROOT / "arm_camera_intrinsics.json"
MAX_REPROJ_PX = 40.0
# Four is the floor: three points give several P3P solutions that all pass the
# reprojection check, so the saved pose would be ambiguous.
MIN_POINTS = 4
MIN_VOLUME_MM3 = 150.0 ** 3
MIN_SPAN_MM2 = 150.0 * 150.0
SAME_PAUSE_MM = 15.0

_CAMS = ("cam0", "cam1")


def _sub(a, b):
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def _triple(ab, ac, ad) -> float:
    return (
        ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
        - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
        + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
    )


def _dist3(a, b) -> float:
    return (
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    ) ** 0.5


def _cross_len(ab, ac) -> float:
    cx = ab[1] * ac[2] - ab[2] * ac[1]
    cy = ab[2] * ac[0] - ab[0] * ac[2]
    cz = ab[0] * ac[1] - ab[1] * ac[0]
    return (cx * cx + cy * cy + cz * cz) ** 0.5


def spread_ok(points_mm) -> bool:
    """True when the joints are not in one straight line, or not flat if there are four."""
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in points_mm]
    n = len(pts)
    if n < 3:
        return False
    if n == 3:
        return _cross_len(_sub(pts[1], pts[0]), _sub(pts[2], pts[0])) >= MIN_SPAN_MM2
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for m in range(k + 1, n):
                    vol = abs(_triple(_sub(pts[j], pts[i]), _sub(pts[k], pts[i]), _sub(pts[m], pts[i])))
                    if vol > best:
                        best = vol
    return best >= MIN_VOLUME_MM3


def _k_matrix(fx, fy, cx, cy) -> np.ndarray:
    return np.array(
        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _cell_from_cv(rvec, tvec) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV p_cam = R_cv p_base + t_cv  ->  cell p_base = R p_cam + t."""
    R_cv, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t_cv = np.asarray(tvec, dtype=np.float64).reshape(3)
    R = np.zeros((3, 3), dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            R[i, j] = float(R_cv[j, i])
        acc = 0.0
        for j in range(3):
            acc += float(R[i, j]) * float(t_cv[j])
        t[i] = -acc
    return R, t


def _det(R) -> float:
    return _triple(
        (float(R[0, 0]), float(R[1, 0]), float(R[2, 0])),
        (float(R[0, 1]), float(R[1, 1]), float(R[2, 1])),
        (float(R[0, 2]), float(R[1, 2]), float(R[2, 2])),
    )


def _mean_reproj(obj, img, rvec, tvec, k) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    err = 0.0
    n = int(obj.shape[0])
    flat = proj.reshape(n, 2)
    for i in range(n):
        du = float(flat[i, 0]) - float(img[i, 0])
        dv = float(flat[i, 1]) - float(img[i, 1])
        err += (du * du + dv * dv) ** 0.5
    return err / float(n)


def solve_camera(points_mm, pixels, fx, fy, cx, cy, *, max_px: float = MAX_REPROJ_PX):
    """Camera pose in the cell_frames convention, or (None, message).

    Returns ``(R, t_mm, mean_px)`` when the stickers land within ``max_px``.
    """
    if len(points_mm) < MIN_POINTS or len(points_mm) != len(pixels):
        return None, f"need at least {MIN_POINTS} stickers"
    if not spread_ok(points_mm):
        if len(points_mm) == 3:
            return None, "these three stickers sit in a straight line; not saved"
        return None, "these pauses are too alike; move the arm to a different height and reach"
    obj = np.asarray(points_mm, dtype=np.float32).reshape(-1, 3)
    img = np.asarray(pixels, dtype=np.float32).reshape(-1, 2)
    if not (np.isfinite(obj).all() and np.isfinite(img).all()):
        return None, "a sticker or joint position was not a real number"
    k = _k_matrix(fx, fy, cx, cy)
    n = int(obj.shape[0])
    flags = []
    if n >= 6:
        flags.append(cv2.SOLVEPNP_ITERATIVE)
    if n >= 4:
        flags.append(cv2.SOLVEPNP_EPNP)
    sqp = getattr(cv2, "SOLVEPNP_SQPNP", None)
    if sqp is not None:
        flags.append(sqp)
    best = None
    best_err = 1e18
    for flag in flags:
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=flag)
        except Exception:
            continue
        if not ok:
            continue
        err = _mean_reproj(obj, img, rvec, tvec, k)
        if err < best_err:
            best_err = err
            best = (rvec, tvec)
    if best is None:
        return None, "the camera pose did not solve"
    R, t = _cell_from_cv(best[0], best[1])
    if _det(R) < 0.5:
        return None, "the solved camera was a mirror; not saved"
    if best_err > float(max_px):
        return None, f"stickers land {best_err:.0f} px off (limit {max_px:.0f}); not saved"
    return (R, t, float(best_err)), "ok"


def _same_pause(prev, xyz) -> bool:
    return _dist3(prev, xyz) < SAME_PAUSE_MM


def add_pause(store: list, cam: str, tags: dict, pts_mm, q) -> list[str]:
    """Append sticker sightings. Skip a joint that has not moved since last time."""
    added = []
    for mid, hit in tags.items():
        if hit.get("coasting"):
            continue
        mid_i = int(mid)
        if mid_i < 0 or mid_i >= len(pts_mm):
            continue
        xyz = (float(pts_mm[mid_i, 0]), float(pts_mm[mid_i, 1]), float(pts_mm[mid_i, 2]))
        u, v = hit["center"]
        if any(
            row["cam"] == cam and int(row["id"]) == mid_i and _same_pause(row["xyz"], xyz)
            for row in store
        ):
            continue
        store.append(
            {
                "cam": cam,
                "id": mid_i,
                "name": str(hit.get("name") or mid_i),
                "u": float(u),
                "v": float(v),
                "xyz": xyz,
                "q": [float(v) for v in np.asarray(q, dtype=np.float64).reshape(6)],
            }
        )
        added.append(str(hit.get("name") or mid_i))
    return added


def rows_for(store: list, cam: str):
    rows = [row for row in store if row["cam"] == cam]
    pts = [row["xyz"] for row in rows]
    pix = [(row["u"], row["v"]) for row in rows]
    return pts, pix


def write_obs(store: list, path: Path = OBS_PATH) -> None:
    path.write_text(json.dumps({"pauses": store}, indent=2) + "\n", encoding="utf-8")


def write_intrinsics(info: dict, path: Path = INTR_PATH) -> None:
    payload = {cam: [float(v) for v in info[cam]] for cam in _CAMS if cam in info}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_obs(path: Path = OBS_PATH) -> list:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("pauses") or [])


def save_poses(store: list, intrinsics: dict, *, cell_path: Path | None = None) -> list[str]:
    """Solve each camera that has enough stickers. Leave a failed camera as it was."""
    cf = load_cell_frames(cell_path)
    lines = []
    wrote = False
    for cam in _CAMS:
        info = intrinsics.get(cam)
        if info is None:
            lines.append(f"{cam}: no camera intrinsics yet")
            continue
        fx, fy, cx, cy = info
        pts, pix = rows_for(store, cam)
        solved, message = solve_camera(pts, pix, fx, fy, cx, cy)
        if solved is None:
            lines.append(f"{cam}: {message}")
            continue
        R, t, err = solved
        pose = cf.pose(cam)
        pose.R = R
        pose.t_mm = t
        pose.ready = True
        wrote = True
        lines.append(f"{cam}: saved, stickers land {err:.1f} px from where they were seen ({len(pts)} stickers)")
    if wrote:
        cf.layout = LAYOUT_ARM_TAGS
        cf.bootstrap_from_blue_base = False
        if "cam0" in intrinsics:
            cf.fx_px = float(intrinsics["cam0"][0])
        save_cell_frames(cf)
        lines.append(f"wrote {cf.path}")
    else:
        lines.append("nothing written; the previous camera file was kept")
    return lines


def _draw_status(bgr, cam: str, tags: dict, n_stored: int, q_text: str):
    out = bgr.copy()
    for hit in tags.values():
        cu, cv = int(round(hit["center"][0])), int(round(hit["center"][1]))
        cv2.circle(out, (cu, cv), 6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, str(hit.get("name") or hit["id"]), (cu + 8, cv), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    names = ", ".join(str(hit.get("name") or hit["id"]) for hit in tags.values()) or "none"
    lines = [
        f"{cam}  seeing: {names}",
        f"stored stickers: {n_stored}    joint angles: {q_text}",
        "SPACE store this pause    S save    C clear    Q quit",
    ]
    y = 22
    for text in lines:
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return out


def live() -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image
    except Exception as exc:
        print("ROS is not available in this window.")
        print(r"cd D:\Jessica\Jessica\robotics; .\save_arm_camera.cmd")
        print(exc)
        return 1

    codec_dir = ROOT.parent / "ros2_ws" / "src" / "orbbec_dual_sync" / "orbbec_dual_sync"
    if str(codec_dir) not in sys.path:
        sys.path.insert(0, str(codec_dir))
    from aruco_marker import find_joint_tags
    from image_codec import imgmsg_to_numpy
    from ur_skeleton import fk_joint_positions_mm, load_joints

    class Grab(Node):
        def __init__(self):
            super().__init__("arm_camera_calib")
            self.lock = threading.Lock()
            self.frames: dict[str, np.ndarray] = {}
            self.info: dict[str, tuple] = {}
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=2,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            for cam, idx in (("cam0", 0), ("cam1", 1)):
                self.create_subscription(
                    Image,
                    f"/orbbec/cam{idx}/color/image_raw",
                    lambda msg, name=cam: self._color(name, msg),
                    qos,
                )
                self.create_subscription(
                    CameraInfo,
                    f"/orbbec/cam{idx}/color/camera_info",
                    lambda msg, name=cam: self._info(name, msg),
                    qos,
                )

        def _color(self, cam, msg):
            try:
                image = imgmsg_to_numpy(msg)
            except Exception:
                return
            with self.lock:
                self.frames[cam] = image

        def _info(self, cam, msg):
            k = list(msg.k)
            if len(k) < 6 or float(k[0]) <= 1.0:
                return
            with self.lock:
                self.info[cam] = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))

        def copy(self):
            with self.lock:
                return dict(self.frames), dict(self.info)

    rclpy.init()
    node = Grab()
    store = load_obs()
    print("Cameras stay where they are. Move the arm, pause, press SPACE. S saves. Q quits.")
    if store:
        print(f"loaded {len(store)} stickers already stored in {OBS_PATH.name}")
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.05)
            frames, info = node.copy()
            q, model, source = load_joints()
            q_text = source.split(":")[0]
            panels = []
            for cam in _CAMS:
                frame = frames.get(cam)
                if frame is None:
                    blank = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(blank, f"waiting for {cam}", (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    panels.append(blank)
                    continue
                tags = find_joint_tags(frame, cam="")
                n_stored = sum(1 for row in store if row["cam"] == cam)
                panels.append(_draw_status(frame, cam, tags, n_stored, q_text))
            view = panels[0] if len(panels) == 1 else np.hstack(
                [cv2.resize(p, (panels[0].shape[1], panels[0].shape[0])) if p.shape[:2] != panels[0].shape[:2] else p for p in panels]
            )
            cv2.imshow("arm camera save", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")):
                store.clear()
                write_obs(store)
                print("cleared stored pauses")
                continue
            if key == ord(" "):
                if source.startswith("photo") or source.startswith("mock"):
                    print("robot joint angles are not live yet; wait until the window says rtde-live")
                    continue
                pts = fk_joint_positions_mm(q, model=model or "ur5e")
                any_new = False
                for cam in _CAMS:
                    frame = frames.get(cam)
                    if frame is None:
                        print(f"{cam}: no picture yet")
                        continue
                    tags = find_joint_tags(frame, cam="")
                    added = add_pause(store, cam, tags, pts, q)
                    if added:
                        any_new = True
                        print(f"{cam}: stored {', '.join(added)}")
                    else:
                        print(f"{cam}: no new sticker (move the arm, or none are visible)")
                if any_new:
                    write_obs(store)
                    if info:
                        write_intrinsics(info)
                continue
            if key in (ord("s"), ord("S")):
                if not info:
                    print("camera intrinsics have not arrived yet")
                    continue
                write_intrinsics(info)
                for line in save_poses(store, info):
                    print(line)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("live",):
        return live()
    if args[0] == "solve":
        store = load_obs(Path(args[1]) if len(args) > 1 else OBS_PATH)
        intr_path = INTR_PATH
        if not intr_path.is_file():
            print(f"missing {intr_path.name}; press S once in the live window first")
            return 1
        intr = json.loads(intr_path.read_text(encoding="utf-8"))
        info = {cam: tuple(intr[cam]) for cam in _CAMS if cam in intr}
        for line in save_poses(store, info):
            print(line)
        return 0
    print("usage: arm_camera_calib.py live")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
