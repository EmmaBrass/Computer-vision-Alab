"""Stereo extrinsics status + board capture + calibrate.

Supports:
  - Asymmetric checkerboard (unequal inner corners, e.g. 9x6) — recommended board
  - Asymmetric circle grid (OpenCV CALIB_CB_ASYMMETRIC_GRID)
  - Symmetric chessboard / ChArUco (legacy)

``stereo_extrinsics.json`` must map cam0 -> cam1 with real R (3x3) and t_mm (3,)
before dual-cam 3D reproject / ``--3d-reproject-assist`` is trustworthy.

    python stereo_calib_stub.py status
    python stereo_calib_stub.py howto
    python stereo_calib_stub.py capture
    python stereo_calib_stub.py calibrate --asymmetric-chess --cols 9 --rows 6 --square-mm 25
    python stereo_calib_stub.py calibrate --asymmetric-circles --cols 4 --rows 11 --square-mm 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from cam_geometry import EXTRINSICS_PATH, is_extrinsics_placeholder

ROOT = Path(__file__).resolve().parent
CALIB_DIR = ROOT / "stereo_calib_pairs"
DEFAULT_SQUARE_MM = 25.0
# Asymmetric checkerboard = primary rigid calib target: UNEQUAL inner corners
# (e.g. 9x6, 7x8, 9x10) so up/down and left/right are unambiguous in OpenCV/ROS.
DEFAULT_CHESS_COLS = 9
DEFAULT_CHESS_ROWS = 6
# Asymmetric circle grid (OpenCV): (circles per row, number of rows). Common 4x11.
DEFAULT_ASYM_CIRCLE_COLS = 4
DEFAULT_ASYM_CIRCLE_ROWS = 11
DEFAULT_ASYM_CIRCLE_SPACING_MM = 20.0
# ChArUco: OpenCV DICT_4X4_50 board, 5x7 squares (common print). Override via flags.
DEFAULT_CHARUCO_SQUARES_X = 5
DEFAULT_CHARUCO_SQUARES_Y = 7
DEFAULT_CHARUCO_SQUARE_MM = 30.0
DEFAULT_CHARUCO_MARKER_MM = 22.0


def _list_orbbec_indexes(max_index: int = 6) -> list[int]:
    found: list[int] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_OBSENSOR)
        opened = cap.isOpened()
        cap.release()
        if opened:
            found.append(index)
        elif found:
            break
    return found


def _warm(cap: cv2.VideoCapture, tries: int = 40):
    for _ in range(tries):
        if not cap.grab():
            time.sleep(0.03)
            continue
        ok_c, color = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
        if ok_c and color is not None:
            return color
        time.sleep(0.03)
    return None


def _intrinsics_from_cap(cap: cv2.VideoCapture, width: int, height: int) -> list[float]:
    """Best-effort fx,fy,cx,cy from OpenCV props; FOV prior fallback."""
    fx = float(cap.get(cv2.CAP_PROP_FOCUS) or 0.0)
    # Orbbec OBSENSOR often exposes fx/fy via proprietary props; use FOV prior.
    # ~60° HFOV prior for Gemini-class RGB at this resolution.
    if fx <= 1.0:
        fx = 0.5 * width / np.tan(np.deg2rad(60.0) / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return [float(fx), float(fy), float(cx), float(cy)]


def cmd_status(_: argparse.Namespace) -> int:
    if not EXTRINSICS_PATH.exists():
        print(f"MISSING: {EXTRINSICS_PATH}")
        print("Run: python cam_geometry.py fit   # writes a placeholder template")
        print("Then: python stereo_calib_stub.py capture  &&  python stereo_calib_stub.py calibrate")
        return 1
    data = json.loads(EXTRINSICS_PATH.read_text(encoding="utf-8"))
    placeholder = is_extrinsics_placeholder()
    R = np.asarray(data["R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(data["t_mm"], dtype=np.float64).reshape(3)
    print(f"file: {EXTRINSICS_PATH}")
    print(f"from_camera={data.get('from_camera')} -> to_camera={data.get('to_camera')}")
    print(f"t_mm={t.tolist()}  ||t||={float(np.linalg.norm(t)):.1f} mm")
    print(f"R det={float(np.linalg.det(R)):.4f}  (should be ~1)")
    print(f"K0={data.get('K0')}  K1={data.get('K1')}")
    n_pairs = len(list(CALIB_DIR.glob("pair_*_cam0.jpg"))) if CALIB_DIR.exists() else 0
    print(f"captured pairs in {CALIB_DIR.name}/: {n_pairs}")
    if placeholder:
        print("STATUS: PLACEHOLDER - identity/near-identity R or notes say PLACEHOLDER.")
        print("Dual reproject / --3d-reproject-assist will NOT aim correctly yet.")
        print("Next: python stereo_calib_stub.py capture  (hold ChArUco/chessboard in both FOVs)")
        return 2
    print("STATUS: looks like a non-placeholder file - still validate with same-tick dual stills.")
    return 0


def cmd_howto(_: argparse.Namespace) -> int:
    print(
        f"""
Board stereo calib (hands-on) — asymmetric checkerboard is a first-class option

What you need:
  - Both Orbbecs mounted rigidly (do not move after calib)
  - A board BOTH cameras can see (park the arm if it blocks overlap)
  - Each camera on its own USB 3 port
  - Measure square / circle spacing in millimeters with a ruler

Board choices (pick one):
  A) Asymmetric CHECKERBOARD — primary rigid target (recommended)
       Strictly uneven rows x columns of INNER CORNERS, e.g. 9x6, 7x8, 9x10
       (never square like 8x8). That kills 180 deg flip ambiguity so OpenCV/ROS
       always know the board's up/down and left/right.
       calibrate --asymmetric-chess --cols {DEFAULT_CHESS_COLS} --rows {DEFAULT_CHESS_ROWS} --square-mm {DEFAULT_SQUARE_MM}

  B) Asymmetric CIRCLE GRID (OpenCV pattern)
       Staggered dots, common {DEFAULT_ASYM_CIRCLE_COLS}x{DEFAULT_ASYM_CIRCLE_ROWS}
       calibrate --asymmetric-circles --cols {DEFAULT_ASYM_CIRCLE_COLS} --rows {DEFAULT_ASYM_CIRCLE_ROWS} --square-mm {DEFAULT_ASYM_CIRCLE_SPACING_MM}

  C) ChArUco / classic chessboard
       calibrate --charuco ...   or   calibrate --cols C --rows R (no asymmetric flag)

Fast path:
  1. python stereo_calib_stub.py capture
       SPACE = save pair when BOTH see the board   Q = quit
       Aim for >=10–12 varied poses (tilt/distance/volume).
  2. python stereo_calib_stub.py calibrate --asymmetric-chess --cols 9 --rows 6 --square-mm 25
  3. python stereo_calib_stub.py status   # must NOT say PLACEHOLDER
  4. Optional cell-frame sync (also writes layout):
       python scene_extrinsics_calib.py board --asymmetric-chess --cols 9 --rows 6 --square-mm 25

When the arm blocks a board:
  python scene_extrinsics_calib.py solve --live
  (shared-scene ORB+depth + purple tape; no printed board)

Until calibrate succeeds, --3d-reproject-assist refuses placeholder R,t.
""".strip()
    )
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    """Interactive dual-RGB pair grab for stereoCalibrate."""
    found = _list_orbbec_indexes()
    if len(found) < 2:
        print(f"Need two Orbbecs; detected: {found if found else 'none'}")
        print("Each Orbbec on its own USB 3 port. Close Orbbec Viewer, then retry.")
        return 1
    cam0_i, cam1_i = int(args.camera0), int(args.camera1)
    if cam0_i not in found or cam1_i not in found:
        print(f"Requested indexes {cam0_i},{cam1_i} not in detected {found}")
        return 1
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    cap0 = cv2.VideoCapture(cam0_i, cv2.CAP_OBSENSOR)
    cap1 = cv2.VideoCapture(cam1_i, cv2.CAP_OBSENSOR)
    if not cap0.isOpened() or not cap1.isOpened():
        print("Failed to open both cameras.")
        return 1
    print(
        f"Stereo capture: cam{cam0_i} | cam{cam1_i}. "
        "Hold board in BOTH FOVs. SPACE=save pair  Q=quit."
    )
    print(f"Saving under {CALIB_DIR}")
    saved = len(list(CALIB_DIR.glob("pair_*_cam0.jpg")))
    win = "Stereo calib capture (cam0 | cam1)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    K0 = K1 = None
    try:
        while True:
            c0 = _warm(cap0, tries=8)
            c1 = _warm(cap1, tries=8)
            if c0 is None or c1 is None:
                view = np.zeros((480, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    view,
                    "Waiting for frames... check USB3",
                    (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 200, 200),
                    2,
                )
                cv2.imshow(win, view)
                if cv2.waitKey(30) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue
            if K0 is None:
                h0, w0 = c0.shape[:2]
                h1, w1 = c1.shape[:2]
                K0 = _intrinsics_from_cap(cap0, w0, h0)
                K1 = _intrinsics_from_cap(cap1, w1, h1)
            # Side-by-side preview (scale cam1 to cam0 height if needed).
            h0, w0 = c0.shape[:2]
            h1, w1 = c1.shape[:2]
            if h1 != h0:
                scale = h0 / float(h1)
                c1s = cv2.resize(c1, (int(w1 * scale), h0))
            else:
                c1s = c1
            view = np.hstack([c0, c1s])
            cv2.putText(
                view,
                f"pairs={saved}  SPACE=save  Q=quit  board in BOTH views",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(win, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                saved += 1
                stem = f"pair_{saved:03d}"
                p0 = CALIB_DIR / f"{stem}_cam0.jpg"
                p1 = CALIB_DIR / f"{stem}_cam1.jpg"
                cv2.imwrite(str(p0), c0)
                cv2.imwrite(str(p1), c1)
                meta = {
                    "pair": saved,
                    "camera0": cam0_i,
                    "camera1": cam1_i,
                    "K0_prior": K0,
                    "K1_prior": K1,
                    "timestamp": time.time(),
                }
                (CALIB_DIR / f"{stem}.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8"
                )
                print(f"Saved {stem} ({p0.name}, {p1.name})")
    finally:
        cap0.release()
        cap1.release()
        cv2.destroyWindow(win)
    print(f"Done. {saved} pair(s) on disk. Next: python stereo_calib_stub.py calibrate")
    if saved < 8:
        print("Tip: capture at least ~8–12 varied poses before calibrate.")
        return 2
    return 0


def _chessboard_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    obj = np.zeros((cols * rows, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_mm)
    return obj


def _asymmetric_circle_object_points(
    cols: int, rows: int, spacing_mm: float
) -> np.ndarray:
    """World points for OpenCV CALIB_CB_ASYMMETRIC_GRID (staggered rows)."""
    obj = np.zeros((rows * cols, 3), np.float32)
    spacing = float(spacing_mm)
    n = 0
    for r in range(rows):
        for c in range(cols):
            obj[n, 0] = (2 * c + (r % 2)) * spacing
            obj[n, 1] = r * spacing
            n += 1
    return obj


def _detect_chessboard(gray, cols: int, rows: int):
    pattern = (int(cols), int(rows))
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok and hasattr(cv2, "findChessboardCornersSB"):
        try:
            ok, corners = cv2.findChessboardCornersSB(
                gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            )
        except cv2.error:
            ok = False
    if not ok:
        return None
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
    return corners


def _detect_asymmetric_circles(gray, cols: int, rows: int):
    pattern = (int(cols), int(rows))
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID
    # Blur slightly — helps printed dots under Orbbec noise.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ok, centers = cv2.findCirclesGrid(blur, pattern, flags=flags)
    if not ok:
        ok, centers = cv2.findCirclesGrid(gray, pattern, flags=flags)
    if not ok:
        return None
    return centers.astype(np.float32)


def _detect_charuco(gray, board, detector):
    corners, ids, _rej = detector.detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None, None
    return corners, ids


def _board_mode(args: argparse.Namespace) -> str:
    if bool(getattr(args, "charuco", False)):
        return "charuco"
    if bool(getattr(args, "asymmetric_circles", False)):
        return "asymmetric_circles"
    if bool(getattr(args, "asymmetric_chess", False)):
        return "asymmetric_chess"
    # Default: treat unequal cols/rows as asymmetric chess; equal as classic.
    if int(args.cols) != int(args.rows):
        return "asymmetric_chess"
    return "chessboard"


def cmd_calibrate(args: argparse.Namespace) -> int:
    pairs = sorted(CALIB_DIR.glob("pair_*_cam0.jpg")) if CALIB_DIR.exists() else []
    if len(pairs) < 4:
        print(f"Need more pairs under {CALIB_DIR} (found {len(pairs)}). Run capture first.")
        return 1
    mode = _board_mode(args)
    use_charuco = mode == "charuco"
    use_asym_circles = mode == "asymmetric_circles"
    obj_points: list[np.ndarray] = []
    img_points0: list[np.ndarray] = []
    img_points1: list[np.ndarray] = []
    image_size = None
    K0_prior = K1_prior = None
    cols, rows = int(args.cols), int(args.rows)
    square_mm = float(args.square_mm)
    board = detector = None
    objp = None

    if use_charuco:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard(
            (int(args.squares_x), int(args.squares_y)),
            float(args.square_mm),
            float(args.marker_mm),
            dictionary,
        )
        detector = cv2.aruco.CharucoDetector(board)
        print(
            f"ChArUco calibrate: squares={args.squares_x}x{args.squares_y} "
            f"square={args.square_mm}mm marker={args.marker_mm}mm"
        )
    elif use_asym_circles:
        if cols < 2 or rows < 2:
            print("Asymmetric circles need --cols and --rows >= 2")
            return 2
        objp = _asymmetric_circle_object_points(cols, rows, square_mm)
        print(
            f"Asymmetric CIRCLE grid: {cols}x{rows} circles, spacing={square_mm}mm"
        )
    else:
        if mode == "asymmetric_chess" and cols == rows:
            print(
                f"WARNING: asymmetric checkerboard usually has cols != rows "
                f"(got {cols}x{rows}). Orientation may be ambiguous."
            )
        objp = _chessboard_object_points(cols, rows, square_mm)
        label = "Asymmetric CHECKERBOARD" if mode == "asymmetric_chess" else "Chessboard"
        print(f"{label}: inner corners {cols}x{rows}, square={square_mm}mm")

    used = 0
    for p0 in pairs:
        stem = p0.name.replace("_cam0.jpg", "")
        p1 = CALIB_DIR / f"{stem}_cam1.jpg"
        meta_path = CALIB_DIR / f"{stem}.json"
        if not p1.exists():
            continue
        im0 = cv2.imread(str(p0))
        im1 = cv2.imread(str(p1))
        if im0 is None or im1 is None:
            continue
        if image_size is None:
            image_size = (im0.shape[1], im0.shape[0])
        if meta_path.exists() and K0_prior is None:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                K0_prior = meta.get("K0_prior")
                K1_prior = meta.get("K1_prior")
            except (OSError, json.JSONDecodeError):
                pass
        g0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        if use_charuco:
            c0, ids0 = _detect_charuco(g0, board, detector)
            c1, ids1 = _detect_charuco(g1, board, detector)
            if c0 is None or c1 is None:
                print(f"  skip {stem}: ChArUco not found in both views")
                continue
            # Intersect shared corner IDs
            ids0 = ids0.flatten()
            ids1 = ids1.flatten()
            shared = sorted(set(ids0.tolist()) & set(ids1.tolist()))
            if len(shared) < 6:
                print(f"  skip {stem}: only {len(shared)} shared ChArUco corners")
                continue
            map0 = {int(i): c0[n].reshape(1, 2) for n, i in enumerate(ids0)}
            map1 = {int(i): c1[n].reshape(1, 2) for n, i in enumerate(ids1)}
            pts0 = np.array([map0[i] for i in shared], dtype=np.float32)
            pts1 = np.array([map1[i] for i in shared], dtype=np.float32)
            # Object points from board chessboard corners for those ids
            chess_corners = board.getChessboardCorners()
            obj = np.array([chess_corners[i] for i in shared], dtype=np.float32)
            obj_points.append(obj)
            img_points0.append(pts0)
            img_points1.append(pts1)
        elif use_asym_circles:
            c0 = _detect_asymmetric_circles(g0, cols, rows)
            c1 = _detect_asymmetric_circles(g1, cols, rows)
            if c0 is None or c1 is None:
                print(f"  skip {stem}: asymmetric circles not found in both views")
                continue
            obj_points.append(objp.copy())
            img_points0.append(c0)
            img_points1.append(c1)
        else:
            c0 = _detect_chessboard(g0, cols, rows)
            c1 = _detect_chessboard(g1, cols, rows)
            if c0 is None or c1 is None:
                print(f"  skip {stem}: checkerboard not found in both views")
                continue
            obj_points.append(objp.copy())
            img_points0.append(c0)
            img_points1.append(c1)
        used += 1
        print(f"  ok {stem}")

    if used < 4:
        print(f"Only {used} usable pairs — capture more with the board visible in BOTH cams.")
        return 1

    assert image_size is not None
    w, h = image_size
    if K0_prior is None:
        K0_prior = [0.5 * w / np.tan(np.deg2rad(60.0) / 2.0), 0.5 * w / np.tan(np.deg2rad(60.0) / 2.0), w / 2.0, h / 2.0]
    if K1_prior is None:
        K1_prior = list(K0_prior)

    camera_matrix0 = np.array(
        [[K0_prior[0], 0, K0_prior[2]], [0, K0_prior[1], K0_prior[3]], [0, 0, 1]],
        dtype=np.float64,
    )
    camera_matrix1 = np.array(
        [[K1_prior[0], 0, K1_prior[2]], [0, K1_prior[1], K1_prior[3]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist0 = np.zeros(5)
    dist1 = np.zeros(5)

    # Per-camera calibrate when we have enough points (improves K before stereo).
    try:
        _, camera_matrix0, dist0, _, _ = cv2.calibrateCamera(
            obj_points, img_points0, image_size, camera_matrix0, dist0
        )
        _, camera_matrix1, dist1, _, _ = cv2.calibrateCamera(
            obj_points, img_points1, image_size, camera_matrix1, dist1
        )
    except cv2.error as exc:
        print(f"Per-cam calibrate warning (continuing with priors): {exc}")

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    try:
        rms, camera_matrix0, dist0, camera_matrix1, dist1, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points0,
            img_points1,
            camera_matrix0,
            dist0,
            camera_matrix1,
            dist1,
            image_size,
            criteria=criteria,
            flags=flags,
        )
    except cv2.error as exc:
        print(f"stereoCalibrate failed: {exc}")
        return 1

    t_mm = (np.asarray(T, dtype=np.float64).reshape(3) * 1.0).tolist()
    # OpenCV T is in the same units as object points (mm here).
    payload = {
        "from_camera": 0,
        "to_camera": 1,
        "R": np.asarray(R, dtype=np.float64).reshape(3, 3).tolist(),
        "t_mm": t_mm,
        "K0": [
            float(camera_matrix0[0, 0]),
            float(camera_matrix0[1, 1]),
            float(camera_matrix0[0, 2]),
            float(camera_matrix0[1, 2]),
        ],
        "K1": [
            float(camera_matrix1[0, 0]),
            float(camera_matrix1[1, 1]),
            float(camera_matrix1[0, 2]),
            float(camera_matrix1[1, 2]),
        ],
        "rms": float(rms),
        "n_pairs": used,
        "board": f"{mode}_{args.cols}x{args.rows}",
        "notes": (
            f"stereoCalibrate mode={mode} {time.strftime('%Y-%m-%d')} "
            f"rms={rms:.4f} n={used}"
        ),
    }
    EXTRINSICS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {EXTRINSICS_PATH}")
    print(f"RMS={rms:.4f}  ||t||={float(np.linalg.norm(T)):.1f} mm  pairs={used} mode={mode}")
    # Mirror into cell_frames when robotics helper is available.
    try:
        from cell_frames import (
            LAYOUT_SCENE_TAPE,
            apply_scene_stereo_to_cell_frames,
            load_cell_frames,
            save_cell_frames,
        )

        cf = load_cell_frames()
        apply_scene_stereo_to_cell_frames(
            cf,
            np.asarray(R, dtype=np.float64).reshape(3, 3),
            np.asarray(T, dtype=np.float64).reshape(3),
        )
        # Keep a distinct layout name for board-based solves.
        cf.layout = "asymmetric_board" if "asymmetric" in mode else LAYOUT_SCENE_TAPE
        save_cell_frames(cf)
        print(f"Updated {cf.path} layout={cf.layout}")
    except Exception as exc:
        print(f"cell_frames update skipped: {exc}")
    if is_extrinsics_placeholder():
        print("WARNING: still classified as PLACEHOLDER - check R / notes.")
        return 2
    print("STATUS: real extrinsics ready.")
    print("Also: python scene_extrinsics_calib.py status")
    print("Live: python depthcamera.py --dual-independent --3d-reproject-assist")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stereo extrinsics status / capture / calibrate")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_st = sub.add_parser("status", help="Check stereo_extrinsics.json placeholder vs real")
    p_st.set_defaults(func=cmd_status)

    p_ht = sub.add_parser("howto", help="Print ChArUco capture / write steps")
    p_ht.set_defaults(func=cmd_howto)

    p_cap = sub.add_parser("capture", help="Interactive dual-cam pair grab (SPACE=save)")
    p_cap.add_argument("--camera0", type=int, default=0)
    p_cap.add_argument("--camera1", type=int, default=1)
    p_cap.set_defaults(func=cmd_capture)

    p_cal = sub.add_parser("calibrate", help="Run stereoCalibrate on captured pairs")
    p_cal.add_argument(
        "--cols",
        type=int,
        default=DEFAULT_CHESS_COLS,
        help="Inner corners X (chess) or circles per row (asymmetric circles)",
    )
    p_cal.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_CHESS_ROWS,
        help="Inner corners Y (chess) or circle rows (asymmetric circles)",
    )
    p_cal.add_argument(
        "--square-mm",
        type=float,
        default=DEFAULT_SQUARE_MM,
        help="Square size or circle spacing in mm",
    )
    board = p_cal.add_mutually_exclusive_group()
    board.add_argument(
        "--asymmetric-chess",
        action="store_true",
        help="Asymmetric checkerboard (prefer cols != rows, e.g. 9x6)",
    )
    board.add_argument(
        "--asymmetric-circles",
        action="store_true",
        help="Asymmetric circle grid (OpenCV CALIB_CB_ASYMMETRIC_GRID)",
    )
    board.add_argument("--charuco", action="store_true", help="Use ChArUco board")
    p_cal.add_argument("--squares-x", type=int, default=DEFAULT_CHARUCO_SQUARES_X)
    p_cal.add_argument("--squares-y", type=int, default=DEFAULT_CHARUCO_SQUARES_Y)
    p_cal.add_argument("--marker-mm", type=float, default=DEFAULT_CHARUCO_MARKER_MM)
    p_cal.set_defaults(func=cmd_calibrate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
