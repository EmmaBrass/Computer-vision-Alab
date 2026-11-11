"""CLI for ArUco-free scene + purple-tape dual-camera extrinsics.

    cd E:\\Desktop\\Jessica\\robotics
    python scene_extrinsics_calib.py status
    python scene_extrinsics_calib.py howto
    python scene_extrinsics_calib.py solve --live
    python scene_extrinsics_calib.py solve --live --frames 5
    python scene_extrinsics_calib.py solve ^
        --color0 pair_color0.jpg --depth0 pair_depth0.npy ^
        --color1 pair_color1.jpg --depth1 pair_depth1.npy
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from cell_frames import CELL_FRAMES_PATH, load_cell_frames, relative_cam1_from_cam0
from scene_extrinsics import (
    apply_result_to_cell_frames,
    draw_debug,
    find_purple_tape,
    solve_scene_extrinsics,
    write_stereo_extrinsics,
)
from cam_geometry import (
    EXTRINSICS_PATH,
    default_intrinsics,
    is_extrinsics_placeholder,
    load_extrinsics,
)

ROOT = Path(__file__).resolve().parent
DEBUG_DIR = ROOT / "scene_calib_debug"


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


def _read_rgbd(cap: cv2.VideoCapture, tries: int = 50):
    color = depth = None
    for _ in range(tries):
        if not cap.grab():
            time.sleep(0.02)
            continue
        ok_c, c = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
        ok_d, d = cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
        if ok_c and c is not None:
            color = c
        if ok_d and d is not None:
            depth = d
        if color is not None and depth is not None:
            return color, depth
        time.sleep(0.02)
    return color, depth


def _load_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(str(path))
    # 16-bit PNG mm
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    return d


def cmd_howto(_: argparse.Namespace) -> int:
    print(
        """
Dual-camera extrinsics — three paths (same output files)

---------------------------------------------------------------------------
PATH 1 — Asymmetric CHECKERBOARD (primary rigid calib target)
---------------------------------------------------------------------------
What it is: a flat, rigid printed board with an UNEVEN number of rows and
columns of inner corners (e.g. 7x8, 9x6, 9x10 — never NxN). That design is
standard in robotics / machine vision so OpenCV/ROS can always recover the
board's exact up/down and left/right (no 180 deg flip ambiguity).

Best when you can hold the board in the dual-camera overlap (park the arm).

  # Capture synced pairs (board visible in BOTH cams)
  python stereo_calib_stub.py capture

  # Match --cols/--rows/--square-mm to YOUR board (inner corners + square mm)
  python scene_extrinsics_calib.py board --asymmetric-chess --cols 9 --rows 6 --square-mm 25

  # Related: asymmetric circle grid (staggered dots) if that is your print
  python scene_extrinsics_calib.py board --asymmetric-circles --cols 4 --rows 11 --square-mm 20

---------------------------------------------------------------------------
PATH 2 — Scene + purple tape (NO board)
---------------------------------------------------------------------------
Best when the arm blocks a board. Uses shared scene ORB+depth + tape origin/axis.

  python scene_extrinsics_calib.py solve --live
       SPACE = buffer dual RGB-D   S = solve+write   Q = quit

---------------------------------------------------------------------------
PATH 3 — Classic ChArUco (optional)
---------------------------------------------------------------------------
  python stereo_calib_stub.py capture
  python stereo_calib_stub.py calibrate --charuco

Writes (all paths):
  - robotics/stereo_extrinsics.json
  - robotics/cell_frames.json   (layout=asymmetric_board or scene_tape)

Validate:
  python scene_extrinsics_calib.py status
""".strip()
    )
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    print(f"stereo_extrinsics: {EXTRINSICS_PATH}")
    if EXTRINSICS_PATH.is_file():
        ext = load_extrinsics()
        ph = is_extrinsics_placeholder()
        if ext is not None:
            print(
                f"  R det={float(np.linalg.det(ext.R)):.4f}  "
                f"|t|={float(np.linalg.norm(ext.t_mm)):.1f} mm  "
                f"placeholder={ph}"
            )
            notes = json.loads(EXTRINSICS_PATH.read_text(encoding="utf-8")).get(
                "notes", ""
            )
            print(f"  notes: {notes[:160]}")
    else:
        print("  MISSING")

    print(f"cell_frames: {CELL_FRAMES_PATH}")
    cf = load_cell_frames()
    print(f"  layout={cf.layout} cam0.ready={cf.cam0.ready} cam1.ready={cf.cam1.ready}")
    if cf.cam0.ready and cf.cam1.ready:
        R, t = relative_cam1_from_cam0(cf)
        print(f"  relative |t|={float(np.linalg.norm(t)):.1f} mm  det(R)={float(np.linalg.det(R)):.4f}")
    return 0 if (EXTRINSICS_PATH.is_file() and not is_extrinsics_placeholder()) else 2


def _solve_pair(
    color0,
    color1,
    depth0,
    depth1,
    *,
    no_tape: bool,
    no_arm_mask: bool,
) -> object:
    return solve_scene_extrinsics(
        color0,
        color1,
        depth0,
        depth1,
        use_tape=not no_tape,
        mask_arm=not no_arm_mask,
    )


def cmd_solve(args: argparse.Namespace) -> int:
    if args.live:
        return _solve_live(args)

    required = (args.color0, args.depth0, args.color1, args.depth1)
    if not all(required):
        print("Provide --live OR all of --color0 --depth0 --color1 --depth1")
        return 2
    color0 = cv2.imread(str(args.color0), cv2.IMREAD_COLOR)
    color1 = cv2.imread(str(args.color1), cv2.IMREAD_COLOR)
    if color0 is None or color1 is None:
        print("Failed to read color images")
        return 1
    depth0 = _load_depth(Path(args.depth0))
    depth1 = _load_depth(Path(args.depth1))
    # Resize depth to color if needed.
    if depth0.shape[:2] != color0.shape[:2]:
        depth0 = cv2.resize(
            depth0, (color0.shape[1], color0.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    if depth1.shape[:2] != color1.shape[:2]:
        depth1 = cv2.resize(
            depth1, (color1.shape[1], color1.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    try:
        result = _solve_pair(
            color0, color1, depth0, depth1, no_tape=args.no_tape, no_arm_mask=args.no_arm_mask
        )
    except RuntimeError as exc:
        print(f"SOLVE FAILED: {exc}")
        return 1

    K0 = default_intrinsics(color0.shape[1], color0.shape[0])
    K1 = default_intrinsics(color1.shape[1], color1.shape[0])
    out_stereo = write_stereo_extrinsics(result, K0=K0, K1=K1, path=Path(args.out_stereo))
    cf = apply_result_to_cell_frames(result, path=Path(args.out_cell))
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    dbg = draw_debug(color0, color1, result)
    dbg_path = DEBUG_DIR / "last_solve.jpg"
    cv2.imwrite(str(dbg_path), dbg)
    print(result.notes)
    print(f"Wrote {out_stereo}")
    print(f"Wrote {cf.path} layout={cf.layout}")
    print(f"Debug {dbg_path}")
    if result.tape0 is None or result.tape1 is None:
        print("WARNING: purple tape missing in one/both views — origin bootstrap weaker.")
    return 0


def _solve_live(args: argparse.Namespace) -> int:
    found = _list_orbbec_indexes()
    if len(found) < 2:
        print(f"Need two Orbbecs; detected: {found if found else 'none'}")
        return 1
    i0, i1 = int(args.camera0), int(args.camera1)
    if i0 not in found or i1 not in found:
        print(f"Requested {i0},{i1} not in {found}")
        return 1
    cap0 = cv2.VideoCapture(i0, cv2.CAP_OBSENSOR)
    cap1 = cv2.VideoCapture(i1, cv2.CAP_OBSENSOR)
    if not cap0.isOpened() or not cap1.isOpened():
        print("Failed to open cameras")
        return 1

    pairs: list[tuple] = []
    print("SPACE=accumulate frame  S=solve+write  Q=quit")
    try:
        while True:
            c0, d0 = _read_rgbd(cap0)
            c1, d1 = _read_rgbd(cap1)
            if c0 is None or c1 is None or d0 is None or d1 is None:
                preview = np.zeros((240, 640, 3), np.uint8)
                cv2.putText(
                    preview,
                    "Waiting for RGB-D...",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
            else:
                t0 = find_purple_tape(c0, depth_mm=d0, K=default_intrinsics(c0.shape[1], c0.shape[0]))
                t1 = find_purple_tape(c1, depth_mm=d1, K=default_intrinsics(c1.shape[1], c1.shape[0]))
                # Lightweight preview without full solve.
                from scene_extrinsics import SceneExtrinsicsResult

                stub = SceneExtrinsicsResult(
                    R=np.eye(3),
                    t_mm=np.zeros(3),
                    n_matches=0,
                    n_inliers=0,
                    inlier_rmse_mm=0.0,
                    tape0=t0,
                    tape1=t1,
                    notes=f"buffered={len(pairs)}",
                )
                preview = draw_debug(c0, c1, stub)
                cv2.putText(
                    preview,
                    f"SPACE add  S solve  Q quit | buf={len(pairs)}",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
            cv2.imshow("scene_extrinsics_calib", preview)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return 0
            if key == ord(" ") and c0 is not None and c1 is not None:
                pairs.append((c0.copy(), c1.copy(), d0.copy(), d1.copy()))
                print(f"Buffered frame {len(pairs)}")
                if args.frames > 0 and len(pairs) >= int(args.frames):
                    key = ord("s")
            if key in (ord("s"), ord("S")):
                if not pairs:
                    print("No buffered frames — press SPACE first")
                    continue
                # Concatenate matches by solving on the last frame but also
                # average R via Umeyama on stacked points from all frames.
                from scene_extrinsics import match_scene_points, ransac_umeyama
                from scene_extrinsics import (
                    TAPE_AXIS_WEIGHT,
                    TAPE_CENTER_WEIGHT,
                    _append_weighted,
                    align_yaw_with_tape_axis,
                    SceneExtrinsicsResult,
                )

                src_all: list = []
                dst_all: list = []
                tape0 = tape1 = None
                used_axis = False
                for c0b, c1b, d0b, d1b in pairs:
                    K0 = default_intrinsics(c0b.shape[1], c0b.shape[0])
                    K1 = default_intrinsics(c1b.shape[1], c1b.shape[0])
                    s, d = match_scene_points(
                        c0b,
                        c1b,
                        d0b,
                        d1b,
                        K0,
                        K1,
                        mask_arm=not args.no_arm_mask,
                    )
                    src_all.extend(list(s))
                    dst_all.extend(list(d))
                    if not args.no_tape:
                        t0 = find_purple_tape(c0b, depth_mm=d0b, K=K0)
                        t1 = find_purple_tape(c1b, depth_mm=d1b, K=K1)
                        if (
                            t0 is not None
                            and t1 is not None
                            and t0.p_cam_mm is not None
                            and t1.p_cam_mm is not None
                        ):
                            tape0, tape1 = t0, t1
                            _append_weighted(
                                src_all,
                                dst_all,
                                t0.p_cam_mm,
                                t1.p_cam_mm,
                                TAPE_CENTER_WEIGHT,
                            )
                            if t0.axis_cam is not None and t1.axis_cam is not None:
                                for sign in (-1.0, 1.0):
                                    p0 = t0.p_cam_mm + sign * 40.0 * t0.axis_cam
                                    p1 = t1.p_cam_mm + sign * 40.0 * t1.axis_cam
                                    _append_weighted(
                                        src_all,
                                        dst_all,
                                        p0,
                                        p1,
                                        TAPE_AXIS_WEIGHT // 2,
                                    )
                                used_axis = True
                if len(src_all) < 3:
                    print("Not enough correspondences across buffered frames")
                    continue
                src_a = np.asarray(src_all, dtype=np.float64)
                dst_a = np.asarray(dst_all, dtype=np.float64)
                R, t, inl, rmse = ransac_umeyama(src_a, dst_a)
                if (
                    used_axis
                    and tape0 is not None
                    and tape1 is not None
                    and tape0.axis_cam is not None
                    and tape1.axis_cam is not None
                ):
                    R = align_yaw_with_tape_axis(R, tape0.axis_cam, tape1.axis_cam)
                    if inl.any():
                        t = dst_a[inl].mean(axis=0) - (R @ src_a[inl].mean(axis=0))
                result = SceneExtrinsicsResult(
                    R=R,
                    t_mm=t,
                    n_matches=len(src_all),
                    n_inliers=int(inl.sum()),
                    inlier_rmse_mm=rmse,
                    tape0=tape0,
                    tape1=tape1,
                    used_tape_axis=used_axis,
                    notes=(
                        f"live multi-frame scene+tape; frames={len(pairs)} "
                        f"inliers={int(inl.sum())} rmse={rmse:.1f}mm"
                    ),
                )
                c0b, c1b, _, _ = pairs[-1]
                K0 = default_intrinsics(c0b.shape[1], c0b.shape[0])
                K1 = default_intrinsics(c1b.shape[1], c1b.shape[0])
                out_stereo = write_stereo_extrinsics(
                    result, K0=K0, K1=K1, path=Path(args.out_stereo)
                )
                cf = apply_result_to_cell_frames(result, path=Path(args.out_cell))
                DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                dbg = draw_debug(c0b, c1b, result)
                dbg_path = DEBUG_DIR / "last_solve.jpg"
                cv2.imwrite(str(dbg_path), dbg)
                print(result.notes)
                print(f"Wrote {out_stereo}")
                print(f"Wrote {cf.path} layout={cf.layout}")
                print(f"Debug {dbg_path}")
                cv2.imshow("scene_extrinsics_calib", dbg)
                cv2.waitKey(500)
                return 0
    finally:
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()


def cmd_board(args: argparse.Namespace) -> int:
    """Run asymmetric (or classic) board stereoCalibrate via stereo_calib_stub."""
    import stereo_calib_stub as stub

    # Build a Namespace compatible with stereo_calib_stub.cmd_calibrate
    cal = argparse.Namespace(
        cols=int(args.cols),
        rows=int(args.rows),
        square_mm=float(args.square_mm),
        asymmetric_chess=bool(args.asymmetric_chess),
        asymmetric_circles=bool(args.asymmetric_circles),
        charuco=bool(args.charuco),
        squares_x=int(args.squares_x),
        squares_y=int(args.squares_y),
        marker_mm=float(args.marker_mm),
    )
    if not (cal.asymmetric_chess or cal.asymmetric_circles or cal.charuco):
        # Default this entrypoint to asymmetric checkerboard.
        cal.asymmetric_chess = True
        print("Defaulting to --asymmetric-chess (unequal checkerboard).")
    print(
        "Tip: capture pairs first with: python stereo_calib_stub.py capture\n"
        "Then this command runs calibrate + updates cell_frames."
    )
    return int(stub.cmd_calibrate(cal))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dual extrinsics: scene+tape OR asymmetric board"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_h = sub.add_parser("howto", help="Print setup instructions")
    p_h.set_defaults(func=cmd_howto)

    p_s = sub.add_parser("status", help="Show stereo_extrinsics + cell_frames status")
    p_s.set_defaults(func=cmd_status)

    p_b = sub.add_parser(
        "board",
        help="Calibrate from asymmetric checkerboard / circle grid / ChArUco pairs",
    )
    p_b.add_argument("--cols", type=int, default=9, help="Inner corners X or circles/row")
    p_b.add_argument("--rows", type=int, default=6, help="Inner corners Y or circle rows")
    p_b.add_argument(
        "--square-mm",
        type=float,
        default=25.0,
        help="Square size or circle spacing (mm) — measure your board",
    )
    bg = p_b.add_mutually_exclusive_group()
    bg.add_argument(
        "--asymmetric-chess",
        action="store_true",
        help="Asymmetric checkerboard (default if no board flag)",
    )
    bg.add_argument(
        "--asymmetric-circles",
        action="store_true",
        help="Asymmetric circle grid (OpenCV staggered dots)",
    )
    bg.add_argument("--charuco", action="store_true", help="ChArUco board")
    p_b.add_argument("--squares-x", type=int, default=5)
    p_b.add_argument("--squares-y", type=int, default=7)
    p_b.add_argument("--marker-mm", type=float, default=22.0)
    p_b.set_defaults(func=cmd_board)

    p_sol = sub.add_parser("solve", help="Solve R,t from live dual or image files")
    p_sol.add_argument("--live", action="store_true", help="Open both Orbbecs")
    p_sol.add_argument("--camera0", type=int, default=0)
    p_sol.add_argument("--camera1", type=int, default=1)
    p_sol.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Auto-solve after N SPACE grabs (0=manual S key)",
    )
    p_sol.add_argument("--color0", type=Path, default=None)
    p_sol.add_argument("--depth0", type=Path, default=None)
    p_sol.add_argument("--color1", type=Path, default=None)
    p_sol.add_argument("--depth1", type=Path, default=None)
    p_sol.add_argument("--no-tape", action="store_true", help="Disable purple-tape prior")
    p_sol.add_argument(
        "--no-arm-mask", action="store_true", help="Do not mask blue arm from ORB"
    )
    p_sol.add_argument(
        "--out-stereo", type=Path, default=EXTRINSICS_PATH, help="stereo_extrinsics.json"
    )
    p_sol.add_argument(
        "--out-cell", type=Path, default=CELL_FRAMES_PATH, help="cell_frames.json"
    )
    p_sol.set_defaults(func=cmd_solve)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
