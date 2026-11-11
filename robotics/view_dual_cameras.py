"""Direct USB dual Orbbec preview (cam0 | cam1 side-by-side).

Uses CAP_OBSENSOR — conflicts with ROS orbbec_camera_publisher.
Stop ROS first (launch_ros_dual_sync / dual sync). For live ROS topics use:
  ros2_ws\\view_dual_cameras.cmd

Press Q / Esc to quit.
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np


def open_orbbec(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(index), cv2.CAP_OBSENSOR)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open Orbbec index {index} (CAP_OBSENSOR)")
    return cap


def grab_color(cap: cv2.VideoCapture) -> np.ndarray | None:
    if not cap.grab():
        return None
    ok, color = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
    if ok and color is not None:
        return color
    return None


def warmup(cap: cv2.VideoCapture, grabs: int = 30) -> np.ndarray | None:
    last = None
    for _ in range(int(grabs)):
        frame = grab_color(cap)
        if frame is not None:
            last = frame
        time.sleep(0.02)
    return last


def _label(bgr: np.ndarray, text: str) -> np.ndarray:
    out = bgr.copy()
    cv2.putText(
        out,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _placeholder(text: str, h: int = 480, w: int = 640) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (24, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def compose(left: np.ndarray | None, right: np.ndarray | None, max_height: int) -> np.ndarray:
    a = _label(left, "cam0") if left is not None else _placeholder("no cam0")
    b = _label(right, "cam1") if right is not None else _placeholder("no cam1")
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb, max_height) if max_height > 0 else min(ha, hb)
    a_s = cv2.resize(a, (max(1, int(wa * (h / float(ha)))), h))
    b_s = cv2.resize(b, (max(1, int(wb * (h / float(hb)))), h))
    return cv2.hconcat([a_s, b_s])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Direct USB dual Orbbec preview")
    parser.add_argument("--cam0", type=int, default=0, help="Orbbec index for left pane")
    parser.add_argument("--cam1", type=int, default=1, help="Orbbec index for right pane")
    parser.add_argument("--max-height", type=int, default=480)
    parser.add_argument("--stagger-s", type=float, default=1.5, help="Delay before opening cam1")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  DIRECT USB dual preview (CAP_OBSENSOR)")
    print("  Stop ROS publishers first if open fails.")
    print("  ROS live view instead: ros2_ws\\view_dual_cameras.cmd")
    print("  Q / Esc quits.")
    print("=" * 60)

    cap0 = cap1 = None
    try:
        print(f"Opening cam0 index {args.cam0}…")
        cap0 = open_orbbec(args.cam0)
        warmup(cap0)
        print(f"Opening cam1 index {args.cam1} after {args.stagger_s:.1f}s stagger…")
        time.sleep(max(0.0, float(args.stagger_s)))
        cap1 = open_orbbec(args.cam1)
        warmup(cap1)
        print("Live. Press Q to quit.")

        title = "view_dual_cameras (cam0 | cam1)"
        while True:
            f0 = grab_color(cap0)
            f1 = grab_color(cap1)
            frame = compose(f0, f1, args.max_height)
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(
            "If ROS dual sync is running, stop it first, or use:\n"
            r"  cd D:\Jessica\Jessica\ros2_ws" "\n"
            r"  .\view_dual_cameras.cmd",
            file=sys.stderr,
        )
        return 1
    finally:
        for cap in (cap0, cap1):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
