"""Orbbec RGB-D open / read helpers for ROS publishers (no object detection)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

USB_TIP = (
    "Each Orbbec needs its own USB 3 port. "
    "Close Orbbec Viewer and other apps using the camera, then try again. "
    "Two cameras are indexes 0 and 1. Index 2 is a third Orbbec when present."
)


def depth_to_mm(depth) -> np.ndarray:
    values = depth.astype(np.float32)
    if values.ndim == 3:
        values = values[:, :, 0]
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmax(finite)) <= 20:
        values = values * 1000.0
    return values


def align_depth(depth_mm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize depth to ``(height, width)``. This is NOT Orbbec D2C alignment."""
    if depth_mm.shape[:2] == (height, width):
        return depth_mm
    return cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)


def intrinsics_from_capture(cap, width: int, height: int):
    fx = float(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FX))
    fy = float(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_FY))
    cx = float(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_CX))
    cy = float(cap.get(cv2.CAP_PROP_OBSENSOR_INTRINSIC_CY))
    if fx <= 1 or fy <= 1 or cx <= 1 or cy <= 1:
        return None
    if cx > width or cy > height:
        scale_x = width / (cx * 2)
        scale_y = height / (cy * 2)
        fx *= scale_x
        cx *= scale_x
        fy *= scale_y
        cy *= scale_y
    return fx, fy, cx, cy


def warm_capture(cap: cv2.VideoCapture, tries: int = 45):
    """First Orbbec frames are often empty — wait until color and depth both arrive."""
    for _ in range(tries):
        if not cap.grab():
            time.sleep(0.03)
            continue
        ok_color, color = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
        ok_depth, depth = cap.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
        if ok_color and color is not None and ok_depth and depth is not None:
            return color, depth
        time.sleep(0.03)
    return None, None


class OrbbecDepthCamera:
    """CAP_OBSENSOR stream only — open, read color+depth, release."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        camera_index: int = 0,
        confidence: float = 0.0,
        detector=None,
        window: str | None = None,
        **_ignored,
    ):
        del model_path, confidence, detector  # accepted for call-site compatibility
        self.camera_index = camera_index
        self.window = window or f"orbbec_cam{camera_index}"
        self.capture = None
        self.intrinsics = None

    def open(
        self,
        *,
        require_frame: bool = True,
        retries: int = 3,
        warm_tries: int = 45,
    ) -> None:
        last_error = None
        for attempt in range(1, retries + 1):
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_OBSENSOR)
            if not self.capture.isOpened():
                last_error = f"Could not open Orbbec camera index {self.camera_index}."
                time.sleep(0.35 * attempt)
                continue
            if not require_frame:
                print(f"Camera {self.camera_index}: stream open (no warm frame)")
                return
            color, depth = warm_capture(self.capture, tries=warm_tries)
            if color is not None and depth is not None:
                print(
                    f"Camera {self.camera_index}: stream open "
                    f"({color.shape[1]}x{color.shape[0]})"
                )
                return
            last_error = (
                f"Orbbec index {self.camera_index} opened but never returned color+depth."
            )
            self.capture.release()
            self.capture = None
            time.sleep(0.35 * attempt)
        raise RuntimeError(f"{last_error} {USB_TIP}")

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        try:
            cv2.destroyWindow(self.window)
        except cv2.error:
            pass

    def read_images(self):
        if self.capture is None or not self.capture.grab():
            return None, None
        ok_color, color = self.capture.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
        ok_depth, depth = self.capture.retrieve(None, cv2.CAP_OBSENSOR_DEPTH_MAP)
        if not ok_color or color is None or not ok_depth or depth is None:
            return None, None
        depth_mm = depth_to_mm(depth)
        if not getattr(self, "_logged_stream_shapes", False):
            self._logged_stream_shapes = True
            ch, cw = color.shape[:2]
            dh, dw = depth_mm.shape[:2]
            if (ch, cw) == (dh, dw):
                note = (
                    "same size as color — treat as already D2C-aligned "
                    "(confirm a hand outline overlaps in both panes)"
                )
            else:
                note = (
                    "native depth size, not D2C. CAP_OBSENSOR does not run "
                    "Orbbec AlignFilter; viewers must map color pixels into "
                    "this frame. Do not treat cv2.resize as alignment."
                )
            print(
                f"Camera {self.camera_index}: color {cw}x{ch} depth {dw}x{dh} — {note}"
            )
        return color, depth_mm
