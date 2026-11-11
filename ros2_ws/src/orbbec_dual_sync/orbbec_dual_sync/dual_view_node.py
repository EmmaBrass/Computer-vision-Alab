"""Live side-by-side preview: cam0 arm/pose CV | cam1 rack OPEN/CLOSED CV.

Subscribes only — does not open CAP_OBSENSOR. Rack overlay on the right pane
(same as orbbec_rack_overlay). Optional digital center-zoom on cam0.

Reading map (this file is the viewer hub):
  - __init__ / subscriptions: ROS params and topics
  - _paint_cam*: per-camera overlays (rack, blues, tags, UR FK)
  - _overlay_ur_fk / online pose: stick figure + OnlineCamPose
  - _on_timer: compose the three panes + depth row and show the preview

Related modules: online_cam_pose.py, ur_skeleton_draw.py, blue_joint_tracks.py,
baby_blue_draw.py, cell_frames.py, camera_publisher.py.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

from .camera_path import ensure_camera_on_path
from .image_codec import imgmsg_to_numpy
from .preview_gui import PreviewWindow
from .rack_overlay_draw import (
    classify_whole_rack,
    draw_rack_overlay,
    load_rack_rois,
)


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


def _placeholder(lines: list[str], h: int = 480, w: int = 640) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y0 = h // 2 - 12 * (len(lines) - 1)
    for i, text in enumerate(lines):
        cv2.putText(
            img,
            text,
            (24, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (80, 80, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def center_zoom(bgr: np.ndarray, zoom: float) -> np.ndarray:
    """Digital zoom toward image center; ``zoom`` 1.0 = no crop."""
    z = float(zoom)
    if bgr is None or z <= 1.001:
        return bgr
    h, w = bgr.shape[:2]
    crop_w = max(1, int(round(w / z)))
    crop_h = max(1, int(round(h / z)))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    crop = bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def compose_side_by_side(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
    *,
    max_height: int = 480,
    left_status: str = "",
    right_status: str = "",
    left_already_labeled: bool = False,
    right_already_labeled: bool = False,
) -> np.ndarray:
    if left is not None:
        a = left if left_already_labeled else _label(left, "cam0")
    else:
        a = _placeholder(["waiting cam0…", left_status] if left_status else ["waiting cam0…"])
    if right is not None:
        b = right if right_already_labeled else _label(right, "cam1")
    else:
        b = _placeholder(
            ["waiting cam1…", right_status] if right_status else ["waiting cam1…"]
        )
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb, max_height) if max_height > 0 else min(ha, hb)
    a_s = cv2.resize(a, (max(1, int(wa * (h / float(ha)))), h))
    b_s = cv2.resize(b, (max(1, int(wb * (h / float(hb)))), h))
    return cv2.hconcat([a_s, b_s])


def _default_rack_json() -> Path:
    root = ensure_camera_on_path()
    return root / "rack_lid_state" / "rack.json"


class DualViewNode(Node):
    def __init__(self) -> None:
        super().__init__("orbbec_dual_view")
        self.declare_parameter("cam0_color_topic", "/orbbec/cam0/color/image_raw")
        self.declare_parameter("cam1_color_topic", "/orbbec/cam1/color/image_raw")
        self.declare_parameter("cam0_depth_topic", "/orbbec/cam0/depth/image_raw")
        self.declare_parameter("cam1_depth_topic", "/orbbec/cam1/depth/image_raw")
        self.declare_parameter("max_height", 480)
        self.declare_parameter("refresh_hz", 15.0)
        # Digital zoom on cam0 display (pose/overlays still run on full frame).
        self.declare_parameter("cam0_zoom", 1.0)
        # Cam1 rack OPEN/CLOSED CV (same as orbbec_rack_overlay).
        self.declare_parameter("enable_rack_cv", True)
        self.declare_parameter("rack_json", "")
        self.declare_parameter("state_mode", "classify")
        self.declare_parameter("force_state", "CLOSED")
        self.declare_parameter("classify_every_n", 2)
        # YOLO arm detector (arm_state/runs/arm_detect/weights/best.pt)
        self.declare_parameter("enable_arm_cv", True)
        self.declare_parameter("arm_weights", "")
        self.declare_parameter("arm_conf", 0.25)
        self.declare_parameter("arm_every_n", 3)
        # Pale baby-blue color mask tracker (no NN) — UR joint plastic.
        self.declare_parameter("enable_blue_cv", True)
        self.declare_parameter("blue_show_mask", True)
        self.declare_parameter("blue_min_area", 280)
        self.declare_parameter("blue_every_n", 1)
        self.declare_parameter("blue_proximity_mm", 100.0)  # 10 cm red alert
        # 6-slot JointBank + Kalman coast (HSV detector unchanged).
        self.declare_parameter("blue_num_joints", 6)
        self.declare_parameter("blue_coast_s", 10.0)
        self.declare_parameter("blue_merge_mm", 80.0)
        self.declare_parameter("blue_match_mm", 120.0)
        # Teach-pendant / UR FK stick figure + vision confirm + FK rack keep-out.
        self.declare_parameter("enable_ur_fk", True)
        self.declare_parameter("ur_fk_draw", True)
        self.declare_parameter("ur_fk_confirm", True)
        self.declare_parameter("ur_fk_rack_alert", True)
        # Online per-camera pose from arm tags + blue caps over many frames.
        # Legacy tag-fitting stays available behind this flag (default off).
        self.declare_parameter("use_legacy_tag_fit", False)
        self.declare_parameter("online_use_blue", True)
        # Blues + live UR FK 3D model help when joint tags are occluded (folded arm).
        self.declare_parameter("online_use_blue_in_pnp", True)
        self.declare_parameter("online_save_cell", False)
        self.declare_parameter("online_ignore_saved_pose", True)
        self.declare_parameter("online_reset", False)
        self.declare_parameter("cam0_info_topic", "/orbbec/cam0/color/camera_info")
        self.declare_parameter("cam1_info_topic", "/orbbec/cam1/color/camera_info")
        self.declare_parameter("cam2_info_topic", "/orbbec/cam2/color/camera_info")

        t0 = str(self.get_parameter("cam0_color_topic").value)
        t1 = str(self.get_parameter("cam1_color_topic").value)
        self.cam0_depth_topic = str(self.get_parameter("cam0_depth_topic").value)
        self.cam1_depth_topic = str(self.get_parameter("cam1_depth_topic").value)
        self.max_height = int(self.get_parameter("max_height").value)
        hz = float(self.get_parameter("refresh_hz").value)
        period = 1.0 / max(hz, 1.0)
        self.cam0_zoom = max(1.0, float(self.get_parameter("cam0_zoom").value))

        self.enable_rack_cv = bool(self.get_parameter("enable_rack_cv").value)
        self.state_mode = str(self.get_parameter("state_mode").value).strip().lower()
        self.force_state = str(self.get_parameter("force_state").value).strip().upper()
        self.classify_every_n = max(1, int(self.get_parameter("classify_every_n").value))

        self.enable_arm_cv = bool(self.get_parameter("enable_arm_cv").value)
        arm_w = str(self.get_parameter("arm_weights").value).strip()
        self.arm_weights: Optional[Path] = Path(arm_w) if arm_w else None
        self.arm_conf = float(self.get_parameter("arm_conf").value)
        self.arm_every_n = max(1, int(self.get_parameter("arm_every_n").value))
        self._arm_cache0: Optional[np.ndarray] = None
        self._arm_cache1: Optional[np.ndarray] = None

        self.enable_blue_cv = bool(self.get_parameter("enable_blue_cv").value)
        self.blue_show_mask = bool(self.get_parameter("blue_show_mask").value)
        self.blue_min_area = max(20, int(self.get_parameter("blue_min_area").value))
        self.blue_every_n = max(1, int(self.get_parameter("blue_every_n").value))
        self.blue_proximity_mm = max(
            10.0, float(self.get_parameter("blue_proximity_mm").value)
        )
        self.blue_num_joints = max(
            1, int(self.get_parameter("blue_num_joints").value)
        )
        self.blue_coast_s = max(0.1, float(self.get_parameter("blue_coast_s").value))
        self.blue_merge_mm = max(
            10.0, float(self.get_parameter("blue_merge_mm").value)
        )
        self.blue_match_mm = max(
            20.0, float(self.get_parameter("blue_match_mm").value)
        )
        self.enable_ur_fk = bool(self.get_parameter("enable_ur_fk").value)
        self.ur_fk_draw = bool(self.get_parameter("ur_fk_draw").value)
        self.ur_fk_confirm = bool(self.get_parameter("ur_fk_confirm").value)
        self.ur_fk_rack_alert = bool(self.get_parameter("ur_fk_rack_alert").value)
        self._ur_state = None
        self._ur_state_t = 0.0
        self._last_fk_alert_log = 0.0
        self._blue_boxes0 = None
        self._blue_boxes1 = None
        self._blue_alert0 = False
        self._blue_alert1 = False
        self._last_retreat_log0 = 0.0
        self._last_retreat_log1 = 0.0
        self._last_fuse_log = 0.0
        self._fuse0: Optional[dict] = None
        self._fuse1: Optional[dict] = None
        self._joint_bank = None
        self._cell_frames = None
        self._latest_depth0: Optional[np.ndarray] = None
        self._latest_depth1: Optional[np.ndarray] = None

        self._latest0: Optional[np.ndarray] = None
        self._latest0_cv: Optional[np.ndarray] = None
        self._latest1: Optional[np.ndarray] = None
        self._latest1_cv: Optional[np.ndarray] = None
        self._recv0 = 0
        self._recv1 = 0
        self._cam0_seq = 0
        self._cam1_seq = 0
        self._cam2_seq = 0
        self._last_state = "CLOSED"
        self._started = time.monotonic()
        self._warned_empty = False
        self._preview_error_logged = False
        self._should_exit = False
        self._robotics: Optional[Path] = None
        self.classify_xywh = (226, 55, 137, 152)
        self.display_xywh = (186, 30, 220, 260)
        self.holes: list = []

        need_robotics = (
            self.enable_rack_cv
            or self.enable_arm_cv
            or self.enable_blue_cv
            or self.enable_ur_fk
        )
        if need_robotics:
            try:
                self._robotics = ensure_camera_on_path()
            except Exception as exc:
                self.get_logger().error(f"Could not resolve robotics root: {exc}")
                self._robotics = None

        if self.enable_arm_cv:
            if self._robotics is None:
                self.enable_arm_cv = False
                self.get_logger().error("Arm CV disabled — robotics root unavailable")
            else:
                w = self.arm_weights or (
                    self._robotics
                    / "arm_state"
                    / "runs"
                    / "arm_detect"
                    / "weights"
                    / "best.pt"
                )
                if not w.is_file():
                    self.enable_arm_cv = False
                    self.get_logger().error(f"Arm CV disabled — missing weights {w}")
                else:
                    self.arm_weights = w
                    self.get_logger().info(
                        f"Arm YOLO ON both panes: conf={self.arm_conf} "
                        f"every_n={self.arm_every_n} weights={w}"
                    )

        if self.enable_blue_cv:
            self.get_logger().info(
                f"Baby-blue color mask ON both panes: min_area={self.blue_min_area} "
                f"show_mask={self.blue_show_mask}; RED alert if blue < "
                f"{self.blue_proximity_mm:.0f}mm of black/silver equipment "
                f"(mask hidden; 3D/depth; base joint ignored); "
                f"JointBank n={self.blue_num_joints} coast={self.blue_coast_s:.1f}s "
                f"merge={self.blue_merge_mm:.0f}mm match={self.blue_match_mm:.0f}mm"
            )
            try:
                from .cell_frames import (
                    LAYOUT_OPPOSITE_180,
                    LAYOUT_PERPENDICULAR,
                    apply_opposite_180_layout,
                    apply_perpendicular_layout,
                    default_cell_frames_path,
                    load_cell_frames,
                )

                cpath = default_cell_frames_path(self._robotics)
                self._cell_frames = load_cell_frames(cpath)
                from .cell_frames import LAYOUT_PERPENDICULAR, apply_perpendicular_layout

                if self._cell_frames.layout == LAYOUT_OPPOSITE_180:
                    apply_opposite_180_layout(self._cell_frames)
                elif self._cell_frames.layout == LAYOUT_PERPENDICULAR:
                    apply_perpendicular_layout(self._cell_frames)
                self.get_logger().info(
                    f"ur_base cell_frames: {cpath} "
                    f"layout={self._cell_frames.layout} "
                    f"bootstrap={self._cell_frames.bootstrap_from_blue_base} "
                    f"cam0_ready={self._cell_frames.cam0.ready} "
                    f"cam1_ready={self._cell_frames.cam1.ready} "
                    f"cam2_ready={self._cell_frames.cam2.ready} "
                    f"(cam1↔cam2 ~180 high-down; cam0 separate side/close)"
                )
            except Exception as exc:
                self._cell_frames = None
                self.get_logger().warn(f"cell_frames not loaded: {exc}")
            try:
                from .blue_joint_tracks import JointBank

                self._joint_bank = JointBank(
                    num_joints=self.blue_num_joints,
                    coast_s=self.blue_coast_s,
                    merge_mm=self.blue_merge_mm,
                    match_mm=self.blue_match_mm,
                )
            except Exception as exc:
                self._joint_bank = None
                self.get_logger().warn(f"JointBank init failed: {exc}")

        if self.enable_ur_fk:
            self.get_logger().info(
                "UR FK stick-figure ON: write robotics/ur_joints.json from teach "
                "pendant (q_rad[6]) or set ur_fk_config.json host for RTDE; "
                f"draw={self.ur_fk_draw} confirm={self.ur_fk_confirm} "
                f"rack_alert={self.ur_fk_rack_alert}"
            )

        if self.enable_rack_cv:
            try:
                if self._robotics is None:
                    raise FileNotFoundError("robotics root unavailable")
                rack_raw = str(self.get_parameter("rack_json").value).strip()
                rack_path = Path(rack_raw) if rack_raw else _default_rack_json()
                if not rack_path.is_file():
                    raise FileNotFoundError(f"rack.json not found: {rack_path}")
                self.classify_xywh, self.display_xywh, self.holes = load_rack_rois(
                    rack_path
                )
                self.get_logger().info(
                    f"Rack CV ON cam1 pane: mode={self.state_mode} "
                    f"classify={self.classify_xywh} display={self.display_xywh} "
                    f"json={rack_path}"
                )
            except Exception as exc:
                self.enable_rack_cv = False
                self.get_logger().error(
                    f"Rack CV disabled (could not load ROI/templates): {exc}"
                )

        title_bits = ["cam0"]
        if self.cam0_zoom > 1.001:
            title_bits[0] = f"cam0 zoom {self.cam0_zoom:.1f}x"
        title_bits.append("cam1 + rack CV" if self.enable_rack_cv else "cam1")
        self._preview = PreviewWindow(
            f"orbbec_dual_view ({' | '.join(title_bits)})",
            max_height=max(self.max_height, 480),
        )
        extras = []
        if self.cam0_zoom > 1.001:
            extras.append(f"cam0 center zoom {self.cam0_zoom:.1f}x")
        if self.enable_rack_cv:
            extras.append("rack OPEN/CLOSED on cam1")
        if self.enable_arm_cv:
            extras.append("arm YOLO on both panes")
        if self.enable_blue_cv:
            extras.append(
                f"baby-blue vs black/silver equipment ({self.blue_proximity_mm:.0f}mm 3D alert)"
            )

        extra = (" + " + "; ".join(extras) + ".") if extras else "."
        self.get_logger().info(
            f"Dual view: {t0} | {t1}{extra} Q / Esc / close exits."
        )

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, t0, self._on_cam0, image_qos)
        self.create_subscription(Image, t1, self._on_cam1, image_qos)
        self.create_subscription(
            Image, self.cam0_depth_topic, self._on_depth0, image_qos
        )
        self.create_subscription(
            Image, self.cam1_depth_topic, self._on_depth1, image_qos
        )
        self.create_timer(period, self._on_timer)

    def _resolve_state(self, bgr: np.ndarray) -> str:
        if self.state_mode == "force_open":
            return "OPEN"
        if self.state_mode == "force_closed":
            return "CLOSED"
        if self.state_mode in ("force", "fixed"):
            return self.force_state or "CLOSED"
        try:
            return classify_whole_rack(
                bgr,
                self.classify_xywh,
                self.holes,
                robotics_root=self._robotics,
            )
        except Exception as exc:
            self.get_logger().warn(f"classify failed, using last={self._last_state}: {exc}")
            return self._last_state

    def _paint_cam0_cv(self, bgr: np.ndarray) -> np.ndarray:
        self._cam0_seq += 1
        tag = "cam0"
        if self.cam0_zoom > 1.001:
            tag = f"cam0 zoom {self.cam0_zoom:.1f}x"
        view = _label(bgr, tag)
        view = self._overlay_arm_on(view, bgr, cam="cam0")
        return self._overlay_blue_on(view, bgr, cam="cam0")

    def _overlay_blue_on(
        self, decorated: np.ndarray, raw: np.ndarray, *, cam: str
    ) -> np.ndarray:
        """HSV baby-blue → JointBank tracks (J0..J5) with Kalman coast."""
        alert_attr = {
            "cam0": "_blue_alert0",
            "cam1": "_blue_alert1",
            "cam2": "_blue_alert2",
        }.get(cam, "_blue_alert1")
        if not self.enable_blue_cv:
            setattr(self, alert_attr, False)
            return self._draw_fk_monitor(decorated, cam)
        seq = {
            "cam0": self._cam0_seq,
            "cam1": self._cam1_seq,
            "cam2": int(getattr(self, "_cam2_seq", 0)),
        }.get(cam, self._cam1_seq)
        cache_attr = {
            "cam0": "_blue_boxes0",
            "cam1": "_blue_boxes1",
            "cam2": "_blue_boxes2",
        }.get(cam, "_blue_boxes1")
        depth = {
            "cam0": self._latest_depth0,
            "cam1": self._latest_depth1,
            "cam2": getattr(self, "_latest_depth2", None),
        }.get(cam)
        if seq % self.blue_every_n == 0 or getattr(self, cache_attr) is None:
            try:
                from .baby_blue_draw import find_baby_blue_boxes

                setattr(
                    self,
                    cache_attr,
                    find_baby_blue_boxes(
                        raw, min_area=self.blue_min_area, depth_mm=depth
                    ),
                )
            except Exception as exc:
                self.get_logger().warn(f"blue mask {cam}: {exc}")
                setattr(self, cache_attr, [])
        try:
            from .baby_blue_draw import (
                baby_blue_mask,
                draw_proximity_error,
                proximity_alerts,
                retreat_guidance_toward_base,
            )
            from .blue_joint_tracks import (
                draw_tracked_blue,
                views_to_prox_boxes,
            )

            boxes = getattr(self, cache_attr) or []
            rh, rw = raw.shape[:2]
            mask = baby_blue_mask(raw)
            pose = None
            fx = 520.0
            if self._cell_frames is not None:
                pose = self._cell_frames.pose(cam)
                fx = float(self._cell_frames.fx_px)

            # Scale boxes to depth frame for 3D / bank ingest
            boxes_for_bank = boxes
            mask_for_bank = mask
            depth_for_bank = depth
            if depth is not None and boxes:
                dh, dw = depth.shape[:2]
                if (dh, dw) != (rh, rw) and rh > 0 and rw > 0:
                    sx, sy = dw / float(rw), dh / float(rh)
                    boxes_for_bank = [
                        (
                            int(x * sx),
                            int(y * sy),
                            max(1, int(w * sx)),
                            max(1, int(h * sy)),
                            d,
                        )
                        for x, y, w, h, d in boxes
                    ]
                    mask_for_bank = cv2.resize(
                        mask, (dw, dh), interpolation=cv2.INTER_NEAREST
                    )

            views = []
            if self._joint_bank is not None:
                views = self._joint_bank.ingest_cam(
                    cam,
                    boxes_for_bank,
                    depth_for_bank,
                    mask_for_bank if depth is not None else mask,
                    pose=pose,
                    fx=fx,
                    frame_h=depth.shape[0] if depth is not None else rh,
                    frame_w=depth.shape[1] if depth is not None else rw,
                    bgr=raw,
                )
                # Pedestal blue = ur_base (0,0,0) for both cameras.
                try:
                    self._bootstrap_origin_from_j0(cam)
                except Exception:
                    pass
                # Scale track views back to color/decorated frame for drawing
                if depth is not None and boxes:
                    dh, dw = depth.shape[:2]
                    if (dh, dw) != (rh, rw) and dh > 0 and dw > 0:
                        ox, oy = rw / float(dw), rh / float(dh)
                        for v in views:
                            v.x = int(v.x * ox)
                            v.y = int(v.y * oy)
                            v.w = max(1, int(v.w * ox))
                            v.h = max(1, int(v.h * oy))
                live, coast, cap = self._joint_bank.hud_counts()
            else:
                live = coast = 0
                cap = self.blue_num_joints
                from .blue_joint_tracks import TrackView

                views = [
                    TrackView(
                        track_id=i,
                        x=b[0],
                        y=b[1],
                        w=b[2],
                        h=b[3],
                        d_mm=b[4],
                        coasting=False,
                        is_base=False,
                        p_base=None,
                    )
                    for i, b in enumerate(boxes)
                ]

            out = draw_tracked_blue(
                decorated,
                views,
                show_mask=self.blue_show_mask,
                depth_mm=depth
                if depth is not None and depth.shape[:2] == raw.shape[:2]
                else None,
                hud_live=live,
                hud_coast=coast,
                hud_cap=cap,
                tape_box=self._joint_bank._tape_box.get(cam)
                if self._joint_bank is not None
                else None,
            )

            alerts = []
            guide = None
            prox_boxes = views_to_prox_boxes(views)
            if depth is not None and prox_boxes:
                if mask.shape[:2] != depth.shape[:2]:
                    mask_d = cv2.resize(
                        mask,
                        (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    raw_for_prox = cv2.resize(
                        raw,
                        (depth.shape[1], depth.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    mask_d = mask
                    raw_for_prox = raw
                dh, dw = depth.shape[:2]
                if (dh, dw) != (rh, rw) and rh > 0 and rw > 0:
                    sx, sy = dw / float(rw), dh / float(rh)
                    boxes_d = [
                        (
                            int(x * sx),
                            int(y * sy),
                            max(1, int(w * sx)),
                            max(1, int(h * sy)),
                            d,
                        )
                        for x, y, w, h, d in prox_boxes
                    ]
                else:
                    boxes_d = prox_boxes
                rack_for_prox = None
                if cam == "cam1" and self.classify_xywh:
                    rcx, rcy, rcw, rch = self.classify_xywh
                    if (dh, dw) != (rh, rw) and rh > 0 and rw > 0:
                        rack_for_prox = (
                            int(rcx * sx),
                            int(rcy * sy),
                            max(1, int(rcw * sx)),
                            max(1, int(rch * sy)),
                        )
                    else:
                        rack_for_prox = (int(rcx), int(rcy), int(rcw), int(rch))
                alerts = proximity_alerts(
                    depth,
                    boxes_d,
                    mask_d,
                    bgr=raw_for_prox,
                    thresh_mm=self.blue_proximity_mm,
                    skip_base=True,
                    rack_xywh=rack_for_prox,
                    rack_height_mm=400.0,
                )
                if alerts:
                    try:
                        guide = retreat_guidance_toward_base(
                            depth, boxes_d, mask_d, alerts
                        )
                    except Exception:
                        guide = None
                try:
                    self._stash_fuse_points(cam, depth, boxes_d, mask_d, alerts)
                except Exception:
                    pass
                if alerts and (dh, dw) != out.shape[:2]:
                    ox, oy = out.shape[1] / float(dw), out.shape[0] / float(dh)
                    alerts = [
                        (
                            int(a[0] * ox),
                            int(a[1] * oy),
                            max(1, int(a[2] * ox)),
                            max(1, int(a[3] * oy)),
                            a[4],
                            a[5] if len(a) > 5 else "",
                        )
                        for a in alerts
                    ]

            # Teach-pendant / UR FK stick figure + confirm + out-of-view rack alert.
            if self.enable_ur_fk and pose is not None:
                try:
                    out, fk_alerts, fk_guide = self._overlay_ur_fk(
                        out, views, pose=pose, fx=fx, cam=cam
                    )
                    if fk_alerts:
                        alerts = list(alerts) + list(fk_alerts)
                        if fk_guide and not guide:
                            guide = fk_guide
                except Exception as exc:
                    self.get_logger().warn(f"ur_fk {cam}: {exc}")

            # Dedup pane alerts: only the tighter-gap cam raises fused flag if
            # both see the same track; still draw red on the alerting pane.
            setattr(self, alert_attr, bool(alerts))
            if alerts:
                out = draw_proximity_error(
                    out, alerts, thresh_cm=self.blue_proximity_mm / 10.0
                )
                if guide:
                    log_attr = (
                        "_last_retreat_log0"
                        if cam == "cam0"
                        else "_last_retreat_log1"
                    )
                    now = time.monotonic()
                    if now - float(getattr(self, log_attr, 0.0)) >= 1.0:
                        setattr(self, log_attr, now)
                        self.get_logger().error(f"[{cam}] {guide}")
                        print(f"\n*** [{cam}] {guide} ***\n", flush=True)
                    cv2.putText(
                        out,
                        "RETREAT -> BASE (see terminal)",
                        (18, out.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
            return out
        except Exception as exc:
            self.get_logger().warn(f"blue draw {cam}: {exc}")
            setattr(self, alert_attr, False)
            return decorated

    def _refresh_ur_state(self):
        now = time.monotonic()
        if self._ur_state is not None and (now - self._ur_state_t) < 0.05:
            return self._ur_state
        from .ur_skeleton_draw import load_ur_state

        self._ur_state = load_ur_state(self._robotics)
        self._ur_state_t = now
        return self._ur_state

    def _overlay_ur_fk(
        self,
        out: np.ndarray,
        views: list,
        *,
        pose,
        fx: float,
        cam: str,
    ):
        """Draw FK skeleton, confirm vs vision, return extra rack alerts."""
        from .ur_skeleton_draw import (
            confirm_vision_vs_fk,
            draw_ur_skeleton,
            fk_alerts_as_proximity,
        )

        state = self._refresh_ur_state()
        if not state:
            return out, [], None
        cfg = state["cfg"]
        pts = state["pts_vision_mm"]
        matches = []
        if self.ur_fk_confirm:
            matches = confirm_vision_vs_fk(
                views, pts, gate_mm=float(cfg.confirm_gate_mm)
            )
        rack_hits = state["rack_hits"] if self.ur_fk_rack_alert else []
        if self.ur_fk_draw:
            out = draw_ur_skeleton(
                out,
                pts,
                pose,
                fx=fx,
                matches=matches,
                rack_hits=rack_hits,
                source=state["source"],
                model=state["model"],
                pts_upright_mm=pts_fk if calibrated else state.get("pts_upright_mm"),
                check_ok=state.get("check_ok"),
                check_mm=state.get("check_mm"),
                check_unit=str(state.get("check_unit") or "mm"),
                base_anchor_uv=anchor,
                img_deg=img_deg,
                img_scale=img_scale,
                blue_uv=state.get("blue_uv"),
                has_tool=bool(state.get("has_tool")),
                tag_uv=tag_uv,
                pts_fk_mm=pts_fk,
                cam=cam,
                calibrated=calibrated,
                fy=fy_draw,
                cx=cx_draw,
                cy=cy_draw,
                teachpad_tcp_err_mm=state.get("teachpad_tcp_err_mm"),
                K=K_draw,
                dist=dist_draw,
            )
        fk_alerts = fk_alerts_as_proximity(rack_hits) if rack_hits else []
        guide = None
        if rack_hits:
            worst = min(rack_hits, key=lambda x: x[1])
            guide = (
                f"FK RACK KEEP-OUT: joint FK{worst[0]} gap={worst[1]:.0f}mm "
                f"(teach-pad / UR angles — works even if blue joint left FOV)"
            )
            now = time.monotonic()
            if now - float(self._last_fk_alert_log) >= 1.0:
                self._last_fk_alert_log = now
                self.get_logger().error(f"[{cam}] {guide}")
        return out, fk_alerts, guide

    def _update_arm_speed(self, q) -> None:
        """Compute joint speed once per new robot reading; cameras only read it."""
        if q is None:
            return
        try:
            q_now = np.asarray(q, dtype=np.float64).reshape(-1)
        except Exception:
            return
        now = time.monotonic()
        speed = None
        if (
            self._last_q is not None
            and self._last_q.shape == q_now.shape
            and (now - self._last_q_t) > 1e-3
        ):
            peak = 0.0
            dt = now - self._last_q_t
            for i in range(q_now.shape[0]):
                d = abs(float(q_now[i]) - float(self._last_q[i])) / dt
                if d > peak:
                    peak = d
            speed = peak
        self._last_q = q_now
        self._last_q_t = now
        self._arm_speed = speed

    def _arm_joint_speed(self, q) -> Optional[float]:
        """Stored max joint speed (rad/s). Does not update _last_q."""
        _ = q
        return getattr(self, "_arm_speed", None)

    def _maybe_reset_online(self) -> None:
        """Clear every OnlineCamPose when R is pressed or online_reset is True."""
        want = False
        try:
            want = bool(self.get_parameter("online_reset").value)
        except Exception:
            want = False
        prev = getattr(self, "_preview", None)
        if prev is not None and getattr(prev, "reset_requested", False):
            want = True
            prev.reset_requested = False
        if not want:
            return
        for op in self._online_pose.values():
            try:
                op.reset()
            except Exception:
                pass
        try:
            from rclpy.parameter import Parameter

            self.set_parameters(
                [Parameter("online_reset", Parameter.Type.BOOL, False)]
            )
        except Exception:
            pass
        self.get_logger().info("OnlineCamPose reset (buffer + pose cleared)")

    def _draw_online_pose_hud(self, out: np.ndarray, cam: str, op) -> np.ndarray:
        """Per-camera online-pose status: buffer pairs, distinct poses, error."""
        err = op.mean_px
        err_txt = "--" if not (err == err) else f"{err:.1f}px"  # NaN-safe
        err3 = getattr(op, "mean_3d_mm", float("nan"))
        err3_txt = "--" if not (err3 == err3) else f"{err3:.0f}mm"
        n3 = int(getattr(op, "n_3d", 0))
        mode = str(getattr(op, "last_mode", "") or "")
        state_txt = "ACC" if op.last_accepted else ("pose" if op.have_pose else "solving")
        if mode:
            state_txt = f"{state_txt}/{mode}"
        line = (
            f"online {cam}: teachpad-FK pairs {op.n_pairs} poses {op.n_poses} "
            f"3d {n3} err {err_txt}/{err3_txt} [{state_txt}]"
        )
        color = (0, 220, 0) if op.last_accepted else (0, 210, 210)
        y = out.shape[0] - 40
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
        return out

    def _overlay_arm_on(
        self, decorated: np.ndarray, raw: np.ndarray, *, cam: str
    ) -> np.ndarray:
        """Detect on raw frame, then draw boxes onto the decorated overlay."""
        if not self.enable_arm_cv or self._robotics is None:
            return decorated
        seq = {
            "cam0": self._cam0_seq,
            "cam1": self._cam1_seq,
            "cam2": int(getattr(self, "_cam2_seq", 0)),
        }.get(cam, self._cam1_seq)
        cache_attr = {
            "cam0": "_arm_boxes0",
            "cam1": "_arm_boxes1",
            "cam2": "_arm_boxes2",
        }.get(cam, "_arm_boxes1")
        if not hasattr(self, "_arm_boxes0"):
            self._arm_boxes0 = None
            self._arm_boxes1 = None
            self._arm_boxes2 = None
        if seq % self.arm_every_n == 0 or getattr(self, cache_attr) is None:
            try:
                from .arm_overlay_draw import default_arm_weights, get_worker

                wpath = self.arm_weights or default_arm_weights(self._robotics)
                boxes = get_worker(wpath, conf=self.arm_conf).predict(
                    raw, conf=self.arm_conf, cam=cam, refine=True
                )
                setattr(self, cache_attr, boxes)
            except Exception as exc:
                self.get_logger().warn(f"arm detect {cam}: {exc}")
                setattr(self, cache_attr, [])
        out = decorated.copy()
        for xyxy, conf in getattr(self, cache_attr) or []:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 0), 2)
            cv2.putText(
                out,
                f"arm {conf:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 0),
                2,
                cv2.LINE_AA,
            )
        return out

    def _paint_cam1_cv(self, bgr: np.ndarray) -> np.ndarray:
        self._cam1_seq += 1
        if self.enable_rack_cv:
            if self._cam1_seq % self.classify_every_n == 0 or self._latest1_cv is None:
                self._last_state = self._resolve_state(bgr)
            view = draw_rack_overlay(
                bgr,
                self.display_xywh,
                self._last_state,
                classify_xywh=self.classify_xywh,
            )
            cv2.putText(
                view,
                "cam1 + rack + arm",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            view = _label(bgr, "cam1")
        view = self._overlay_arm_on(view, bgr, cam="cam1")
        return self._overlay_blue_on(view, bgr, cam="cam1")

    def _paint_cam2_cv(self, bgr: np.ndarray) -> np.ndarray:
        # Opposite high-down partner of cam1 (not cam0).
        self._cam2_seq = int(getattr(self, "_cam2_seq", 0)) + 1
        view = _label(bgr, "cam2 high-down (opp cam1)")
        view = self._overlay_arm_on(view, bgr, cam="cam2")
        return self._overlay_blue_on(view, bgr, cam="cam2")

    def _on_depth0(self, msg: Image) -> None:
        try:
            d = imgmsg_to_numpy(msg)
            if d.ndim == 2:
                self._latest_depth0 = d
        except Exception as exc:
            self.get_logger().warn(f"cam0 depth decode failed: {exc}")

    def _on_depth1(self, msg: Image) -> None:
        try:
            d = imgmsg_to_numpy(msg)
            if d.ndim == 2:
                self._latest_depth1 = d
        except Exception as exc:
            self.get_logger().warn(f"cam1 depth decode failed: {exc}")

    def _on_cam0(self, msg: Image) -> None:
        try:
            img = imgmsg_to_numpy(msg)
            if img.ndim == 3:
                self._latest0 = img
                self._latest0_cv = self._paint_cam0_cv(img)
                self._recv0 += 1
                if self._recv0 == 1:
                    self.get_logger().info(
                        f"cam0 first frame {img.shape[1]}x{img.shape[0]} {msg.encoding}"
                    )
        except Exception as exc:
            self.get_logger().warn(f"cam0 decode/CV failed: {exc}")

    def _on_cam1(self, msg: Image) -> None:
        try:
            img = imgmsg_to_numpy(msg)
            if img.ndim == 3:
                self._latest1 = img
                self._latest1_cv = self._paint_cam1_cv(img)
                self._recv1 += 1
                if self._recv1 == 1:
                    self.get_logger().info(
                        f"cam1 first frame {img.shape[1]}x{img.shape[0]} {msg.encoding}"
                    )
        except Exception as exc:
            self.get_logger().warn(f"cam1 decode/CV failed: {exc}")

    def _status_line(self, recv: int) -> str:
        age = int(time.monotonic() - self._started)
        if recv > 0:
            return f"ok ({recv})"
        return f"no ROS frames {age}s — is publisher up?"

    def _bootstrap_origin_from_j0(self, cam: str) -> None:
        """Lock cell_frames so the pedestal blue joint is ur_base (0,0,0)."""
        if self._joint_bank is None or self._cell_frames is None:
            return
        if not self._joint_bank.origin_ready or self._joint_bank.j0_p_cam is None:
            return
        # Only update the camera that currently owns the pedestal sighting.
        if self._joint_bank.j0_cam and self._joint_bank.j0_cam != cam:
            return
        from .cell_frames import bootstrap_pose_from_base_landmark

        pose = self._cell_frames.pose(cam)
        new_pose = bootstrap_pose_from_base_landmark(
            self._joint_bank.j0_p_cam, R=pose.R, frame_id=pose.frame_id
        )
        if cam == "cam0":
            self._cell_frames.cam0 = new_pose
        else:
            self._cell_frames.cam1 = new_pose

    def _stash_fuse_points(
        self,
        cam: str,
        depth: Optional[np.ndarray],
        boxes: list,
        mask: np.ndarray,
        alerts: list,
    ) -> None:
        """Cache cam-frame 3D base + alerting joint for fused ur_base retreat."""
        from .baby_blue_draw import (
            _blue_point_mm,
            is_base_blue_box,
        )
        from .cell_frames import bootstrap_pose_from_base_landmark, cam_to_base

        fuse_attr = "_fuse0" if cam == "cam0" else "_fuse1"
        if depth is None or not boxes:
            setattr(self, fuse_attr, None)
            return
        dh, dw = depth.shape[:2]
        fx = 520.0
        if self._cell_frames is not None:
            fx = float(self._cell_frames.fx_px)
        cx_f, cy_f = 0.5 * dw, 0.5 * dh
        base_box = None
        for b in boxes:
            if is_base_blue_box(b[0], b[1], b[2], b[3], dh):
                base_box = b
                break
        if base_box is None:
            base_box = max(boxes, key=lambda b: b[1] + 0.5 * b[3])
        bp_base = _blue_point_mm(
            depth,
            mask,
            base_box[0],
            base_box[1],
            base_box[2],
            base_box[3],
            base_box[4],
            fx=fx,
            fy=fx,
            cx=cx_f,
            cy=cy_f,
        )
        p_joint_cam = None
        if alerts:
            alert = min(alerts, key=lambda a: float(a[4]))
            ax, ay, aw, ah = (
                int(alert[0]),
                int(alert[1]),
                int(alert[2]),
                int(alert[3]),
            )
            d_alert = None
            for bx, by, bw, bh, bd in boxes:
                if abs(bx - ax) < 8 and abs(by - ay) < 8:
                    d_alert = bd
                    break
            bp_j = _blue_point_mm(
                depth,
                mask,
                ax,
                ay,
                aw,
                ah,
                d_alert,
                fx=fx,
                fy=fx,
                cx=cx_f,
                cy=cy_f,
            )
            if bp_j is not None:
                p_joint_cam = bp_j[0]

        p_base_cam = bp_base[0] if bp_base is not None else None
        # Bootstrap / refresh T_base_cam from pedestal landmark
        if (
            self._cell_frames is not None
            and self._cell_frames.bootstrap_from_blue_base
            and p_base_cam is not None
        ):
            pose = self._cell_frames.pose(cam)
            # Keep configured R; set t so pedestal → ur_base origin
            new_pose = bootstrap_pose_from_base_landmark(
                p_base_cam, R=pose.R, frame_id=pose.frame_id
            )
            if cam == "cam0":
                self._cell_frames.cam0 = new_pose
            else:
                self._cell_frames.cam1 = new_pose

        p_joint_base = None
        if (
            p_joint_cam is not None
            and self._cell_frames is not None
            and self._cell_frames.pose(cam).ready
        ):
            p_joint_base = cam_to_base(p_joint_cam, self._cell_frames.pose(cam))

        setattr(
            self,
            fuse_attr,
            {
                "alerted": bool(alerts),
                "p_joint_cam": p_joint_cam,
                "p_base_cam": p_base_cam,
                "p_joint_base": p_joint_base,
                "gap_mm": float(min(a[4] for a in alerts)) if alerts else None,
            },
        )

    def _fuse_base_retreat(self) -> None:
        """One terminal instruction in ur_base when either cam proximity-alerts."""
        from .cell_frames import format_base_retreat, save_cell_frames

        if not (self._blue_alert0 or self._blue_alert1):
            return
        now = time.monotonic()
        if now - float(self._last_fuse_log) < 1.0:
            return
        self._last_fuse_log = now

        # Prefer the camera with the tighter gap
        candidates = []
        for cam, fuse in (("cam0", self._fuse0), ("cam1", self._fuse1)):
            if not fuse or not fuse.get("alerted"):
                continue
            pjb = fuse.get("p_joint_base")
            if pjb is None:
                continue
            candidates.append((fuse.get("gap_mm") or 1e9, cam, pjb))
        if not candidates:
            return
        candidates.sort(key=lambda x: x[0])
        _gap, cam, p_joint_base = candidates[0]
        msg = format_base_retreat(p_joint_base)
        src = []
        if self._blue_alert0:
            src.append("cam0")
        if self._blue_alert1:
            src.append("cam1")
        full = f"[{'+'.join(src)} via {cam}] {msg}"
        self.get_logger().error(full)
        print(f"\n*** {full} ***\n", flush=True)
        # Persist bootstrapped poses occasionally
        if self._cell_frames is not None:
            try:
                save_cell_frames(self._cell_frames)
            except Exception:
                pass

    def _on_timer(self) -> None:
        if self._should_exit:
            return
        age = time.monotonic() - self._started
        if (
            not self._warned_empty
            and age >= 8.0
            and (self._recv0 == 0 or self._recv1 == 0)
        ):
            self._warned_empty = True
            self.get_logger().error(
                f"Still empty after {age:.0f}s (cam0={self._recv0} cam1={self._recv1}). "
                "Publishers must be streaming; close USB dual viewers first."
            )
        try:
            left = self._latest0_cv if self._latest0_cv is not None else self._latest0
            if left is not None and self.cam0_zoom > 1.001:
                left = center_zoom(left, self.cam0_zoom)
            right = self._latest1_cv if self._latest1_cv is not None else self._latest1
            frame = compose_side_by_side(
                left,
                right,
                max_height=self.max_height,
                left_status=self._status_line(self._recv0),
                right_status=self._status_line(self._recv1),
                left_already_labeled=self._latest0_cv is not None,
                right_already_labeled=self._latest1_cv is not None,
            )
            depth0 = self._depth_map_bgr(self._latest_depth0, "cam0 depth")
            if depth0 is not None and self.cam0_zoom > 1.001:
                depth0 = center_zoom(depth0, self.cam0_zoom)
            depth_row = compose_side_by_side(
                depth0,
                self._depth_map_bgr(self._latest_depth1, "cam1 depth"),
                max_height=max(160, int(self.max_height) // 2),
                left_already_labeled=True,
                right_already_labeled=True,
            )
            third = self._latest2_cv if self._latest2_cv is not None else self._latest2
            if third is None:
                third = _placeholder(
                    ["waiting cam2…", self._status_line(self._recv2)]
                )
            frame = _attach_pane(frame, third)
            depth_row = _attach_pane(
                depth_row, self._depth_map_bgr(self._latest_depth2, "cam2 depth")
            )
            if depth_row.shape[1] != frame.shape[1]:
                depth_row = cv2.resize(depth_row, (frame.shape[1], depth_row.shape[0]))
            frame = cv2.vconcat([frame, depth_row])
            if self._blue_alert0 or self._blue_alert1:
                # Extra full-window red signal so it's impossible to miss
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 18)
                cv2.putText(
                    frame,
                    "ERROR: BLUE WITHIN 10cm",
                    (24, max(40, h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "FUSED RETREAT -> ur_base (0,0,0)  see terminal",
                    (24, max(80, h // 2 + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                try:
                    self._fuse_base_retreat()
                except Exception as exc:
                    self.get_logger().warn(f"ur_base fuse failed: {exc}")
            ok = self._preview.show(frame)
            if not ok and self._preview.user_closed:
                self.get_logger().info("Preview closed — exiting dual view.")
                self._should_exit = True
            elif (
                not ok
                and self._preview.last_error
                and not self._preview_error_logged
            ):
                self._preview_error_logged = True
                self.get_logger().error(
                    f"Preview backend failed: {self._preview.last_error}"
                )
        except Exception as exc:
            self.get_logger().error(
                f"dual view refresh failed: {exc}\n{traceback.format_exc()}"
            )

    def destroy_node(self) -> bool:
        try:
            self._preview.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DualViewNode()
        # Single-threaded: tkinter preview must run on the main thread.
        while rclpy.ok() and not node._should_exit:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
