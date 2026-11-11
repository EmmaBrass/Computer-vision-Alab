"""Fixed rack lid state: closed vs open (not free-form object search).

Camera 1 looks at a rack that does not move. Lock the rack once in rack.json,
then focus only that crop:

    cd D:\\Jessica\\Jessica\\robotics
    python rack_lid_state.py focus

Green = closed (white lid). Orange = open (beige interior). Q quits.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "rack_lid_state" / "slots.json"
RACK_CONFIG = ROOT / "rack_lid_state" / "rack.json"
CAPTURE_DIR = ROOT / "rack_lid_state" / "captures"
REFERENCE_IMAGE = CAPTURE_DIR / "rack_reference_cam1.png"
LABELED_IMAGE = CAPTURE_DIR / "rack_labeled.png"
FOCUS_CAMERA = 1

# Hole centers measured on the locked 137x152 rack crop (camera 1).
# Four columns, plus the single lid on the top shelf. y=123 includes empty holes.
_REF_CROP_W = 137
_REF_CROP_H = 152
_REF_ROW_Y = (60, 74, 86, 99, 110, 123)

# Slot ROI is a square of this side around each calibrated center (pixels).
DEFAULT_SLOT_SIZE = 72
# Bright-white lid vs beige open interior (HSV V channel + saturation).
WHITE_V_MIN = 185
WHITE_S_MAX = 70
CLOSED_WHITE_FRAC = 0.42
OPEN_WHITE_FRAC = 0.22
MIN_CIRCULARITY = 0.55


@dataclass
class Slot:
    name: str
    cx: int
    cy: int
    size: int = DEFAULT_SLOT_SIZE

    def roi(self, width: int, height: int) -> tuple[int, int, int, int]:
        half = max(8, int(self.size) // 2)
        x0 = max(0, int(self.cx) - half)
        y0 = max(0, int(self.cy) - half)
        x1 = min(width, int(self.cx) + half)
        y1 = min(height, int(self.cy) + half)
        return x0, y0, x1 - x0, y1 - y0


@dataclass
class SlotState:
    name: str
    state: str  # closed | open | empty | unknown
    white_frac: float
    circularity: float
    box: tuple[int, int, int, int]


def load_slots(path: Path = DEFAULT_CONFIG) -> list[Slot]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    slots = []
    for item in data.get("slots", []):
        slots.append(
            Slot(
                name=str(item.get("name", f"slot{len(slots)}")),
                cx=int(item["cx"]),
                cy=int(item["cy"]),
                size=int(item.get("size", data.get("slot_size", DEFAULT_SLOT_SIZE))),
            )
        )
    return slots


@dataclass
class RackHole:
    """Jar hole fixed inside the rack ROI (coordinates relative to the crop)."""

    cx: int
    cy: int
    rx: int = 8
    ry: int = 5


@dataclass
class RackROI:
    x: int
    y: int
    w: int
    h: int
    camera_index: int
    image_width: int
    image_height: int
    note: str
    holes: list[RackHole]
    slot_rx: int = 8
    slot_ry: int = 5


@dataclass
class LidHit:
    state: str  # closed | open | empty
    box: tuple[int, int, int, int]  # x, y, w, h inside the rack crop
    white_count: int
    beige_count: int


def _ref_columns(y: int) -> list[int]:
    """Four lid columns. x drifts slightly with y because of perspective."""
    t = (y - 60) / 50.0
    return [
        int(round(26 + 7 * t)),
        int(round(45 + 4 * t)),
        int(round(83 + 3 * t)),
        int(round(103 - t)),
    ]


def default_holes(width: int, height: int, rx: int = 8, ry: int = 5) -> list[RackHole]:
    """Fixed jar holes for this rack, scaled if the locked crop size changes."""
    sx = width / float(_REF_CROP_W)
    sy = height / float(_REF_CROP_H)
    points = [(65, 44)]
    for y in _REF_ROW_Y:
        points.extend((x, y) for x in _ref_columns(y))
    holes = []
    for x, y in points:
        holes.append(
            RackHole(
                cx=int(round(x * sx)),
                cy=int(round(y * sy)),
                rx=max(4, int(round(rx * sx))),
                ry=max(3, int(round(ry * sy))),
            )
        )
    return holes


def _rack_xywh(data: dict) -> tuple[int, int, int, int]:
    """Prefer top-level x/y/w/h; fall back to nested roi (capture agents schema)."""
    if all(k in data for k in ("x", "y", "w", "h")):
        return int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"])
    nested = data.get("roi") or {}
    if all(k in nested for k in ("x", "y", "w", "h")):
        return int(nested["x"]), int(nested["y"]), int(nested["w"]), int(nested["h"])
    raise KeyError("rack.json missing ROI (need x/y/w/h or roi.{x,y,w,h})")


def load_rack(path: Path = RACK_CONFIG) -> RackROI | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rx = int(data.get("slot_rx", 8))
    ry = int(data.get("slot_ry", 5))
    holes = []
    for item in data.get("holes", []):
        holes.append(
            RackHole(
                cx=int(item["cx"]),
                cy=int(item["cy"]),
                rx=int(item.get("rx", rx)),
                ry=int(item.get("ry", ry)),
            )
        )
    x, y, w, h = _rack_xywh(data)
    frame = data.get("frame_size") or []
    image_width = int(data.get("image_width") or (frame[0] if len(frame) > 0 else 0))
    image_height = int(data.get("image_height") or (frame[1] if len(frame) > 1 else 0))
    return RackROI(
        x=x,
        y=y,
        w=w,
        h=h,
        camera_index=int(data.get("camera_index", FOCUS_CAMERA)),
        image_width=image_width,
        image_height=image_height,
        note=str(data.get("note", "")),
        holes=holes,
        slot_rx=rx,
        slot_ry=ry,
    )


def save_rack(rack: RackROI, path: Path = RACK_CONFIG) -> None:
    """Write focus ROI/holes; keep capture match/reference and sync nested roi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(existing)
    payload.update(
        {
            "camera_index": int(rack.camera_index),
            "x": int(rack.x),
            "y": int(rack.y),
            "w": int(rack.w),
            "h": int(rack.h),
            "image_width": int(rack.image_width),
            "image_height": int(rack.image_height),
            "frame_size": [int(rack.image_width), int(rack.image_height)],
            "slot_rx": int(rack.slot_rx),
            "slot_ry": int(rack.slot_ry),
            "note": rack.note,
            "holes": [
                {"cx": int(h.cx), "cy": int(h.cy), "rx": int(h.rx), "ry": int(h.ry)}
                for h in rack.holes
            ],
            "roi": {
                "x": int(rack.x),
                "y": int(rack.y),
                "w": int(rack.w),
                "h": int(rack.h),
                "comment": "Same crop as top-level x/y/w/h; rack mount does not move.",
            },
        }
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _mask_span(mask: np.ndarray) -> tuple[int, int]:
    count = int(np.count_nonzero(mask))
    if count == 0:
        return 0, 0
    xs = np.where(mask)[1]
    return count, int(xs.max() - xs.min() + 1)


def classify_lid_patch(bgr: np.ndarray) -> tuple[str, int, int]:
    """Classify one fixed hole. Closed = white lid, open = beige interior, else empty.

    Returns (state, white_pixel_count, beige_pixel_count).
    """
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return "empty", 0, 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    white = (val >= 140) & (sat <= 85)
    beige = (
        (hue >= 8)
        & (hue <= 40)
        & (sat >= 28)
        & (sat <= 150)
        & (val >= 100)
        & (val <= 185)
        & ~white
    )
    white_n, white_span = _mask_span(white)
    beige_n, beige_span = _mask_span(beige)
    peak = int(val.max())
    if white_n >= 12 and white_span >= 6 and peak >= 160:
        return "closed", white_n, beige_n
    if beige_n >= 12 and beige_span >= 6 and peak >= 110 and white_n < 12:
        return "open", white_n, beige_n
    return "empty", white_n, beige_n


def classify_holes(crop: np.ndarray, holes: list[RackHole]) -> list[LidHit]:
    height, width = crop.shape[:2]
    hits = []
    for hole in holes:
        x0 = max(0, int(hole.cx) - int(hole.rx))
        y0 = max(0, int(hole.cy) - int(hole.ry))
        x1 = min(width, int(hole.cx) + int(hole.rx))
        y1 = min(height, int(hole.cy) + int(hole.ry))
        state, white_n, beige_n = classify_lid_patch(crop[y0:y1, x0:x1])
        hits.append(LidHit(state, (x0, y0, x1 - x0, y1 - y0), white_n, beige_n))
    return hits


def render_rack(crop: np.ndarray, hits: list[LidHit], scale: int = 3) -> np.ndarray:
    """Crop-only preview. Closed is green, open is orange. Empty holes are not drawn."""
    scale = max(1, int(scale))
    view = cv2.resize(
        crop,
        (crop.shape[1] * scale, crop.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    colors = {"closed": (0, 220, 0), "open": (0, 140, 255)}
    for hit in hits:
        color = colors.get(hit.state)
        if color is None:
            continue
        x, y, w, h = hit.box
        x, y, w, h = x * scale, y * scale, w * scale, h * scale
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 1)
        cv2.putText(
            view,
            hit.state,
            (x, max(12, y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    closed = sum(1 for hit in hits if hit.state == "closed")
    opened = sum(1 for hit in hits if hit.state == "open")
    cv2.putText(
        view,
        f"closed {closed}   open {opened}",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return view


def match_rack_template(image: np.ndarray, reference: np.ndarray) -> tuple[int, int, int, int, float]:
    """Find the fixed rack by matching the reference crop. Does not scan for vials."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    rh, rw = ref.shape[:2]
    best = (-1.0, 0, 0, rw, rh, 1.0)
    for scale in np.linspace(0.7, 1.4, 22):
        tw = int(round(rw * float(scale)))
        th = int(round(rh * float(scale)))
        if tw < 30 or th < 30 or tw >= gray.shape[1] or th >= gray.shape[0]:
            continue
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        templ = cv2.resize(ref, (tw, th), interpolation=interp)
        score_map = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED)
        _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(score_map)
        if float(max_v) > best[0]:
            best = (float(max_v), int(max_l[0]), int(max_l[1]), tw, th, float(scale))
    score, x, y, w, h, _scale = best
    return x, y, w, h, score


def lock_rack_from_image(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    note: str,
) -> RackROI:
    x, y, w, h, score = match_rack_template(image, reference)
    height, width = image.shape[:2]
    holes = default_holes(w, h)
    full_note = f"{note} Template match score {score:.3f}."
    return RackROI(
        x=x,
        y=y,
        w=w,
        h=h,
        camera_index=FOCUS_CAMERA,
        image_width=width,
        image_height=height,
        note=full_note,
        holes=holes,
    )


def crop_rack(image: np.ndarray, rack: RackROI) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, rack.x))
    y0 = max(0, min(height - 1, rack.y))
    x1 = max(x0 + 1, min(width, rack.x + rack.w))
    y1 = max(y0 + 1, min(height, rack.y + rack.h))
    return image[y0:y1, x0:x1]


def save_slots(slots: list[Slot], path: Path = DEFAULT_CONFIG, slot_size: int = DEFAULT_SLOT_SIZE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "slot_size": int(slot_size),
        "slots": [
            {"name": s.name, "cx": int(s.cx), "cy": int(s.cy), "size": int(s.size)}
            for s in slots
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _white_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return ((v >= WHITE_V_MIN) & (s <= WHITE_S_MAX)).astype(np.uint8) * 255


def _circularity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri <= 1e-6 or area <= 1.0:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def classify_slot(image: np.ndarray, slot: Slot) -> SlotState:
    h, w = image.shape[:2]
    x, y, bw, bh = slot.roi(w, h)
    patch = image[y : y + bh, x : x + bw]
    if patch.size == 0:
        return SlotState(slot.name, "unknown", 0.0, 0.0, (x, y, bw, bh))

    mask = _white_mask(patch)
    white_frac = float(np.count_nonzero(mask)) / float(mask.size)
    circ = _circularity(mask)
    mean_v = float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]))

    if white_frac >= CLOSED_WHITE_FRAC and circ >= MIN_CIRCULARITY:
        state = "closed"
    elif white_frac <= OPEN_WHITE_FRAC and mean_v < 160:
        # Dark hole / empty rack pocket.
        state = "empty"
    elif white_frac < CLOSED_WHITE_FRAC:
        # Beige interior / rim visible — treated as open jar.
        state = "open"
    else:
        # Bright but not a clean disk (tilted lid, glare) — open-ish.
        state = "open" if circ < MIN_CIRCULARITY else "closed"

    return SlotState(slot.name, state, white_frac, circ, (x, y, bw, bh))


def classify_all(image: np.ndarray, slots: list[Slot]) -> list[SlotState]:
    return [classify_slot(image, slot) for slot in slots]


def annotate(image: np.ndarray, results: list[SlotState]) -> np.ndarray:
    view = image.copy()
    colors = {
        "closed": (0, 220, 0),
        "open": (0, 140, 255),
        "empty": (80, 80, 80),
        "unknown": (0, 255, 255),
    }
    for item in results:
        x, y, w, h = item.box
        color = colors.get(item.state, (0, 255, 255))
        cv2.rectangle(view, (x, y), (x + w, y + h), color, 2)
        label = f"{item.name}: {item.state}"
        cv2.putText(
            view,
            label,
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return view


def calibrate_image(image: np.ndarray, config_path: Path, slot_size: int) -> list[Slot]:
    """Click slot centers. ENTER/SPACE saves, U undoes, Q cancels."""
    clicks: list[tuple[int, int]] = []
    window = "rack calibrate — click centers, Enter save, U undo, Q quit"

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((int(x), int(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        view = image.copy()
        for i, (cx, cy) in enumerate(clicks):
            half = slot_size // 2
            cv2.rectangle(
                view,
                (cx - half, cy - half),
                (cx + half, cy + half),
                (0, 255, 255),
                2,
            )
            cv2.drawMarker(view, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(
                view,
                f"s{i}",
                (cx - half, max(16, cy - half - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            view,
            f"slots={len(clicks)}  Enter=save  U=undo  Q=quit",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter / Space
            break
        if key in (ord("u"), ord("U")) and clicks:
            clicks.pop()
        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(window)
            return []
    cv2.destroyWindow(window)
    slots = [
        Slot(name=f"s{i}", cx=cx, cy=cy, size=slot_size) for i, (cx, cy) in enumerate(clicks)
    ]
    save_slots(slots, config_path, slot_size=slot_size)
    print(f"Saved {len(slots)} slots -> {config_path}")
    return slots


def open_orbbec(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_OBSENSOR)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open Orbbec index {camera_index}")
    return cap


def read_color(cap) -> np.ndarray | None:
    """One Orbbec color frame. grab() then retrieve(None, CAP_OBSENSOR_BGR_IMAGE)."""
    if not cap.grab():
        return None
    ok, color = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
    if ok and color is not None and getattr(color, "size", 0) > 0:
        return color
    return None


def warm_color(cap, tries: int = 60) -> np.ndarray | None:
    """First Orbbec frames are often empty. Keep a non-black frame after a short warmup."""
    frame = None
    for i in range(max(1, int(tries))):
        color = read_color(cap)
        if color is not None and float(np.mean(color)) > 5:
            frame = color
            if i >= 8:
                return frame
        time.sleep(0.02)
    return frame


def cmd_calibrate(args: argparse.Namespace) -> int:
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            print(f"Could not read image: {args.image}")
            return 1
    else:
        cap = open_orbbec(args.camera)
        image = None
        for _ in range(60):
            image = read_color(cap)
            if image is not None:
                break
        cap.release()
        if image is None:
            print("No frame from camera.")
            return 1
    slots = calibrate_image(image, Path(args.config), args.slot_size)
    if not slots:
        print("Calibration cancelled.")
        return 1
    results = classify_all(image, slots)
    view = annotate(image, results)
    cv2.imshow("rack calibrate result", view)
    print("States:", ", ".join(f"{r.name}={r.state}" for r in results))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def cmd_still(args: argparse.Namespace) -> int:
    slots = load_slots(Path(args.config))
    if not slots:
        print(f"No slots in {args.config}. Run: python rack_lid_state.py calibrate ...")
        return 1
    image = cv2.imread(str(args.image))
    if image is None:
        print(f"Could not read image: {args.image}")
        return 1
    results = classify_all(image, slots)
    view = annotate(image, results)
    for r in results:
        print(
            f"{r.name}: {r.state}  white={r.white_frac:.2f}  circ={r.circularity:.2f}"
        )
    cv2.imshow("rack lid state", view)
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = CAPTURE_DIR / f"still_{stamp}.png"
            cv2.imwrite(str(path), view)
            print(f"Saved {path}")
    cv2.destroyAllWindows()
    return 0


def label_counts(hits: list[LidHit]) -> tuple[int, int]:
    closed = sum(1 for hit in hits if hit.state == "closed")
    opened = sum(1 for hit in hits if hit.state == "open")
    return closed, opened


def cmd_focus(args: argparse.Namespace) -> int:
    """Camera 1 only. Load rack.json, crop that ROI, label open/closed. Q quits."""
    rack = load_rack(Path(args.rack))
    if rack is None or not rack.holes:
        print(f"No locked rack in {args.rack}.")
        return 1
    cap = open_orbbec(FOCUS_CAMERA)
    window = "rack focus camera 1 (Q quits)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        if warm_color(cap) is None:
            print("No frame from camera 1.")
            return 1
        while True:
            color = read_color(cap)
            if color is None:
                continue
            crop = crop_rack(color, rack)
            hits = classify_holes(crop, rack.holes)
            view = render_rack(crop, hits, scale=3)
            closed, opened = label_counts(hits)
            cv2.putText(
                view,
                f"cam {FOCUS_CAMERA}  ROI {rack.x},{rack.y} {rack.w}x{rack.h}  Q quits",
                (8, view.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print(f"closed {closed}  open {opened}")
                break
            if key in (ord("s"), ord("S")):
                CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = CAPTURE_DIR / f"focus_{stamp}.png"
                cv2.imwrite(str(path), view)
                print(f"Saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    slots = load_slots(Path(args.config))
    if not slots:
        print(f"No slots in {args.config}. Run: python rack_lid_state.py calibrate --camera {args.camera}")
        return 1
    cap = open_orbbec(args.camera)
    window = "rack lid state (live)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            color = read_color(cap)
            if color is None:
                continue
            results = classify_all(color, slots)
            view = annotate(color, results)
            summary = " | ".join(f"{r.name}:{r.state}" for r in results)
            cv2.putText(
                view,
                summary[:120],
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = CAPTURE_DIR / f"live_{stamp}.png"
                cv2.imwrite(str(path), view)
                print(f"Saved {path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-slot rack lid open/closed classifier (not free-form YOLO)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Slot JSON (default {DEFAULT_CONFIG})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="Click slot centers and save slots.json")
    cal.add_argument("--image", type=Path, default=None, help="Still image (else live cam)")
    cal.add_argument("--camera", type=int, default=0)
    cal.add_argument("--slot-size", type=int, default=DEFAULT_SLOT_SIZE)
    cal.set_defaults(func=cmd_calibrate)

    still = sub.add_parser("still", help="Classify one still with saved slots")
    still.add_argument("--image", type=Path, required=True)
    still.set_defaults(func=cmd_still)

    live = sub.add_parser("live", help="Live Orbbec open/closed on saved click-slots")
    live.add_argument("--camera", type=int, default=FOCUS_CAMERA)
    live.set_defaults(func=cmd_live)

    focus = sub.add_parser(
        "focus",
        help="Camera 1 only: load rack.json, crop the fixed rack, label open/closed",
    )
    focus.add_argument(
        "--rack",
        type=Path,
        default=RACK_CONFIG,
        help=f"Locked rack ROI (default {RACK_CONFIG})",
    )
    focus.set_defaults(func=cmd_focus)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
