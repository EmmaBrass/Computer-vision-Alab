"""Multi-agent Orbbec capture: keep only closed/home rack frames (camera 1).

Alongside rack_lid_state.py (per-slot open/closed). This pipeline does NOT run
free-form object labelling — it only asks: does this frame look like
rack_lid_state/reference_closed.png (front view, trays retracted)?

Agents (separate processes, named in logs):
  CaptureAgent  — Orbbec CAP_OBSENSOR grab/retrieve on an interval
  MatchAgent    — ROI crop + gray NCC + edge NCC + histogram vs reference
  KeeperAgent   — write keepers / capped rejects + score log

Examples:
  cd D:\\Jessica\\Jessica\\robotics
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe rack_capture_agents.py still
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe rack_capture_agents.py run --camera 1 --seconds 30
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe rack_capture_agents.py refresh-open-template --camera 1
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe rack_capture_agents.py run --camera 1 --seconds 20 --label open
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe rack_capture_agents.py judge --dir rack_lid_state/keeps
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe rack_capture_agents.py judge --image path.png
  C:\\Users\\Ceder-ALAB\\anaconda3\\python.exe rack_capture_agents.py judge --live --camera 1 --seconds 10

JudgeAgent (asymmetric):
  CLOSED = single canonical home/retracted look (strict closed template match).
  OPEN   = variable shelf/tray positions (looser max-similarity to open exemplars,
           or not-closed + weak open cue). Uses keeps/open_*.png + template_open_roi.png.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "rack_lid_state"
DEFAULT_RACK_JSON = STATE_DIR / "rack.json"
DEFAULT_REFERENCE = STATE_DIR / "reference_closed.png"
DEFAULT_REFERENCE_OPEN = STATE_DIR / "template_open_roi.png"
KEEPS_DIR = STATE_DIR / "keeps"
REJECTS_DIR = STATE_DIR / "rejects"
SCORE_LOG = STATE_DIR / "capture_scores.jsonl"
VALID_LABELS = ("closed", "open")

# Asymmetric judge defaults:
#   CLOSED — single canonical home look (strict).
#   OPEN   — variable shelf/tray positions (looser, multi-exemplar).
DEFAULT_CLOSED_JUDGE_THRESHOLD = 0.80
DEFAULT_OPEN_JUDGE_THRESHOLD = 0.58
DEFAULT_OPEN_WEAK_THRESHOLD = 0.45
DEFAULT_CLOSED_NOT_CLOSED_MAX = 0.70
DEFAULT_MAX_OPEN_EXEMPLARS = 24
JUDGE_THRESHOLDS_PATH = STATE_DIR / "judge_thresholds.json"

# Orbbec first frames are often empty — warm many grabs (same idea as depthcamera).
WARMUP_GRABS = 60
DEFAULT_INTERVAL_S = 0.35
DEFAULT_REJECT_CAP = 40
SENTINEL = None


@dataclass
class RackConfig:
    camera_index: int
    roi: tuple[int, int, int, int]  # x, y, w, h
    reference: Path
    work_size: tuple[int, int]  # w, h
    threshold: float
    blur_min: float
    weights: tuple[float, float, float]  # gray, edge, hist
    frame_size: tuple[int, int] | None = None


def _agent_log(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s [{name}] %(levelname)s %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# Canonical camera-1 rack crop (shared with rack_lid_state focus).
DEFAULT_ROI = (226, 55, 137, 152)


def _roi_from_rack_data(data: dict[str, Any]) -> tuple[int, int, int, int]:
    """Read one ROI: prefer top-level x/y/w/h (focus), else nested roi (capture)."""
    if all(k in data for k in ("x", "y", "w", "h")):
        return (
            int(data["x"]),
            int(data["y"]),
            int(data["w"]),
            int(data["h"]),
        )
    nested = data.get("roi") or {}
    if all(k in nested for k in ("x", "y", "w", "h")):
        return (
            int(nested["x"]),
            int(nested["y"]),
            int(nested["w"]),
            int(nested["h"]),
        )
    return DEFAULT_ROI


def load_rack_config(path: Path = DEFAULT_RACK_JSON) -> RackConfig:
    data: dict[str, Any] = {}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    match = data.get("match") or {}
    weights = match.get("weights") or {}
    ref_name = data.get("reference", "template_roi.png")
    reference = Path(ref_name)
    if not reference.is_absolute():
        reference = STATE_DIR / reference
    if not reference.is_file():
        fallback = STATE_DIR / "reference_closed.png"
        if fallback.is_file():
            reference = fallback
    work = match.get("work_size") or [160, 180]
    frame = data.get("frame_size")
    if frame is None and data.get("image_width") and data.get("image_height"):
        frame = [data["image_width"], data["image_height"]]
    return RackConfig(
        camera_index=int(data.get("camera_index", 1)),
        roi=_roi_from_rack_data(data),
        reference=reference,
        work_size=(int(work[0]), int(work[1])),
        threshold=float(match.get("threshold", 0.70)),
        blur_min=float(match.get("blur_min", 35.0)),
        weights=(
            float(weights.get("gray", 0.45)),
            float(weights.get("edge", 0.35)),
            float(weights.get("hist", 0.20)),
        ),
        frame_size=(int(frame[0]), int(frame[1])) if frame else None,
    )


def save_rack_config(cfg: RackConfig, path: Path = DEFAULT_RACK_JSON) -> None:
    """Write capture fields; keep focus holes/note and sync top-level + nested ROI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    x, y, w, h = (int(v) for v in cfg.roi)
    frame = list(cfg.frame_size) if cfg.frame_size else existing.get("frame_size") or [640, 480]
    payload = dict(existing)
    payload.update(
        {
            "camera_index": int(cfg.camera_index),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "image_width": int(frame[0]),
            "image_height": int(frame[1]),
            "frame_size": [int(frame[0]), int(frame[1])],
            "roi": {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "comment": "Same crop as top-level x/y/w/h; rack mount does not move.",
            },
            "reference": cfg.reference.name,
            "reference_closed": existing.get("reference_closed", "reference_closed.png"),
            "reference_open": existing.get("reference_open", "template_open_roi.png"),
            "match": {
                "method": "ncc_gray+ncc_edge+hist",
                "work_size": list(cfg.work_size),
                "threshold": cfg.threshold,
                "blur_min": cfg.blur_min,
                "weights": {
                    "gray": cfg.weights[0],
                    "edge": cfg.weights[1],
                    "hist": cfg.weights[2],
                },
            },
        }
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    ih, iw = image.shape[:2]
    x0 = max(0, min(iw - 1, x))
    y0 = max(0, min(ih - 1, y))
    x1 = max(x0 + 1, min(iw, x0 + max(1, w)))
    y1 = max(y0 + 1, min(ih, y0 + max(1, h)))
    return image[y0:y1, x0:x1]


def embed_crop_in_frame(
    crop_bgr: np.ndarray,
    *,
    frame_size: tuple[int, int],
    place_roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Paste a cropped/overlay screenshot into a blank full frame so ROI scoring works.

    Operator screenshots are often zoomed rack crops (~200px), not 640x480 cam1
    frames. Placing them into display/rack ROI lets MatchScorer crop_roi behave.
    """
    fw, fh = int(frame_size[0]), int(frame_size[1])
    canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
    x, y, w, h = (int(v) for v in place_roi)
    x = max(0, min(fw - 1, x))
    y = max(0, min(fh - 1, y))
    w = max(1, min(fw - x, w))
    h = max(1, min(fh - y, h))
    resized = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_AREA)
    canvas[y : y + h, x : x + w] = resized
    return canvas


def maybe_fullframe_for_rack(
    bgr: np.ndarray,
    cfg: RackConfig,
    *,
    display_roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """If image looks like a crop (not cam full-frame), embed into a blank frame."""
    fw, fh = cfg.frame_size
    ih, iw = bgr.shape[:2]
    if iw >= int(fw * 0.85) and ih >= int(fh * 0.85):
        return bgr
    place = display_roi
    if place is None:
        # Prefer rack.json display_roi when present; else expand match ROI a bit.
        place = _load_display_roi(cfg) or (
            max(0, cfg.roi[0] - 40),
            max(0, cfg.roi[1] - 25),
            min(fw, cfg.roi[2] + 80),
            min(fh, cfg.roi[3] + 100),
        )
    return embed_crop_in_frame(bgr, frame_size=cfg.frame_size, place_roi=place)


def _load_display_roi(cfg: RackConfig) -> tuple[int, int, int, int] | None:
    path = DEFAULT_RACK_JSON
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    d = data.get("display_roi") or {}
    if not all(k in d for k in ("x", "y", "w", "h")):
        return None
    return (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))


def _to_work(bgr: np.ndarray, work_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(bgr, work_size, interpolation=cv2.INTER_AREA)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    a64 = a64 - a64.mean()
    b64 = b64 - b64.mean()
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a64, b64) / denom)


def _hist_corr_hsv(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2HSV)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2HSV)
    ha = cv2.calcHist([a], [0, 1], None, [32, 32], [0, 180, 0, 256])
    hb = cv2.calcHist([b], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class MatchResult:
    score: float
    gray: float
    edge: float
    hist: float
    blur: float
    keep: bool
    reason: str


class MatchScorer:
    """Compare a BGR frame (or crop) to a reference image. No YOLO."""

    def __init__(self, cfg: RackConfig):
        self.cfg = cfg
        ref = cv2.imread(str(cfg.reference))
        if ref is None:
            raise FileNotFoundError(f"Missing reference image: {cfg.reference}")
        self._set_ref(ref)

    @classmethod
    def from_bgr(cls, cfg: RackConfig, ref_bgr: np.ndarray) -> MatchScorer:
        """Build a scorer from an in-memory ROI crop (for open exemplars)."""
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj._set_ref(ref_bgr)
        return obj

    def _set_ref(self, ref: np.ndarray) -> None:
        self.ref_full = ref
        self.ref_work = _to_work(ref, self.cfg.work_size)
        self.ref_gray = cv2.cvtColor(self.ref_work, cv2.COLOR_BGR2GRAY)
        self.ref_edge = cv2.Canny(self.ref_gray, 60, 160)

    def score_bgr(self, bgr: np.ndarray, *, already_roi: bool = False) -> MatchResult:
        patch = bgr if already_roi else crop_roi(bgr, self.cfg.roi)
        if patch.size == 0:
            return MatchResult(0.0, 0.0, 0.0, 0.0, 0.0, False, "empty_crop")
        work = _to_work(patch, self.cfg.work_size)
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(gray, 60, 160)
        blur = laplacian_var(gray)
        g = _ncc(gray, self.ref_gray)
        e = _ncc(edge, self.ref_edge)
        h = _hist_corr_hsv(work, self.ref_work)
        # Clamp hist to [0,1] for weighted blend (correl can be slightly negative).
        h_clamped = max(0.0, min(1.0, h))
        wg, we, wh = self.cfg.weights
        score = wg * max(0.0, g) + we * max(0.0, e) + wh * h_clamped
        if blur < self.cfg.blur_min:
            return MatchResult(score, g, e, h, blur, False, "blurry")
        if score >= self.cfg.threshold:
            return MatchResult(score, g, e, h, blur, True, "match")
        return MatchResult(score, g, e, h, blur, False, "low_score")


@dataclass
class JudgeResult:
    label: str  # CLOSED | OPEN | UNKNOWN
    closed_score: float
    open_score: float
    blur: float
    reason: str
    open_exemplar: str | None = None


@dataclass
class JudgeThresholds:
    """Asymmetric thresholds: closed is strict; open is looser / multi-exemplar."""

    closed_threshold: float = DEFAULT_CLOSED_JUDGE_THRESHOLD
    open_threshold: float = DEFAULT_OPEN_JUDGE_THRESHOLD
    open_weak_threshold: float = DEFAULT_OPEN_WEAK_THRESHOLD
    closed_not_closed_max: float = DEFAULT_CLOSED_NOT_CLOSED_MAX
    blur_min: float = 35.0
    max_open_exemplars: int = DEFAULT_MAX_OPEN_EXEMPLARS


def load_judge_thresholds(
    path: Path = JUDGE_THRESHOLDS_PATH,
    *,
    blur_min: float | None = None,
) -> JudgeThresholds:
    """Load TrainAgent sidecar thresholds; fall back to asymmetric defaults."""
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    return JudgeThresholds(
        closed_threshold=float(data.get("closed_threshold", DEFAULT_CLOSED_JUDGE_THRESHOLD)),
        open_threshold=float(data.get("open_threshold", DEFAULT_OPEN_JUDGE_THRESHOLD)),
        open_weak_threshold=float(
            data.get("open_weak_threshold", DEFAULT_OPEN_WEAK_THRESHOLD)
        ),
        closed_not_closed_max=float(
            data.get("closed_not_closed_max", DEFAULT_CLOSED_NOT_CLOSED_MAX)
        ),
        blur_min=float(blur_min if blur_min is not None else data.get("blur_min", 35.0)),
        max_open_exemplars=int(data.get("max_open_exemplars", DEFAULT_MAX_OPEN_EXEMPLARS)),
    )


class RackStateJudge:
    """Classify rack frame as CLOSED | OPEN | UNKNOWN.

    CLOSED: only one allowed look — tight match to the closed home template.
    OPEN: shelves/trays may open independently — max score vs open exemplar set,
    or (not tightly closed) + a weaker open cue.
    """

    def __init__(
        self,
        cfg: RackConfig,
        *,
        closed_reference: Path | None = None,
        open_exemplars: list[Path] | None = None,
        thresholds: JudgeThresholds | None = None,
    ):
        self.cfg = cfg
        self.thresholds = thresholds or load_judge_thresholds(blur_min=cfg.blur_min)
        closed_path = Path(closed_reference) if closed_reference else self._resolve_closed_ref(cfg)
        if not closed_path.is_file():
            raise FileNotFoundError(f"Closed reference missing: {closed_path}")
        closed_cfg = RackConfig(
            camera_index=cfg.camera_index,
            roi=cfg.roi,
            reference=closed_path,
            work_size=cfg.work_size,
            threshold=self.thresholds.closed_threshold,
            blur_min=self.thresholds.blur_min,
            weights=cfg.weights,
            frame_size=cfg.frame_size,
        )
        self.closed_scorer = MatchScorer(closed_cfg)
        self.closed_path = closed_path
        self.open_scorers: list[tuple[str, MatchScorer]] = []
        for path in open_exemplars if open_exemplars is not None else self._discover_open_exemplars():
            path = Path(path)
            if not path.is_file():
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            # Keepers are full frames; templates are ROI crops; overlay screenshots
            # are zoomed crops — embed those into a blank cam frame first.
            fw, fh = cfg.frame_size
            if img.shape[1] < int(fw * 0.85) or img.shape[0] < int(fh * 0.85):
                if img.shape[0] <= cfg.roi[3] + 20 and img.shape[1] <= cfg.roi[2] + 20:
                    crop = img
                else:
                    img = maybe_fullframe_for_rack(img, cfg)
                    crop = crop_roi(img, cfg.roi)
            elif img.shape[0] > cfg.roi[3] + 20 or img.shape[1] > cfg.roi[2] + 20:
                crop = crop_roi(img, cfg.roi)
            else:
                crop = img
            tmp_cfg = RackConfig(
                camera_index=cfg.camera_index,
                roi=cfg.roi,
                reference=path,
                work_size=cfg.work_size,
                threshold=self.thresholds.open_threshold,
                blur_min=self.thresholds.blur_min,
                weights=cfg.weights,
                frame_size=cfg.frame_size,
            )
            self.open_scorers.append((path.name, MatchScorer.from_bgr(tmp_cfg, crop)))
        if not self.open_scorers:
            raise FileNotFoundError(
                "No open exemplars found. Capture open keepers or run refresh-open-template."
            )

    @staticmethod
    def _resolve_closed_ref(cfg: RackConfig) -> Path:
        candidates = [
            STATE_DIR / "reference_closed.png",
            STATE_DIR / "template_roi.png",
            cfg.reference,
        ]
        for path in candidates:
            if path is not None and Path(path).is_file():
                return Path(path)
        return STATE_DIR / "reference_closed.png"

    def _discover_open_exemplars(self) -> list[Path]:
        paths: list[Path] = []
        if DEFAULT_REFERENCE_OPEN.is_file():
            paths.append(DEFAULT_REFERENCE_OPEN)
        keeps = sorted(KEEPS_DIR.glob("open_*.png"))
        # Prefer newest keepers so fresh OPEN variants (tray-out looks) stay in set.
        cap = max(1, int(self.thresholds.max_open_exemplars))
        if keeps:
            newest = list(reversed(keeps))[:cap]
            # Keep a thin older sample so earlier open styles are not forgotten.
            older = keeps[: max(0, len(keeps) - cap)]
            step = max(1, len(older) // max(1, min(8, cap // 3 or 1))) if older else 1
            sampled_old = older[::step][: max(0, cap // 3)]
            paths.extend(newest)
            paths.extend(sampled_old)
        # Dedupe by resolved path; newest first already.
        seen: set[str] = set()
        out: list[Path] = []
        for p in paths:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
            if len(out) >= cap + (1 if DEFAULT_REFERENCE_OPEN.is_file() else 0):
                break
        return out

    def judge_bgr(
        self,
        bgr: np.ndarray,
        *,
        already_roi: bool = False,
        exclude_open_exemplar: str | None = None,
    ) -> JudgeResult:
        if not already_roi:
            bgr = maybe_fullframe_for_rack(bgr, self.cfg)
        closed = self.closed_scorer.score_bgr(bgr, already_roi=already_roi)
        best_open = -1.0
        best_name: str | None = None
        for name, scorer in self.open_scorers:
            # Leave-one-out when scoring a keeper that is also an exemplar.
            if exclude_open_exemplar and name == exclude_open_exemplar:
                continue
            hit = scorer.score_bgr(bgr, already_roi=already_roi)
            if hit.score > best_open:
                best_open = hit.score
                best_name = name
        if best_open < 0.0:
            best_open = 0.0
        blur = closed.blur
        th = self.thresholds
        if blur < th.blur_min:
            return JudgeResult("UNKNOWN", closed.score, best_open, blur, "blurry", best_name)
        # Strict closed: only the single home look qualifies as CLOSED.
        if closed.score >= th.closed_threshold:
            return JudgeResult(
                "CLOSED",
                closed.score,
                best_open,
                blur,
                "strict_closed_match",
                best_name,
            )
        # Looser open: max similarity to any open exemplar.
        if best_open >= th.open_threshold:
            return JudgeResult(
                "OPEN",
                closed.score,
                best_open,
                blur,
                "open_exemplar_match",
                best_name,
            )
        # Not closed + weak open cue (variable tray positions).
        if closed.score < th.closed_not_closed_max and best_open >= th.open_weak_threshold:
            return JudgeResult(
                "OPEN",
                closed.score,
                best_open,
                blur,
                "not_closed_plus_open_cue",
                best_name,
            )
        return JudgeResult(
            "UNKNOWN",
            closed.score,
            best_open,
            blur,
            "ambiguous",
            best_name,
        )


def open_orbbec(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_OBSENSOR)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open Orbbec index {camera_index} (CAP_OBSENSOR)")
    return cap


def grab_color(cap: cv2.VideoCapture) -> np.ndarray | None:
    """Frame grab must use grab + retrieve(None, CAP_OBSENSOR_BGR_IMAGE)."""
    if not cap.grab():
        return None
    ok, color = cap.retrieve(None, cv2.CAP_OBSENSOR_BGR_IMAGE)
    if ok and color is not None:
        return color
    return None


def warmup_camera(cap: cv2.VideoCapture, grabs: int = WARMUP_GRABS) -> np.ndarray | None:
    last = None
    for _ in range(int(grabs)):
        frame = grab_color(cap)
        if frame is not None:
            last = frame
        time.sleep(0.02)
    return last


# --- Agents -----------------------------------------------------------------


def capture_agent(
    camera_index: int,
    out_q: mp.Queue,
    stop_event: mp.Event,
    interval_s: float,
    warmup: int,
) -> None:
    log = _agent_log("CaptureAgent")
    cap = None
    try:
        log.info("opening Orbbec index=%s", camera_index)
        cap = open_orbbec(camera_index)
        log.info("warming up (%s grabs)...", warmup)
        first = warmup_camera(cap, warmup)
        if first is None:
            log.error("no frames during warmup — exiting")
            out_q.put(SENTINEL)
            return
        log.info("warmup ok frame=%sx%s", first.shape[1], first.shape[0])
        seq = 0
        next_t = time.monotonic()
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_t:
                # Keep streaming so buffer stays fresh.
                grab_color(cap)
                time.sleep(0.01)
                continue
            next_t = now + float(interval_s)
            frame = grab_color(cap)
            if frame is None:
                log.warning("grab failed seq=%s", seq)
                continue
            seq += 1
            out_q.put({"seq": seq, "t": time.time(), "bgr": frame})
            log.info("captured seq=%s shape=%sx%s", seq, frame.shape[1], frame.shape[0])
    except Exception as exc:  # noqa: BLE001 — surface to parent via log + sentinel
        log.exception("CaptureAgent failed: %s", exc)
    finally:
        if cap is not None:
            cap.release()
            log.info("camera released")
        try:
            out_q.put(SENTINEL)
        except Exception:
            pass


def match_agent(
    cfg_dict: dict[str, Any],
    in_q: mp.Queue,
    out_q: mp.Queue,
    stop_event: mp.Event,
) -> None:
    log = _agent_log("MatchAgent")
    raw = dict(cfg_dict)
    raw["reference"] = Path(raw["reference"])
    if raw.get("frame_size") is not None:
        raw["frame_size"] = tuple(raw["frame_size"])
    raw["roi"] = tuple(raw["roi"])
    raw["work_size"] = tuple(raw["work_size"])
    raw["weights"] = tuple(raw["weights"])
    cfg = RackConfig(**raw)
    try:
        scorer = MatchScorer(cfg)
    except Exception as exc:  # noqa: BLE001
        log.exception("MatchAgent init failed: %s", exc)
        out_q.put(SENTINEL)
        return
    log.info(
        "ready threshold=%.3f blur_min=%.1f roi=%s ref=%s",
        cfg.threshold,
        cfg.blur_min,
        cfg.roi,
        cfg.reference.name,
    )
    try:
        while not stop_event.is_set():
            try:
                item = in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is SENTINEL:
                out_q.put(SENTINEL)
                break
            result = scorer.score_bgr(item["bgr"], already_roi=False)
            out_q.put(
                {
                    "seq": item["seq"],
                    "t": item["t"],
                    "bgr": item["bgr"],
                    "score": result.score,
                    "gray": result.gray,
                    "edge": result.edge,
                    "hist": result.hist,
                    "blur": result.blur,
                    "keep": result.keep,
                    "reason": result.reason,
                }
            )
            log.info(
                "seq=%s keep=%s score=%.3f gray=%.3f edge=%.3f hist=%.3f blur=%.1f (%s)",
                item["seq"],
                result.keep,
                result.score,
                result.gray,
                result.edge,
                result.hist,
                result.blur,
                result.reason,
            )
    finally:
        try:
            out_q.put(SENTINEL)
        except Exception:
            pass
        log.info("stopped")


def keeper_agent(
    in_q: mp.Queue,
    stop_event: mp.Event,
    keeps_dir: str,
    rejects_dir: str,
    score_log: str,
    reject_cap: int,
    summary: dict[str, Any],
    keep_label: str = "closed",
) -> None:
    log = _agent_log("KeeperAgent")
    label_keep = str(keep_label).strip().lower()
    if label_keep not in VALID_LABELS:
        raise ValueError(f"keep_label must be one of {VALID_LABELS}, got {keep_label!r}")
    rack_state = label_keep.upper()
    keeps = Path(keeps_dir)
    rejects = Path(rejects_dir)
    keeps.mkdir(parents=True, exist_ok=True)
    rejects.mkdir(parents=True, exist_ok=True)
    log_path = Path(score_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    rejected = 0
    rejects_written = 0
    log.info(
        "writing keeps -> %s  rejects -> %s (cap=%s) keep_label=%s",
        keeps,
        rejects,
        reject_cap,
        label_keep,
    )
    try:
        while True:
            try:
                item = in_q.get(timeout=0.2)
            except queue.Empty:
                if stop_event.is_set():
                    # Drain briefly then exit if empty.
                    try:
                        item = in_q.get_nowait()
                    except Exception:
                        break
                else:
                    continue
            if item is SENTINEL:
                break
            stamp = time.strftime("%Y%m%d_%H%M%S")
            seq = int(item["seq"])
            label = label_keep if item["keep"] else "reject"
            name = f"{label}_{stamp}_seq{seq:04d}_s{item['score']:.3f}.png"
            record = {
                "seq": seq,
                "t": item["t"],
                "score": item["score"],
                "gray": item["gray"],
                "edge": item["edge"],
                "hist": item["hist"],
                "blur": item["blur"],
                "keep": item["keep"],
                "reason": item["reason"],
                "label": label,
                "rack_state": rack_state if item["keep"] else None,
                "file": name,
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            if item["keep"]:
                path = keeps / name
                cv2.imwrite(str(path), item["bgr"])
                sidecar = path.with_suffix(".json")
                sidecar.write_text(
                    json.dumps(
                        {
                            "label": label_keep,
                            "rack_state": rack_state,
                            "score": item["score"],
                            "seq": seq,
                            "file": name,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                kept += 1
                log.info("KEEP %s %s score=%.3f", label_keep, path.name, item["score"])
            else:
                rejected += 1
                if rejects_written < int(reject_cap):
                    path = rejects / name
                    cv2.imwrite(str(path), item["bgr"])
                    rejects_written += 1
                    log.info("REJECT %s score=%.3f (%s)", path.name, item["score"], item["reason"])
                else:
                    log.info(
                        "REJECT (not saved, cap) seq=%s score=%.3f (%s)",
                        seq,
                        item["score"],
                        item["reason"],
                    )
    finally:
        summary["kept"] = kept
        summary["rejected"] = rejected
        summary["rejects_written"] = rejects_written
        log.info("done kept=%s rejected=%s rejects_written=%s", kept, rejected, rejects_written)


def _cfg_to_dict(cfg: RackConfig) -> dict[str, Any]:
    return {
        "camera_index": cfg.camera_index,
        "roi": list(cfg.roi),
        "reference": str(cfg.reference),
        "work_size": list(cfg.work_size),
        "threshold": cfg.threshold,
        "blur_min": cfg.blur_min,
        "weights": list(cfg.weights),
        "frame_size": list(cfg.frame_size) if cfg.frame_size else None,
    }


def run_agents(
    *,
    camera: int,
    seconds: float,
    interval: float,
    reject_cap: int,
    rack_json: Path,
    warmup: int = WARMUP_GRABS,
    label: str = "closed",
    reference: Path | None = None,
) -> dict[str, int]:
    keep_label = str(label).strip().lower()
    if keep_label not in VALID_LABELS:
        raise ValueError(f"label must be one of {VALID_LABELS}, got {label!r}")

    cfg = load_rack_config(rack_json)
    cfg.camera_index = int(camera)
    if reference is not None:
        cfg.reference = Path(reference)
    elif keep_label == "open":
        open_ref = DEFAULT_REFERENCE_OPEN
        data: dict[str, Any] = {}
        if rack_json.is_file():
            data = json.loads(rack_json.read_text(encoding="utf-8"))
        named = data.get("reference_open")
        if named:
            candidate = Path(named)
            if not candidate.is_absolute():
                candidate = STATE_DIR / candidate
            if candidate.is_file():
                open_ref = candidate
        cfg.reference = open_ref

    if not cfg.reference.is_file():
        hint = (
            "Run: refresh-open-template (rack open now)."
            if keep_label == "open"
            else "Place reference_closed.png / template_roi.png under rack_lid_state/."
        )
        raise FileNotFoundError(f"Reference missing: {cfg.reference}. {hint}")

    KEEPS_DIR.mkdir(parents=True, exist_ok=True)
    REJECTS_DIR.mkdir(parents=True, exist_ok=True)

    # Manager dict so KeeperAgent can report counts back to parent.
    with mp.Manager() as manager:
        summary = manager.dict(kept=0, rejected=0, rejects_written=0)
        stop_event = mp.Event()
        q_cap_match: mp.Queue = mp.Queue(maxsize=8)
        q_match_keep: mp.Queue = mp.Queue(maxsize=8)

        procs = [
            mp.Process(
                target=capture_agent,
                name="CaptureAgent",
                args=(camera, q_cap_match, stop_event, interval, warmup),
            ),
            mp.Process(
                target=match_agent,
                name="MatchAgent",
                args=(_cfg_to_dict(cfg), q_cap_match, q_match_keep, stop_event),
            ),
            mp.Process(
                target=keeper_agent,
                name="KeeperAgent",
                args=(
                    q_match_keep,
                    stop_event,
                    str(KEEPS_DIR),
                    str(REJECTS_DIR),
                    str(SCORE_LOG),
                    reject_cap,
                    summary,
                    keep_label,
                ),
            ),
        ]
        log = _agent_log("Orchestrator")
        log.info(
            "starting agents camera=%s seconds=%s interval=%s threshold=%.3f label=%s ref=%s",
            camera,
            seconds,
            interval,
            cfg.threshold,
            keep_label,
            cfg.reference.name,
        )
        for p in procs:
            p.start()
        try:
            time.sleep(max(0.1, float(seconds)))
        finally:
            stop_event.set()
            # Allow CaptureAgent to finish and send sentinel.
            procs[0].join(timeout=15)
            procs[1].join(timeout=15)
            procs[2].join(timeout=15)
            for p in procs:
                if p.is_alive():
                    log.warning("terminating hung process %s", p.name)
                    p.terminate()
                    p.join(timeout=3)
        result = {
            "kept": int(summary.get("kept", 0)),
            "rejected": int(summary.get("rejected", 0)),
            "rejects_written": int(summary.get("rejects_written", 0)),
        }
        log.info(
            "finished kept=%s rejected=%s (rejects_written=%s) keeps=%s",
            result["kept"],
            result["rejected"],
            result["rejects_written"],
            KEEPS_DIR,
        )
        return result


def make_bad_crop(ref: np.ndarray) -> np.ndarray:
    """Deliberately bad view: corner scramble + blur — must reject."""
    h, w = ref.shape[:2]
    bad = np.zeros_like(ref)
    # Take a noisy corner / unrelated patch and stretch.
    y0, x0 = max(0, h // 6), max(0, w // 6)
    patch = ref[y0 : y0 + max(8, h // 3), x0 : x0 + max(8, w // 3)]
    if patch.size == 0:
        patch = np.full_like(ref, 40)
    bad = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
    bad = cv2.GaussianBlur(bad, (15, 15), 0)
    # Wipe white-lid pattern with a solid band (simulates arm / open tray blob).
    bad[h // 3 : 2 * h // 3, :] = (90, 90, 90)
    return bad


def cmd_still(args: argparse.Namespace) -> int:
    """Smoke test: reference vs itself (keep) and a bad crop (reject)."""
    cfg = load_rack_config(Path(args.rack_json))
    if args.threshold is not None:
        cfg.threshold = float(args.threshold)

    # Prefer user keep-look reference_closed.png for the self/bad smoke when present.
    user_ref = STATE_DIR / "reference_closed.png"
    if user_ref.is_file():
        cfg.reference = user_ref

    scorer = MatchScorer(cfg)
    ref = scorer.ref_full
    good = scorer.score_bgr(ref, already_roi=True)
    bad_img = make_bad_crop(ref)
    bad = scorer.score_bgr(bad_img, already_roi=True)

    print("=== still-mode smoke (no camera, no YOLO) ===")
    print(f"reference: {cfg.reference}")
    print(f"threshold: {cfg.threshold:.3f}  method: ncc_gray+ncc_edge+hist")
    print(
        f"SELF:  keep={good.keep} score={good.score:.3f} "
        f"gray={good.gray:.3f} edge={good.edge:.3f} hist={good.hist:.3f} blur={good.blur:.1f}"
    )
    print(
        f"BAD:   keep={bad.keep} score={bad.score:.3f} "
        f"gray={bad.gray:.3f} edge={bad.edge:.3f} hist={bad.hist:.3f} blur={bad.blur:.1f} ({bad.reason})"
    )

    # Live matcher uses ROI template from rack.json when different from user ref.
    live_cfg = load_rack_config(Path(args.rack_json))
    if args.threshold is not None:
        live_cfg.threshold = float(args.threshold)
    if live_cfg.reference.resolve() != cfg.reference.resolve() and live_cfg.reference.is_file():
        live_scorer = MatchScorer(live_cfg)
        live_self = live_scorer.score_bgr(live_scorer.ref_full, already_roi=True)
        print(
            f"LIVE_TMPL SELF: keep={live_self.keep} score={live_self.score:.3f} "
            f"ref={live_cfg.reference.name}"
        )

    # Optional: also score a full-frame probe if present (ROI path + live template).
    probe = STATE_DIR / "captures" / "probe_cam1.png"
    if probe.is_file():
        full = cv2.imread(str(probe))
        if full is not None:
            live_scorer = MatchScorer(live_cfg)
            p = live_scorer.score_bgr(full, already_roi=False)
            print(
                f"PROBE: keep={p.keep} score={p.score:.3f} "
                f"gray={p.gray:.3f} edge={p.edge:.3f} hist={p.hist:.3f} blur={p.blur:.1f} ({p.reason})"
            )

    ok = good.keep and (not bad.keep)
    if ok:
        print("PASS: self kept, bad rejected")
        return 0
    print("FAIL: expected self=keep and bad=reject")
    return 1


def _refresh_template_from_camera(
    *,
    rack_json: Path,
    camera: int | None,
    warmup: int,
    out_name: str,
    set_as_default_reference: bool,
    open_key: bool = False,
) -> int:
    """Grab one warmed frame, write ROI crop; optionally update rack.json pointers."""
    cfg = load_rack_config(rack_json)
    cam = int(camera if camera is not None else cfg.camera_index)
    log = _agent_log("RefreshTemplate")
    cap = None
    try:
        cap = open_orbbec(cam)
        log.info("warming camera %s...", cam)
        frame = warmup_camera(cap, int(warmup))
        if frame is None:
            print("No frame from camera.")
            return 1
        cfg.frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        crop = crop_roi(frame, cfg.roi)
        out = STATE_DIR / out_name
        cv2.imwrite(str(out), crop)
        if set_as_default_reference:
            cfg.reference = out
            save_rack_config(cfg, rack_json)
        else:
            # Keep closed default reference; only stamp open path / frame size.
            existing: dict[str, Any] = {}
            if rack_json.is_file():
                existing = json.loads(rack_json.read_text(encoding="utf-8"))
            if open_key:
                existing["reference_open"] = out.name
            existing["frame_size"] = list(cfg.frame_size)
            existing["image_width"] = int(cfg.frame_size[0])
            existing["image_height"] = int(cfg.frame_size[1])
            rack_json.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out} from camera {cam} ROI={cfg.roi}")
        return 0
    finally:
        if cap is not None:
            cap.release()
            log.info("camera released")


def cmd_refresh_template(args: argparse.Namespace) -> int:
    """Grab one warmed frame from camera and overwrite template_roi.png (+ ROI)."""
    return _refresh_template_from_camera(
        rack_json=Path(args.rack_json),
        camera=args.camera,
        warmup=int(args.warmup),
        out_name="template_roi.png",
        set_as_default_reference=True,
        open_key=False,
    )


def cmd_refresh_open_template(args: argparse.Namespace) -> int:
    """Grab live OPEN rack ROI -> template_open_roi.png (does not flip closed default)."""
    return _refresh_template_from_camera(
        rack_json=Path(args.rack_json),
        camera=args.camera,
        warmup=int(args.warmup),
        out_name="template_open_roi.png",
        set_as_default_reference=False,
        open_key=True,
    )


def cmd_run(args: argparse.Namespace) -> int:
    label = str(getattr(args, "label", "closed") or "closed").strip().lower()
    ref = Path(args.reference) if getattr(args, "reference", None) else None
    result = run_agents(
        camera=int(args.camera),
        seconds=float(args.seconds),
        interval=float(args.interval),
        reject_cap=int(args.reject_cap),
        rack_json=Path(args.rack_json),
        warmup=int(args.warmup),
        label=label,
        reference=ref,
    )
    print(
        f"Summary: kept={result['kept']} rejected={result['rejected']} "
        f"rejects_written={result['rejects_written']} label={label}"
    )
    print(f"Keepers: {KEEPS_DIR}")
    print(f"Rejects: {REJECTS_DIR}")
    print(f"Scores:  {SCORE_LOG}")
    return 0


def _build_judge(args: argparse.Namespace) -> RackStateJudge:
    cfg = load_rack_config(Path(args.rack_json))
    th = JudgeThresholds(
        closed_threshold=float(args.closed_threshold),
        open_threshold=float(args.open_threshold),
        open_weak_threshold=float(args.open_weak_threshold),
        closed_not_closed_max=float(args.closed_not_closed_max),
        blur_min=float(args.blur_min) if args.blur_min is not None else cfg.blur_min,
        max_open_exemplars=int(args.max_open_exemplars),
    )
    closed_ref = Path(args.closed_reference) if args.closed_reference else None
    return RackStateJudge(cfg, closed_reference=closed_ref, thresholds=th)


def _expected_label_from_name(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith("closed_"):
        return "CLOSED"
    if name.startswith("open_"):
        return "OPEN"
    return None


def cmd_judge(args: argparse.Namespace) -> int:
    """JudgeAgent: asymmetric CLOSED (strict) vs OPEN (looser multi-exemplar)."""
    judge = _build_judge(args)
    th = judge.thresholds
    print("=== JudgeAgent (asymmetric rack state) ===")
    print("CLOSED = single canonical home/retracted look (strict closed template).")
    print("OPEN   = variable shelf/tray positions (looser / multi-exemplar).")
    print(
        f"thresholds: closed>={th.closed_threshold:.2f}  "
        f"open>={th.open_threshold:.2f}  "
        f"weak_open>={th.open_weak_threshold:.2f} when closed<{th.closed_not_closed_max:.2f}"
    )
    print(f"closed_ref: {judge.closed_path.name}  open_exemplars: {len(judge.open_scorers)}")

    if args.live:
        return _cmd_judge_live(args, judge)

    paths: list[Path] = []
    if args.image:
        paths.append(Path(args.image))
    if args.dir:
        d = Path(args.dir)
        paths.extend(sorted(d.glob("*.png")))
    if not paths and args.smoke:
        paths.extend(sorted(KEEPS_DIR.glob("closed_*.png")))
        paths.extend(sorted(KEEPS_DIR.glob("open_*.png")))
    if not paths:
        print("Provide --image, --dir, --live, or --smoke.")
        return 2

    correct = 0
    total_labeled = 0
    counts = {"CLOSED": 0, "OPEN": 0, "UNKNOWN": 0}
    per_expected = {
        "CLOSED": {"ok": 0, "bad": 0, "unk": 0},
        "OPEN": {"ok": 0, "bad": 0, "unk": 0},
    }

    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"SKIP unreadable {path}")
            continue
        result = judge.judge_bgr(
            bgr,
            already_roi=False,
            exclude_open_exemplar=path.name if path.name.startswith("open_") else None,
        )
        counts[result.label] = counts.get(result.label, 0) + 1
        expected = _expected_label_from_name(path)
        mark = ""
        if expected:
            total_labeled += 1
            if result.label == expected:
                correct += 1
                per_expected[expected]["ok"] += 1
                mark = " OK"
            elif result.label == "UNKNOWN":
                per_expected[expected]["unk"] += 1
                mark = " UNK"
            else:
                per_expected[expected]["bad"] += 1
                mark = f" ERR(exp={expected})"
        print(
            f"{path.name}: {result.label}  closed={result.closed_score:.3f} "
            f"open={result.open_score:.3f} blur={result.blur:.1f} "
            f"({result.reason}"
            f"{', ex=' + result.open_exemplar if result.open_exemplar else ''})"
            f"{mark}"
        )

    print("---")
    print(f"judged={sum(counts.values())}  CLOSED={counts['CLOSED']} OPEN={counts['OPEN']} UNKNOWN={counts['UNKNOWN']}")
    if total_labeled:
        print(
            f"smoke accuracy: {correct}/{total_labeled} "
            f"({100.0 * correct / total_labeled:.1f}%) "
            f"closed_ok={per_expected['CLOSED']['ok']}/"
            f"{per_expected['CLOSED']['ok'] + per_expected['CLOSED']['bad'] + per_expected['CLOSED']['unk']} "
            f"open_ok={per_expected['OPEN']['ok']}/"
            f"{per_expected['OPEN']['ok'] + per_expected['OPEN']['bad'] + per_expected['OPEN']['unk']}"
        )
        print(
            f"  closed mistakes: bad={per_expected['CLOSED']['bad']} unk={per_expected['CLOSED']['unk']}  "
            f"open mistakes: bad={per_expected['OPEN']['bad']} unk={per_expected['OPEN']['unk']}"
        )
    return 0


def _cmd_judge_live(args: argparse.Namespace, judge: RackStateJudge) -> int:
    camera = int(args.camera)
    seconds = float(args.seconds)
    interval = float(args.interval)
    log = _agent_log("JudgeAgent")
    cap = None
    try:
        cap = open_orbbec(camera)
        log.info("warming camera %s...", camera)
        first = warmup_camera(cap, int(args.warmup))
        if first is None:
            print("No frames from camera.")
            return 1
        end_t = time.monotonic() + max(0.1, seconds)
        next_t = time.monotonic()
        seq = 0
        while time.monotonic() < end_t:
            frame = grab_color(cap)
            if frame is None:
                continue
            now = time.monotonic()
            if now < next_t:
                continue
            next_t = now + interval
            seq += 1
            result = judge.judge_bgr(frame, already_roi=False)
            print(
                f"seq={seq:04d} {result.label}  closed={result.closed_score:.3f} "
                f"open={result.open_score:.3f} ({result.reason})"
            )
        return 0
    finally:
        if cap is not None:
            cap.release()
            log.info("camera released")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-agent rack capture + asymmetric OPEN/CLOSED judge (no free-form YOLO). "
            "CLOSED = one home look (strict); OPEN = variable shelves (looser)."
        )
    )
    parser.add_argument(
        "--rack-json",
        type=Path,
        default=DEFAULT_RACK_JSON,
        help=f"ROI + match config (default {DEFAULT_RACK_JSON})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    still = sub.add_parser("still", help="Score reference vs self + bad crop (smoke)")
    still.add_argument("--threshold", type=float, default=None)
    still.set_defaults(func=cmd_still)

    refresh = sub.add_parser(
        "refresh-template",
        help="Overwrite template_roi.png from one live camera frame (lighting drift)",
    )
    refresh.add_argument("--camera", type=int, default=None)
    refresh.add_argument("--warmup", type=int, default=WARMUP_GRABS)
    refresh.set_defaults(func=cmd_refresh_template)

    refresh_open = sub.add_parser(
        "refresh-open-template",
        help="Write template_open_roi.png from live cam (rack OPEN now; keeps closed default)",
    )
    refresh_open.add_argument("--camera", type=int, default=None)
    refresh_open.add_argument("--warmup", type=int, default=WARMUP_GRABS)
    refresh_open.set_defaults(func=cmd_refresh_open_template)

    run = sub.add_parser("run", help="Launch Capture/Match/Keeper agents on live camera")
    run.add_argument("--camera", type=int, default=1, help="Orbbec index (default 1)")
    run.add_argument("--seconds", type=float, default=30.0, help="Capture duration")
    run.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S, help="Seconds between kept candidates")
    run.add_argument("--reject-cap", type=int, default=DEFAULT_REJECT_CAP, help="Max reject images to save")
    run.add_argument("--warmup", type=int, default=WARMUP_GRABS, help="Warm-up grab count")
    run.add_argument(
        "--label",
        choices=list(VALID_LABELS),
        default="closed",
        help="Keeper filename/sidecar label (closed|open). open uses template_open_roi.png",
    )
    run.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional explicit reference image path (overrides label default)",
    )
    run.set_defaults(func=cmd_run)

    judge_p = sub.add_parser(
        "judge",
        help=(
            "JudgeAgent: CLOSED (strict single home look) vs OPEN "
            "(looser multi-exemplar / not-closed+cue) vs UNKNOWN"
        ),
    )
    judge_p.add_argument("--image", type=Path, default=None, help="Single image to judge")
    judge_p.add_argument("--dir", type=Path, default=None, help="Directory of PNGs to judge")
    judge_p.add_argument(
        "--smoke",
        action="store_true",
        help="Judge keeps/closed_*.png and keeps/open_*.png; report accuracy",
    )
    judge_p.add_argument("--live", action="store_true", help="Judge live camera frames")
    judge_p.add_argument("--camera", type=int, default=1)
    judge_p.add_argument("--seconds", type=float, default=10.0)
    judge_p.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    judge_p.add_argument("--warmup", type=int, default=WARMUP_GRABS)
    judge_p.add_argument(
        "--closed-reference",
        type=Path,
        default=None,
        help="Override closed template (default reference_closed.png / template_roi.png)",
    )
    judge_p.add_argument(
        "--closed-threshold",
        type=float,
        default=DEFAULT_CLOSED_JUDGE_THRESHOLD,
        help=f"Strict CLOSED min score (default {DEFAULT_CLOSED_JUDGE_THRESHOLD})",
    )
    judge_p.add_argument(
        "--open-threshold",
        type=float,
        default=DEFAULT_OPEN_JUDGE_THRESHOLD,
        help=f"Looser OPEN max-exemplar min score (default {DEFAULT_OPEN_JUDGE_THRESHOLD})",
    )
    judge_p.add_argument(
        "--open-weak-threshold",
        type=float,
        default=DEFAULT_OPEN_WEAK_THRESHOLD,
        help=f"Weak open cue when not closed (default {DEFAULT_OPEN_WEAK_THRESHOLD})",
    )
    judge_p.add_argument(
        "--closed-not-closed-max",
        type=float,
        default=DEFAULT_CLOSED_NOT_CLOSED_MAX,
        help=f"closed_score below this + weak open => OPEN (default {DEFAULT_CLOSED_NOT_CLOSED_MAX})",
    )
    judge_p.add_argument("--blur-min", type=float, default=None)
    judge_p.add_argument(
        "--max-open-exemplars",
        type=int,
        default=DEFAULT_MAX_OPEN_EXEMPLARS,
        help="Cap on open keepers used as exemplars",
    )
    judge_p.set_defaults(func=cmd_judge)
    return parser


def main() -> None:
    # Windows spawn: freeze_support for multiprocessing entry.
    mp.freeze_support()
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
