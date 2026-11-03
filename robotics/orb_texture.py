"""ORB texture matching for a crop or a full frame.

Clear surfaces often have few corners. A weak match returns an empty result
and never invents a box.
"""

from __future__ import annotations

import cv2
import numpy as np

ORB_MAX_FEATURES = 32
ORB_IPC_MAX_FEATURES = 16
ORB_RATIO_TEST = 0.75
ORB_AGREE_GOOD_MATCHES = 5
ORB_AGREE_SCORE_MIN = 0.22


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(width - 1, round(x)))
    y = max(0, min(height - 1, round(y)))
    w = max(1, min(width - x, round(max(1, w))))
    h = max(1, min(height - y, round(max(1, h))))
    return x, y, w, h


def _brighter_end(image, gray, box) -> np.ndarray:
    """Crop the brighter end of a box so matching uses the textured cap rim."""
    height, width = gray.shape[:2]
    x, y, w, h = _clip_box(box, width, height)
    crop = image[y : y + h, x : x + w]
    if crop.size == 0 or crop.ndim != 3:
        return gray[y : y + h, x : x + w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= 60) & (hsv[:, :, 2] >= 160)
    if not white.any():
        return gray[y : y + h, x : x + w]
    horizontal = w >= h
    if horizontal:
        band = max(4, int(w * 0.38))
        left = float(white[:, :band].mean())
        right = float(white[:, -band:].mean())
        x1, x2 = (x, x + band) if left >= right else (x + w - band, x + w)
        y1, y2 = y, y + h
    else:
        band = max(4, int(h * 0.38))
        top = float(white[:band].mean())
        bottom = float(white[-band:].mean())
        y1, y2 = (y, y + band) if top >= bottom else (y + h - band, y + h)
        x1, x2 = x, x + w
    pad = max(2, min(w, h) // 10)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return gray[y : y + h, x : x + w]
    return gray[y1:y2, x1:x2]


def extract_orb_descriptors(
    image,
    box=None,
    *,
    max_features: int = ORB_MAX_FEATURES,
) -> tuple[list | None, np.ndarray | None]:
    """ORB on a crop or the full frame. Soft-fail returns (None, None)."""
    try:
        if image is None or image.size == 0:
            return None, None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        roi = gray if box is None else _brighter_end(image, gray, box)
        if roi.size == 0 or min(roi.shape[:2]) < 8:
            return None, None
        keypoints, descriptors = cv2.ORB_create(nfeatures=int(max_features)).detectAndCompute(roi, None)
        if descriptors is None or len(keypoints) < 2:
            return None, None
        return keypoints, descriptors
    except cv2.error:
        return None, None


def pack_orb_descriptors(descriptors: np.ndarray | None) -> list[list[int]] | None:
    """Serialize ORB uint8 descriptors. Keeps at most ORB_IPC_MAX_FEATURES rows."""
    if descriptors is None or getattr(descriptors, "size", 0) == 0:
        return None
    rows = np.asarray(descriptors, dtype=np.uint8)
    if rows.ndim != 2 or rows.shape[1] != 32:
        return None
    return rows[:ORB_IPC_MAX_FEATURES].tolist()


def unpack_orb_descriptors(payload: list[list[int]] | None) -> np.ndarray | None:
    if not payload:
        return None
    try:
        arr = np.asarray(payload, dtype=np.uint8)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] != 32 or arr.shape[0] == 0:
        return None
    return arr


def match_orb_descriptors(
    desc_a: np.ndarray | None,
    desc_b: np.ndarray | None,
    *,
    ratio: float = ORB_RATIO_TEST,
) -> tuple[int, float]:
    """Hamming match with Lowe's ratio test. Returns (good count, score 0..1)."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return 0, 0.0
    try:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False).knnMatch(desc_a, desc_b, k=2)
    except cv2.error:
        return 0, 0.0
    good = 0
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < float(ratio) * second.distance:
            good += 1
    return good, float(min(1.0, good / 12.0))


def orb_cap_match_score(image_a, box_a, image_b, box_b) -> float:
    """0..1 match between two crops. A weak match is 0."""
    _, left = extract_orb_descriptors(image_a, box_a)
    _, right = extract_orb_descriptors(image_b, box_b)
    _count, score = match_orb_descriptors(left, right)
    return score


def orb_agree_label(
    local_desc: np.ndarray | None,
    peer_desc: np.ndarray | None,
) -> str | None:
    """Short label when two descriptor sets agree. None when the match is weak."""
    count, score = match_orb_descriptors(local_desc, peer_desc)
    if count >= ORB_AGREE_GOOD_MATCHES and score >= ORB_AGREE_SCORE_MIN:
        return f"orb agree ({count})"
    return None
