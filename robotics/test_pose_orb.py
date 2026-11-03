"""ORB texture-match tests. Empty crops soft-fail; a checkerboard cap matches."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import orb_texture as orb


def _textured_cap_pair():
    """Two similar crops with unique corners, shifted a couple of pixels."""
    image = np.full((180, 180, 3), 25, dtype=np.uint8)
    spots = (
        (30, 28, 6), (70, 40, 4), (120, 36, 8), (48, 90, 5), (100, 80, 7),
        (150, 70, 4), (36, 140, 9), (88, 130, 5), (140, 125, 6), (60, 160, 4),
    )
    for x, y, radius in spots:
        image[y - radius : y + radius, x - 2 : x + 2] = 255
        image[y - 2 : y + 2, x - radius : x + radius] = 255
    shifted = np.zeros_like(image)
    shifted[2:, 2:] = image[:-2, :-2]
    return image, shifted, (20, 20, 120, 120)


class OrbAssistTests(unittest.TestCase):
    def test_empty_match_soft_fails(self):
        blank = np.full((80, 80, 3), 40, dtype=np.uint8)
        keypoints, descriptors = orb.extract_orb_descriptors(blank, (10, 10, 40, 40))
        self.assertIsNone(keypoints)
        self.assertIsNone(descriptors)
        count, score = orb.match_orb_descriptors(None, None)
        self.assertEqual(count, 0)
        self.assertEqual(score, 0.0)
        self.assertIsNone(orb.orb_agree_label(None, None))
        self.assertEqual(orb.orb_cap_match_score(blank, (0, 0, 40, 40), blank, (0, 0, 40, 40)), 0.0)

    def test_textured_caps_match(self):
        left, right, box = _textured_cap_pair()
        _keypoints, left_desc = orb.extract_orb_descriptors(left, None)
        _keypoints, right_desc = orb.extract_orb_descriptors(right, None)
        self.assertIsNotNone(left_desc)
        self.assertIsNotNone(right_desc)
        count, score = orb.match_orb_descriptors(left_desc, right_desc)
        self.assertGreater(count, 0)
        self.assertGreater(score, 0.0)
        orb.extract_orb_descriptors(left, box)

    def test_pack_unpack_roundtrip(self):
        left, _right, _box = _textured_cap_pair()
        _keypoints, descriptors = orb.extract_orb_descriptors(left, None)
        packed = orb.pack_orb_descriptors(descriptors)
        self.assertIsNotNone(packed)
        assert packed is not None
        self.assertLessEqual(len(packed), orb.ORB_IPC_MAX_FEATURES)
        restored = orb.unpack_orb_descriptors(packed)
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.shape[1], 32)


if __name__ == "__main__":
    unittest.main()
