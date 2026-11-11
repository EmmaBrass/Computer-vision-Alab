"""Smoke tests for closed/home multi-agent match scoring (no camera, no YOLO)."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

import rack_capture_agents as rca

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "rack_lid_state"


class RackCaptureAgentsTests(unittest.TestCase):
    def test_self_keep_bad_reject(self):
        ref_path = STATE / "reference_closed.png"
        self.assertTrue(ref_path.is_file(), "reference_closed.png missing")
        cfg = rca.load_rack_config()
        cfg.reference = ref_path
        scorer = rca.MatchScorer(cfg)
        good = scorer.score_bgr(scorer.ref_full, already_roi=True)
        bad = scorer.score_bgr(rca.make_bad_crop(scorer.ref_full), already_roi=True)
        self.assertTrue(good.keep, f"self should keep, score={good.score}")
        self.assertFalse(bad.keep, f"bad should reject, score={bad.score}")

    def test_template_and_keeper_roi(self):
        tmpl = STATE / "template_roi.png"
        if not tmpl.is_file():
            self.skipTest("template_roi.png not present")
        cfg = rca.load_rack_config()
        cfg.reference = tmpl
        scorer = rca.MatchScorer(cfg)
        self_hit = scorer.score_bgr(scorer.ref_full, already_roi=True)
        self.assertTrue(self_hit.keep, f"template self should keep, score={self_hit.score}")
        keeps = sorted((STATE / "keeps").glob("*.png"))
        if keeps:
            full = cv2.imread(str(keeps[-1]))
            self.assertIsNotNone(full)
            result = scorer.score_bgr(full, already_roi=False)
            self.assertTrue(
                result.keep,
                f"recent keeper should keep, score={result.score} {result.reason}",
            )

    def test_no_yolo_import(self):
        # Pipeline module must not pull ultralytics runtime.
        import rack_capture_agents as mod

        src = Path(mod.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("ultralytics", src)
        self.assertNotIn("import yolo", src)
        self.assertNotIn("from ultralytics", src)


if __name__ == "__main__":
    unittest.main()
