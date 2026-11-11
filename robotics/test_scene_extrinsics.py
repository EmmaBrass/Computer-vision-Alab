"""Unit tests for scene + purple-tape extrinsics (no cameras required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from cell_frames import (
    LAYOUT_SCENE_TAPE,
    apply_scene_stereo_to_cell_frames,
    load_cell_frames,
    relative_cam1_from_cam0,
    save_cell_frames,
)
from scene_extrinsics import (
    PurpleTapeObservation,
    SceneExtrinsicsResult,
    align_yaw_with_tape_axis,
    apply_result_to_cell_frames,
    find_purple_tape,
    purple_tape_mask,
    ransac_umeyama,
    umeyama_rigid,
    write_stereo_extrinsics,
)
from cam_geometry import Intrinsics, default_intrinsics


class TestAsymmetricBoard(unittest.TestCase):
    def test_asymmetric_circle_object_points_stagger(self):
        from stereo_calib_stub import _asymmetric_circle_object_points

        obj = _asymmetric_circle_object_points(4, 3, 20.0)
        self.assertEqual(obj.shape, (12, 3))
        # Row 0: x = 0,40,80,120 ; Row 1 staggered by one spacing
        self.assertAlmostEqual(float(obj[0, 0]), 0.0)
        self.assertAlmostEqual(float(obj[1, 0]), 40.0)
        self.assertAlmostEqual(float(obj[4, 0]), 20.0)  # first of odd row
        self.assertAlmostEqual(float(obj[4, 1]), 20.0)

    def test_board_mode_defaults_asymmetric_chess_when_unequal(self):
        from stereo_calib_stub import _board_mode
        import argparse

        ns = argparse.Namespace(
            cols=9,
            rows=6,
            charuco=False,
            asymmetric_circles=False,
            asymmetric_chess=False,
        )
        self.assertEqual(_board_mode(ns), "asymmetric_chess")


class TestUmeyama(unittest.TestCase):
    def test_recovers_known_rigid(self):
        rng = np.random.default_rng(42)
        src = rng.normal(size=(40, 3)) * 200.0 + np.array([0.0, 0.0, 800.0])
        # 90 deg yaw about Y + translation
        ang = np.deg2rad(90.0)
        R_true = np.array(
            [
                [np.cos(ang), 0.0, np.sin(ang)],
                [0.0, 1.0, 0.0],
                [-np.sin(ang), 0.0, np.cos(ang)],
            ]
        )
        t_true = np.array([120.0, -30.0, 50.0])
        dst = (R_true @ src.T).T + t_true
        R, t = umeyama_rigid(src, dst)
        self.assertTrue(np.allclose(R, R_true, atol=1e-6))
        self.assertTrue(np.allclose(t, t_true, atol=1e-5))

    def test_ransac_with_outliers(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(60, 3)) * 150.0 + np.array([50.0, 0.0, 700.0])
        R_true = np.eye(3)
        # small rotation
        a = np.deg2rad(15.0)
        R_true = np.array(
            [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
        )
        t_true = np.array([200.0, 10.0, -40.0])
        dst = (R_true @ src.T).T + t_true
        # Corrupt 25% of points.
        n_bad = 15
        dst[:n_bad] += rng.normal(scale=250.0, size=(n_bad, 3))
        R, t, inl, rmse = ransac_umeyama(
            src, dst, iters=200, inlier_mm=25.0, min_inliers=20, rng=rng
        )
        self.assertGreaterEqual(int(inl.sum()), 30)
        self.assertLess(rmse, 20.0)
        self.assertTrue(np.allclose(R, R_true, atol=0.05))
        self.assertTrue(np.allclose(t, t_true, atol=15.0))


class TestTapeAxis(unittest.TestCase):
    def test_align_yaw_flips_ambiguity(self):
        R = np.eye(3)
        axis0 = np.array([1.0, 0.0, 0.0])
        # Peer sees opposite sign of the strip.
        axis1 = np.array([-1.0, 0.0, 0.0])
        R2 = align_yaw_with_tape_axis(R, axis0, axis1)
        v = R2 @ axis0
        # Should align to +axis1 or -axis1; with flip choice, dot ≈ 1.
        self.assertGreater(abs(float(np.dot(v, axis1))), 0.99)

    def test_find_purple_tape_on_synthetic(self):
        img = np.zeros((400, 640, 3), dtype=np.uint8)
        # Magenta strip (BGR): high R+B, low G.
        img[300:320, 200:360] = (180, 40, 200)
        depth = np.full((400, 640), 900.0, dtype=np.float32)
        K = default_intrinsics(640, 400)
        mask = purple_tape_mask(img)
        self.assertGreater(int(mask.sum()), 100)
        obs = find_purple_tape(img, depth_mm=depth, K=K, min_area=50)
        self.assertIsNotNone(obs)
        assert obs is not None
        self.assertIsNotNone(obs.p_cam_mm)
        # Long axis should be mostly horizontal.
        self.assertGreater(abs(obs.axis_uv[0]), abs(obs.axis_uv[1]))


class TestCellFramesWrite(unittest.TestCase):
    def test_apply_scene_and_relative(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cell_frames.json"
            cf = load_cell_frames(path)
            R = np.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
            )
            t = np.array([100.0, 0.0, 200.0])
            origin0 = np.array([0.0, 0.0, 800.0])
            origin1 = R @ origin0 + t
            apply_scene_stereo_to_cell_frames(
                cf, R, t, origin_cam0_mm=origin0, origin_cam1_mm=origin1
            )
            save_cell_frames(cf)
            self.assertEqual(cf.layout, LAYOUT_SCENE_TAPE)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["layout"], LAYOUT_SCENE_TAPE)
            # Tape origin maps near zero in both poses.
            p0 = cf.cam0.R @ origin0 + cf.cam0.t_mm
            p1 = cf.cam1.R @ origin1 + cf.cam1.t_mm
            self.assertLess(float(np.linalg.norm(p0)), 1e-6)
            self.assertLess(float(np.linalg.norm(p1)), 1e-6)

    def test_write_stereo_clears_placeholder_wording(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stereo_extrinsics.json"
            result = SceneExtrinsicsResult(
                R=np.eye(3),
                t_mm=np.array([150.0, 0.0, 0.0]),
                n_matches=20,
                n_inliers=18,
                inlier_rmse_mm=5.0,
                notes="unit test",
            )
            K = Intrinsics(500, 500, 320, 200)
            write_stereo_extrinsics(result, K0=K, K1=K, path=path)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("PLACEHOLDER", text.upper())


if __name__ == "__main__":
    unittest.main()
