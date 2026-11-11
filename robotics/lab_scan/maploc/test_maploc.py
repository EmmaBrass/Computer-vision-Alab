"""Round-trip a walkthrough frame through localization, and fuse a synthetic pair."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

import fuse
import localize
from colmap_io import read_images
from geometry import backproject, camera_center, rotation_angle_deg, world_to_cam

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
# A middle frame whose depth ratio sat on the global scale, so the 5 cm gate is meaningful.
QUERY_NAME = "000248.jpg"


class RoundTripTest(unittest.TestCase):
    def test_excluded_frame_returns_near_its_colmap_pose(self) -> None:
        config = localize.load_config(CONFIG)
        images = read_images(Path(config["paths"]["images_bin"]))
        self.assertIn(QUERY_NAME, images)
        query = cv2.imread(str(Path(config["paths"]["color_dir"]) / QUERY_NAME), cv2.IMREAD_COLOR)
        self.assertIsNotNone(query)
        cv2.setRNGSeed(0)
        pose = localize.localize_still(
            query,
            config["walkthrough"],
            config,
            exclude_names={QUERY_NAME},
        )
        gt = images[QUERY_NAME]
        center = np.asarray(pose["camera_center_m"], dtype=np.float64)
        delta = center - gt.center
        dist = float((delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) ** 0.5)
        R = np.asarray(pose["R"], dtype=np.float64)
        angle = rotation_angle_deg(R, gt.R)
        print(f"round-trip center error {dist * 100:.2f} cm, angle {angle:.3f} deg", flush=True)
        self.assertLess(dist, 0.05)
        self.assertLess(angle, 2.0)
        self.assertGreaterEqual(pose["inliers"], 50)
        self.assertLessEqual(pose["median_reproj_px"], 3.0)
        self.assertNotIn(QUERY_NAME, pose["retrieved_frames"])


class FusionTest(unittest.TestCase):
    def test_same_point_in_two_cameras_becomes_one_track(self) -> None:
        config = fuse.load_config(CONFIG)
        # Two pinhole cameras looking along +z. A vial at (1, 0, 2) and a box that only cam1 sees.
        intr = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "has_depth": False}
        R = np.eye(3, dtype=np.float64)
        cam1_t = np.zeros(3, dtype=np.float64)
        cam2_center = np.array([0.4, 0.0, 0.0], dtype=np.float64)
        cam2_t = -cam2_center  # R is I, so t = -center
        poses = {
            "cam1": {"R": R, "t": cam1_t, "camera_center_m": camera_center(R, cam1_t)},
            "cam2": {"R": R.copy(), "t": cam2_t, "camera_center_m": camera_center(R, cam2_t)},
        }
        cameras_cfg = {"cam1": intr, "cam2": dict(intr)}
        truth = np.array([1.0, 0.0, 2.0], dtype=np.float64)
        other = np.array([-0.5, 0.3, 1.5], dtype=np.float64)

        def pack(label: str, xyz: np.ndarray, pose: dict, score: float) -> dict:
            x_cam = world_to_cam(pose["R"], pose["t"], xyz.reshape(1, 3))[0]
            u = intr["fx"] * float(x_cam[0]) / float(x_cam[2]) + intr["cx"]
            v = intr["fy"] * float(x_cam[1]) / float(x_cam[2]) + intr["cy"]
            return {"label": label, "u": u, "v": v, "depth_m": float(x_cam[2]), "score": score}

        book = fuse.TrackBook(float(config["fusion"]["track_gate_m"]))
        first = fuse.step(
            {
                "cam1": [pack("vial", truth, poses["cam1"], 0.9), pack("box", other, poses["cam1"], 0.5)],
                "cam2": [pack("vial", truth + np.array([0.02, 0.0, 0.0]), poses["cam2"], 0.8)],
            },
            poses,
            cameras_cfg,
            config,
            book,
        )
        vials = [item for item in first if item["label"] == "vial"]
        boxes = [item for item in first if item["label"] == "box"]
        self.assertEqual(len(vials), 1)
        self.assertEqual(sorted(vials[0]["contributing_cameras"]), ["cam1", "cam2"])
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["contributing_cameras"], ["cam1"])
        got = np.asarray(vials[0]["xyz_m"], dtype=np.float64)
        # Score-weighted mean of truth and truth+2cm, weights 0.9 and 0.8.
        err = got - truth
        dist = float((err[0] ** 2 + err[1] ** 2 + err[2] ** 2) ** 0.5)
        self.assertLess(dist, 0.03)

        moved = truth + np.array([0.01, -0.01, 0.0])
        second = fuse.step(
            {
                "cam1": [pack("vial", moved, poses["cam1"], 1.0)],
                "cam2": [pack("vial", moved, poses["cam2"], 1.0)],
            },
            poses,
            cameras_cfg,
            config,
            book,
        )
        vials2 = [item for item in second if item["label"] == "vial"]
        self.assertEqual(len(vials2), 1)
        self.assertEqual(vials2[0]["id"], vials[0]["id"])

        # The back-project of the packed detection must return the camera-frame point we started from.
        packed = pack("vial", truth, poses["cam1"], 1.0)
        x_cam = backproject(packed["u"], packed["v"], packed["depth_m"], 600.0, 600.0, 320.0, 240.0)
        self.assertAlmostEqual(float(x_cam[2]), packed["depth_m"], places=6)


if __name__ == "__main__":
    unittest.main()
