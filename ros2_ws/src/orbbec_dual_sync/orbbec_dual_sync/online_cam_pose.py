"""Continuous camera-pose estimate from the UR FK 3D model + tags/blues.

One ``OnlineCamPose`` per camera. Called from ``dual_view_node._overlay_ur_fk``.

The object model is always the live UR kinematic chain
(``fk_joint_positions_mm`` in robot base). Image evidence is ArUco joint tags
and/or baby-blue caps (optionally with depth → camera-frame 3D).

Flow:
  add_observations(fk, live tags, live blues) -> rolling buffer
  update() -> solvePnPRansac on tags (blues optional) + Kabsch 3D-3D on
              FK↔blue depth pairs (works even when tags are hidden)
  project() / draw uses the same K, dist as the solve

Rules of thumb:
  - Kabsch on tags is accepted only if its tag reprojection beats the current pose
  - Kabsch on FK+blue depth alone is accepted when RMS is small (folded/no-tag views)
  - Blues assign by projected FK when a pose exists; otherwise by FK-model 3D RANSAC
  - Blues stay out of solvePnPRansac unless use_blue_in_pnp is True
  - No ``@`` / ``np.dot`` / ``np.matmul`` (broken BLAS on this PC)

cell_frames convention: p_base = R p_cam + t_mm.
OpenCV solvePnP: p_cam = R_cv p_base + t_cv, so R = R_cv^T, t = -R_cv^T t_cv.
"""

from __future__ import annotations

import math
import random
import time
from typing import Optional

import cv2
import numpy as np

# Tag id i is FK joint i: 0 base, 1 shoulder, 2 elbow, 3 wrist1, 4 wrist2, 5 wrist3.
# FK-model bootstrap: max Kabsch RMS (mm) to accept a blue-depth-only pose.
_FK_BLUE_KABSCH_RMS_MM = 80.0
_FK_BLUE_RANSAC_TRIES = 80


def _mat3_mul(a, b):
    out = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for i in range(3):
        for j in range(3):
            out[i][j] = (
                float(a[i][0]) * float(b[0][j])
                + float(a[i][1]) * float(b[1][j])
                + float(a[i][2]) * float(b[2][j])
            )
    return out


def _mat3_vec(a, v):
    out = [0.0, 0.0, 0.0]
    for i in range(3):
        out[i] = (
            float(a[i][0]) * float(v[0])
            + float(a[i][1]) * float(v[1])
            + float(a[i][2]) * float(v[2])
        )
    return out


def _transpose3(a):
    return [[float(a[j][i]) for j in range(3)] for i in range(3)]


def _cv_from_cell(R_cell, t_cell):
    """cell (R,t) -> OpenCV (rvec, tvec). p_cam = R_cv p_base + t_cv."""
    R_cv = _transpose3(R_cell)
    t_cv = _mat3_vec(R_cv, (-float(t_cell[0]), -float(t_cell[1]), -float(t_cell[2])))
    rvec, _ = cv2.Rodrigues(np.asarray(R_cv, dtype=np.float64))
    return rvec, np.asarray(t_cv, dtype=np.float64).reshape(3, 1)


def _cell_from_cv(rvec, tvec):
    """OpenCV (rvec, tvec) -> cell (R, t). p_base = R p_cam + t."""
    R_cv, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    R_cv = [[float(R_cv[i, j]) for j in range(3)] for i in range(3)]
    t_cv = [float(v) for v in np.asarray(tvec, dtype=np.float64).reshape(3)]
    R = _transpose3(R_cv)
    t = _mat3_vec(R, (-t_cv[0], -t_cv[1], -t_cv[2]))
    return (
        np.asarray(R, dtype=np.float64),
        np.asarray(t, dtype=np.float64).reshape(3),
    )


def _rot_angle_deg(R_a, R_b):
    """Angle between two rotations, in degrees, with no matmul."""
    rel = _mat3_mul(_transpose3(R_a), R_b)
    trace = float(rel[0][0]) + float(rel[1][1]) + float(rel[2][2])
    c = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(c))


def _det3(R) -> float:
    return (
        float(R[0][0]) * (float(R[1][1]) * float(R[2][2]) - float(R[1][2]) * float(R[2][1]))
        - float(R[0][1]) * (float(R[1][0]) * float(R[2][2]) - float(R[1][2]) * float(R[2][0]))
        + float(R[0][2]) * (float(R[1][0]) * float(R[2][1]) - float(R[1][1]) * float(R[2][0]))
    )


def umeyama_cell(p_cam, p_base):
    """Rigid p_base = R p_cam + t from 3D-3D pairs (Kabsch / Umeyama).

    3x3 SVD is ``cv2.SVDecomp``; the products are explicit loops (no ``@``).
    Returns ``(R, t_mm, rms_mm)`` or None.
    """
    n = len(p_cam)
    if n < 3 or n != len(p_base):
        return None
    src = [[float(p[0]), float(p[1]), float(p[2])] for p in p_cam]
    dst = [[float(p[0]), float(p[1]), float(p[2])] for p in p_base]
    cs = [0.0, 0.0, 0.0]
    cd = [0.0, 0.0, 0.0]
    for i in range(n):
        for k in range(3):
            cs[k] += src[i][k]
            cd[k] += dst[i][k]
    inv = 1.0 / float(n)
    for k in range(3):
        cs[k] *= inv
        cd[k] *= inv
    H = np.zeros((3, 3), dtype=np.float64)
    for i in range(n):
        for r in range(3):
            for c in range(3):
                H[r, c] += (src[i][r] - cs[r]) * (dst[i][c] - cd[c])
    _w, u, vt = cv2.SVDecomp(H)
    U = [[float(u[i, j]) for j in range(3)] for i in range(3)]
    Vt = [[float(vt[i, j]) for j in range(3)] for i in range(3)]
    V = _transpose3(Vt)
    R = _mat3_mul(V, _transpose3(U))
    if _det3(R) < 0.0:
        V[0][2] = -V[0][2]
        V[1][2] = -V[1][2]
        V[2][2] = -V[2][2]
        R = _mat3_mul(V, _transpose3(U))
    Rc = _mat3_vec(R, cs)
    t = [cd[0] - Rc[0], cd[1] - Rc[1], cd[2] - Rc[2]]
    err = 0.0
    for i in range(n):
        pred = _mat3_vec(R, src[i])
        dx = pred[0] + t[0] - dst[i][0]
        dy = pred[1] + t[1] - dst[i][1]
        dz = pred[2] + t[2] - dst[i][2]
        err += (dx * dx + dy * dy + dz * dz) ** 0.5
    return (
        np.asarray(R, dtype=np.float64),
        np.asarray(t, dtype=np.float64).reshape(3),
        err / float(n),
    )


class OnlineCamPose:
    def __init__(
        self,
        K,
        dist=None,
        R=None,
        t_mm=None,
        *,
        cam: str = "",
        buffer_size: int = 200,
        min_pairs: int = 8,
        min_poses: int = 3,
        same_pose_mm: float = 20.0,
        fast_speed: float = 0.15,
        assign_px: float = 25.0,
        max_jump_mm: float = 150.0,
        max_jump_deg: float = 15.0,
        update_period_s: float = 1.0,
        use_blue: bool = True,
        use_blue_in_pnp: bool = True,
        min_3d_pairs: int = 3,
    ):
        self.cam = cam
        self.set_intrinsics(K, dist)
        self.R: Optional[np.ndarray] = None
        self.t: Optional[np.ndarray] = None
        if R is not None and t_mm is not None:
            self.R = np.asarray(R, dtype=np.float64).reshape(3, 3).copy()
            self.t = np.asarray(t_mm, dtype=np.float64).reshape(3).copy()
        self.buffer: list[dict] = []
        self.buffer_size = int(buffer_size)
        self.min_pairs = int(min_pairs)
        self.min_poses = int(min_poses)
        self.same_pose_mm = float(same_pose_mm)
        self.fast_speed = float(fast_speed)
        self.assign_px = float(assign_px)
        self.max_jump_mm = float(max_jump_mm)
        self.max_jump_deg = float(max_jump_deg)
        self.update_period_s = float(update_period_s)
        self.use_blue = bool(use_blue)
        self.use_blue_in_pnp = bool(use_blue_in_pnp)
        self.min_3d_pairs = int(min_3d_pairs)
        # Max-jump check stays off until a pose is accepted with tag error < 10 px.
        self._jump_armed = False
        self._last_fk: Optional[np.ndarray] = None
        self._next_pose_id = 0
        self._last_update_t = 0.0
        # HUD / status
        self.mean_px = float("nan")
        self.mean_3d_mm = float("nan")
        self.last_accepted = False
        self.n_pairs = 0
        self.n_poses = 0
        self.n_3d = 0
        self.last_mode = ""  # "tags" | "fk_blue_3d" | ""

    # -- intrinsics ---------------------------------------------------------
    def set_intrinsics(self, K, dist=None):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
        self.dist = (
            None if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1, 1)
        )

    @property
    def have_pose(self) -> bool:
        return self.R is not None and self.t is not None

    def reset(self) -> None:
        """Clear the buffer and pose so the next solve is tags-only / fresh."""
        self.buffer.clear()
        self.R = None
        self.t = None
        self._last_fk = None
        self._last_update_t = 0.0
        self._next_pose_id = 0
        self.n_poses = 0
        self.n_pairs = 0
        self.n_3d = 0
        self.mean_px = float("nan")
        self.mean_3d_mm = float("nan")
        self.last_accepted = False
        self.last_mode = ""
        self._jump_armed = False

    def _append_projected_blues(self, pairs, fk, blue_uv, xyz_list) -> None:
        """Assign blues to FK joints by projecting the 3D model with current pose."""
        proj = self.project(fk)
        blue_list = blue_uv
        if hasattr(blue_uv, "reshape"):
            arr = np.asarray(blue_uv, dtype=np.float64).reshape(-1, 2)
            blue_list = [(float(p[0]), float(p[1])) for p in arr]
        for bi, b in enumerate(blue_list):
            if b is None:
                continue
            bu, bv = float(b[0]), float(b[1])
            if not (math.isfinite(bu) and math.isfinite(bv)):
                continue
            best_i = -1
            best_d = self.assign_px
            for i, p in enumerate(proj):
                if p is None:
                    continue
                d = math.hypot(bu - float(p[0]), bv - float(p[1]))
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i < 0:
                continue
            cam_xyz = None
            if bi < len(xyz_list) and xyz_list[bi] is not None:
                xyz = xyz_list[bi]
                cam_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                if not all(math.isfinite(v) for v in cam_xyz):
                    cam_xyz = None
            pairs.append(
                {
                    "obj": (
                        float(fk[best_i, 0]),
                        float(fk[best_i, 1]),
                        float(fk[best_i, 2]),
                    ),
                    "img": (bu, bv),
                    "weight": 0.5,
                    "is_blue": True,
                    "cam_xyz": cam_xyz,
                }
            )

    def _append_fk_model_blues_3d(self, pairs, fk, blue_uv, xyz_list) -> None:
        """Cold-start: match blue depth points to the FK 3D model via RANSAC Kabsch.

        When joint tags are hidden (folded arm), the kinematic chain still gives
        known 3D joint locations in the robot base. Depth-backed blue caps give
        camera-frame 3D. A rigid fit between them localizes the arm in the view.
        """
        blues = []
        blue_list = blue_uv
        if hasattr(blue_uv, "reshape"):
            arr = np.asarray(blue_uv, dtype=np.float64).reshape(-1, 2)
            blue_list = [(float(p[0]), float(p[1])) for p in arr]
        for bi, b in enumerate(blue_list):
            if b is None or bi >= len(xyz_list) or xyz_list[bi] is None:
                continue
            bu, bv = float(b[0]), float(b[1])
            xyz = xyz_list[bi]
            cam_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            if not (
                math.isfinite(bu)
                and math.isfinite(bv)
                and all(math.isfinite(v) for v in cam_xyz)
            ):
                continue
            blues.append((bu, bv, cam_xyz))
        if len(blues) < self.min_3d_pairs:
            return
        # FK joints 1..5 are the blue caps (skip base pedestal / flange).
        fk_idx = [i for i in range(1, min(6, int(fk.shape[0])))]
        if len(fk_idx) < 3:
            return
        best_assign = None
        best_rms = float("inf")
        rng = random.Random(0xF11)
        n_b = len(blues)
        for _ in range(_FK_BLUE_RANSAC_TRIES):
            bi_pick = rng.sample(range(n_b), 3)
            fi_pick = rng.sample(fk_idx, 3)
            kabsch = umeyama_cell(
                [blues[i][2] for i in bi_pick],
                [
                    (float(fk[j, 0]), float(fk[j, 1]), float(fk[j, 2]))
                    for j in fi_pick
                ],
            )
            if kabsch is None:
                continue
            R, t, _ = kabsch
            R_l = [[float(R[r, c]) for c in range(3)] for r in range(3)]
            # Assign every blue to nearest transformed FK joint.
            assign = {}
            used = set()
            errs = []
            for bi, (_u, _v, cam_xyz) in enumerate(blues):
                p_base = _mat3_vec(R_l, cam_xyz)
                p_base = (
                    p_base[0] + float(t[0]),
                    p_base[1] + float(t[1]),
                    p_base[2] + float(t[2]),
                )
                best_j, best_d = -1, 120.0
                for j in fk_idx:
                    if j in used:
                        continue
                    d = math.sqrt(
                        (p_base[0] - float(fk[j, 0])) ** 2
                        + (p_base[1] - float(fk[j, 1])) ** 2
                        + (p_base[2] - float(fk[j, 2])) ** 2
                    )
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j < 0:
                    continue
                used.add(best_j)
                assign[bi] = best_j
                errs.append(best_d)
            if len(assign) < self.min_3d_pairs or not errs:
                continue
            rms = sum(errs) / float(len(errs))
            if rms < best_rms:
                best_rms = rms
                best_assign = assign
        if best_assign is None or best_rms > _FK_BLUE_KABSCH_RMS_MM:
            return
        for bi, j in best_assign.items():
            bu, bv, cam_xyz = blues[bi]
            pairs.append(
                {
                    "obj": (float(fk[j, 0]), float(fk[j, 1]), float(fk[j, 2])),
                    "img": (bu, bv),
                    "weight": 0.6,
                    "is_blue": True,
                    "cam_xyz": cam_xyz,
                }
            )

    # -- ingest -------------------------------------------------------------
    def add_observations(
        self, fk_points_mm, tag_uv, blue_uv, arm_speed=None, blue_xyz=None
    ) -> None:
        """Pair sticker/blue pixels with FK joints and push to the buffer.

        ``fk_points_mm`` is (7,3) in the robot base — the live UR 3D model.
        ``tag_uv`` is a list of 6 (u,v) or None. ``blue_uv`` is unlabeled pixels.
        ``blue_xyz`` is optional camera-frame 3D (mm) from depth. Fast arm motion
        is skipped. A still arm refreshes the last pose slot instead of dropping.
        """
        if arm_speed is not None and float(arm_speed) > self.fast_speed:
            return
        fk = np.asarray(fk_points_mm, dtype=np.float64).reshape(-1, 3)
        if fk.shape[0] < 6 or not np.isfinite(fk).all():
            return
        same_pose = False
        if self._last_fk is not None and self._last_fk.shape == fk.shape:
            moved = 0.0
            for i in range(fk.shape[0]):
                d = (
                    (fk[i, 0] - self._last_fk[i, 0]) ** 2
                    + (fk[i, 1] - self._last_fk[i, 1]) ** 2
                    + (fk[i, 2] - self._last_fk[i, 2]) ** 2
                ) ** 0.5
                if d > moved:
                    moved = d
            if moved < self.same_pose_mm:
                same_pose = True

        pairs: list[dict] = []
        # Tags: id i -> FK joint i (hard correspondence into the 3D model).
        if tag_uv is not None:
            for i, uv in enumerate(tag_uv):
                if uv is None or i >= fk.shape[0]:
                    continue
                u, v = float(uv[0]), float(uv[1])
                if not (math.isfinite(u) and math.isfinite(v)):
                    continue
                pairs.append(
                    {
                        "obj": (float(fk[i, 0]), float(fk[i, 1]), float(fk[i, 2])),
                        "img": (u, v),
                        "weight": 1.0,
                        "is_blue": False,
                        "cam_xyz": None,
                    }
                )

        xyz_list = list(blue_xyz) if blue_xyz is not None else []
        n_tags = sum(1 for pr in pairs if not pr["is_blue"])
        if self.use_blue and blue_uv is not None:
            if self.have_pose:
                allow = True
                if n_tags:
                    tag_err = self._mean_reproj(
                        *self._cv_pose(), [pr for pr in pairs if not pr["is_blue"]]
                    )
                    # Keep the strict gate when tags are good; if tags are bad /
                    # missing this frame, still use the FK model + blues.
                    allow = (not math.isfinite(tag_err)) or tag_err < 10.0 or n_tags < 2
                if allow:
                    self._append_projected_blues(pairs, fk, blue_uv, xyz_list)
            elif n_tags == 0:
                # No camera pose yet and no tags: locate via FK 3D model + depth.
                self._append_fk_model_blues_3d(pairs, fk, blue_uv, xyz_list)

        if not pairs:
            return
        if same_pose and self._next_pose_id > 0:
            pose_id = self._next_pose_id - 1
            self.buffer = [pr for pr in self.buffer if pr.get("pose_id") != pose_id]
        else:
            pose_id = self._next_pose_id
            self._next_pose_id += 1
        for pr in pairs:
            pr["pose_id"] = pose_id
            self.buffer.append(pr)
        while len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self._last_fk = fk.copy()

    # -- status helpers -----------------------------------------------------
    def _distinct_poses(self, use_blue: bool) -> int:
        ids = set()
        for pr in self.buffer:
            if pr["is_blue"] and not use_blue:
                continue
            ids.add(pr["pose_id"])
        return len(ids)

    def _mean_reproj(self, rvec, tvec, rows) -> float:
        if not rows:
            return float("inf")
        obj = np.asarray([r["obj"] for r in rows], dtype=np.float32).reshape(-1, 3)
        img = [r["img"] for r in rows]
        proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        flat = proj.reshape(-1, 2)
        err = 0.0
        for i in range(len(rows)):
            du = float(flat[i, 0]) - float(img[i][0])
            dv = float(flat[i, 1]) - float(img[i][1])
            err += (du * du + dv * dv) ** 0.5
        return err / float(len(rows))

    # -- solve --------------------------------------------------------------
    def _try_accept_kabsch(self, kabsch, tag_rows, blue_rows=None) -> bool:
        """Accept Umeyama: prefer tag reproj; else FK↔blue depth RMS."""
        if kabsch is None:
            return False
        Rk, tk = kabsch[0], kabsch[1]
        rms3 = float(kabsch[2]) if len(kabsch) > 2 else float("nan")
        if tag_rows:
            rvec_k, tvec_k = _cv_from_cell(
                [[float(Rk[i, j]) for j in range(3)] for i in range(3)],
                (float(tk[0]), float(tk[1]), float(tk[2])),
            )
            kabsch_err = self._mean_reproj(rvec_k, tvec_k, tag_rows)
            if not math.isfinite(kabsch_err):
                return False
            if self.have_pose:
                cur_err = self._mean_reproj(*self._cv_pose(), tag_rows)
                if math.isfinite(cur_err) and not (kabsch_err < cur_err):
                    return False
            self.mean_px = kabsch_err
            self.mean_3d_mm = rms3
            if self._accept_pose(Rk, tk):
                self.last_mode = "tags+3d" if blue_rows else "tags"
                return True
            return False
        # No tags: accept FK-model ↔ blue-depth fit when RMS is tight enough.
        if not math.isfinite(rms3) or rms3 > _FK_BLUE_KABSCH_RMS_MM:
            return False
        if not blue_rows or len(blue_rows) < self.min_3d_pairs:
            return False
        self.mean_px = float("nan")
        self.mean_3d_mm = rms3
        if self._accept_pose(Rk, tk):
            self.last_mode = "fk_blue_3d"
            return True
        return False

    def update(self, use_blue: Optional[bool] = None) -> bool:
        """Re-solve at most once per update period. Returns True if accepted."""
        now = time.monotonic()
        if now - self._last_update_t < self.update_period_s:
            return False
        self._last_update_t = now
        ub = self.use_blue if use_blue is None else bool(use_blue)
        rows = [pr for pr in self.buffer if ub or not pr["is_blue"]]
        rows3 = [pr for pr in rows if pr.get("cam_xyz") is not None]
        self.n_pairs = len(rows)
        self.n_poses = self._distinct_poses(ub)
        self.n_3d = len(rows3)
        self.last_accepted = False

        kabsch = None
        if len(rows3) >= self.min_3d_pairs:
            kabsch = umeyama_cell(
                [pr["cam_xyz"] for pr in rows3], [pr["obj"] for pr in rows3]
            )
            if kabsch is not None:
                self.mean_3d_mm = float(kabsch[2])

        tag_rows = [pr for pr in rows if not pr["is_blue"]]
        blue_rows = [pr for pr in rows if pr["is_blue"] and pr.get("cam_xyz") is not None]
        # Blues stay out of solvePnPRansac unless online_use_blue_in_pnp.
        pnp_rows = rows if self.use_blue_in_pnp else tag_rows
        pnp_poses = self._distinct_poses(bool(self.use_blue_in_pnp and ub))

        if len(pnp_rows) < self.min_pairs or pnp_poses < self.min_poses:
            if self._try_accept_kabsch(kabsch, tag_rows, blue_rows):
                return True
            if self.have_pose and tag_rows:
                self.mean_px = self._mean_reproj(*self._cv_pose(), tag_rows)
            return False

        obj = np.asarray([r["obj"] for r in pnp_rows], dtype=np.float32).reshape(-1, 3)
        img = np.asarray([r["img"] for r in pnp_rows], dtype=np.float32).reshape(-1, 2)

        guess = self.have_pose or kabsch is not None
        if self.have_pose:
            rvec0, tvec0 = self._cv_pose()
        elif kabsch is not None:
            rvec0, tvec0 = _cv_from_cell(
                [[float(kabsch[0][i, j]) for j in range(3)] for i in range(3)],
                (float(kabsch[1][0]), float(kabsch[1][1]), float(kabsch[1][2])),
            )
        else:
            rvec0 = np.zeros((3, 1), dtype=np.float64)
            tvec0 = np.zeros((3, 1), dtype=np.float64)

        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                img,
                self.K,
                self.dist,
                useExtrinsicGuess=bool(guess),
                rvec=rvec0.copy(),
                tvec=tvec0.copy(),
                reprojectionError=8.0,
                iterationsCount=200,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            ok, rvec, tvec, inliers = False, None, None, None
        cmp_rows = tag_rows if tag_rows else pnp_rows
        if not ok or rvec is None or inliers is None or len(inliers) < self.min_pairs:
            if self._try_accept_kabsch(kabsch, tag_rows, blue_rows):
                return True
            if self.have_pose:
                self.mean_px = self._mean_reproj(rvec0, tvec0, cmp_rows)
            return False

        idx = [int(i) for i in inliers.reshape(-1)]
        obj_in = obj[idx].reshape(-1, 1, 3)
        img_in = img[idx].reshape(-1, 1, 2)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_in, img_in, self.K, self.dist, rvec, tvec
            )
        except Exception:
            pass

        new_err = self._mean_reproj(rvec, tvec, cmp_rows)
        old_err = self._mean_reproj(rvec0, tvec0, cmp_rows) if guess else float("inf")
        self.mean_px = new_err
        if not math.isfinite(new_err):
            return self._try_accept_kabsch(kabsch, tag_rows, blue_rows)
        # Accept only when the fit over the tag rows improves.
        if guess and new_err > old_err + 0.5:
            if self._try_accept_kabsch(kabsch, tag_rows, blue_rows):
                return True
            return False
        R_new, t_new = _cell_from_cv(rvec, tvec)
        if self._accept_pose(R_new, t_new):
            self.last_mode = "pnp"
            return True
        return False

    def _accept_pose(self, R_new, t_new) -> bool:
        """Keep the candidate if it does not jump too far from the current pose."""
        if self.have_pose and self._jump_armed:
            dt = (
                (float(t_new[0]) - float(self.t[0])) ** 2
                + (float(t_new[1]) - float(self.t[1])) ** 2
                + (float(t_new[2]) - float(self.t[2])) ** 2
            ) ** 0.5
            dang = _rot_angle_deg(
                [[float(self.R[i, j]) for j in range(3)] for i in range(3)],
                [[float(R_new[i, j]) for j in range(3)] for i in range(3)],
            )
            if dt > self.max_jump_mm or dang > self.max_jump_deg:
                return False
        self.R = np.asarray(R_new, dtype=np.float64).reshape(3, 3).copy()
        self.t = np.asarray(t_new, dtype=np.float64).reshape(3).copy()
        self.last_accepted = True
        if math.isfinite(float(self.mean_px)) and float(self.mean_px) < 10.0:
            self._jump_armed = True
        return True

    def _cv_pose(self):
        return _cv_from_cell(
            [[float(self.R[i, j]) for j in range(3)] for i in range(3)],
            (float(self.t[0]), float(self.t[1]), float(self.t[2])),
        )

    # -- projection ---------------------------------------------------------
    def project(self, fk_points_mm) -> list:
        """FK base points -> pixel list (None where behind the camera)."""
        fk = np.asarray(fk_points_mm, dtype=np.float32).reshape(-1, 3)
        if not self.have_pose:
            return [None] * int(fk.shape[0])
        rvec, tvec = self._cv_pose()
        proj, _ = cv2.projectPoints(fk, rvec, tvec, self.K, self.dist)
        flat = proj.reshape(-1, 2)
        out: list = []
        for i in range(fk.shape[0]):
            u, v = float(flat[i, 0]), float(flat[i, 1])
            if math.isfinite(u) and math.isfinite(v):
                out.append((int(round(u)), int(round(v))))
            else:
                out.append(None)
        return out

    # -- persist ------------------------------------------------------------
    def save(self, cell_path=None) -> bool:
        """Write this camera's pose into cell_frames.json (layout arm_tags)."""
        if not self.have_pose:
            return False
        from .cell_frames import LAYOUT_ARM_TAGS, load_cell_frames, save_cell_frames

        cf = load_cell_frames(cell_path)
        pose = cf.pose(self.cam)
        pose.R = self.R.copy()
        pose.t_mm = self.t.copy()
        pose.ready = True
        cf.layout = LAYOUT_ARM_TAGS
        cf.bootstrap_from_blue_base = False
        cf.fx_px = float(self.K[0, 0])
        save_cell_frames(cf)
        return True
