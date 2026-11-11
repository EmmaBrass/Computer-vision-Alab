"""UR forward kinematics + joint-angle sources for stick-figure / crash assist.

Joint sources (first that works):
  1. Teach-pendant realtime client interface (port 30003) when ``host`` is set —
     same ``q`` / TCP Polyscope's 3D robot uses
  2. JSON file ``ur_joints.json`` (manual / external writer)
  3. Mock zeros (draw at home) for offline overlay tests

Kinematics for Jeeves: UR5e DH via ``Useful Validation Scripts/ur5e_kinematics``
(matches teach-pad TCP within ~1 mm). Dashboard may say ``UR5``; do not use classic UR5.

    python -c "from ur_skeleton import fk_joint_positions_mm, load_joints; print(fk_joint_positions_mm(load_joints()[0]))"
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
JOINTS_PATH = ROOT / "ur_joints.json"
CONFIG_PATH = ROOT / "ur_fk_config.json"

# DH (meters) — Universal Robots published parameters (approx by model family).
UR_MODELS_M = {
    "ur5": {
        "d": [0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823],
        "a": [0.0, -0.42500, -0.39225, 0.0, 0.0, 0.0],
    },
    "ur5e": {
        "d": [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996],
        "a": [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0],
    },
    "ur10e": {
        "d": [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655],
        "a": [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0],
    },
}
UR_ALPHAS = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]


@dataclass
class UrFkConfig:
    model: str = "ur5e"
    host: str = ""  # RTDE host; empty = file/mock only
    joints_path: Path = JOINTS_PATH
    # Offset: p_vision_base = R @ p_ur_base + t_mm (align pendant base to cell ur_base)
    R_ur_to_vision: np.ndarray | None = None
    t_ur_to_vision_mm: np.ndarray | None = None
    rack_keepout_min_mm: tuple[float, float, float] = (200.0, -400.0, 0.0)
    rack_keepout_max_mm: tuple[float, float, float] = (700.0, 400.0, 500.0)
    confirm_gate_mm: float = 80.0
    proximity_mm: float = 100.0


def load_fk_config(path: Path | None = None) -> UrFkConfig:
    path = path or CONFIG_PATH
    cfg = UrFkConfig()
    if not path.is_file():
        return cfg
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return cfg
    cfg.model = str(data.get("model", cfg.model))
    cfg.host = str(data.get("host", cfg.host) or "")
    jp = data.get("joints_path")
    if jp:
        cfg.joints_path = Path(jp)
        if not cfg.joints_path.is_absolute():
            cfg.joints_path = ROOT / cfg.joints_path
    if data.get("R_ur_to_vision") is not None:
        cfg.R_ur_to_vision = np.asarray(data["R_ur_to_vision"], dtype=np.float64).reshape(3, 3)
    if data.get("t_ur_to_vision_mm") is not None:
        cfg.t_ur_to_vision_mm = np.asarray(
            data["t_ur_to_vision_mm"], dtype=np.float64
        ).reshape(3)
    if data.get("rack_keepout_min_mm") is not None:
        cfg.rack_keepout_min_mm = tuple(float(v) for v in data["rack_keepout_min_mm"])  # type: ignore
    if data.get("rack_keepout_max_mm") is not None:
        cfg.rack_keepout_max_mm = tuple(float(v) for v in data["rack_keepout_max_mm"])  # type: ignore
    cfg.confirm_gate_mm = float(data.get("confirm_gate_mm", cfg.confirm_gate_mm))
    cfg.proximity_mm = float(data.get("proximity_mm", cfg.proximity_mm))
    return cfg


def write_example_files() -> None:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "model": "ur5e",
                    "host": "",
                    "joints_path": "ur_joints.json",
                    "R_ur_to_vision": np.eye(3).tolist(),
                    "t_ur_to_vision_mm": [0.0, 0.0, 0.0],
                    "rack_keepout_min_mm": [200.0, -400.0, 0.0],
                    "rack_keepout_max_mm": [700.0, 400.0, 500.0],
                    "confirm_gate_mm": 80.0,
                    "proximity_mm": 100.0,
                    "notes": (
                        "Write q_rad into ur_joints.json from teach pendant / script, "
                        "or set host to robot IP for RTDE when ur_rtde is installed."
                    ),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not JOINTS_PATH.exists():
        JOINTS_PATH.write_text(
            json.dumps(
                {
                    "q_rad": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
                    "model": "ur5e",
                    "source": "example",
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fk_transforms(q_rad: np.ndarray, model: str = "ur5e") -> list[np.ndarray]:
    """Return T_0_i (4x4) for i=0..6 (base + 6 joints) in UR base meters."""
    q = np.asarray(q_rad, dtype=np.float64).reshape(6)
    key = model if model in UR_MODELS_M else "ur5e"
    d = UR_MODELS_M[key]["d"]
    a = UR_MODELS_M[key]["a"]
    T = np.eye(4, dtype=np.float64)
    out = [T.copy()]
    for i in range(6):
        T = T @ _dh(float(q[i]), float(d[i]), float(a[i]), float(UR_ALPHAS[i]))
        out.append(T.copy())
    return out


def fk_joint_positions_mm(q_rad: np.ndarray, model: str = "ur5e") -> np.ndarray:
    """(7,3) positions in UR base: base origin + 6 joint frames, millimeters."""
    Ts = fk_transforms(q_rad, model=model)
    pts = np.zeros((7, 3), dtype=np.float64)
    for i, T in enumerate(Ts):
        pts[i] = T[:3, 3] * 1000.0
    return pts


# UR base is Z-up. Cam0's ur_base is the optical frame (X right, Y down, Z forward)
# until a real mount rotation is saved. Identity R therefore graphs the arm on its side.
R_UR_Z_UP_TO_CAM_Y_DOWN = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


# File example is an L. The validation viewer does not draw that L.
# Its simulator starts at VIEWER_DEMO_Q (urp_viewer.ProgramSimulator).
FILE_L_Q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)
# Photographed cell pose, used only while the teach pendant stream is down.
PHOTO_Q = np.array([0.0, -2.618, 1.5708, 0.0, 0.0, 0.0], dtype=np.float64)
PHOTO_YAW = -math.pi / 2
# Live overlay heading. Encoder q stays locked. yaw spins the arm about
# vertical before the mount; img_deg / img_scale seat that projection on the
# photograph (upper arm up-right, forearm back along the arm). Joint 0 stays
# on the pedestal anchor. Blues still only blend pixels 1:1.
# Seating for uncalibrated draw only. cam1/cam2 = opposite high-down pair;
# cam0 = separate side/close view (not 180 from the others).
LIVE_VIEW_ALIGN = {
    "cam0": {"yaw": math.radians(34.0), "img_deg": 44.0, "img_scale": 0.62},
    "cam1": {"yaw": math.radians(-100.0), "img_deg": 40.0, "img_scale": 0.45},
    "cam2": {"yaw": math.radians(80.0), "img_deg": -40.0, "img_scale": 0.45},
}


def is_example_l(q_rad: np.ndarray) -> bool:
    q = np.asarray(q_rad, dtype=np.float64).reshape(6)
    return float(np.max(np.abs(q - FILE_L_Q))) < 0.08


def seed_photo_if_example(
    q_rad: np.ndarray, yaw: float, *, locked: bool = False
) -> tuple[np.ndarray, float]:
    """Replace the canned L with the validation viewer's start pose.

    RTDE q is left alone. Yaw is not changed: their redraw applies no extra
    spin, only the rail shift along UR +X (0 at the home rail).
    """
    q = np.asarray(q_rad, dtype=np.float64).reshape(6).copy()
    if locked or not is_example_l(q):
        return q, float(yaw)
    return VIEWER_DEMO_Q.copy(), float(yaw)


def mount_R(cfg: UrFkConfig) -> np.ndarray:
    """Rotation from UR base (Z up) into the vision frame the blue points use."""
    R = cfg.R_ur_to_vision
    if R is None or np.allclose(R, np.eye(3), atol=1e-3):
        return R_UR_Z_UP_TO_CAM_Y_DOWN.copy()
    return np.asarray(R, dtype=np.float64).reshape(3, 3)


def _rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def ur_base_to_vision_mm(
    pts_ur_mm: np.ndarray, cfg: UrFkConfig
) -> np.ndarray:
    """Map UR-base points into vision ur_base (mm)."""
    P = np.asarray(pts_ur_mm, dtype=np.float64).reshape(-1, 3)
    R = cfg.R_ur_to_vision if cfg.R_ur_to_vision is not None else np.eye(3)
    t = (
        cfg.t_ur_to_vision_mm
        if cfg.t_ur_to_vision_mm is not None
        else np.zeros(3)
    )
    return (R @ P.T).T + t.reshape(3)


def _read_joints_file(path: Path) -> Optional[tuple[np.ndarray, str]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    q = data.get("q_rad") or data.get("q")
    if q is None or len(q) < 6:
        return None
    model = str(data.get("model", "ur5e"))
    return np.asarray(q[:6], dtype=np.float64), model


# UR real-time interface (port 30003). Message size is a big-endian int, then
# doubles. q_actual is six doubles at byte 252 (after time and the five
# target blocks). Same angles RTDEReceiveInterface.getActualQ returns.
_Q_ACTUAL_OFFSET = 252
# Teach-pendant / Polyscope Client Interface: tool vector actual (x,y,z,rx,ry,rz).
_TCP_ACTUAL_OFFSET = 444


def q_from_realtime_packet(packet: bytes) -> Optional[np.ndarray]:
    """Six joint angles (rad) from one UR 30003 state packet, or None."""
    if len(packet) < _Q_ACTUAL_OFFSET + 48:
        return None
    try:
        from ur_rtde import RTDEReceiveInterface  # type: ignore

        return RTDEReceiveInterface(host)
    except Exception:
        pass
    try:
        import rtde_receive  # type: ignore

        return rtde_receive.RTDEReceiveInterface(host)
    except Exception:
        return None


def tcp_from_realtime_packet(packet: bytes) -> Optional[np.ndarray]:
    """Teach-pendant tool pose XYZ (m) in the UR base, or None."""
    if len(packet) < _TCP_ACTUAL_OFFSET + 24:
        return None
    xyz = struct.unpack(">3d", packet[_TCP_ACTUAL_OFFSET : _TCP_ACTUAL_OFFSET + 24])
    if not all(math.isfinite(v) and abs(v) < 3.0 for v in xyz):
        return None
    return np.asarray(xyz, dtype=np.float64)


def teachpad_kinematic_model(robot_model: str = "", *, polyscope: str = "") -> str:
    """Map Polyscope/dashboard names to the DH table the teach pad actually uses.

    Jeeves reports ``get robot model -> UR5`` on Polyscope 5.12, but the live
    teach-pad TCP matches the UR5e DH table (±1 mm), not classic UR5. Always
    prefer the e-Series chain for this cell unless an explicit classic model is
    requested.
    """
    name = (robot_model or "").strip().upper().replace(" ", "")
    poly = (polyscope or "").lower()
    if name in {"UR5E", "UR5", "UR7E", "UR7"}:
        return "ur5e"
    if "5.1" in poly or "5.12" in poly or "ursoftware 5" in poly:
        return "ur5e"
    if name in {"UR10E", "UR10"}:
        return "ur10e"
    if name in UR_MODELS_M:
        return name.lower()
    return "ur5e"


class TeachPadCache:
    """Background reader of the teach-pendant realtime stream (port 30003).

    Provides the same joint angles + TCP the Polyscope 3D model uses.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.q: Optional[np.ndarray] = None
        self.tcp_m: Optional[np.ndarray] = None
        self.mode = ""
        self.host = ""
        self.robot_model = ""
        self._started = False
        self._stop = False

    def note(self, q: np.ndarray, *, live: bool, host: str) -> None:
        with self._lock:
            self.q = np.asarray(q, dtype=np.float64).reshape(6).copy()
            self.mode = "live" if live else "hold"
            self.host = host
            self._started = True

    def mark_hold(self) -> None:
        with self._lock:
            if self.q is not None:
                self.mode = "hold"

    def snapshot(self, host: str = "") -> Optional[tuple[np.ndarray, str]]:
        with self._lock:
            if self.q is None:
                return None
            if host and self.host and self.host != host:
                return None
            tag = "teachpad-live" if self.mode == "live" else "teachpad-hold"
            return self.q.copy(), f"{tag}:{self.host or host}"

    def snapshot_tcp_mm(self, host: str = "") -> Optional[np.ndarray]:
        """Teach-pad tool XYZ in UR base, millimeters."""
        with self._lock:
            if self.tcp_m is None:
                return None
            if host and self.host and self.host != host:
                return None
            return (self.tcp_m * 1000.0).copy()

    def ensure(self, host: str, port: int = 30003) -> None:
        with self._lock:
            if self._started and self.host == host:
                return
            self.host = host
            self._started = True
            self._stop = False
        threading.Thread(
            target=self._run, args=(host, int(port)), daemon=True
        ).start()

    def _probe_model(self, host: str) -> None:
        """Best-effort dashboard ``get robot model`` (port 29999), short timeout."""
        try:
            sock = socket.create_connection((host, 29999), 0.4)
            try:
                sock.settimeout(0.4)
                sock.recv(1024)  # welcome banner
                sock.sendall(b"get robot model\n")
                raw = sock.recv(256).decode("utf-8", "ignore").strip()
            finally:
                sock.close()
            if raw:
                with self._lock:
                    self.robot_model = raw.splitlines()[-1].strip()
        except Exception:
            pass

    def _run(self, host: str, port: int) -> None:
        self._probe_model(host)
        while not self._stop:
            sock = None
            try:
                sock = socket.create_connection((host, port), 1.0)
                sock.settimeout(0.5)
                buf = b""
                while not self._stop:
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("teach pad closed")
                    buf += chunk
                    if len(buf) > 20000:
                        buf = buf[-4000:]
                    while len(buf) >= 4:
                        n = struct.unpack(">i", buf[:4])[0]
                        if n < 300 or n > 4000:
                            buf = buf[1:]
                            continue
                        if len(buf) < n:
                            break
                        pkt = buf[:n]
                        buf = buf[n:]
                        q = q_from_realtime_packet(pkt)
                        if q is None:
                            continue
                        tcp = tcp_from_realtime_packet(pkt)
                        with self._lock:
                            self.q = q
                            if tcp is not None:
                                self.tcp_m = tcp
                            self.mode = "live"
                            self.host = host
            except Exception:
                self.mark_hold()
                time.sleep(0.4)
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass


def _read_joints_rtde(host: str) -> Optional[tuple[np.ndarray, str]]:
    rtde = _rtde_receive(host)
    if rtde is None:
        return None
    try:
        q = np.asarray(rtde.getActualQ(), dtype=np.float64).reshape(6)
    except Exception:
        _close_rtde(rtde)
        return None
    _close_rtde(rtde)
    return q, "ur5e"


def read_robot_tcp_mm(host: str) -> Optional[np.ndarray]:
    """Tool position in the robot base, millimeters, from the controller.

    This is ``getActualTCPPose`` (x, y, z). It is the pendant's Cartesian
    number, not a pose recomputed here.
    """
    rtde = _rtde_receive(host)
    if rtde is None:
        return None
    try:
        pose = np.asarray(rtde.getActualTCPPose(), dtype=np.float64).reshape(-1)
    except Exception:
        _close_rtde(rtde)
        return None
    _close_rtde(rtde)
    if pose.size < 3 or not np.isfinite(pose[:3]).all():
        return None
    return pose[:3] * 1000.0


def robot_arm_points_mm(cfg: UrFkConfig | None = None) -> Optional[tuple[np.ndarray, str]]:
    """Arm points in the vision base from the controller or ``ur_joints.json``.

    Returns None for the offline mock pose so a mask is not drawn on a
    pretend arm. The tool tip is the controller Cartesian pose when RTDE
    answers. The other points are the reported joint angles placed on the
    UR link model, which is only there so the mask can follow each tube.
    """
    cfg = cfg or load_fk_config()
    q: Optional[np.ndarray] = None
    source = ""
    tcp_mm = None
    if cfg.host:
        got = _read_joints_rtde(cfg.host)
        if got is not None:
            q = got[0]
            source = f"rtde:{cfg.host}"
            tcp_mm = read_robot_tcp_mm(cfg.host)
    if q is None:
        filed = _read_joints_file(cfg.joints_path)
        if filed is None:
            return None
        q, model = filed
        source = f"file:{cfg.joints_path.name}"
    else:
        model = cfg.model or "ur5e"
    pts = ur_base_to_vision_mm(fk_joint_positions_mm(q, model), cfg)
    if tcp_mm is not None:
        pts = np.array(pts, dtype=np.float64, copy=True)
        pts[-1] = ur_base_to_vision_mm(tcp_mm.reshape(1, 3), cfg)[0]
    return pts, source


def load_joints(
    cfg: UrFkConfig | None = None,
) -> tuple[np.ndarray, str, str]:
    """Return (q_rad, model, source_name).

    When ``host`` is set, joints come from the teach-pendant realtime stream
    (same q Polyscope's 3D robot uses). The kinematic ``model`` is forced to the
    teach-pad DH table (UR5e for Jeeves), not the dashboard's short name ``UR5``.
    """
    cfg = cfg or load_fk_config()
    model = teachpad_kinematic_model(
        getattr(_TEACH, "robot_model", "") or cfg.model
    )
    if cfg.host:
        _TEACH.ensure(cfg.host, cfg.port)
        snap = _TEACH.snapshot(cfg.host)
        if snap is None:
            # First packets take a moment; wait briefly before PHOTO_Q fallback.
            for _ in range(25):
                time.sleep(0.08)
                snap = _TEACH.snapshot(cfg.host)
                if snap is not None:
                    break
        if snap is not None:
            model = teachpad_kinematic_model(
                getattr(_TEACH, "robot_model", "") or cfg.model
            )
            return snap[0], model, snap[1]
        # Stream not up yet. Do not draw the file L. Photographed pose until
        # the first actual-q packet, then the cache holds the last encoders.
        return PHOTO_Q.copy(), model, "photo:offline"
    got = _read_joints_file(cfg.joints_path)
    if got is not None:
        return got[0], teachpad_kinematic_model(str(got[1] or cfg.model)), f"file:{cfg.joints_path.name}"
    # Mock “elbow up” pose so the stick figure is visible offline.
    q = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float64)
    return q, model, "mock"


def teachpad_tcp_mm(cfg: UrFkConfig | None = None) -> Optional[np.ndarray]:
    """Live teach-pad tool XYZ (mm) in UR base, or None if the stream is down."""
    cfg = cfg or load_fk_config()
    if not cfg.host:
        return None
    _TEACH.ensure(cfg.host, cfg.port)
    return _TEACH.snapshot_tcp_mm(cfg.host)


def teachpad_fk_residual_mm(
    q_rad: np.ndarray, tcp_mm: np.ndarray, model: str = "ur5e"
) -> float:
    """|FK flange − teach-pad TCP| in mm (should be ~0–2 for the right DH model)."""
    pts = fk_joint_positions_mm(q_rad, model=model)
    flange = pts[-1]
    t = np.asarray(tcp_mm, dtype=np.float64).reshape(3)
    return float(
        math.sqrt(
            (float(flange[0]) - float(t[0])) ** 2
            + (float(flange[1]) - float(t[1])) ** 2
            + (float(flange[2]) - float(t[2])) ** 2
        )
    )


def point_to_aabb_gap_mm(
    p_mm: np.ndarray,
    mn: tuple[float, float, float],
    mx: tuple[float, float, float],
) -> float:
    """0 if inside AABB; else Euclidean distance to box surface (mm)."""
    p = np.asarray(p_mm, dtype=np.float64).reshape(3)
    mn_a = np.asarray(mn, dtype=np.float64)
    mx_a = np.asarray(mx, dtype=np.float64)
    closest = np.minimum(np.maximum(p, mn_a), mx_a)
    if np.all(p >= mn_a) and np.all(p <= mx_a):
        return 0.0
    return float(np.linalg.norm(p - closest))


def fk_rack_proximity(
    pts_vision_mm: np.ndarray,
    cfg: UrFkConfig,
) -> list[tuple[int, float]]:
    """Return list of (joint_index, gap_mm) within cfg.proximity_mm of rack keep-out.

    joint_index: 0=base ... 6=tool.
    """
    hits: list[tuple[int, float]] = []
    for i, p in enumerate(np.asarray(pts_vision_mm, dtype=np.float64).reshape(-1, 3)):
        if i == 0:
            continue  # skip base origin
        gap = point_to_aabb_gap_mm(p, cfg.rack_keepout_min_mm, cfg.rack_keepout_max_mm)
        if gap <= float(cfg.proximity_mm):
            hits.append((i, gap))
    return hits
