"""Log rack OPEN <-> CLOSED transitions.

Subscribes to cam1 color, classifies with the same RackStateJudge used by the
ROS overlay, and appends a line only when the *stable* state changes.

Also publishes:
  /orbbec/cam1/rack_state          std_msgs/String   current stable state
  /orbbec/cam1/rack_state_event    std_msgs/String   only on change: "OPEN->CLOSED"

Logs (append):
  rack_lid_state/state_transitions.log
  rack_lid_state/state_transitions.jsonl

Usage (Pixi/ROS env):
  cd ros2_ws
  call _ros_env.cmd && call install\\setup.bat
  set PYTHONPATH=..\\robotics;%PYTHONPATH%
  python ..\\robotics\\rack_state_logger.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "rack_lid_state"
LOG_TXT = STATE_DIR / "state_transitions.log"
LOG_JSONL = STATE_DIR / "state_transitions.jsonl"
RACK_JSON = STATE_DIR / "rack.json"

TOPIC_IMAGE = "/orbbec/cam1/color/image_raw"
TOPIC_STATE = "/orbbec/cam1/rack_state"
TOPIC_EVENT = "/orbbec/cam1/rack_state_event"

# Require this many consecutive identical classifications before accepting a change.
STABLE_HITS = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _append_log(prev: str, new: str, held_s: float) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{_now_iso()} | {prev} -> {new} | held_{prev.lower()}_s={held_s:.1f}"
    with LOG_TXT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    rec = {
        "ts": _now_iso(),
        "from": prev,
        "to": new,
        "held_s": round(held_s, 3),
        "topic_state": TOPIC_STATE,
        "topic_event": TOPIC_EVENT,
    }
    with LOG_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(line, flush=True)


def main() -> int:
    # ROS paths (same pattern as arm_agents)
    ws = ROOT.parent / "ros2_ws"
    for p in (
        ws / "install" / "Lib" / "site-packages",
        ws / "src" / "orbbec_dual_sync",
    ):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

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
    from std_msgs.msg import String

    from orbbec_dual_sync.image_codec import imgmsg_to_numpy
    from orbbec_dual_sync.rack_overlay_draw import (
        classify_whole_rack,
        load_rack_rois,
    )

    class RackStateLogger(Node):
        def __init__(self) -> None:
            super().__init__("rack_state_logger")
            if not RACK_JSON.is_file():
                raise FileNotFoundError(f"Missing {RACK_JSON}")
            self.classify_xywh, _, self.holes = load_rack_rois(RACK_JSON)
            self._stable: Optional[str] = None
            self._pending: Optional[str] = None
            self._pending_hits = 0
            self._since = time.monotonic()
            self._frame_i = 0

            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=2,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._pub_state = self.create_publisher(String, TOPIC_STATE, 10)
            self._pub_event = self.create_publisher(String, TOPIC_EVENT, 10)
            self.create_subscription(Image, TOPIC_IMAGE, self._on_image, qos)
            self.get_logger().info(
                f"Watching {TOPIC_IMAGE}; publishing {TOPIC_STATE} + {TOPIC_EVENT}; "
                f"log -> {LOG_TXT}"
            )
            print(
                f"[{_now_iso()}] rack_state_logger started — "
                f"will log only on OPEN/CLOSED changes.\n"
                f"  state topic: {TOPIC_STATE}\n"
                f"  event topic: {TOPIC_EVENT}\n"
                f"  log file:    {LOG_TXT}",
                flush=True,
            )

        def _classify(self, bgr: np.ndarray) -> str:
            label = classify_whole_rack(
                bgr,
                self.classify_xywh,
                self.holes,
                robotics_root=ROOT,
            )
            label = (label or "").strip().upper()
            return label if label in ("OPEN", "CLOSED") else "CLOSED"

        def _on_image(self, msg: Image) -> None:
            try:
                bgr = imgmsg_to_numpy(msg)
            except Exception as exc:
                self.get_logger().warn(f"decode failed: {exc}")
                return
            if bgr is None or getattr(bgr, "ndim", 0) != 3:
                return
            # Classify every other frame — enough for door transitions.
            self._frame_i += 1
            if self._frame_i % 2 != 0:
                return
            raw = self._classify(bgr)

            if self._stable is None:
                self._stable = raw
                self._since = time.monotonic()
                self._publish_state(raw)
                boot = f"{_now_iso()} | START {raw}"
                STATE_DIR.mkdir(parents=True, exist_ok=True)
                with LOG_TXT.open("a", encoding="utf-8") as f:
                    f.write(boot + "\n")
                print(boot, flush=True)
                return

            if raw == self._stable:
                self._pending = None
                self._pending_hits = 0
                # Keep latched state on topic for late subscribers.
                if self._frame_i % 30 == 0:
                    self._publish_state(self._stable)
                return

            if raw == self._pending:
                self._pending_hits += 1
            else:
                self._pending = raw
                self._pending_hits = 1

            if self._pending_hits < STABLE_HITS:
                return

            prev = self._stable
            new = self._pending
            held = time.monotonic() - self._since
            self._stable = new
            self._since = time.monotonic()
            self._pending = None
            self._pending_hits = 0
            _append_log(prev, new, held)
            self._publish_state(new)
            ev = String()
            ev.data = f"{prev}->{new}"
            self._pub_event.publish(ev)

        def _publish_state(self, state: str) -> None:
            msg = String()
            msg.data = state
            self._pub_state.publish(msg)

    rclpy.init()
    node = RackStateLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"\n[{_now_iso()}] rack_state_logger stopped.", flush=True)
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
