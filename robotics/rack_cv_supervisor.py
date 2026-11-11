"""Supervisor + 5 agents for rack OPEN/CLOSED computer vision.

Agents (separate processes):
  1. CaptureAgent  — take pictures (ROS topic preferred; USB fallback)
  2. LabelAgent    — assign open/closed labels + quarantine mislabels
  3. AlgoAgent     — score match weights / thresholds on labeled set
  4. TrainAgent    — write closed/open templates + rack.json thresholds
  5. JudgeAgent    — validate asymmetric OPEN vs CLOSED classifier

Supervisor coordinates: capture → label → algo → train → judge.

Examples:
  cd D:\\Jessica\\Jessica\\robotics
  python rack_cv_supervisor.py bootstrap-closed --source ros --n 8
  python rack_cv_supervisor.py run --source ros --seconds 12
  python rack_cv_supervisor.py status
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

import rack_capture_agents as rca

ROOT = Path(__file__).resolve().parent
STATE_DIR = rca.STATE_DIR
LABELED_DIR = STATE_DIR / "labeled"
QUARANTINE_DIR = STATE_DIR / "quarantine_open"
SUPERVISOR_LOG = STATE_DIR / "supervisor_run.jsonl"
DEFAULT_RACK_JSON = rca.DEFAULT_RACK_JSON


def _log(name: str) -> logging.Logger:
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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


def grab_frames_ros(n: int, *, topic: str, timeout_s: float = 30.0) -> list[np.ndarray]:
    """Subscribe to a sensor_msgs/Image topic; return up to n BGR frames.

    Uses qos_profile_sensor_data (BEST_EFFORT) to match Orbbec publishers.
    Prefer calling via CaptureAgent subprocess on Windows so DDS inherits the
    Pixi/ROS env from rack_cv_supervisor.cmd.
    """
    import sys

    ros_ws = ROOT.parent / "ros2_ws"
    for p in (
        ros_ws / "install" / "Lib" / "site-packages",
        ros_ws / "src" / "orbbec_dual_sync",
    ):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    from orbbec_dual_sync.image_codec import imgmsg_to_numpy

    frames: list[np.ndarray] = []
    errors: list[str] = []
    rclpy.init()
    node = Node("rack_cv_capture_ros")

    def cb(msg: Image) -> None:
        if len(frames) >= n:
            return
        try:
            img = imgmsg_to_numpy(msg)
            if img.ndim == 3:
                frames.append(img.copy())
        except Exception as exc:
            errors.append(str(exc))

    node.create_subscription(Image, topic, cb, qos_profile_sensor_data)
    t0 = time.time()
    while len(frames) < n and time.time() - t0 < timeout_s:
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.02)
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass
    if not frames and errors:
        _log("CaptureAgent").warning("decode errors: %s", errors[:3])
    return frames


def grab_frames_usb(n: int, *, camera: int, interval_s: float = 0.35) -> list[np.ndarray]:
    cap = rca.open_orbbec(camera)
    try:
        rca.warmup_camera(cap, rca.WARMUP_GRABS)
        out: list[np.ndarray] = []
        next_t = time.monotonic()
        while len(out) < n:
            frame = rca.grab_color(cap)
            if frame is None:
                continue
            now = time.monotonic()
            if now < next_t:
                continue
            next_t = now + interval_s
            out.append(frame.copy())
        return out
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Quarantine helpers
# ---------------------------------------------------------------------------


def _sharpest_labeled_path(paths: list[str]) -> Optional[Path]:
    cfg = rca.load_rack_config()
    best_path: Optional[Path] = None
    best_blur = -1.0
    for raw in paths:
        img = cv2.imread(raw)
        if img is None:
            continue
        crop = rca.crop_roi(img, cfg.roi)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur > best_blur:
            best_blur = blur
            best_path = Path(raw)
    return best_path


def quarantine_open_matching_closed(
    *,
    closed_ref_override: Optional[Path] = None,
    score_min: float = 0.78,
) -> int:
    """Move open keepers that match the closed home look into quarantine_open/."""
    log = _log("LabelAgent")
    cfg = rca.load_rack_config()
    closed_ref = closed_ref_override
    if closed_ref is None or not Path(closed_ref).is_file():
        for candidate in (
            STATE_DIR / "reference_closed.png",
            STATE_DIR / "template_roi.png",
            cfg.reference,
        ):
            if candidate is not None and Path(candidate).is_file():
                closed_ref = Path(candidate)
                break
    if closed_ref is None or not Path(closed_ref).is_file():
        log.warning("no closed reference for quarantine")
        return 0

    # Scorer expects a ROI crop or full frame; if given a full frame, crop first
    # into a temporary in-memory reference via MatchScorer.from_bgr.
    ref_img = cv2.imread(str(closed_ref))
    if ref_img is None:
        return 0
    if ref_img.shape[0] > cfg.roi[3] + 20 or ref_img.shape[1] > cfg.roi[2] + 20:
        ref_crop = rca.crop_roi(ref_img, cfg.roi)
    else:
        ref_crop = ref_img
    closed_cfg = rca.RackConfig(
        camera_index=cfg.camera_index,
        roi=cfg.roi,
        reference=Path(closed_ref),
        work_size=cfg.work_size,
        threshold=score_min,
        blur_min=cfg.blur_min,
        weights=cfg.weights,
        frame_size=cfg.frame_size,
    )
    scorer = rca.MatchScorer.from_bgr(closed_cfg, ref_crop)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    quarantined = 0
    for open_png in sorted(rca.KEEPS_DIR.glob("open_*.png")):
        bgr = cv2.imread(str(open_png))
        if bgr is None:
            continue
        hit = scorer.score_bgr(bgr, already_roi=False)
        if hit.score >= score_min:
            dest = QUARANTINE_DIR / open_png.name
            shutil.move(str(open_png), str(dest))
            side = open_png.with_suffix(".json")
            if side.is_file():
                shutil.move(str(side), str(QUARANTINE_DIR / side.name))
            quarantined += 1
            log.info("quarantine %s (closed_score=%.3f)", open_png.name, hit.score)
    return quarantined


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


@dataclass
class CaptureResult:
    paths: list[str]
    source: str
    n: int


def capture_agent_main(q: mp.Queue, *, source: str, n: int, camera: int, topic: str) -> None:
    log = _log("CaptureAgent")
    try:
        if source == "ros":
            # Windows spawn + rclpy DDS is unreliable in mp.Process. Run a
            # dedicated worker subprocess that inherits the live ROS/Pixi env.
            import subprocess

            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_dir = STATE_DIR / "captures" / f"supervisor_{stamp}"
            result_json = STATE_DIR / "captures" / f"supervisor_{stamp}_result.json"
            worker = ROOT / "capture_ros_worker.py"
            if not worker.is_file():
                worker = ROOT / "agent_temp" / "capture_ros_worker.py"
            log.info("grabbing %s frames from %s (subprocess worker)", n, topic)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    "--topic",
                    topic,
                    "--n",
                    str(n),
                    "--timeout",
                    "30",
                    "--out-dir",
                    str(out_dir),
                    "--result-json",
                    str(result_json),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if result_json.is_file():
                payload = json.loads(result_json.read_text(encoding="utf-8"))
            else:
                payload = {
                    "ok": False,
                    "error": f"worker exit={proc.returncode} stderr={proc.stderr[-500:]}",
                    "agent": "CaptureAgent",
                }
            if proc.stdout:
                log.info("worker stdout: %s", proc.stdout.strip()[:200])
            if not payload.get("ok"):
                log.error("ROS capture failed: %s", payload.get("error"))
            else:
                log.info("wrote %s frames -> %s", payload.get("n"), out_dir)
            q.put(payload)
            return

        log.info("grabbing %s frames from USB cam %s", n, camera)
        frames = grab_frames_usb(n, camera=camera)
        if not frames:
            q.put({"ok": False, "error": "no frames", "agent": "CaptureAgent"})
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = STATE_DIR / "captures" / f"supervisor_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for i, frame in enumerate(frames, start=1):
            path = out_dir / f"raw_{i:03d}.png"
            cv2.imwrite(str(path), frame)
            paths.append(str(path))
        log.info("wrote %s frames -> %s", len(paths), out_dir)
        q.put(
            {
                "ok": True,
                "agent": "CaptureAgent",
                "paths": paths,
                "source": source,
                "n": len(paths),
            }
        )
    except Exception as exc:
        log.exception("capture failed")
        q.put({"ok": False, "error": str(exc), "agent": "CaptureAgent"})


def label_agent_main(
    q: mp.Queue,
    *,
    paths: list[str],
    label: str,
    quarantine_open_vs_closed: bool,
) -> None:
    """Assign filenames/sidecars; optionally quarantine open keepers that match closed."""
    log = _log("LabelAgent")
    try:
        label_u = label.strip().lower()
        if label_u not in ("open", "closed"):
            raise ValueError(f"label must be open|closed, got {label!r}")
        LABELED_DIR.mkdir(parents=True, exist_ok=True)
        labeled_paths: list[str] = []
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for i, raw in enumerate(paths, start=1):
            src = Path(raw)
            img = cv2.imread(str(src))
            if img is None:
                continue
            name = f"{label_u}_{stamp}_{i:03d}.png"
            dst = LABELED_DIR / name
            cv2.imwrite(str(dst), img)
            side = dst.with_suffix(".json")
            side.write_text(
                json.dumps(
                    {
                        "label": label_u.upper(),
                        "source": str(src),
                        "agent": "LabelAgent",
                        "ts": stamp,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            # Also seed keeps/ so JudgeAgent discover paths stay consistent.
            keep = rca.KEEPS_DIR / name
            cv2.imwrite(str(keep), img)
            labeled_paths.append(str(dst))
        quarantined = 0
        if quarantine_open_vs_closed:
            quarantined = quarantine_open_matching_closed(
                closed_ref_override=_sharpest_labeled_path(labeled_paths)
                if label_u == "closed"
                else None
            )
        log.info(
            "labeled=%s as %s; quarantined_open=%s",
            len(labeled_paths),
            label_u.upper(),
            quarantined,
        )
        q.put(
            {
                "ok": True,
                "agent": "LabelAgent",
                "labeled": labeled_paths,
                "label": label_u.upper(),
                "quarantined_open": quarantined,
            }
        )
    except Exception as exc:
        log.exception("label failed")
        q.put({"ok": False, "error": str(exc), "agent": "LabelAgent"})


def algo_agent_main(q: mp.Queue, *, labeled_paths: list[str], label: str) -> None:
    """Sweep simple weight/threshold candidates; pick best closed-vs-open split."""
    log = _log("AlgoAgent")
    try:
        cfg = rca.load_rack_config()
        closed_imgs: list[np.ndarray] = []
        open_imgs: list[np.ndarray] = []
        for p in labeled_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            if Path(p).name.lower().startswith("closed"):
                closed_imgs.append(img)
            elif Path(p).name.lower().startswith("open"):
                open_imgs.append(img)
        # Supplement from keeps.
        for p in sorted(rca.KEEPS_DIR.glob("closed_*.png"))[-12:]:
            img = cv2.imread(str(p))
            if img is not None:
                closed_imgs.append(img)
        for p in sorted(rca.KEEPS_DIR.glob("open_*.png"))[-12:]:
            img = cv2.imread(str(p))
            if img is not None:
                open_imgs.append(img)

        if not closed_imgs and label.lower() == "closed" and labeled_paths:
            for p in labeled_paths:
                img = cv2.imread(p)
                if img is not None:
                    closed_imgs.append(img)

        closed_ref = STATE_DIR / "reference_closed.png"
        if not closed_ref.is_file():
            closed_ref = cfg.reference
        weight_grid = [
            (0.45, 0.35, 0.20),
            (0.55, 0.30, 0.15),
            (0.40, 0.40, 0.20),
            (0.50, 0.25, 0.25),
        ]
        closed_ths = [0.75, 0.80, 0.85]
        open_ths = [0.55, 0.58, 0.62]
        best: dict[str, Any] = {"score": -1.0}

        for weights in weight_grid:
            trial_cfg = rca.RackConfig(
                camera_index=cfg.camera_index,
                roi=cfg.roi,
                reference=closed_ref,
                work_size=cfg.work_size,
                threshold=0.8,
                blur_min=cfg.blur_min,
                weights=weights,
                frame_size=cfg.frame_size,
            )
            try:
                scorer = rca.MatchScorer(trial_cfg)
            except Exception:
                continue
            closed_scores = [
                scorer.score_bgr(im, already_roi=False).score for im in closed_imgs[:16]
            ]
            open_scores = [
                scorer.score_bgr(im, already_roi=False).score for im in open_imgs[:16]
            ]
            if not closed_scores:
                continue
            c_med = float(np.median(closed_scores))
            o_med = float(np.median(open_scores)) if open_scores else 0.0
            gap = c_med - o_med
            for cth in closed_ths:
                for oth in open_ths:
                    # Reward: closed above cth, open below cth, separation gap.
                    c_ok = sum(1 for s in closed_scores if s >= cth) / len(closed_scores)
                    o_ok = (
                        sum(1 for s in open_scores if s < cth) / len(open_scores)
                        if open_scores
                        else 1.0
                    )
                    metric = 0.55 * c_ok + 0.25 * o_ok + 0.20 * max(0.0, gap)
                    if metric > best["score"]:
                        best = {
                            "score": metric,
                            "weights": {
                                "gray": weights[0],
                                "edge": weights[1],
                                "hist": weights[2],
                            },
                            "closed_threshold": cth,
                            "open_threshold": oth,
                            "closed_median": c_med,
                            "open_median": o_med,
                            "gap": gap,
                            "closed_n": len(closed_scores),
                            "open_n": len(open_scores),
                        }
        if best["score"] < 0:
            best = {
                "score": 0.0,
                "weights": {"gray": 0.45, "edge": 0.35, "hist": 0.20},
                "closed_threshold": rca.DEFAULT_CLOSED_JUDGE_THRESHOLD,
                "open_threshold": rca.DEFAULT_OPEN_JUDGE_THRESHOLD,
                "note": "fallback defaults (insufficient labeled data)",
            }
        log.info(
            "best metric=%.3f closed_th=%.2f open_th=%.2f weights=%s",
            best["score"],
            best["closed_threshold"],
            best["open_threshold"],
            best["weights"],
        )
        q.put({"ok": True, "agent": "AlgoAgent", "algo": best})
    except Exception as exc:
        log.exception("algo failed")
        q.put({"ok": False, "error": str(exc), "agent": "AlgoAgent"})


def train_agent_main(
    q: mp.Queue,
    *,
    labeled_paths: list[str],
    label: str,
    algo: dict[str, Any],
) -> None:
    """Update templates + rack.json from labeled frames and AlgoAgent picks."""
    log = _log("TrainAgent")
    try:
        cfg = rca.load_rack_config()
        label_u = label.strip().lower()
        used: list[str] = []
        if labeled_paths:
            # Use sharpest labeled frame for the template of this class.
            best_path: Optional[Path] = None
            best_blur = -1.0
            for p in labeled_paths:
                img = cv2.imread(p)
                if img is None:
                    continue
                crop = rca.crop_roi(img, cfg.roi)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if blur > best_blur:
                    best_blur = blur
                    best_path = Path(p)
            if best_path is not None:
                img = cv2.imread(str(best_path))
                assert img is not None
                crop = rca.crop_roi(img, cfg.roi)
                if label_u == "closed":
                    out_roi = STATE_DIR / "template_roi.png"
                    out_ref = STATE_DIR / "reference_closed.png"
                    cv2.imwrite(str(out_roi), crop)
                    cv2.imwrite(str(out_ref), crop)
                    full = STATE_DIR / "captures" / "probe_closed_live.png"
                    cv2.imwrite(str(full), img)
                    cfg.reference = out_roi
                    used = [str(out_roi), str(out_ref), str(full)]
                else:
                    out_open = STATE_DIR / "template_open_roi.png"
                    cv2.imwrite(str(out_open), crop)
                    used = [str(out_open)]
                cfg.frame_size = (int(img.shape[1]), int(img.shape[0]))

        # Persist match weights / threshold into rack.json.
        weights = algo.get("weights") or {}
        if weights:
            cfg.weights = (
                float(weights.get("gray", cfg.weights[0])),
                float(weights.get("edge", cfg.weights[1])),
                float(weights.get("hist", cfg.weights[2])),
            )
        if "closed_threshold" in algo:
            # Store closed strict threshold in match.threshold for capture keepers.
            cfg.threshold = float(algo["closed_threshold"])
        rca.save_rack_config(cfg, DEFAULT_RACK_JSON)

        # Judge thresholds sidecar for overlay/supervisor.
        judge_path = STATE_DIR / "judge_thresholds.json"
        payload = {
            "closed_threshold": float(
                algo.get("closed_threshold", rca.DEFAULT_CLOSED_JUDGE_THRESHOLD)
            ),
            "open_threshold": float(
                algo.get("open_threshold", rca.DEFAULT_OPEN_JUDGE_THRESHOLD)
            ),
            "open_weak_threshold": rca.DEFAULT_OPEN_WEAK_THRESHOLD,
            "closed_not_closed_max": rca.DEFAULT_CLOSED_NOT_CLOSED_MAX,
            "weights": {
                "gray": cfg.weights[0],
                "edge": cfg.weights[1],
                "hist": cfg.weights[2],
            },
            "updated_by": "TrainAgent",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        judge_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        used.append(str(judge_path))
        log.info("trained templates/thresholds: %s", used)
        q.put(
            {
                "ok": True,
                "agent": "TrainAgent",
                "wrote": used,
                "judge_thresholds": payload,
            }
        )
    except Exception as exc:
        log.exception("train failed")
        q.put({"ok": False, "error": str(exc), "agent": "TrainAgent"})


def judge_agent_main(q: mp.Queue, *, algo: dict[str, Any]) -> None:
    log = _log("JudgeAgent")
    try:
        cfg = rca.load_rack_config()
        th = rca.JudgeThresholds(
            closed_threshold=float(
                algo.get("closed_threshold", rca.DEFAULT_CLOSED_JUDGE_THRESHOLD)
            ),
            open_threshold=float(
                algo.get("open_threshold", rca.DEFAULT_OPEN_JUDGE_THRESHOLD)
            ),
            open_weak_threshold=rca.DEFAULT_OPEN_WEAK_THRESHOLD,
            closed_not_closed_max=rca.DEFAULT_CLOSED_NOT_CLOSED_MAX,
            blur_min=cfg.blur_min,
        )
        judge = rca.RackStateJudge(cfg, thresholds=th)
        rows: list[dict[str, Any]] = []
        correct = 0
        total = 0
        for path in sorted(rca.KEEPS_DIR.glob("closed_*.png"))[-20:]:
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            result = judge.judge_bgr(bgr, already_roi=False)
            ok = result.label == "CLOSED"
            total += 1
            correct += int(ok)
            rows.append(
                {
                    "file": path.name,
                    "expected": "CLOSED",
                    "pred": result.label,
                    "closed": result.closed_score,
                    "open": result.open_score,
                    "reason": result.reason,
                    "ok": ok,
                }
            )
        for path in sorted(rca.KEEPS_DIR.glob("open_*.png"))[-20:]:
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            result = judge.judge_bgr(
                bgr,
                already_roi=False,
                exclude_open_exemplar=path.name,
            )
            ok = result.label == "OPEN"
            total += 1
            correct += int(ok)
            rows.append(
                {
                    "file": path.name,
                    "expected": "OPEN",
                    "pred": result.label,
                    "closed": result.closed_score,
                    "open": result.open_score,
                    "reason": result.reason,
                    "ok": ok,
                }
            )
        # Live probe if present.
        probe = STATE_DIR / "captures" / "probe_closed_live.png"
        probe_result = None
        if probe.is_file():
            bgr = cv2.imread(str(probe))
            if bgr is not None:
                r = judge.judge_bgr(bgr, already_roi=False)
                probe_result = {
                    "file": probe.name,
                    "pred": r.label,
                    "closed": r.closed_score,
                    "open": r.open_score,
                    "reason": r.reason,
                }
                log.info(
                    "probe_closed_live -> %s closed=%.3f open=%.3f (%s)",
                    r.label,
                    r.closed_score,
                    r.open_score,
                    r.reason,
                )
        acc = (100.0 * correct / total) if total else 0.0
        log.info("smoke %s/%s (%.1f%%)", correct, total, acc)
        q.put(
            {
                "ok": True,
                "agent": "JudgeAgent",
                "correct": correct,
                "total": total,
                "accuracy_pct": acc,
                "probe": probe_result,
                "rows": rows[:40],
            }
        )
    except Exception as exc:
        log.exception("judge failed")
        q.put({"ok": False, "error": str(exc), "agent": "JudgeAgent"})


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def _run_agent(target, kwargs: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=target, kwargs={"q": q, **kwargs}, daemon=True)
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {"ok": False, "error": f"timeout after {timeout_s}s", "agent": target.__name__}
    try:
        return q.get_nowait()
    except Exception:
        return {"ok": False, "error": "no result from agent", "agent": target.__name__}


def _run_capture_agent_ros(*, n: int, topic: str) -> dict[str, Any]:
    """Spawn CaptureAgent as a real OS subprocess (not mp) so ROS/DDS works on Windows."""
    import subprocess

    log = _log("CaptureAgent")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = STATE_DIR / "captures" / f"supervisor_{stamp}"
    result_json = STATE_DIR / "captures" / f"supervisor_{stamp}_result.json"
    worker = ROOT / "capture_ros_worker.py"
    if not worker.is_file():
        worker = ROOT / "agent_temp" / "capture_ros_worker.py"
    log.info("grabbing %s frames from %s (OS subprocess)", n, topic)
    proc = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--topic",
            topic,
            "--n",
            str(n),
            "--timeout",
            "30",
            "--out-dir",
            str(out_dir),
            "--result-json",
            str(result_json),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    if result_json.is_file():
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    else:
        payload = {
            "ok": False,
            "error": f"worker exit={proc.returncode} stderr={(proc.stderr or '')[-500:]}",
            "agent": "CaptureAgent",
        }
    if proc.stdout:
        log.info("worker stdout: %s", proc.stdout.strip()[:200])
    if not payload.get("ok"):
        log.error("ROS capture failed: %s", payload.get("error"))
    else:
        log.info("wrote %s frames -> %s", payload.get("n"), out_dir)
    return payload


def supervisor_pipeline(
    *,
    source: str,
    n: int,
    label: str,
    camera: int,
    topic: str,
    quarantine: bool,
) -> dict[str, Any]:
    slog = _log("SupervisorAgent")
    slog.info(
        "launching 5 agents | source=%s n=%s label=%s",
        source,
        n,
        label,
    )
    report: dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": source,
        "label": label,
        "agents": {},
    }

    if source == "ros":
        # Must be a real subprocess from the supervisor (Pixi/ROS env), not mp.Process.
        cap = _run_capture_agent_ros(n=n, topic=topic)
    else:
        cap = _run_agent(
            capture_agent_main,
            {"source": source, "n": n, "camera": camera, "topic": topic},
            timeout_s=60.0,
        )
    report["agents"]["CaptureAgent"] = {k: v for k, v in cap.items() if k != "paths"} | {
        "n_paths": len(cap.get("paths") or [])
    }
    if not cap.get("ok"):
        slog.error("CaptureAgent failed: %s", cap.get("error"))
        _append_jsonl(SUPERVISOR_LOG, report)
        return report
    paths = list(cap["paths"])

    lab = _run_agent(
        label_agent_main,
        {
            "paths": paths,
            "label": label,
            "quarantine_open_vs_closed": quarantine,
        },
        timeout_s=90.0,
    )
    report["agents"]["LabelAgent"] = {
        k: v for k, v in lab.items() if k != "labeled"
    } | {"n_labeled": len(lab.get("labeled") or [])}
    if not lab.get("ok"):
        slog.error("LabelAgent failed: %s", lab.get("error"))
        _append_jsonl(SUPERVISOR_LOG, report)
        return report
    labeled = list(lab["labeled"])

    algo = _run_agent(
        algo_agent_main,
        {"labeled_paths": labeled, "label": label},
        timeout_s=90.0,
    )
    report["agents"]["AlgoAgent"] = algo
    if not algo.get("ok"):
        slog.error("AlgoAgent failed: %s", algo.get("error"))
        _append_jsonl(SUPERVISOR_LOG, report)
        return report

    train = _run_agent(
        train_agent_main,
        {
            "labeled_paths": labeled,
            "label": label,
            "algo": algo.get("algo") or {},
        },
        timeout_s=60.0,
    )
    report["agents"]["TrainAgent"] = train
    if not train.get("ok"):
        slog.error("TrainAgent failed: %s", train.get("error"))
        _append_jsonl(SUPERVISOR_LOG, report)
        return report

    # After templates refresh, re-quarantine open keepers that match new closed home.
    if quarantine and label.strip().lower() == "closed":
        q2 = quarantine_open_matching_closed(
            closed_ref_override=STATE_DIR / "reference_closed.png",
            score_min=0.78,
        )
        report["agents"]["LabelAgent"]["quarantined_open_after_train"] = q2
        slog.info("post-train quarantine_open=%s", q2)

    judge = _run_agent(
        judge_agent_main,
        {"algo": algo.get("algo") or {}},
        timeout_s=90.0,
    )
    report["agents"]["JudgeAgent"] = {
        k: v for k, v in judge.items() if k != "rows"
    }
    report["ok"] = bool(judge.get("ok"))
    _append_jsonl(SUPERVISOR_LOG, report)

    slog.info("=== Supervisor summary ===")
    for name, payload in report["agents"].items():
        slog.info("%s: %s", name, json.dumps(payload, default=str)[:240])
    if judge.get("probe"):
        slog.info("LIVE PROBE: %s", judge["probe"])
    return report


def cmd_bootstrap_closed(args: argparse.Namespace) -> int:
    """User confirms rack is CLOSED now — capture/label/train/judge."""
    report = supervisor_pipeline(
        source=str(args.source),
        n=int(args.n),
        label="closed",
        camera=int(args.camera),
        topic=str(args.topic),
        quarantine=not bool(args.no_quarantine),
    )
    probe = (report.get("agents") or {}).get("JudgeAgent", {}).get("probe") or {}
    pred = str(probe.get("pred") or "")
    print(json.dumps(report, indent=2, default=str)[:4000])
    if pred == "CLOSED":
        print("PASS: probe classified CLOSED")
        return 0
    if probe:
        print(f"WARN: probe predicted {pred!r} (wanted CLOSED) — collect more closed / open contrast")
        return 1
    print("WARN: no probe result")
    return 1


def cmd_bootstrap_open(args: argparse.Namespace) -> int:
    report = supervisor_pipeline(
        source=str(args.source),
        n=int(args.n),
        label="open",
        camera=int(args.camera),
        topic=str(args.topic),
        quarantine=False,
    )
    print(json.dumps(report, indent=2, default=str)[:4000])
    return 0 if report.get("ok") else 1


def cmd_run(args: argparse.Namespace) -> int:
    report = supervisor_pipeline(
        source=str(args.source),
        n=max(3, int(float(args.seconds) / max(0.2, float(args.interval)))),
        label=str(args.label),
        camera=int(args.camera),
        topic=str(args.topic),
        quarantine=bool(args.quarantine),
    )
    print(json.dumps(report, indent=2, default=str)[:4000])
    return 0 if report.get("ok") else 1


def cmd_status(args: argparse.Namespace) -> int:
    cfg = rca.load_rack_config()
    closed_n = len(list(rca.KEEPS_DIR.glob("closed_*.png")))
    open_n = len(list(rca.KEEPS_DIR.glob("open_*.png")))
    quar_n = len(list(QUARANTINE_DIR.glob("open_*.png"))) if QUARANTINE_DIR.is_dir() else 0
    th_path = STATE_DIR / "judge_thresholds.json"
    th = json.loads(th_path.read_text(encoding="utf-8")) if th_path.is_file() else {}
    print("=== rack CV status ===")
    print(f"ROI: {cfg.roi}  reference: {cfg.reference.name}")
    print(f"keeps closed={closed_n} open={open_n}  quarantined_open={quar_n}")
    print(f"judge_thresholds: {th or '(none yet)'}")
    if SUPERVISOR_LOG.is_file():
        lines = SUPERVISOR_LOG.read_text(encoding="utf-8").strip().splitlines()
        print(f"last supervisor run: {lines[-1][:300] if lines else '(empty)'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Supervisor + 5 rack CV agents")
    p.add_argument(
        "--topic",
        default="/orbbec/cam1/color/image_raw",
        help="ROS color topic when --source ros",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--source", choices=("ros", "usb"), default="ros")
        sp.add_argument("--camera", type=int, default=1, help="USB Orbbec index")
        sp.add_argument("--n", type=int, default=8, help="Frames to capture")

    b = sub.add_parser(
        "bootstrap-closed",
        help="Rack is CLOSED now: Capture→Label→Algo→Train→Judge (5 agents)",
    )
    _add_common(b)
    b.add_argument(
        "--no-quarantine",
        action="store_true",
        help="Do not move open keepers that match closed template",
    )
    b.set_defaults(func=cmd_bootstrap_closed)

    o = sub.add_parser(
        "bootstrap-open",
        help="Rack is OPEN now: Capture→Label→Algo→Train→Judge",
    )
    _add_common(o)
    o.set_defaults(func=cmd_bootstrap_open)

    r = sub.add_parser("run", help="Generic supervised capture/label/train cycle")
    r.add_argument("--source", choices=("ros", "usb"), default="ros")
    r.add_argument("--camera", type=int, default=1, help="USB Orbbec index")
    r.add_argument("--label", choices=("closed", "open"), required=True)
    r.add_argument("--seconds", type=float, default=12.0)
    r.add_argument("--interval", type=float, default=0.4)
    r.add_argument("--quarantine", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="Show labeled counts + thresholds")
    s.set_defaults(func=cmd_status)
    return p


def main() -> None:
    mp.freeze_support()
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
