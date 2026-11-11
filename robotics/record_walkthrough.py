"""Record a walkthrough with one Orbbec Gemini 335L (color + depth).

The lab cameras are Gemini 335L. OpenCV CAP_OBSENSOR on this PC (OpenCV 4.9)
enumerates them but never delivers frames, so this script uses pyorbbecsdk2.

Leave CAMERA_SERIAL blank to see every connected camera in index order
(0, then 1, then 2, left to right). Move the one you can carry and press
its number. SPACE starts a 3 s countdown, then records RECORD_SECONDS.

Saves to lab_scan/walkthrough_<time>/:
    color/000000.jpg ...   color frames
    depth/000000.png ...   16-bit depth in millimeters, aligned to color
    intrinsics.json
    preview.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

# Blank serial: show cameras in order and press 0, 1, or 2.
# After you know which one moves, paste its serial here to skip the picker.
CAMERA_SERIAL = ""
# Fallback if CAMERA_SERIAL is blank. None = show the picker.
CAMERA_INDEX = None
RECORD_SECONDS = 30.0

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "lab_scan"
PREVIEW_W, PREVIEW_H, PREVIEW_FPS = 640, 480, 15
RECORD_W, RECORD_H, RECORD_FPS = 1280, 720, 30

HOLDER_NEEDLES = (
    "orbbec_camera_publisher",
    "orbbec_dual_sync",
    "OrbbecViewer",
    "view_dual_cameras.py",
    "scan_lab_cloud.py",
)


def camera_holders() -> list[str]:
    """Processes that open the fixed Orbbec cameras. Skips this recorder."""
    script = ROOT / "camera_busy.ps1"
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-File", str(script)],
            text=True,
            errors="replace",
            timeout=20,
        )
    except subprocess.CalledProcessError as exc:
        text = (exc.output or "") + (exc.stderr or "")
        return [line.strip() for line in text.splitlines() if line.strip()] or [
            "a camera process is running (see camera_busy.ps1)"
        ]
    except OSError as exc:
        print(f"Could not check for other camera processes: {exc}")
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def banner(img: np.ndarray, lines: list[str], color=(0, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22 + 26 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(
            out, text, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA
        )
    return out


def depth_vis(depth_mm: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    d = np.nan_to_num(depth_mm, nan=0.0)
    d = np.clip(d, 0, 4000) / 4000.0 * 255.0
    vis = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[depth_mm <= 0] = 0
    return cv2.resize(vis, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


def color_bgr(frame) -> np.ndarray | None:
    if frame is None:
        return None
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    fmt = frame.get_format()
    height, width = frame.get_height(), frame.get_width()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    print(f"Unsupported color format: {fmt}")
    return None


def depth_mm_from_frame(frame) -> np.ndarray | None:
    if frame is None:
        return None
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(
        (frame.get_height(), frame.get_width())
    )
    scale = float(frame.get_depth_scale())
    if scale == 0:
        scale = 1.0
    return raw.astype(np.float32) * scale


def query_devices():
    ctx = Context()
    device_list = ctx.query_devices()
    found = []
    for index in range(device_list.get_count()):
        device = device_list.get_device_by_index(index)
        info = device.get_device_info()
        found.append(
            {
                "index": index,
                "device": device,
                "name": info.get_name(),
                "serial": info.get_serial_number(),
                "usb": info.get_connection_type(),
                "pid": info.get_pid(),
            }
        )
    return found


def print_devices(devices) -> None:
    print(f"Orbbec devices in index order ({len(devices)}):")
    for item in devices:
        print(
            f"  {item['index']}  {item['name']}  serial={item['serial']}  {item['usb']}"
        )


def _enable_color(pipeline: Pipeline, config: Config, width: int, height: int, fps: int):
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    profile = profiles.get_video_stream_profile(width, height, OBFormat.MJPG, fps)
    config.enable_stream(profile)
    return profile


class OrbbecStream:
    """One Gemini camera via pyorbbecsdk2. Color preview, or color + aligned depth."""

    def __init__(self, item: dict):
        self.index = item["index"]
        self.serial = item["serial"]
        self.name = item["name"]
        self.usb = item["usb"]
        self.pipeline = Pipeline(item["device"])
        self.align = None
        self.intrinsics = None

    def start_preview(self) -> None:
        config = Config()
        _enable_color(self.pipeline, config, PREVIEW_W, PREVIEW_H, PREVIEW_FPS)
        self.pipeline.start(config)

    def start_record(self) -> None:
        config = Config()
        _enable_color(self.pipeline, config, RECORD_W, RECORD_H, RECORD_FPS)
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        config.enable_stream(depth_profiles.get_default_video_stream_profile())
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        try:
            self.pipeline.enable_frame_sync()
        except Exception as exc:
            print(f"Frame sync not enabled: {exc}")
        self.pipeline.start(config)
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def read(self, timeout_ms: int = 100):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if not frames:
            return None, None
        if self.align is not None:
            frames = self.align.process(frames)
            if not frames:
                return None, None
        color = color_bgr(frames.get_color_frame())
        depth = depth_mm_from_frame(frames.get_depth_frame()) if self.align is not None else None
        if self.intrinsics is None and color is not None and self.align is not None:
            self.intrinsics = self._read_intrinsics()
        return color, depth

    def _read_intrinsics(self):
        param = self.pipeline.get_camera_param()
        rgb = param.rgb_intrinsic
        return {
            "fx": float(rgb.fx),
            "fy": float(rgb.fy),
            "cx": float(rgb.cx),
            "cy": float(rgb.cy),
        }

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def pick_on_screen(devices) -> dict | None:
    """Show cameras left to right in index order. Press that camera's number."""
    streams: list[OrbbecStream] = []
    errors: dict[int, str] = {}
    try:
        for item in devices:
            stream = OrbbecStream(item)
            try:
                stream.start_preview()
                streams.append(stream)
                print(f"  preview {item['index']} {item['serial']}: streaming")
            except Exception as exc:
                errors[item["index"]] = str(exc)
                print(f"  preview {item['index']} {item['serial']} failed: {exc}")
                stream.close()
        if not streams:
            print("No camera returned a preview.")
            return None
        title = "Orbbec cameras in order — press 0, 1, or 2 for the one you can move"
        chosen = None
        blanks = {
            item["index"]: np.zeros((PREVIEW_H, PREVIEW_W, 3), np.uint8) for item in devices
        }
        while chosen is None:
            tiles = []
            by_index = {stream.index: stream for stream in streams}
            for item in devices:
                index = item["index"]
                stream = by_index.get(index)
                tile = blanks[index].copy()
                if stream is not None:
                    color, _depth = stream.read(80)
                    if color is not None:
                        tile = cv2.resize(color, (PREVIEW_W, PREVIEW_H))
                label = f"{index}  {item['serial']}"
                if index in errors:
                    label = f"{index} FAILED"
                cv2.rectangle(tile, (0, 0), (PREVIEW_W, 70), (0, 0, 0), -1)
                cv2.putText(
                    tile, str(index), (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4, cv2.LINE_AA
                )
                cv2.putText(
                    tile, item["serial"], (90, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
                )
                cv2.putText(
                    tile,
                    "press this number" if index not in errors else errors[index][:40],
                    (90, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                tiles.append(tile)
            view = cv2.hconcat(tiles) if len(tiles) > 1 else tiles[0]
            cv2.imshow(title, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if ord("0") <= key <= ord("9"):
                number = key - ord("0")
                match = next((item for item in devices if item["index"] == number), None)
                if match is not None and number not in errors:
                    chosen = match
    finally:
        for stream in streams:
            stream.close()
        cv2.destroyAllWindows()
    return chosen


def open_for_record(serial: str) -> OrbbecStream:
    time.sleep(0.8)
    devices = query_devices()
    print_devices(devices)
    match = next((item for item in devices if item["serial"] == serial), None)
    if match is None:
        raise RuntimeError(f"Serial {serial} is not connected.")
    stream = OrbbecStream(match)
    stream.start_record()
    return stream


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Orbbec walkthrough recorder")
    ap.add_argument("--serial", default=CAMERA_SERIAL or None, help="Orbbec serial number")
    ap.add_argument(
        "--index",
        type=int,
        default=CAMERA_INDEX,
        help="SDK index 0, 1, 2. Ignored when --serial is set.",
    )
    ap.add_argument("--seconds", type=float, default=RECORD_SECONDS)
    ap.add_argument("--max-fps", type=float, default=15.0, help="Cap on saved frames per second")
    args = ap.parse_args(argv)

    holders = camera_holders()
    if holders:
        print("Another program already has the Orbbec cameras open:")
        for line in holders:
            print(" ", line.replace("\t", "  "))
        print("Stop that program, then run this again.")
        return 1

    devices = query_devices()
    if not devices:
        print("No Orbbec camera found. Windows does not see a fourth camera either.")
        print("Plug the one you want to move into a USB 3 port and run this again.")
        return 1
    print_devices(devices)

    chosen = None
    if args.serial:
        chosen = next((item for item in devices if item["serial"] == args.serial), None)
        if chosen is None:
            print(f"Serial {args.serial} is not in the list above.")
            return 1
    elif args.index is not None:
        chosen = next((item for item in devices if item["index"] == args.index), None)
        if chosen is None:
            print(f"Index {args.index} is not connected. Connected indexes: "
                  + ", ".join(str(item["index"]) for item in devices))
            return 1
    else:
        print("Showing cameras left to right in index order. Move one, then press its number.")
        chosen = pick_on_screen(devices)
        if chosen is None:
            print("No camera picked.")
            return 1

    print(f"Using index {chosen['index']}  serial {chosen['serial']}")
    print("Next time: record_walkthrough.cmd --serial " + chosen["serial"])

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(f"Free disk space: {free_gb:.1f} GB")
    if free_gb < 1.0:
        print("WARNING: less than 1 GB free. A 30 s recording needs about 150-300 MB.")

    cam = open_for_record(chosen["serial"])
    color, depth = None, None
    deadline = time.time() + 8
    while time.time() < deadline and (color is None or depth is None):
        color, depth = cam.read(500)
    if color is None or depth is None:
        print("ERROR: camera opened but gave no color+depth frame.")
        cam.close()
        return 1
    print(
        f"Recording stream: color {color.shape[1]}x{color.shape[0]}  "
        f"depth {depth.shape[1]}x{depth.shape[0]}  "
        f"aligned={tuple(color.shape[:2]) == tuple(depth.shape[:2])}"
    )

    title = f"Walkthrough  index {chosen['index']}  {chosen['serial']}"
    state = "preview"
    t_mark = 0.0
    out_dir: Path | None = None
    writer = None
    stamps: list[float] = []
    sharp_vals: list[float] = []
    n = 0
    last_save = 0.0
    min_dt = 1.0 / max(1.0, args.max_fps)

    try:
        while True:
            nxt_color, nxt_depth = cam.read(100)
            if nxt_color is None or nxt_depth is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                continue
            color, depth = nxt_color, nxt_depth
            now = time.time()
            sharp = sharpness(color)
            blurry = sharp < 60.0

            if state == "countdown" and now - t_mark >= 3.0:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = OUT_ROOT / f"walkthrough_{stamp}"
                (out_dir / "color").mkdir(parents=True, exist_ok=True)
                (out_dir / "depth").mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(out_dir / "preview.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.max_fps,
                    (color.shape[1], color.shape[0]),
                )
                state = "recording"
                t_mark = now
                print(f"Recording to {out_dir}")

            if state == "recording":
                if now - last_save >= min_dt:
                    last_save = now
                    d16 = np.nan_to_num(depth, nan=0.0)
                    d16 = np.clip(d16, 0, 65535).astype(np.uint16)
                    cv2.imwrite(
                        str(out_dir / "color" / f"{n:06d}.jpg"),
                        color,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )
                    cv2.imwrite(str(out_dir / "depth" / f"{n:06d}.png"), d16)
                    writer.write(color)
                    stamps.append(now)
                    sharp_vals.append(sharp)
                    n += 1
                if now - t_mark >= args.seconds:
                    state = "done"

            view = cv2.hconcat([color, depth_vis(depth, color.shape[:2])])
            if state == "preview":
                lines = ["SPACE = start (3 s countdown)   Q = quit"]
                col = (0, 255, 255)
            elif state == "countdown":
                lines = [f"Starting in {3 - int(now - t_mark)} ..."]
                col = (0, 200, 255)
            elif state == "recording":
                left_s = max(0.0, args.seconds - (now - t_mark))
                lines = [f"REC  {left_s:4.1f} s left   frames {n}   Q = stop"]
                col = (0, 0, 255)
            else:
                lines = ["Done."]
                col = (0, 255, 0)
            if blurry and state in ("preview", "recording"):
                lines.append("BLURRY - move slower")
            view = banner(view, lines, col)
            cv2.imshow(title, view)
            key = cv2.waitKey(1) & 0xFF

            if state == "done":
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" ") and state == "preview":
                state = "countdown"
                t_mark = now
    finally:
        if writer is not None:
            writer.release()
        cam.close()
        cv2.destroyAllWindows()

    if out_dir is None or n == 0:
        print("Nothing recorded.")
        return 0

    dh, dw = int(depth.shape[0]), int(depth.shape[1])
    meta = {
        "camera_index": chosen["index"],
        "camera_serial": chosen["serial"],
        "camera_name": chosen["name"],
        "usb": chosen["usb"],
        "backend": "pyorbbecsdk2",
        "frames": n,
        "seconds": stamps[-1] - stamps[0] if n > 1 else 0.0,
        "color_hw": [int(color.shape[0]), int(color.shape[1])],
        "depth_hw": [dh, dw],
        "depth_units": "millimeters, uint16, 0 = no reading",
        "depth_aligned_to_color": True,
        "same_pixel_grid": [int(color.shape[0]), int(color.shape[1])] == [dh, dw],
        "intrinsics_color": cam.intrinsics,
        "timestamps": stamps,
        "sharpness": sharp_vals,
    }
    (out_dir / "intrinsics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    blurry_pct = 100.0 * sum(1 for s in sharp_vals if s < 60.0) / n
    print(f"Saved {n} frames to {out_dir}")
    print(f"Blurry frames: {blurry_pct:.0f}%  (under ~30% is good)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
