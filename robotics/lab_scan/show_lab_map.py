"""Open a top-down view of the meter-scale walkthrough splat."""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

import numpy as np

SRC = Path(r"D:\Jessica\Jessica\robotics\lab_scan\lab_map\splat.ply")
OUT = Path(r"D:\Jessica\Jessica\robotics\lab_scan\lab_map_view.html")


def main() -> None:
    with SRC.open("rb") as f:
        while True:
            line = f.readline()
            if line.strip() == b"end_header":
                break
        raw = np.fromfile(f, dtype="<f4")
    raw = raw.reshape(raw.size // 62, 62)
    xyz = raw[:, 0:3]
    dc = raw[:, 6:9]
    op = raw[:, 54]
    finite = np.isfinite(xyz).all(axis=1)
    near = finite & (np.abs(xyz[:, 0]) < 20) & (np.abs(xyz[:, 1]) < 20) & (np.abs(xyz[:, 2]) < 20)
    sig = 1.0 / (1.0 + np.exp(-np.clip(op, -20, 20)))
    keep = near & (sig > 0.5)
    xyz = xyz[keep]
    rgb = np.clip(0.5 + 0.28209479177387814 * dc[keep], 0, 1)
    step = max(1, int(xyz.shape[0] // 20000))
    xyz = xyz[::step]
    rgb = (rgb[::step] * 255).astype(np.int32)
    print(f"shown {xyz.shape[0]} of {int(keep.sum())} in-bounds opaque points", flush=True)
    payload_xyz = json.dumps(np.round(xyz.astype(np.float64), 3).tolist())
    payload_rgb = json.dumps(rgb.tolist())
    note = (
        "Walkthrough #2 splat in the nerfstudio dataparser frame (see lab_map_frame.json for meters). "
        "Drag sideways to turn, drag up to tilt, scroll to zoom. Click a point for its position."
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Lab map</title>
<style>
  html, body {{ margin: 0; height: 100%; background: #16181c; color: #e8eaed; font-family: Segoe UI, sans-serif; }}
  header {{ padding: 10px 14px 0; }}
  h1 {{ font-size: 18px; margin: 0; font-weight: 600; }}
  p {{ margin: 4px 0 0; color: #c4ccd6; font-size: 13px; }}
  #wrap {{ position: absolute; inset: 72px 0 0 0; }}
  canvas {{ width: 100%; height: 100%; display: block; cursor: grab; touch-action: none; }}
</style>
</head>
<body>
<header>
  <h1>Lab walkthrough</h1>
  <p>{note}</p>
  <p id="status">Looking down. Drag sideways to turn, drag up to tilt, scroll to zoom.</p>
</header>
<div id="wrap"><canvas id="c"></canvas></div>
<script>
const xyz = {payload_xyz};
const rgb = {payload_rgb};
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");
const status = document.getElementById("status");
let yaw = 0, pitch = 1.35, zoom = 1, drag = null;
const items = xyz.map(function (p, i) {{ return {{ p: p, rgb: rgb[i] }}; }});
function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}
function flat(p) {{
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x1 = cy * p[0] - sy * p[1];
  const y1 = sy * p[0] + cy * p[1];
  return [x1, Math.sin(pitch) * y1 + Math.cos(pitch) * p[2], p[2]];
}}
function frame() {{
  let minH = 1e9, maxH = -1e9, minV = 1e9, maxV = -1e9;
  items.forEach(function (it) {{
    const q = flat(it.p);
    if (q[0] < minH) minH = q[0];
    if (q[0] > maxH) maxH = q[0];
    if (q[1] < minV) minV = q[1];
    if (q[1] > maxV) maxV = q[1];
  }});
  const w = canvas.clientWidth, h = canvas.clientHeight;
  const scale = 0.88 * Math.min(w / Math.max(1, maxH - minH), h / Math.max(1, maxV - minV)) * zoom;
  return {{ minH: minH, maxV: maxV, scale: scale, w: w, h: h }};
}}
function project(p, fr) {{
  const q = flat(p);
  return [0.06 * fr.w + (q[0] - fr.minH) * fr.scale, 0.06 * fr.h + (fr.maxV - q[1]) * fr.scale, q[2]];
}}
function draw() {{
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#16181c";
  ctx.fillRect(0, 0, w, h);
  const fr = frame();
  const drawn = items.map(function (it) {{ return {{ q: project(it.p, fr), it: it }}; }});
  drawn.sort(function (a, b) {{ return a.q[2] - b.q[2]; }});
  const dot = Math.max(2, Math.min(6, fr.scale * 0.04));
  drawn.forEach(function (d) {{
    ctx.fillStyle = "rgb(" + d.it.rgb[0] + "," + d.it.rgb[1] + "," + d.it.rgb[2] + ")";
    ctx.fillRect(d.q[0], d.q[1], dot, dot);
  }});
}}
canvas.addEventListener("pointerdown", function (e) {{
  drag = {{ x: e.clientX, y: e.clientY, yaw: yaw, pitch: pitch, moved: false }};
  canvas.setPointerCapture(e.pointerId);
}});
canvas.addEventListener("pointermove", function (e) {{
  if (!drag) return;
  if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) drag.moved = true;
  yaw = drag.yaw + (e.clientX - drag.x) * 0.005;
  pitch = Math.max(0.15, Math.min(1.57, drag.pitch - (e.clientY - drag.y) * 0.005));
  draw();
}});
canvas.addEventListener("pointerup", function (e) {{
  if (drag && !drag.moved) {{
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left, y = e.clientY - rect.top;
    const fr = frame();
    let best = null, bestD = 18 * 18;
    items.forEach(function (it) {{
      const q = project(it.p, fr);
      const d = (q[0] - x) * (q[0] - x) + (q[1] - y) * (q[1] - y);
      if (d < bestD) {{ bestD = d; best = it; }}
    }});
    if (best) {{
      const p = best.p;
      status.textContent = "x " + p[0].toFixed(2) + " m   y " + p[1].toFixed(2) + " m   z " + p[2].toFixed(2) + " m";
    }}
  }}
  drag = null;
}});
canvas.addEventListener("wheel", function (e) {{
  e.preventDefault();
  zoom = Math.max(0.4, Math.min(8, zoom * (e.deltaY > 0 ? 0.9 : 1.1)));
  draw();
}}, {{ passive: false }});
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(OUT, flush=True)
    webbrowser.open(OUT.as_uri())


if __name__ == "__main__":
    main()
