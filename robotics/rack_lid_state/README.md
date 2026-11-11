# Rack lid state (open / closed)

This is **not** the free-form vial YOLO tracker.

Camera 1 sees a rack that does not move. The locked crop is `rack.json`.
Live view:

```powershell
cd D:\Jessica\Jessica\robotics
& C:\Users\Ceder-ALAB\anaconda3\python.exe .\rack_lid_state.py focus
```

That opens camera 1 only, crops the saved rack, and labels **closed** (green, white lid)
or **open** (orange, beige interior). Q quits.

The click-to-calibrate slot tools below are optional and separate from `focus`.

## 1. Calibrate slots (once per camera pose)

Use a clear still of the rack (or live camera). Click the **center of each hole**
you care about, then press Enter:

```powershell
cd D:\Jessica\Jessica\robotics
python rack_lid_state.py calibrate --image rack_lid_state\example_closed.png
# or live:
python rack_lid_state.py calibrate --camera 0
```

Saves `rack_lid_state\slots.json`.

## 2. Check stills

```powershell
cd D:\Jessica\Jessica\robotics
python rack_lid_state.py still --image rack_lid_state\example_closed.png
python rack_lid_state.py still --image rack_lid_state\example_open.png
```

Green box = **closed**, orange = **open**, gray = **empty**.

## 3. Live

```powershell
cd D:\Jessica\Jessica\robotics
python rack_lid_state.py live --camera 0
```

Q quits, S saves a snapshot under `rack_lid_state\captures\`.

## Tips

- Keep the camera mount fixed after calibration.
- If boxes are too big/small, recalibrate with `--slot-size 60` (or 90).
- For YOLO later: collect open/closed crops from these fixed ROIs; the slot
  geometry stays the source of truth for *where*, YOLO/classifiers only for *state*.

## Multi-agent closed/home capture (camera 1)

Separate from per-slot open/closed above. Keeps only frames that look like
`reference_closed.png` (front view, trays retracted). No free-form YOLO.

```powershell
cd D:\Jessica\Jessica\robotics
C:\Users\Ceder-ALAB\anaconda3\python.exe rack_capture_agents.py still
C:\Users\Ceder-ALAB\anaconda3\python.exe rack_capture_agents.py refresh-template --camera 1
C:\Users\Ceder-ALAB\anaconda3\python.exe rack_capture_agents.py run --camera 1 --seconds 30
```

Open capture (rack open now; stamps `open_*.png` + `"rack_state": "OPEN"`):

```powershell
cd D:\Jessica\Jessica\robotics
C:\Users\Ceder-ALAB\anaconda3\python.exe rack_capture_agents.py refresh-open-template --camera 1
C:\Users\Ceder-ALAB\anaconda3\python.exe rack_capture_agents.py run --camera 1 --seconds 20 --label open
```

Keepers: `rack_lid_state\keeps\`  Rejects: `rack_lid_state\rejects\`
ROI / threshold: `rack_lid_state\rack.json`
Live match template: `template_roi.png` (refresh after lighting changes).
Open template: `template_open_roi.png`.
Keep-look photo: `reference_closed.png`.

## JudgeAgent (asymmetric OPEN vs CLOSED)

- **CLOSED**: only one allowed look — the home/retracted rack. Strict match to the
  closed template (`closed_threshold` default **0.80**).
- **OPEN**: shelves/trays can open separately, so appearance varies. Looser match to
  any open exemplar (`open_threshold` default **0.58**), or not-closed
  (`closed < 0.70`) plus a weak open cue (`>= 0.45`). Exemplars =
  `template_open_roi.png` + sampled `keeps/open_*.png`.

```powershell
cd D:\Jessica\Jessica\robotics
C:\Users\Ceder-ALAB\anaconda3\python.exe rack_capture_agents.py judge --smoke
C:\Users\Ceder-ALAB\anaconda3\python.exe rack_capture_agents.py judge --dir rack_lid_state\keeps
C:\Users\Ceder-ALAB\anaconda3\python.exe rack_capture_agents.py judge --image rack_lid_state\keeps\open_....png
C:\Users\Ceder-ALAB\anaconda3\python.exe rack_capture_agents.py judge --live --camera 1 --seconds 10
```

ROS rack overlay default is **`force_open`** (tray-extended work session —
always **RACK OPEN**). JudgeAgent `classify` is wired and ready; switch when
both poses are trained:

```text
rack_state_mode:=force_open     # default after rebuild (always OPEN)
rack_state_mode:=classify       # JudgeAgent: strict CLOSED, looser OPEN
rack_state_mode:=force_closed   # always RACK CLOSED
```

Or edit `ros2_ws/src/orbbec_dual_sync/config/dual_sync.yaml` → `state_mode`.
