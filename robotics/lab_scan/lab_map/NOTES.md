# A-Lab walkthrough #2: Gaussian splat map

Trained 2026-11-06 from `data/` (1030 frames, Orbbec Gemini 335L CP28563000NA, 1280×720, 90 s).
All 1030 frames were used, with no subsampling.

## Results

| | |
|---|---|
| COLMAP registered images | **1030 / 1030** (one model) |
| COLMAP 3D points | 222,701 |
| Mean reprojection error | **0.880 px** |
| Metric scale (COLMAP units → m) | **0.24475** |
| Scale spread | MAD 0.0105 (**4.3 %** relative); 5/25/75/95 pct = 0.161 / 0.235 / 0.256 / 0.375; per-image median std 4.1 % |
| Scale samples | 3,117,882 keypoints from 1030 images, sensor depth 0.15–10 m |
| splatfacto eval (all eval images, step 29000) | PSNR **26.61**, SSIM **0.851**, LPIPS **0.207** |
| Exported Gaussians | 502,890 (no NaN/Inf; 9,358 low-opacity ones dropped by the exporter) |

## How it was run

1. **COLMAP** used fixed PINHOLE intrinsics (fx 609.857, fy 609.870, cx 638.134, cy 354.407) on a single camera.
   It ran SIFT with up to 8192 features, then sequential matching (overlap 25, quadratic) with loop detection
   (vocab_tree_32K). The mapper kept focal length, principal point and extra parameters fixed.
2. **Metric scale** is the median of (sensor depth / COLMAP depth) over all registered keypoints. Depth is read
   at the keypoint pixel, and readings outside 0.15–10 m are discarded. The model's camera translations and
   points were multiplied by this scale, so `colmap/sparse/0` is **in meters**.
3. **nerfstudio**: `ns-process-data images --skip-colmap` was run on the metric model, then
   `ns-train splatfacto` with the defaults (30k iterations, SH degree 3, camera optimizer off), then
   `ns-export gaussian-splat`.

## Files

- `splat.ply`: exactly as exported, not rescaled. It is in nerfstudio's **dataparser frame**, not meters.
- `dataparser_transforms.json`: the 3×4 `transform` and `scale` (0.64319) that nerfstudio applied.
- `colmap/sparse/0/`: cameras/images/points3D `.bin`, in meters. Image names are `000000.jpg … 001029.jpg`,
  matching the depth PNGs.
- `transforms.json`: nerfstudio input. Its frames are renamed `images/frame_00001.jpg …`, where
  frame_NNNNN = color file NNNNN−1 (frame_00001 = `000000.jpg`).
- `scale.json`: full scale statistics.

## Mapping the splat to meters (COLMAP frame)

```
p_splat  = scale * (R @ p_colmap_m + t)      # R = transform[:, :3], t = transform[:, 3]
p_colmap_m = R.T @ (p_splat / scale - t)
```

`dataparser_transforms.json` maps directly from the metric COLMAP frame. **Do not** also apply the
`applied_transform` from `transforms.json`, because nerfstudio already folds that into `transform`.
To check this, I mapped the COLMAP points (error < 1 px, seen by ≥ 5 images, n = 78,109) into the splat
frame. Their median distance to the nearest opaque Gaussian is **1.2 cm** (25–75 %: 0.6–2.4 cm). If
`applied_transform` is added as well, the median distance becomes 39 cm. The map's 2–98 % extent is about
9.8 × 14.6 × 5.3 m (in splat axes).

Gaussian scales (`scale_0..2`, which are log values) convert to meters by adding `−ln(scale)`. Rotations
need to be composed with `R.T`.
