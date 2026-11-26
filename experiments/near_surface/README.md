# Near-surface stereo candidate

Offline prototype, not imported by the live controller. Inputs must be an already
rectified 1280×720 stereo pair and a bucket bounding box in rectified left-image
coordinates. `factor_m_px` is the calibrated rectified focal length times baseline
in metres. Requires NumPy and OpenCV.

`NearSurface.measure()` returns the median forward Z of supported textured surface
pixels. It does not estimate the bucket opening, nearest obstacle, or actuator
clearance. Invalid observations have `valid=false` and `distance_m=null`; callers
must not replace these with stale accepted distances for motion decisions.

The 256-pixel full-resolution search and quality gates are provisional. On the
2026-11-25 saved scene, five textured-paper pairs gave 24.82–24.84 cm against an
approximate 24 cm paper-center tape measurement; two blank-bucket pairs were
rejected. These recordings informed the design, and are not independent validation.
Vertical rectification residual remains under investigation. No automatic motion
integration is justified by this single scene.

Run the synthetic checks from this directory:

```sh
python3 -m unittest test_near_surface -v
```

Six checks passed on RK3588 with OpenCV 4.10.0: known-disparity depth, flat images,
missing right-image signal, zero disparity, disparity beyond the search range,
and invalid geometry. They exercise numerical/invalid-input behavior and do not
establish field accuracy. Full observations and follow-up validation needs are in
`docs/left-distance-validation-20261125.md` at the repository root.
