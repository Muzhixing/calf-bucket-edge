# Reused tracking components

- OpenCV 4.10.0: `cv2.KalmanFilter` implements prediction/correction, `calcOpticalFlowPyrLK` implements pyramidal Lucas–Kanade flow, `estimateAffinePartial2D` provides RANSAC geometry, and `cv::UMat`/OpenCL runs rectification. These are library calls, not reimplementations. OpenCV is Apache-2.0 licensed; the installed OpenCV distribution retains its own license.
  https://docs.opencv.org/4.10.0/dd/d6a/classcv_1_1KalmanFilter.html
  https://github.com/opencv/opencv/blob/4.10.0/LICENSE
  Background compensation also uses these existing LK/RANSAC implementations, on a masked 320x180 image. It supplies velocity evidence only, never target visibility. API references:
  https://docs.opencv.org/4.10.0/dc/d6b/group__video__track.html
  https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html
  Fine-grid weighted correlation also reuses OpenCV `dft`, `mulSpectrums` and `idft` with packed real spectra and the documented `nonzeroRows` optimization. `matching.py` shares image/weight transforms and bounds reusable reference spectra; it preserves the weighted NCC formula, original pixel grid and thresholds. Broad/low-contrast searches keep `matchTemplate`.
  https://docs.opencv.org/4.10.0/d2/de8/group__core__array.html
- ByteTrack (ECCV 2022), FoundationVision/ByteTrack: high-confidence detection association followed by lower-confidence continuation is the design reference. This project adapts that association policy to one operator-selected/automatically acquired bucket and verifies it with current image geometry. It does not claim to run the full ByteTrack multi-object tracker or copy its detector/Kalman implementation.
  https://github.com/FoundationVision/ByteTrack
  https://arxiv.org/abs/2110.06864

Motion predictions only guide association and bounded compensation for fresh verified observations. They never authorize following an invisible target. Low-score detections cannot initialize a new target, choose a different identity, or independently replace geometric/appearance validation.

- Ultralytics test-time augmentation is the reference for horizontal-flip inference and mapping predicted boxes back to the original view: https://docs.ultralytics.com/yolov5/tutorials/test-time-augmentation/ . The exported RKNN model does not implement Ultralytics' PyTorch `--augment` option. The host uses OpenCV `flip` and the unchanged RKNN model for a bounded extra pass, at most every third missed normal-view frame. This is a small scheduling/coordinate adapter, not the full Ultralytics multiscale TTA implementation; no Ultralytics implementation code was copied.

- Optional CSRT: OpenCV 4.10.0 contrib `TrackerCSRT_create` supplies the correlation-filter implementation; `correlation.py` adapts its candidate rectangle and applies existing identity checks. No CSRT algorithm was reimplemented or pretrained weights added. The tested isolated distribution is `opencv-contrib-python-headless==4.10.0.84`; its distribution licenses remain in the dependency directory. Runtime use is opt-in with `--csrt`. The accepted 2026-11-22 configuration leaves it disabled after replay and field diagnosis found synchronous slow frames without sufficient additional accepted targets.
  https://docs.opencv.org/4.10.0/d2/da2/classcv_1_1TrackerCSRT.html
  https://openaccess.thecvf.com/content_cvpr_2017/html/Lukezic_Discriminative_Correlation_Filter_CVPR_2017_paper.html
