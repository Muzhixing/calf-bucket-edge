# RK3588 frozen V2 Python/C++ comparison

This isolated experiment asks whether migrating the visual pipeline to C++ improves new stereo measurements, rather than merely redisplaying frames faster. The production entry in `deploy/vision_v2` is unchanged. Nothing here imports a controller or opens a serial port.

The model is `seed2_REG_FP16`, SHA-256 `bb3d9d9deacf3f099232414139040eae81ea3c362976cbf3a7a13af5cca06c4d`. The V2 radial2 calibration SHA-256 is `21c1c049edc92503f26fee3ce0ba9e70984db5155f48ded53b67002eb976a12a`.

## Board result, 2026-11-18

Three controlled 15-second windows per implementation and scale, after five seconds of warmup. Values below are means of per-window metrics; latency columns are means of per-window percentiles. Recorded positive pairs were replayed at 30 Hz; display was disabled.

| Mode / implementation | Output FPS | New depth Hz | Latency P50 / P95 ms | Process-tree PSS MiB |
| --- | ---: | ---: | ---: | ---: |
| Fast / stock Python | 29.97 | 10.01 | 147.4 / 148.1 | 293.3 |
| Fast / C++ pipeline | 29.98 | 10.05 | 145.3 / 145.6 | 148.6 |
| Full / stock Python | 29.98 | 3.10 | 375.1 / 384.2 | 311.1 |
| Full / C++ pipeline | 29.97 | 3.10 | 363.8 / 394.6 | 162.8 |

The matched-library Python control gave 10.02 / 3.03 Hz. All 116 final per-pair comparisons matched boxes, scores, distances, state and source; 8,127 asynchronous records passed source-aware checks. Present/lost/reacquired traces exercised all three measurement states without retaining a value after target loss. PSS includes the same roughly 90 MiB replay cache and all children, not the observer. Real camera acquisition alone delivered 29.999 FPS over 300 frames; that fresh view did not yield valid depth, so it is not the positive-depth performance basis.

Recommendation: retain the production Python pipeline for throughput. C++ saves about 145–148 MiB in this replay, but does not materially increase new measurements per second. The main bottleneck is CPU SGBM, already implemented in native OpenCV; its median was about 33 ms fast / 263 ms full. Investigate duplicate ROI statistics and result pickup latency independently before a whole-application rewrite. Detailed evidence is retained locally under `outputs/cpp_vision_20261118` and on the board under this experiment's `results/`, not uploaded with source.

## Implementations and controls

- `python-stock`: deployed Python algorithms with the installed OpenCV 4.10.0 wheel and RKNNLite 2.3.2. Async mode runs the existing `preview_pipeline.run` with a paced recorded capture source. Wrappers only time stages.
- `python-matched`: exactly the same Python benchmark, using the isolated OpenCV shared library and Python binding built together with the C++ version.
- `cpp`: RKNN C API, nine `want_float=1` output tensors, BGR letterbox, DFL, NMS, original integer coordinate restoration, full/half-scale SGBM and original ROI percentile/MAD filtering. Uses the same shared OpenCV library as `python-matched`.
- `cpp-pipeline`: the same C++ detector/ranger, with persistent rectification workers and a shared-memory stereo process. Like the original Python pipeline, parent OpenCV uses one thread, stereo uses three, and affinity is detection core 6, rectification cores 5/2, stereo cores 4/6/7 with nice 5. The NPU context is initialized only in the parent, after the stereo process is forked.

The stock wheel statically embeds OpenCV and cannot be linked by a standalone C++ application. A stock-versus-C++ result alone confounds build differences with language changes. The matched control separates these effects. The first `cpp` async variant serializes left rectification and detection and uses a stereo thread; its results remain as an architectural counterexample. Use `cpp-pipeline` for the primary asynchronous comparison. Neither prototype implements the production display or a real camera capture loop; both use recorded pairs.

## Board setup

Working directory: `/home/elf/experiments/cpp_vision`. Runtime sources/model/calibration are read from `/home/elf/deploy/vision_v2`. Keep the production directory intact.

```sh
cd /home/elf/experiments/cpp_vision
OPENBLAS_NUM_THREADS=1 python3 prepare.py --root /home/elf/deploy/vision_v2 --out prepared
# Dependencies: GCC, Python headers, NumPy, the existing RKNN SDK header/runtime.
# Put the cmake 3.29.5 aarch64 wheel and OpenCV 4.10.0 source tarball in deps/.
# Source: https://github.com/opencv/opencv/tree/4.10.0
# Tarball filename expected below: deps/bucket-opencv-4.10.0.tar.gz
sh build_opencv.sh > results/opencv_build.log 2>&1
sh build.sh
```

Do not benchmark while compiling. The build is local to `deps/install`, including its Python binding; it does not replace the installed wheel. The existing RKNN header is used through `RKNN_SDK_DIR`, defaulting to the field research package on this board. `build.sh` links the existing `/usr/lib/librknnrt.so`.

## Reproduce measurements

Use unique IDs; the launcher refuses to overwrite results, verifies model/calibration/input hashes, records the command, and samples the entire process tree's PSS/RSS every 0.5 seconds. No image saving or display occurs during timed replay. Both implementations buffer JSON rows and write them after the timed window.

```sh
python3 run.py --impl python-stock --mode async --scale 0.5 --seconds 20 --id python_fast_r1
python3 run.py --impl python-matched --mode async --scale 0.5 --seconds 20 --id matched_fast_r1
python3 run.py --impl cpp --mode async --scale 0.5 --seconds 20 --id cpp_fast_r1
# Repeat with unique IDs, then repeat all three with --scale 1.
```

Primary repeated comparison with the separated C++ pipeline and a controlled starting temperature:

```sh
python3 suite.py --tag controlled_NEW --seconds 15 --repeats 3 --controlled
```

This rotates `python-stock`, `python-matched`, and `cpp-pipeline`, waits for a SoC temperature at or below 60°C before each run, then measures 15 seconds after five seconds of warmup. It does not change governors or thermal protections. A five-minute cooling timeout stops the suite. The board's first two passive thermal trips are 75°C/85°C, so continuous hot-load results are retained separately and are not used as the primary language comparison. Memory samples also record actual CPU/NPU frequency.

Async replay preloads 17 historical SBS pairs and repeatedly presents them at a **synthetic 30 Hz cadence**; original capture timestamps remain in `prepared/binding.json`. Five seconds of pipeline warmup are excluded. This isolates compute and scheduling and is not a fresh live-camera or motion-accuracy test. PNG decoding is measured separately and excluded from compute throughput; cold-start and the preload cache are included in memory samples, with the report selecting samples inside the measurement window.

Maximum sequential processing and per-frame equivalence:

```sh
python3 run.py --impl python-stock --mode sync --scale 0.5 --frames 68 --warmup 17 --id py_pair_fast
python3 run.py --impl python-matched --mode sync --scale 0.5 --frames 68 --warmup 17 --id matched_pair_fast
python3 run.py --impl cpp --mode sync --scale 0.5 --frames 68 --warmup 17 --id cpp_pair_fast
python3 compare.py results/py_pair_fast/frames.jsonl results/cpp_pair_fast/frames.jsonl --out results/parity_fast
```

Sync mode measures every pair; it cannot be equated to asynchronous live throughput. Compare async raw distances against the corresponding **source input**, not necessarily the current display input: the implementations may finish measurements on different display frames. `new` is a new accepted measurement, `held` retains an unexpired value from the same target, and `invalid` has no value. The prototype retains 0.5 IoU target association, 1.5 s expiration and 0.9 smoothing. Result differences and all failure rows are retained by `compare.py`; equal-score NMS ordering requires specific scrutiny before replacing the deployed implementation.

Real capture/decode alone, when the camera is free:

```sh
python3 camera_probe.py --out results/camera_probe_NEW --frames 300
```

This opens the configured USB camera for a bounded run, measures V4L2 grab/wait and retrieve/decode separately, saves one final SBS image, and releases the camera. It is not a depth-throughput measurement.

`prepare.py` also prepares `qa_corpus.txt`: a sampled historical moving-bucket sequence and a synthetic blank. Use `run.py --qa --mode sync --warmup 12 --frames 24` for coordinate/invalid-state regression, with unique IDs and each scale. These inputs are engineering regression data, not the sealed model test set. They do not supply independent range ground truth.

`state_cases.py` prepares a present/lost/reacquired replay from those verified records. `check_trace.py REFERENCE_JSONL TRACE_JSONL --out CHECK_JSON` verifies the current detection, distance against its source pair, new/held/invalid labels and clearing on loss. `visualize_pairs.py` renders the recorded paired predictions after tests, without affecting timing windows.

The live shared RKNN runtime was queried as **2.1.0**, driver **0.9.8**, despite RKNNLite and the model compiler being 2.3.2. All arms use the same existing `/usr/lib/librknnrt.so`; this experiment does not upgrade it.

## Outputs and stopping

`results/ID/frames.jsonl` contains frame/input IDs, boxes, score, raw/published distance, measurement state and source, latency and stage timings. `results/ID_memory.json` includes parent plus all descendants, including the Python stereo worker. Summing RSS double-counts shared pages; summed PSS is the main comparison metric. Temporary inputs, binaries, dependencies and results are Git-ignored.

Each command exits at its requested duration/frame count. Ctrl+C stops a foreground test. There is no background service or autostart installation. Copy evidence back with:

```sh
rsync -a elf@192.168.31.175:/home/elf/experiments/cpp_vision/results/ outputs/cpp_vision_20261118/results/
```
