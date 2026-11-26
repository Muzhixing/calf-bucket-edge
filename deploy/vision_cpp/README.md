> 当前维护入口已迁移至标准`bucket_robot`包。请先阅读仓库根目录README、`docs/architecture.md`与`docs/release.md`。以下保留原工程/现场说明；其中旧目录启动和测试命令不再作为本地重构版的操作入口。

# RK3588 原生实时视觉

实时采集、左目校正、冻结 RKNN 检测、独立进程双目测距及 HDMI 显示运行在 C++ 中。原 `run.sh` 同时启动 Python 追踪与控制进程，读取同帧原始左图和识别结果，提供网页及原 STM32 协议控制。采集、追踪与网页预览独立运行，各自只读取最新帧；耗时重获不会阻塞预览。`vision_v2` 保留当前模型、标定和采集模块。

## 使用

本机原入口保持不变：

```sh
cd deploy/vision_v2
./view_live.sh                         # C++，快速双目
./view_live.sh --depth-scale 1         # 全分辨率双目
./view_live.sh --observe-only          # 只读，不开放机构启动
```

板端独立部署目录 `/home/elf/deploy/vision_cpp`：

```sh
cd /home/elf/deploy/vision_cpp
./run.sh --seconds 3600 --id live_001  # HTTP 仅监听 127.0.0.1:8788
./run.sh --hdmi --seconds 3600        # HDMI-A-1，先停止 display-manager
```

也可继续使用 `/home/elf/deploy/vision_v2/view_hdmi.sh`。HDMI 也提供环回 HTTP 控制页，并显示追踪框。前台 Ctrl+C 先请求禁用机构、等待回传，再退出原生程序并释放相机与 DRM 显示。其他会话应向 `tracking/runtime.py` 主进程发送 SIGINT。需要桌面时，在直显退出后运行 `sudo systemctl start display-manager`。

## 识别后追踪

1. 启动原主程序后，默认自动识别、积累确认、锁定并连续跟随，不需要锁定后再点“开始”。首次运动前检查新鲜串口回传、零位/航向、故障和 A/B/C 当前范围。`--observe-only` 或回放模式不启动机构。
2. 控制页显示跟随状态。“停止机构”撤销自动运行意图，持续保持停止；“恢复追踪”或“自动识别并跟随”明确恢复。自动模式与页面生命周期独立，关闭页面不会停止板端程序。
3. 初次获取仍使用原模型：1.2 秒窗口内含 ≥0.65 分候选需 3 次确认，否则 ≥0.536 的输出需 5 次确认，跨度至少 0.3 秒、相邻命中间隔不超过 0.6 秒。锁定后前景光流与外观参考桥接模型漏检。
4. 目标短暂丢失先禁用机构。当前位置/外观重获与连续新检测确认并行，不必等旧身份5秒超时；边缘部分可见目标由当前检测证据单独确认。超出身份保留时间后自动重新检测、确认并锁定，目标 ID 改变；不把重新检测的新 ID 计作同目标重获。相机/串口/零位/故障异常撤销自动运行，需要处理后明确恢复或重启。
5. A 按相机内参计算角误差，结合带时间戳的舵机软件位置估计和限幅预测持续更新；使用本次实机拟合的约 140 ms 有效反馈延迟对齐画面与角度，预测须有连续趋势证据。原协议最多每 40 ms 一帧、最多 3 帧待确认，按回传计数累计确认；不等待每次 A 到位再禁用。A 命令限速 30°/s，加速 180°/s²、减速 360°/s²，并按剩余角度提前减速，范围 60–100°，整度编码。B 按关联的新鲜距离换算竖直误差，缺测时用现场增益，保持 0–96 mm；C 固定 10 mm。原固件 B 有限步进行程中，A 更新保留 B 目标；竖直反向需要停止当前步进。
6. 原固件无通信超时停机，且使能机构同时启用底盘航向保持；早期联调断开行走动力；2026-11-22最终测试时轮子已接入、车身垫高。退出主程序走禁用确认流程。PWM 软件角度和步进脉冲位置不等于外部物理位置测量。

RK3588 使用 NPU 0/1/2、A76 CPU 4/6、单个 OpenCV 跟踪线程。板内 `/capture.bgr` 传递与检测结果同帧的原始左图，避免 JPEG 往返；网页和诊断 `/capture.jpg` 保留。光流在半分辨率运行，特征仅在目标 ROI 提取，外观比较使用相同采样网格。`BUCKET_TRACK_CPUS` / `BUCKET_TRACK_THREADS` 可覆盖计算配置，`--cores auto` 可覆盖 NPU 核配置。

`--step-control` 保留旧版手动开启、每步禁用的控制路径用于回退，该模式仍要求页面心跳；默认连续模式不使用该路径。原发送器、冻结模型和 STM32 固件不变。当前实测版本、依赖与启动方式见 [版本记录](../../docs/tracking-release-20261122.md)。

## 冻结口径

网页分别显示画面输出与追踪处理帧率；输出帧率不等于浏览器实际绘制帧率。超过150毫秒的跟踪结果不叠加到新画面，避免旧锁定框误导。画面保留准星与目标框，不绘制容差圆；后台10%居中保持、15%离圈恢复规则继续生效。

- 模型为 B1_C0_REG_FP16（2026-11-22 用户授权切换），SHA-256 `4a59b48b634c25ca79d843cc746d8e062ca1161ec5ef7c0c0c96533c78d3fce7`。启动打印绝对路径并验证哈希，INT8 输出由 RKNN runtime 解量化，沿用本轮已验证的 B1 C0 转换产物。
- 标定源为 `printed_30mm_20261122.json`，SHA-256 `5917ede19309a4f78d810b3c9c91d4e8033c23751307e6879f7e3f55f5c7031e`；准备时验证源文件，启动时验证导出矩阵的 SHA-256。
- 固定 by-id 相机，MJPG 2560×720、30 FPS，SBS 左半是物理左目。检测输入 RGB（OpenCV BGR 经显式换色），640×360 后上下各补 140 黑边，门槛 0.536、NMS 0.45；先截断模型坐标再还原至校正左目。NMS 同分排序匹配板端 NumPy 2.2.1 的 ARM 标量 argsort。
- 常规检测漏检时，每三帧最多增加一次同帧水平镜像补检，使用同一模型、输入尺寸和阈值。补检框恢复到原图坐标后才参与锁定与测距；不使用历史框填充当前检测。`detector.recovery_view_attempted/used/ms` 记录补检行为，弱补检框不用于建立目标。模型权重保持冻结，但检测策略已增加这个补充视图。
- 双目沿用原 SGBM、ROI/分位数/MAD/质量判定和 0.9 平滑。默认视差计算尺度 0.5；`--depth-scale 1` 全分辨率。保留一次重复 ROI 审计，不把减少原校验工作计作语言收益。
- 单个在途双目任务，不排队累积旧帧；源帧年龄最多 1.5 秒，IoU 至少 0.5 且目标代次一致才接收。目标丢失、更换或结果过期清空距离。
- `measurement_status` 为 `new` / `held` / `invalid`。`distance_source_frame`、`distance_age_ms`、`sampling_roi` 指向有效距离来源；`attempt_source`、`discarded`、`distance_reason` 解释最近一次测距。距离是框内视差对应的相机前向 Z，未承诺等于桶口或前壁。
- 十字为校正左图中心；桶框中心偏差右、下为正。原生 `guidance` 保留 ±51.2/±28.8 px 的视觉预演容差；新增 `tracking` 使用上述更小的居中容差。控制使用追踪框中心；A按图像方向调整，B可使用关联的新鲜距离，不执行投喂。

## 构建与部署

本机运行 `deploy/vision_v2/deploy.sh` 同步原工程后构建原生后端；也可在已部署 V2 的板上单独运行 `deploy/vision_cpp/deploy.sh`。安装只同步源文件白名单，复用当前模型、标定与板端依赖。

该板复用 `/usr/lib/librknnrt.so` 和先前核验的 OpenCV 4.10 构建。`build.sh` 默认使用当前目录 `deps/install` 的 OpenCV 头文件/库及 `deps/rknn/rknn_api.h`，分别可用 `OPENCV_PREFIX`、`RKNN_SDK_DIR` 覆盖。GStreamer、glib、libdrm、OpenSSL 链接板上现有运行库；缺少的 GStreamer 开发包仅下载解包到 `deps/headers`，不替换系统库。

```sh
python3 prepare.py /home/elf/deploy/vision_v2
./build.sh
```

构建将必需的 OpenCV 动态库复制到 `lib/`；运行通过部署目录的库路径加载。追踪进程另需板端 Python 3、NumPy、OpenCV（包含 LK 光流），当前板已具备。实际相机更新率取决于场景、ROI 与温度；相机 30 FPS 不意味着每帧都有新的有效距离。此前未接入追踪的同源 HDMI 回放约30 FPS、10次/秒新测距，进程树PSS约179.9 MiB；该内存结果不包含新增追踪进程。

板端验证：`PYTHONPATH=tracking python3 -m unittest discover -s tracking/tests -v`。测试使用合成移动/遮挡图像和 PTY 模拟 STM32，覆盖目标身份、过期、范围、固定 C、禁用回传和回传丢失；不向真实串口发送。

当前板卡已将核验过的 OpenCV 开发文件和 RKNN 头文件放入上述 `deps/` 目录。新板需准备相同依赖；`build_opencv.sh` 保留4.10.0构建参数，需要事先放入对应OpenCV源码包与CMake3.29.5安装包。厂商头文件、依赖包和二进制不提交Git。

当前仅保留 30 mm 打印棋盘标定；[标定及约 49 cm 实测确认](../../docs/stereo-recalibration-20261122.md)。准备时缺少新参数或哈希不符即停止，不回退旧参数。
