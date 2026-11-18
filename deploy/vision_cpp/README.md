# RK3588 原生实时视觉

实时采集、左目校正、冻结 RKNN 检测、独立进程双目测距及网页/HDMI 显示均运行在 C++ 中。部署准备脚本使用 Python 生成标定矩阵与中文标签，实时运行不保留 Python 解释器。`vision_v2` 保留当前模型、标定和 Python 采集模块；旧 Python 实时后端已移除。

## 使用

本机原入口保持不变：

```sh
cd deploy/vision_v2
./view_live.sh                         # C++，快速双目
./view_live.sh --depth-scale 1         # 全分辨率双目
```

板端独立部署目录 `/home/elf/deploy/vision_cpp`：

```sh
cd /home/elf/deploy/vision_cpp
./run.sh --seconds 3600 --id live_001  # HTTP 仅监听 127.0.0.1:8788
./run.sh --hdmi --seconds 3600        # HDMI-A-1，先停止 display-manager
```

也可继续使用 `/home/elf/deploy/vision_v2/view_hdmi.sh`，进入 C++。前台 Ctrl+C 退出，释放相机与 DRM 显示。其他会话可用 `pkill -INT -u "$(id -u)" -f '^/home/elf/deploy/vision_cpp/vision( |$)'` 停止本用户的原生预览。需要桌面时，在直显退出后运行 `sudo systemctl start display-manager`。

每次使用新编号。`runs/<id>/` 保留启动绑定、逐帧 JSONL、首组原始左右图、校正叠加图与结束汇总；HDMI 另保存合成图、实际 sink 渲染统计。`rsync -av elf@192.168.31.175:/home/elf/deploy/vision_cpp/runs/ ./outputs/vision_cpp_runs/` 可从仓库根目录取回。生成目录、模型副本、动态库和构建产物均不提交 Git。

## 冻结口径

- 模型仍为 seed2_REG_FP16，SHA-256 `bb3d9d9deacf3f099232414139040eae81ea3c362976cbf3a7a13af5cca06c4d`。启动打印绝对路径并验证哈希，INT8 输出由 RKNN runtime 解量化，未重新转换模型。
- 标定源为 `v2_radial2.json`，SHA-256 `21c1c049edc92503f26fee3ce0ba9e70984db5155f48ded53b67002eb976a12a`；准备时验证源文件，启动时验证导出矩阵的 SHA-256。
- 固定 by-id 相机，MJPG 2560×720、30 FPS，SBS 左半是物理左目。检测输入 BGR，640×360 后上下各补 140 黑边，门槛 0.536、NMS 0.45；先截断模型坐标再还原至校正左目。NMS 同分排序匹配板端 NumPy 2.2.1 的 ARM 标量 argsort。
- 双目沿用原 SGBM、ROI/分位数/MAD/质量判定和 0.9 平滑。默认视差计算尺度 0.5；`--depth-scale 1` 全分辨率。保留一次重复 ROI 审计，不把减少原校验工作计作语言收益。
- 单个在途双目任务，不排队累积旧帧；源帧年龄最多 1.5 秒，IoU 至少 0.5 且目标代次一致才接收。目标丢失、更换或结果过期清空距离。
- `measurement_status` 为 `new` / `held` / `invalid`。`distance_source_frame`、`distance_age_ms`、`sampling_roi` 指向有效距离来源；`attempt_source`、`discarded`、`distance_reason` 解释最近一次测距。距离是框内视差对应的相机前向 Z，未承诺等于桶口或前壁。
- 十字为校正左图中心；桶框中心偏差右、下为正。两轴容差 ±51.2/±28.8 px，`guidance` 保留 `offset_px`、毫米估计、对齐状态和一次性投喂预演周期。没有串口、STM32 或执行机构代码。

## 构建与部署

本机运行 `deploy/vision_v2/deploy.sh` 同步原工程后构建原生后端；也可在已部署 V2 的板上单独运行 `deploy/vision_cpp/deploy.sh`。安装只同步源文件白名单，复用当前模型、标定与板端依赖。

该板复用 `/usr/lib/librknnrt.so` 和先前核验的 OpenCV 4.10 构建。`build.sh` 默认使用当前目录 `deps/install` 的 OpenCV 头文件/库及 `deps/rknn/rknn_api.h`，分别可用 `OPENCV_PREFIX`、`RKNN_SDK_DIR` 覆盖。GStreamer、glib、libdrm、OpenSSL 链接板上现有运行库；缺少的 GStreamer 开发包仅下载解包到 `deps/headers`，不替换系统库。

```sh
python3 prepare.py /home/elf/deploy/vision_v2
./build.sh
```

构建将必需的 OpenCV 动态库复制到 `lib/`；运行通过部署目录的库路径加载，不依赖研究目录中的模型、回放图或 Python 环境。实际相机更新率取决于场景、ROI 与温度；相机 30 FPS 不意味着每帧都有新的有效距离。已核验同源 HDMI 回放稳定约30 FPS、10次/秒新测距，进程树PSS约179.9 MiB。

当前板卡已将核验过的 OpenCV 开发文件和 RKNN 头文件放入上述 `deps/` 目录。新板需准备相同依赖；`build_opencv.sh` 保留4.10.0构建参数，需要事先放入对应OpenCV源码包与CMake3.29.5安装包。厂商头文件、依赖包和二进制不提交Git。
