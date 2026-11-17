# RK3588固定相机视觉工程

模型固定为seed2_REG_FP16，SHA-256 `bb3d9d9deacf3f099232414139040eae81ea3c362976cbf3a7a13af5cca06c4d`。V2表示新标定与运行入口，未重新训练或转换模型。默认标定为`calibrations/v2_radial2.json`，SHA-256 `21c1c049edc92503f26fee3ce0ba9e70984db5155f48ded53b67002eb976a12a`。

本目录只包含运行代码、必要配置、模型、标定与启动工具。依赖信息见`requirements_vision.txt`，优先复用板端RKNNLite、librknnrt和OpenCV。相机使用固定by-id设备，SBS2560×720、左右各1280×720；物理左目对应SBS左半幅。输入BGR，缩为640×360后上下黑边各140，检测门槛0.536、NMS0.45，沿用原检测解码、坐标还原及双目测距。

## 部署和实时画面

在本机仓库中运行：

```bash
cd deploy/vision_v2
./deploy.sh
./view_live.sh
```

板端部署到`/home/elf/deploy/vision_v2`，原V1/V2部署目录保留。默认SSH为`elf@192.168.31.175`，可用`BUCKET_V2_HOST`设置登录地址；密码不保存在工程中。

实时页面通过SSH隧道打开`http://127.0.0.1:8788/`，只标出当前测距桶，顶部显示桶身距离（cm）、检测状态、处理帧率和新距离更新率。Ctrl+C停止并释放相机。预览最长默认1小时，可用`./view_live.sh --seconds 60`指定时长。板端直接运行：

```bash
cd /home/elf/deploy/vision_v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u tools/live_viewer.py --id live_NEW_ID --seconds 3600
```

直接运行板端命令时另需SSH转发8788端口。不要同时运行预览与采集，它们共用相机。

直接运行 `./view_live.sh` 启用快速测距，运行 `./view_live.sh --depth-scale 1` 切换到全分辨率双目。预览把采集、校正、逐帧检测与双目匹配分开运行。相机使用两个 V4L2 缓冲区，检测读取最新校正左目，双目进程每次只处理一组同源左右图。输出保留测距来源帧号与样本年龄。检测丢失、目标框切换或样本过期时清除旧距离。页面显示最近 3 秒的新距离更新率。持续沿用旧读数时显示黄色提示，未检出桶或距离无效时显示横线。帧号、距离来源及延迟可在 `/status` 和逐帧日志中查看。

| 固定相机、单桶、单个 SSH 预览客户端 | 电脑收图 FPS | 新距离 Hz | 新样本从取图到发布的中位延迟 |
| --- | ---: | ---: | ---: |
| 原同步入口 `./view_live.sh --mode sync`，25 秒 | 14.93 | 1.90 | — |
| `./view_live.sh --depth-scale 1`，全分辨率双目，45 秒 | 30.02 | 2.98 | 375 ms |
| 默认 `./view_live.sh`，快速双目，默认入口完整运行 45 秒 | 30.01 | 9.92 | 141 ms |

快速模式只将视差计算降至 640×360，按原图尺度还原视差；检测继续使用同一模型和完整的校正左目。前一次快速预览在第 635 帧遇到 USB 相机掉线。相机接到同一集线器的 4 号口后，显式选择快速模式实测收图 30.02 FPS、新测 9.96 Hz。设为默认后复测 45.01 秒，电脑收到全部 1,352 张带框画面。443 帧发布新距离，895 帧沿用，启动时 14 帧等待首个有效距离。用 17 对已采图像对比两种尺度，原尺度深度覆盖 0.881–0.895 m，均得到有效结果。两者差值绝对值的中位数为 0.56 cm，最大 1.13 cm。0.50 m 作业边界需要单独实测。

全分辨率下申请三核 `--cores 0_1_2` 实测收图 30.02 FPS、新测 2.97 Hz，默认使用 AUTO。每次预览在板端 `logs/<运行编号>/` 保存 `environment.json`、逐帧 `frames.jsonl` 和结束汇总 `completion.json`。完整留存左右图、张量使用采集入口，前次 14 帧实测 4.98 FPS。采集并行编码同一帧的文件，写入成功后才发布该帧记录。原始双目 PNG 采用默认无损编码，本次每帧约 1.23 MB。

## 外接 HDMI 直显

在板端终端或 SSH 会话运行以下命令。若网页预览或正式采集正在使用相机，先用其启动终端的 Ctrl+C 正常收束并保存记录。停止桌面管理器会结束板上图形登录会话，先保存桌面工作。

```bash
sudo systemctl stop display-manager
cd /home/elf/deploy/vision_v2
./view_hdmi.sh
```

启动默认使用快速测距，`./view_hdmi.sh --depth-scale 1` 使用全分辨率测距，`--seconds 60` 可限定运行时长。前台按 Ctrl+C 停止。从另一 SSH 会话停止同一用户的直显程序，可运行：

```bash
pkill -INT -u "$(id -u)" -f '^python3 -u tools/live_viewer.py --hdmi( |$)'
```

需要桌面时在直显退出后运行 `sudo systemctl start display-manager`。这些命令只改变本次运行状态，板卡保留原开机目标。

本地输出复用一份双目输入、检测和测距，通过独立显示线程和单帧缓冲送入 GStreamer appsrc / kmssink。HDMI 模式不启动 HTTP、WebRTC 或浏览器。运行依赖包括板上已有的 GStreamer 1.20.3、kmssink、Python GI、Pillow、Noto CJK 字体及 libdrm。程序从 Rockchip 显示卡枚举已连接的 HDMI-A-1，核对支持 1280×720 后选用其 connector ID。板上实测 HDMI-A-1 为 432，输出 1280×720、60 Hz；DSI 使用另一个显示管线。

主图显示校正左目、桶框、画面十字和桶框中心。原始左右图取自该帧的同一双目缓冲。橙框是最近有效距离来源帧的实际采样区域，缩略视差图来自同一次 SGBM 结果，以黑色标出无效像素。顶部标记新测、沿用或无有效值，底部显示偏差、当前帧和深度来源帧。桶口作业中心待独立定位。日志目录 `logs/hdmi_*/` 下的 `hdmi_stats.json` 按约 5 秒刷新，分别统计提交给 GStreamer 和 sink 已渲染的帧；退出汇总写入 `completion.json`，合成画面样例为 `hdmi_preview.jpg`。

2026-11-17 实测外屏画面经现场确认。60 秒现场窗口没有检出桶，处理 1,802 帧，KMS 统计已渲染 1,801 帧，约 30 FPS。保持同一网页视觉进程运行，仅临时停止桌面管理器，系统 MemAvailable 从 2,447,612 KiB 增至 3,118,756 KiB（约 +655 MiB）。HDMI 无桶窗口中，视觉进程树 PSS 约 237 MiB，双目工作进程尚未因检测到桶而启动。历史有桶图像离屏复核得到 0.8873 m，实际采样矩形为 [430,234,874,642]，显示用视差缩略图与该工程采集帧已保存的视差下采样逐像素相同。

## 两轴对齐与投喂预演

白色十字为校正左目画面中心 (640, 360)，圆点为当前整桶检测框的中点。橙框表示仍需调整，绿框表示左右、上下均在容差内。左右容差沿用 `controller.center_tolerance`，上下容差由 `reference_contract.vertical_tolerance_fraction_of_half_height` 配置。当前各取对应半幅的 0.08，即横向 ±51.2 px、纵向 ±28.8 px，含边界。纵向数值为初始联调值。

在运行预览的电脑或板上读取 `curl -s http://127.0.0.1:8788/status`。同一份 `guidance` 随每帧写入 `logs/<运行编号>/frames.jsonl`。

| 字段 | 对接含义 |
| --- | --- |
| `guidance.offset_px` | 桶框中心减画面中心的 [dx, dy]，右、下为正。目标丢失时为 null。 |
| `guidance.estimated_offset_mm` | 按当前有效桶身深度估算的横、纵偏移，依次为 dx × Z / fx、dy × Z / fy，Z 用毫米，焦距来自校正投影矩阵 P1。相同深度平面内以画面中心射线为基准；距离无效时为 null。 |
| `guidance.state` | `no_target` 未检出，`adjust` 至少一轴超差，`aligned_wait_depth` 两轴已对齐并等待本次对齐后的新测距，`ready_dry_run` 具备视觉预演条件。 |
| `guidance.reference_depth_mm` | 就绪时的桶身前向深度，单位 mm，供后续计算伸长量。其余状态为 null。 |
| `guidance.cycle_id`、`guidance.feed_preview_started` | 每次连续对齐使用同一个周期编号，首次收到来源帧也在本次连续对齐期间的新深度时，仅一帧标记 true。轮询端按周期编号去重；离开容差、丢失或更换目标后重新判定。 |

同步消费同一条记录的 `frame_id`、`measurement_status`、`distance_source_frame` 和 `guidance`。实时轮询端同时检查 `frame_age_s`、帧号是否继续推进及距离的新旧，沿用值供显示。视觉周期用于预演，实际一次投喂任务仍由控制器管理启动、完成回执和重复执行。前向参考深度以相机为起点，出水口的零位、轴向、桶前壁到投放点偏移以及可用行程须在装配后实测，再换算电机方向和伸长量。

## 桶身距离采集

相机固定，桶完整可见即可，暂不要求精确桶口中心。量尺起点D为左镜头外壳前沿，终点B为朝向相机的桶前壁实体表面；水平量尺读数与相机前向深度分开记录。0.85是以下命令的填写示例，须替换为实际米数：

```bash
./run_body.sh capture B01_R1 --body-datum-m 0.85 --direction horizontal
./run_body.sh pull
```

未量尺时可省略`--body-datum-m`，只记输出，不评价误差。不需要桶口点或桶口距离。每次新摆放使用新编号，连续帧不算独立重复。原算法仍从扩大框统计视差，尚未确认输出等于桶前壁距离。

`captures/ID/`保存左右图与时间戳；`results/ID__v2_radial2/seed2_REG_FP16/`保存叠加图、检测/距离和`body_assessment/`；`results/body_placements.csv`是桶身阶段逐摆位表。生成的数据、日志和测试文件不提交Git。

出水口安装在相机旁，由伸缩臂移到桶口上方。当前只做相对测距；装配时再核对出水口相对相机的偏移和伸缩覆盖范围，确定停车区间。原0.50m字段不是已经标定的出水口到位条件。

全部入口只写视觉结果，不导入或启动STM32串口、车辆控制器和执行机构。`config/`中的哈希清单用于防止误载模型或代码漂移；本目录没有实验输出、采集样例或历史研究资料。
