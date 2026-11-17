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

实时页面通过SSH隧道打开`http://127.0.0.1:8788/`，显示桶框、距离、new/held/invalid、来源帧和FPS。Ctrl+C停止并释放相机。预览最长默认1小时，可用`./view_live.sh --seconds 60`指定时长。板端直接运行：

```bash
cd /home/elf/deploy/vision_v2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u tools/live_viewer.py --id live_NEW_ID --seconds 3600
```

直接运行板端命令时另需SSH转发8788端口。不要同时运行预览与采集，它们共用相机。

固定机位单桶实测，完整预览处理 15.0 FPS，SSH 隧道接收 MJPEG 约 14.8 FPS。含前后处理的检测平均 18.2 ms，其中 RKNN 调用 17.2 ms。每 8 帧计算一次全分辨率双目视差，新距离约 1.86 次/秒。写全量图像和张量时，14 帧实拍平均 4.98 FPS。相机协商 30 FPS。

预览页面连续接收最新图像。OpenCV 使用四个线程，检测先按冻结分数门槛筛选候选位置，再对保留的位置解码。采集并行编码同一帧的文件，写入成功后才发布该帧记录。原始双目 PNG 采用默认无损编码，本次每帧约 1.23 MB。

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
