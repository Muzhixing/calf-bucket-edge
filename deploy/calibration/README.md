# 原始双目标定采集

本入口独立于识别和追踪程序，不覆盖现用标定。打印棋盘为 10×7 方格、9×6 内角点；2026-11-22 用户确认硬板平整固定，棋盘外沿实测 300×210 mm，因此本次采用每格 30 mm。不得沿用此前屏幕棋盘的 22.73 mm。

板端目录 `/home/elf/deploy/calibration`。先退出占用相机的程序。每个摆位用新目录；同一位置连续帧只用于稳定性检查，不能当作多个独立姿态：

```sh
cd /home/elf/deploy/calibration
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 capture_pair.py \
  --out runs/F01 --pairs 3 --role fit
```

每组保存未校正 SBS、原始左右 PNG、角点叠加图和 `capture.json`。打开 PNG 后逐像素验证保存结果；报告每眼是否找到全部 54 个角点。相机请求 MJPG、2560×720、30 FPS；实际图像必须严格为这个尺寸。左右各 1280×720，沿用已确认的 SBS 左半/右半顺序；不得把旧标定生成的校正图用于求新标定。

`arm_step.py` 复用当前 `vision_cpp/tracking/arm.py` 的原协议发送、遥测、超时和停用确认。它不启动视觉追踪或底盘任务。必须已确认底盘及水泵动力隔离，只操作机械臂。每次命令只改变 A 至多 2°、B 至多 10 mm 或 C 至多 5 mm，本轮收窄为 A=60..100、B=0..60、C=0..20 范围；C 的离散动作复用原控制器的遥测和停用状态机。不修改生产追踪范围。每次动作后核对画面及机构状态，再执行下一步。

```sh
# 示例只适用于最新遥测确为 A90/B0/C0 时，不可盲目重复。
python3 arm_step.py --isolated-chassis-pump \
  --expect 90 0 0 --target 88 0 0 --out runs/move_A88
```

重启后若零点标志未确认，且用户已确认 B 在最顶部、C 完全收回，单独登记零点：

```sh
python3 arm_step.py --isolated-chassis-pump --register-confirmed-home \
  --out runs/zero_registration_NEW_ID
```

登记仅清软件计数，不驱动物理回零；禁止在未知位置执行。工具拒绝已有零点的重新登记。动作程序处理 Ctrl+C/SIGTERM/SIGHUP 并尝试停用；正常结束必须出现 `disabled_confirmed=true`。失去通信时软件无法保证停机，现场应使用已有断电/停机手段。停止相机采集不会发送任何串口指令。

拟合数据与留出验证使用不同物理姿态，分别标记 `--role fit` 和 `--role holdout`。求解前检查角点覆盖、清晰度和姿态变化；固定棋盘配合 A/B 平移与转向未必提供足够俯仰变化，需依据实际图像决定是否追加人工倾斜。验收应同时报告留出图上的校正垂直残差、重投影残差及独立实测距离误差。未通过这些检查前不替换现用配置。

回传本机：

```sh
rsync -a rk3588:/home/elf/deploy/calibration/runs/ \
  /Users/jerry/Downloads/bucket/outputs/stereo_recalibration_20261122/board_runs/
```

`collect_route.py` 接受事先审阅的 JSON 路线（start、board_orientation_group、poses）。单个路点只能变化一轴，自动分成上述小步，每步验证停用；每个路点采图，任何眼未找到完整棋盘即停止。它不根据识别框闭环运动，避免标定期间使用旧校正坐标。

候选求解（不激活）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 solve.py \
  --captures runs --baseline ../vision_v2/calibrations/printed_30mm_20261122.json \
  --out solutions/NEW_ID
```

每个姿态仅取第一对完整左右图参与计算，额外连续帧留作稳定性检查；严格分开 fit/holdout，校验原图哈希。输出候选 JSON、逐姿态 CSV、汇总 JSON。沿用当前双径向、零切向镜头模型，用独立姿态的棋盘格边长重建与垂直残差对照旧标定。30 mm 格边误差不等于桶的绝对测距误差。

[OpenCV 标定说明](https://docs.opencv.org/4.x/d4/d94/tutorial_camera_calibration.html)解释了多姿态采集与重投影评价；近似相同图像不能替代姿态变化。
