> 当前维护入口已迁移至标准`bucket_robot`包。请先阅读仓库根目录README、`docs/architecture.md`与`docs/release.md`。以下保留原工程/现场说明；其中旧目录启动和测试命令不再作为本地重构版的操作入口。

# 当前模型、标定与采集支撑

本目录保留 `vision_v2` 名称以兼容已有板端路径。实时预览只有 C++ 后端；Python 模块用于部署准备、保存左右图和独立参照验收。

## 启动与部署

```sh
./deploy.sh
./view_live.sh                     # 快速双目
./view_live.sh --depth-scale 1     # 全分辨率
./view_live.sh --seconds 60        # 限定时长
```

默认连接 `elf@192.168.31.175`，可设置 `BUCKET_V2_HOST`；可用 `BUCKET_SSH_SOCKET` 复用 SSH 主会话。密码不保存在工程中。网页经 SSH 隧道打开，Ctrl+C 停止。

板端 HDMI：

```sh
sudo systemctl stop display-manager
cd /home/elf/deploy/vision_v2
./view_hdmi.sh
```

退出后运行 `sudo systemctl start display-manager` 恢复桌面。上述命令不修改开机目标。实时日志在 `/home/elf/deploy/vision_cpp/runs/<编号>/`，启动时打印最终模型绝对路径及哈希。

## 冻结配置

- seed2_REG_FP16：`bb3d9d9deacf3f099232414139040eae81ea3c362976cbf3a7a13af5cca06c4d`。
- 当前标定 `calibrations/printed_30mm_20261122.json`：`5917ede19309a4f78d810b3c9c91d4e8033c23751307e6879f7e3f55f5c7031e`。
- 固定 by-id 相机，MJPG 2560×720、左右各1280×720；SBS左半为物理左目。
- BGR 输入缩放到640×360，上下各补140黑边；分数0.536、NMS0.45。
- `config/runtime_hashes.json` 校验当前文件；`original_core_hashes.json` 验证保留的检测与测距基线没有改变。

## 桶身采集与独立参照

相机保持固定，先摆单桶。卷尺从左镜头外壳前沿量到朝向相机的桶前壁，记录尺的方向；沿相机前向的深度与水平读数分开。只有确认镜头光学原点偏移和量尺方向后，才评价正式 Z 误差。

每次独立摆放使用新编号，同一摆位连续帧用于观察抖动。下面0.85须换成实测米数；未量尺可省略该参数：

```sh
./run_body.sh capture B01_R1 --body-datum-m 0.85 --direction horizontal
./run_body.sh pull
```

板端 `captures/编号/` 保存左右图和时间戳；`results/编号__printed_30mm_20261122/seed2_REG_FP16/` 保存叠加图、检测和距离记录；`body_assessment/` 与 `results/body_placements.csv` 保存逐摆位评价。图片写入成功后才记录该帧。不要同时启动采集和预览，它们共用相机。

输出区分本帧新测、沿用旧值和无效值。当前算法从扩大桶框区域统计视差，尚未确认其输出始终等于前壁或桶口；误差容限未确定时，只报告测量和误差，不自行宣布准确性通过。

[视觉输出接口](../../docs/interfaces/api-rules.md) · [C++ 构建及运行](../vision_cpp/README.md)

当前仅保留 30 mm 打印棋盘标定；[标定及约 49 cm 实测确认](../../docs/stereo-recalibration-20261122.md)。准备脚本缺少新标定或哈希不符时停止，不回退旧参数。
