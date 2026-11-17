# Bucket 投喂车视觉工程

当前固定相机联调使用冻结的 **seed2_REG_FP16** 模型；V2表示标定和运行版本。

本机启动实时画面：

```bash
cd deploy/vision_v2
./deploy.sh
./view_live.sh
```

打开本机 http://127.0.0.1:8788/ ，Ctrl+C停止。板端目录为 `/home/elf/deploy/vision_v2`。部署会保留旧V1/V2目录。

当前实测：完整实时预览约7.8 FPS；保存全量验收文件的采集约1.7–2.6 FPS；相机协商格式30 FPS。模型调用约17.5ms，完整检测含前后处理约27.5ms；新距离约1次/秒。

- [视觉部署、测距及启动说明](deploy/vision_v2/README.md)
- [完整应用结构](docs/runtime.md)
- [应用接口](docs/interfaces/api-rules.md)

工程目录：

- `deploy/vision_v2/`：独立视觉入口、采集/预览工具、冻结模型、标定及必要配置；不启动STM32和执行机构。
- `main.py`、`app/`：原完整应用、设备任务、串口及Web模块。
- `config/`：应用任务配置；`model/`为旧完整应用模型，V2显式加载自己的冻结模型。
- `src/views/`、`tools/webrtc_sender.py`：原WebRTC观看组件与发送工具。

当前先验收桶身相对距离，暂不要求精确桶口中心。相机旁的出水口由伸缩臂送到桶口上方；装配时再确认伸缩范围以确定停车区间。

仓库只保存工程运行文件和使用说明。测试文件、采集数据、日志、实验结果、缓存、历史研究和论文均不纳入本次提交。
