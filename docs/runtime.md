# 应用结构与运行入口

固定相机视觉入口在 [deploy/vision_v2](../deploy/vision_v2/README.md)，显式加载seed2_REG_FP16，模型和标定路径及SHA-256会打印在启动日志中。

本机运行 `deploy/vision_v2/deploy.sh` 部署，`view_live.sh` 打开实时画面。启动和采集入口位于 `/home/elf/deploy/vision_v2`，实时后端默认使用 `/home/elf/deploy/vision_cpp`；`--backend python` 回退原预览，旧部署保留。原生构建和运行说明见 [C++ 部署说明](../deploy/vision_cpp/README.md)。入口只装配相机、检测、测距与画面服务，不导入串口、车辆控制器或执行机构。

完整应用仍保留原分层结构：

- `main.py`：装配完整设备配置、事件总线、任务、视觉及服务。
- `app/domain/`：事件定义。
- `app/application/`：生命周期、状态及任务配置。
- `app/infrastructure/`：检测、测距、串口、任务和推送。
- `app/presentation/web.py`：原Flask实时网页。
- `src/views/WebRTCViewer.vue`：原WebRTC观看组件。
- `config/`：完整应用任务样例。

完整应用的 `python main.py` 会访问设备与网络，并可装配串口/任务；本阶段使用独立视觉入口。完整应用依赖见根目录 `requirements.txt`，独立视觉依赖信息见 `deploy/vision_v2/requirements_vision.txt`；优先复用板端已有RKNNLite、librknnrt和OpenCV。

接口说明见 [api-rules.md](interfaces/api-rules.md)。运行生成的左右图、距离表和日志保存在板端相应目录，需要时用 `run_body.sh pull` 同步回本机；这些生成文件不提交Git。
