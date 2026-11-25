# 投喂车视觉与追踪控制

RK3588上的实时采集、RKNN检测和双目计算使用C++；目标追踪、机构/底盘控制与任务编排使用标准Python包`bucket_robot`。静止追踪与行进任务共享同一TrackingController、身份策略与串口事务。

## 开发与检查

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -e '.[desktop,dev]'
.venv/bin/python -m unittest discover -s tests -t . -v
.venv/bin/ruff check src tools/field tools/release tools/calibration tools/build
.venv/bin/ruff format --check src tests tools/field tools/release tools/calibration tools/build
.venv/bin/python -m build --wheel
```

RK3588沿用已核验的Python/OpenCV/NumPy与系统RKNN环境；正式发布从wheel解包加载，不通过安装桌面OpenCV依赖替换板端环境。

## 人工维护入口

- [架构与依赖规则](docs/architecture.md)：能力所有权、数据流、扩展任务的方法。
- [构建与发布手册](docs/release.md)：CMake、wheel、完整发布包、校验与部署。
- [故障隔离与恢复](docs/resilience/README.md)：缺设备、失败降级、能力门控和停止确认。
- [本地ARM64构建](tools/build/README.md)：容器、显式工具链和无设备回放验证。
- [重构决策](docs/adr/0001-python-package-and-control-ownership.md)：采用分层应用与渐进迁移的理由。
- `src/bucket_robot/tracking/controller.py`：两个业务入口共用的追踪控制器。
- `src/bucket_robot/motion/`：命令约束、持续目标事务、ACK及禁用确认。
- `src/bucket_robot/missions/`：阶段编排；不得重复编写轴控制或串口发送。
- `tests/`：统一回归入口；`tools/`为人工工具；`experiments/`为候选算法。

## 运行入口

完整发布包提供`run_track.sh`和`run_mission.sh`。先按发布手册核验包与当前设备状态，再按已有现场流程运行。`python -m bucket_robot --help`和`python -m bucket_robot verify <release>`只显示帮助或校验文件，不启动设备。

旧`deploy/`入口保留兼容转发；原Python实现已经移入正式包。旧assisted脚本需要显式设置`BUCKET_RELEASE_ROOT`，统一使用A/B追踪，不再默认拼接日期目录。冻结v1及既有板端入口作为回退保留，重构候选包通过软件回归不代表已完成新的实车验收。

## 资源与记录

模型选择、相机标定及原生源码仍在`deploy/vision_cpp/`；默认机构配置随包发布于`src/bucket_robot/resources/`。`outputs/`保存本地验证资料，`research/`保存论文与研究档案，均不作为正式程序的运行依赖。历史现场证据及各阶段适用范围保留在`docs/`。
