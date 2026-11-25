# 工作区导航

当前源码入口是`src/bucket_robot/`，测试入口是`tests/`。架构、启动和发布以[README](README.md)、[架构](docs/architecture.md)与[发布手册](docs/release.md)为准。

- `deploy/vision_cpp/`：原生源码与CMake；Python目录只留兼容导入。
- `deploy/serial_rx/`：旧入口转发和配置兼容链接。
- `tools/calibration/`、`tools/field/`、`tools/chassis_debug/`：各类人工工具。
- `experiments/near_surface/`：未接入正式运行的近表面测距候选。
- `outputs/project_audit_20261125/`：板端资源、版本和依赖审计。
- `outputs/architecture_refactor_20261125/`：迁移前源码快照、回归结果、构建与发布回执。

正式运行资源由完整发布清单定义，不按文件名猜测版本。历史论文、录像、日志和硬件证据保留在`research/`、`outputs/`与板端归档卡；本轮没有删除这些资料。原归档索引见[存储整理记录](docs/storage-cleanup-20261122.md)。

既有板端常规目录和冻结基线未被重构候选覆盖。新发布包可在独立目录检查，现场切换与物理验收另行记录。
