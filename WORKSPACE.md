# 工作区导航

## 当前工程

- `deploy/vision_cpp/`：当前C++视觉程序、Python追踪/控制和126项回归测试。
- `deploy/serial_rx/`：原协议只读回传、禁用工具、机械工作范围。
- `deploy/vision_v2/`：历史冻结模型及采集/回放支撑，实时入口调用C++工程。
- `deploy/calibration/`：打印棋盘标定采集和求解工具，保留本地工作成果。
- [当前实测版本](docs/tracking-release-20261122.md)：精确依赖、启动方式与验收证据。
- `docs/new-model-tracking-joint-20261122/`：最新模型联调和三轮追踪修正记录。

## 本地资料

- `outputs/tracking_edge_reacquire_20261122/`：本轮已验收源码清单、录像、日志和回放结果。
- `outputs/storage_cleanup_20261122/`：GitHub发布检查、磁盘迁移校验与旧路径映射。
- `outputs/`其余日期目录：历史证据；保留路径以支持既有回放脚本，详见其中的README。
- `research/`：论文、研究证据与封存材料，按原要求保留，不提交工程Git仓库。
- `research/archive/engineering_20261122/`：已终止的新固件试验、打印产物、早期追踪/航向/远程连接记录；原目录层级保留。迁移列表在`outputs/storage_cleanup_20261122/local_path_map.tsv`。

## RK3588存储分工

当前原生程序、模型、动态库、追踪源码、串口发送器及隔离OpenCV依赖留在系统盘。大型研究档案、历史录像、测试输出和旧部署副本迁往64G卡：

`/media/elf/5E28133C2813129F/bucket_archive_20261122/`

归档内部保留相对`/home/elf/`的目录结构。原路径通过符号链接继续可用。当前两个实时入口的`runs/`保留为系统盘目录，仅旧运行编号链接到归档；新日志继续写入系统盘。`_manifests/`记录逐文件SHA256及原权限，`relocations.json`记录来源/目标、字节数和校验状态。64G卡是NTFS，文件权限表现与ext4不同，原权限以清单为准。

64G卡必须挂载后才能访问冷资料。启动实时程序所需的常用代码与依赖不依赖归档研究目录；读取旧运行记录须该卡可用，新实时运行不依赖这些历史记录。归档卡已有的录像、系统升级镜像和其他资料原样保留。

这轮整理不自动启动相机或机构，不烧录固件。迁移后的容量与检查结果见[存储整理记录](docs/storage-cleanup-20261122.md)。
