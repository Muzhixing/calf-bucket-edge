# 结构重构结果

2026-11-26完成软件实现与隔离验证。

## 建立的能力

- Python采用标准src布局、pyproject.toml、wheel与模块/console入口；C++采用CMake目标与安装规则。
- 静止追踪和行进任务共同调用`TrackingController`；原FollowPolicy和居中/行程规则继续使用，未另写A/B算法。
- 持续目标的命令编号、pending和发送集中到`ContinuousArmPort._send_target()`；正式FrameWriter不再动态加载旧工程发送器。
- 目标身份策略、帧读取和阶段交接有明确所有者；跨阶段继续使用同一目标会话，前一观察线程必须先停止。
- 任务不选择A/B时必须显式声明A-only诊断，不能因漏参数默默丢掉B。标准任务及兼容脚本均接入A/B。
- 33个原回归模块移至统一tests；现场、标定和底盘工具移至tools；候选测距移至experiments。旧路径保留转发或符号链接。
- 正式包不从工具、测试或日期目录导入；CI配置包含依赖方向、能力复用、格式和回归检查。

## 验证证据

- 迁移前：追踪210项（6项可选CSRT跳过）、任务/串口80项，均无失败。
- 最终板端：303项，6项可选CSRT因依赖缺失跳过，其余通过；原测试方法全部保留。
- 从迁移前源码生成的180条A/B决策，在静止和任务上下文均逐项一致。
- PTY模拟串口覆盖实际控制循环、B调整前停车确认、A在B移动时继续更新、故障与禁用确认；正式FrameWriter也直接参与模拟MCU验证。
- CMake在RK3588完成构建与安装；私有OpenCV运行路径为`$ORIGIN/lib`。
- 完整发布包88项文件校验通过；46个Python模块全部从发布包内部导入。
- 冻结v1的89项文件再次重算，全部匹配。

完整记录位于`outputs/architecture_refactor_20261125/`：`before_sources.tar.gz`、`before_manifest.json`、`final_board_tests.log`、`cmake_build.log`、`release_manifest.json`、`release_receipt.json`与`validation_summary.json`。

## 部署状态

板端候选包：`/home/elf/bucket/releases/20261125-architecture/`，约81.2 MiB。包包含运行源码、配置、两个模型、标定、原生二进制、私有库与启动脚本。旧目录仅作为输入来源，其文件被复制并纳入清单，运行时不再从这些日期目录加载。

本轮没有切换现有实车入口、启动相机或实际机构运动，也没有刷写固件。软件回归证明结构迁移与已覆盖控制行为保持；下一阶段使用同一发布包完成现场回归后，统一正式入口。历史资料与冻结回退未删除。
