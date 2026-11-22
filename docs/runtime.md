# 当前运行结构

当前验收版本是2026-11-22近距离边缘重获版，模型为B1_C0_REG_FP16，RGB输入，使用30 mm打印棋盘标定。C++负责采集、校正、NPU检测和GPU校正/双目测距；Python负责跟踪、预览和原STM32协议控制，线程只读取最新帧。

仓库入口为`deploy/vision_cpp/run.sh`。默认自动识别、确认、跟随，无须在锁定后手动开始；运动前检查新鲜回传、零位、航向、故障与行程。`--observe-only`及回放不启动机构。关闭浏览器不等于关闭自动跟随；页面“停止机构”或主进程SIGINT走禁用确认流程。

板端实测入口及精确依赖见[当前版本说明](tracking-release-20261122.md)。当前板端验收副本位于`/home/elf/deploy/vision_edge_stage_20261122`，生产目录`vision_cpp`提供原生程序和模型；提交GitHub本身不覆盖生产运行入口、不使能机构。

`deploy/vision_v2`保留历史冻结模型与采集/回放工具，不将其旧模型描述当作当前实时模型选择。当前模型绑定在`deploy/vision_cpp/model_source.json`，标定绑定在`calibration_source.json`。板端原单帧发送器仍由`serial_rx/arm_reference.json`指定。

A范围60–100°、B范围0–96 mm、C固定10 mm；圈内10%保持/外圈15%恢复，界面不绘制容差圆。完全遮挡或证据不足时暂停，确认当前目标后恢复；不靠预测延长可见性。

[主程序说明](../deploy/vision_cpp/README.md) · [视觉接口](interfaces/api-rules.md) · [工作区导航](../WORKSPACE.md)
