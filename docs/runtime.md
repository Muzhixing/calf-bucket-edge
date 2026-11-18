# 当前运行结构

实时入口为 `deploy/vision_cpp/main.cpp`。采集线程保存最新双目帧，校正线程处理左目，主线程调用冻结 RKNN 检测，独立双目进程处理一个在途任务；显示线程输出网页 MJPEG 或 HDMI。各阶段只保留最新帧，距离保留来源帧及年龄。

本机使用 `deploy/vision_v2/view_live.sh`；板端使用 `deploy/vision_cpp/run.sh` 或 `deploy/vision_v2/view_hdmi.sh`。默认快速双目，`--depth-scale 1` 启用全分辨率。旧 Python 实时后端和旧整车应用已移除。

`deploy/vision_v2` 是当前模型、标定和采集支撑包。其 `app/` 仅包含独立视觉采集需要的检测、测距、标定加载和事件模块；`vision_v2.py` 保存左右图与验收记录。原检测/测距模块的冻结哈希仍核验，未重新训练或转换模型。

板端当前工程路径：

- `/home/elf/deploy/vision_cpp`：C++ 实时程序；结果保存到 `runs/<编号>/`。
- `/home/elf/deploy/vision_v2`：模型与采集支撑；数据保存到 `captures/`、`results/`、`trial_records/`。

配置与依赖详见 [C++ 说明](../deploy/vision_cpp/README.md) 和 [采集说明](../deploy/vision_v2/README.md)。运行不导入串口、车辆控制器或任务服务。
