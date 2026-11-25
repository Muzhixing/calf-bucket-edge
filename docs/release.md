# 构建、验证与发布

## 约定

源码由`pyproject.toml`定义；Python发布产物是wheel。原生程序由CMake构建。完整RK3588发布目录包含`python/`、`native/`、`assets/`、`assisted_runtime/`、两个启动脚本和逐文件SHA256清单。系统Python、OpenCV/NumPy、驱动、GStreamer和检测所需的系统RKNN属于板端已验证环境，不会被发布工具替换。

发布目录不可作为录像/日志输出目录。每次发布使用新目录，验证后再按现场流程切换入口；发布工具不启动程序，不修改开机服务，不覆盖冻结v1，不自动修改current。

## Python

```sh
python -m pip install -e '.[desktop,dev]'
python -m unittest discover -s tests -t . -v
python -m build --wheel
```

在RK3588上验证时使用板端现有NumPy/OpenCV，设置`PYTHONPATH=src`即可运行独立源码检查。发布包的入口会自行设置其`python/`目录，避免从工作区或另一日期目录导入。

## C++

在RK3588上，依赖路径指向已核验的SDK与OpenCV。以下为常见布局；新机器应显式设置本机路径。

```sh
cmake -S deploy/vision_cpp -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DOPENCV_PREFIX=/home/elf/deploy/vision_cpp/deps/install \
  -DRKNN_SDK_DIR=/home/elf/deploy/vision_cpp/deps/rknn \
  -DGST_INCLUDE_DIR=/home/elf/deploy/vision_cpp/deps/headers/usr/include/gstreamer-1.0
cmake --build build/native --parallel 1
cmake --install build/native --prefix build/native-package
```

需要CMake>=3.18。没有系统CMake时可在单独构建环境安装CMake；兼容`build.sh`支持`BUCKET_CMAKE=/path/to/cmake`，不需要修改系统环境。`build.sh`同样调用上述CMake目标，只为旧工具保留`vision`与`lib/`输出位置。

原生二进制通过`$ORIGIN/lib`加载包内OpenCV。检测使用系统RKNN；神经辅助测距单独使用`assisted_runtime/`中的RKNN运行库。不要合并两套库路径。

## 组装完整包

所有输入必须显式提供。下面的资产路径由维护者选定，不会按日期目录自动猜测。

```sh
python tools/release/build_release.py \
  --wheel dist/bucket_robot-0.3.0-py3-none-any.whl \
  --native-dir build/native-package \
  --native-source-dir deploy/vision_cpp \
  --assets-dir /path/to/validated/display-assets \
  --detector /path/to/B1_C0_REG_FP16.rknn \
  --range-model /path/to/fast_acv.rknn \
  --calibration /path/to/printed_30mm_20261122.json \
  --range-runtime /path/to/validated/assisted_runtime \
  --out releases/<new-version> --version <new-version> \
  --native-source-relation built_from_supplied_source
python -m bucket_robot verify releases/<new-version>
```

`--assets-dir`需含对应`calibration.yml`及显示图/模板；JSON与YAML应来自同一份已验证标定。启动时沿用原模型与标定哈希检查。`binary_only`模式仅记录外部二进制，不宣称能由随包源码重建；从当前源码构建时使用`built_from_supplied_source`并保存输入。

清单覆盖每个部署文件和构建来源；新入口也在清单中。校验拒绝丢失、修改、多出或逃逸路径的文件。校验不会推断物理零位、启动相机或发送串口。

## 部署与使用

```sh
python tools/release/deploy_release.py \
  --release-dir releases/<new-version> --host rk3588 \
  --destination /home/elf/bucket/releases/<new-version>
```

目标已存在时停止。上传后在板端重算清单；已有运行入口保持原样。原`deploy/vision_cpp/deploy.sh`和`vision_v2/deploy.sh`均转发此工具，不再同步部分源码。

从板端发布目录运行：

```sh
./run_track.sh --observe-only
./run_mission.sh --out /home/elf/bucket/runs/<new-run>
```

第一条观察模式不允许机构运动。第二条沿用已有任务的运动条件和A73/B0/C0起点检查；执行前按已有现场流程核对。原准备与断线/故障停止机制继续生效。新任务入口从完整包读取机构配置，A/B能力默认接入；不会自动复用某个可写stage目录。

检查命令与帮助、文件完整性、包导入、模拟UART及C++编译属于软件验证；完整自动靠桶与现场运动仍须单独验收。

## 0.3.0降级运行入口

`track`和`mission`现使用同一个先启动HTTP的应用外壳。发布前的`verify`仍执行完整清单验证；运行中的资源检查按组件归属降级，避免缺模型或串口导致整个页面消失。详见[故障隔离说明](resilience/README.md)。

发布构建需同时包含`worker_channel.hpp`、`detector_process.hpp`以及由当前源码构建的原生程序。0.3.0的Python wheel和完整Linux ARM64原生程序均已完成本地构建，并通过无设备回放。尚未部署或切换板端入口，真实NPU与运动仍待现场验收。构建、审查和验收范围见[ARM复核记录](resilience/arm_review_20261126.md)。
