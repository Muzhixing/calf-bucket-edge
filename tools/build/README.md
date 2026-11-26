# 本地Linux ARM64构建

在macOS或Linux的Docker环境中构建Linux/AArch64 ELF，不连接RK3588。容器固定使用Ubuntu 22.04和OpenCV 4.10.0；CMake显式设置AArch64 GNU工具链。Apple Silicon上Linux ARM64容器直接使用ARM指令；x86主机需Docker的ARM仿真支持。本次验证不证明任意x86交叉工具链均可使用。

准备本机已有RKNN SDK目录，包含`rknn_api.h`和`librknnrt.so`。SDK由调用者显式提供，不提交供应商二进制、不替换板端运行库。

```sh
mkdir -p build/arm-context
curl -fL https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.10.0 \
  -o build/arm-context/opencv-4.10.0.tar.gz
docker build --platform linux/arm64 -f tools/build/arm64.Dockerfile \
  -t bucket-arm64-build:opencv-4.10 build/arm-context
./tools/build/build_arm64.sh /absolute/path/to/rknn-sdk build/arm64
```

OpenCV压缩包SHA256固定在Dockerfile中。构建容器没有设备映射且禁用网络，挂载源码和SDK为只读。输出在`build/arm64/package/`，包含原生程序和私有OpenCV动态库；其余模型、标定与测距库仍由正式发布构建工具显式组装。

这条路径检查完整C++源码的编译、链接与ELF依赖。摄像头、NPU驱动、HDMI、时延及实际运动需现场验收。离线回放测试只能使用合成图像或已有静态文件，并使用缺失模型/禁用推理，不能将容器运行当成真实NPU推理证据。

构建后可在同一容器执行`native_replay_check.py --native <vision> --output <新目录>`，设置`PYTHONPATH`为仓库`src`并提供RKNN动态库搜索路径。该检查复用正式`NativeCapture`，验证左图、原始双目、故障状态和Python界面联合运行；故意使用不可能通过摘要校验的模型，不执行NPU推理。
