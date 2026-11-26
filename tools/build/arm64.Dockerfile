FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config ca-certificates file binutils \
    libssl-dev libdrm-dev libglib2.0-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libjpeg-dev libpng-dev zlib1g-dev \
    python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
COPY opencv-4.10.0.tar.gz /tmp/opencv.tar.gz
RUN printf '%s  /tmp/opencv.tar.gz\n' b2171af5be6b26f7a06b1229948bbb2bdaa74fcf5cd097e0af6378fce50a6eb9 | sha256sum -c - \
    && tar -xf /tmp/opencv.tar.gz -C /tmp \
    && cmake -S /tmp/opencv-4.10.0 -B /tmp/opencv-build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/opencv \
       -DBUILD_LIST=core,imgproc,imgcodecs,videoio,calib3d,features2d,flann \
       -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF \
       -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF \
       -DWITH_OPENCL=ON -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_FFMPEG=OFF \
       -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_LAPACK=OFF \
       -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF -DWITH_TIFF=OFF -DWITH_OPENJPEG=OFF \
       -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_ZLIB=ON \
    && cmake --build /tmp/opencv-build --parallel 3 \
    && cmake --install /tmp/opencv-build \
    && rm -rf /tmp/opencv-build /tmp/opencv-4.10.0 /tmp/opencv.tar.gz
RUN python3 -m pip install --no-cache-dir numpy==1.26.4 opencv-python-headless==4.10.0.84 build
