#!/bin/sh
set -eu
cd "$(dirname "$0")"
BASE=$(pwd)
python3 -m pip install --no-index --target "$BASE/deps/cmake" "$BASE"/deps/cmake-3.29.5-*.whl
tar xf deps/bucket-opencv-4.10.0.tar.gz -C deps
CMAKE="$BASE/deps/cmake/cmake/data/bin/cmake"
# Debian's Python include directory contains a NumPy 1.x symlink which precedes
# the installed NumPy 2.x include directory. Isolate Python headers without it.
mkdir -p deps/python-include
rsync -a --exclude numpy /usr/include/python3.10/ deps/python-include/
"$CMAKE" -S deps/opencv-4.10.0 -B deps/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$BASE/deps/install" -DBUILD_LIST=core,imgproc,imgcodecs,videoio,calib3d,python3 \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
  -DBUILD_JAVA=OFF -DBUILD_SHARED_LIBS=ON -DWITH_OPENCL=ON -DWITH_TBB=OFF -DWITH_OPENMP=OFF \
  -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_LAPACK=OFF \
  -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_ZLIB=ON -DWITH_WEBP=OFF -DWITH_OPENEXR=OFF \
  -DWITH_TIFF=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF \
  -DPYTHON3_EXECUTABLE=/usr/bin/python3 -DPYTHON3_INCLUDE_DIR="$BASE/deps/python-include" \
  -DPYTHON3_INCLUDE_PATH="$BASE/deps/python-include" \
  -DPYTHON3_NUMPY_INCLUDE_DIRS="$(python3 -c 'import numpy;print(numpy.get_include())')" \
  -DOPENCV_PYTHON3_INSTALL_PATH="$BASE/deps/install/python" -DOPENCV_GENERATE_PKGCONFIG=ON
"$CMAKE" --build deps/build -j 4
"$CMAKE" --install deps/build
