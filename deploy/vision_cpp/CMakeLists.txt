cmake_minimum_required(VERSION 3.18)
project(bucket_vision LANGUAGES CXX)

set(OPENCV_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/deps/install" CACHE PATH "Validated OpenCV installation")
set(RKNN_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/rknn" CACHE PATH "RKNN API headers")
set(GST_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/headers/usr/include/gstreamer-1.0" CACHE PATH "GStreamer headers")

find_package(OpenCV 4.10 REQUIRED COMPONENTS core imgproc imgcodecs calib3d features2d flann videoio
    PATHS "${OPENCV_PREFIX}/lib/cmake/opencv4" NO_DEFAULT_PATH)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLIB REQUIRED IMPORTED_TARGET glib-2.0 gobject-2.0)
pkg_check_modules(DRM REQUIRED IMPORTED_TARGET libdrm)
find_library(RKNN_RUNTIME NAMES rknnrt REQUIRED)
find_library(GST_APP NAMES gstapp-1.0 libgstapp-1.0.so.0 REQUIRED)
find_library(GST_CORE NAMES gstreamer-1.0 libgstreamer-1.0.so.0 REQUIRED)
if(NOT EXISTS "${RKNN_SDK_DIR}/rknn_api.h" OR NOT EXISTS "${GST_INCLUDE_DIR}/gst/gst.h")
    message(FATAL_ERROR "Supply RKNN_SDK_DIR and GST_INCLUDE_DIR for the board's validated dependencies")
endif()

add_executable(vision main.cpp)
target_compile_features(vision PRIVATE cxx_std_17)
target_compile_options(vision PRIVATE -Wall -Wextra -Wno-deprecated-declarations)
target_include_directories(vision PRIVATE "${RKNN_SDK_DIR}" "${GST_INCLUDE_DIR}" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision PRIVATE ${OpenCV_LIBS} ${RKNN_RUNTIME} OpenSSL::Crypto
    ${GST_APP} ${GST_CORE} Threads::Threads PkgConfig::GLIB PkgConfig::DRM)
set_target_properties(vision PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
install(TARGETS vision RUNTIME DESTINATION .)
# Preserve the validated SONAMEs in the private runtime bundle.
foreach(component core imgproc imgcodecs calib3d features2d flann videoio)
    install(FILES "${OPENCV_PREFIX}/lib/libopencv_${component}.so.4.10.0"
        DESTINATION lib RENAME "libopencv_${component}.so.410")
endforeach()
