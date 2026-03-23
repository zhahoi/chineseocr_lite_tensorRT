cmake_minimum_required(VERSION 3.18)

# -------------------- CUDA 编译器 --------------------
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.6/bin/nvcc)

project(chineseocr_lite LANGUAGES CXX CUDA)

# -------------------- 编译标准 --------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
add_definitions(-pthread)

# -------------------- OpenCV --------------------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------------- CUDA --------------------
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# -------------------- TensorRT --------------------
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

# -------------------- 项目头文件路径 --------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

# -------------------- 源文件收集 --------------------
set(OCR_CPP_FILES
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/ocr_lite.cpp
    ${PROJECT_SOURCE_DIR}/src/ocr_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/dbnet.cpp
    ${PROJECT_SOURCE_DIR}/src/anglenet.cpp
    ${PROJECT_SOURCE_DIR}/src/crnn_net.cpp
    ${PROJECT_SOURCE_DIR}/src/clipper.cpp 
)

# -------------------- 可执行文件 --------------------
add_executable(chineseocr_lite
    ${OCR_CPP_FILES}
)

# -------------------- Jetson NX 架构设置 --------------------
set_target_properties(chineseocr_lite PROPERTIES
    CUDA_SEPARABLE_COMPILATION  ON
    CUDA_ARCHITECTURES          "72"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# -------------------- 编译选项 --------------------
target_compile_options(chineseocr_lite PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

# -------------------- 链接库 --------------------
target_link_libraries(chineseocr_lite
    PRIVATE
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
    ${OpenCV_LIBS}
    opencv_freetype
    pthread
)

# -------------------- 打印配置信息 --------------------
message(STATUS "========================================")
message(STATUS "  chineseocr_lite build configuration")
message(STATUS "========================================")
message(STATUS "  OpenCV version : ${OpenCV_VERSION}")
message(STATUS "  CUDA version   : ${CUDA_VERSION}")
message(STATUS "  CUDA arch      : 72 (Jetson NX)")
message(STATUS "  Source files   :")
foreach(f ${OCR_CPP_FILES})
    message(STATUS "    ${f}")
endforeach()
message(STATUS "========================================")