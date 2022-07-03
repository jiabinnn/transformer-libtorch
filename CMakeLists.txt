# 指定CMake的最小版本要求
cmake_minimum_required(VERSION 3.0)

# 定义工程名称
project(TRANSFORMER_LIBTORCH)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "/home/tars/libtorch/libtorch_gpu")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wall")

find_package(CUDA REQUIRED)

set(TENSORRT_DIR "/home/tars/TensorRT-8.4.1.5")
set(CUDA_TOOLKIT_DIR "/usr/local/cuda")

set(COMMONSRC src/config_parser.cc src/string_utils.cc src/vocab.cc)

# 向工程添加多个特定的头文件搜索路径
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${TENSORRT_DIR}/include
    ${TENSORRT_DIR}/samples/common
    ${CUDA_TOOLKIT_DIR}/include
)

link_directories(
    ${TENSORRT_DIR}/lib
    ${CUDA_TOOLKIT_DIR}/lib64
)

# 可执行文件输出的存放路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_BUILD_TYPE Debug)
set(SKIP_BUILD_RPATH ON)

# 生成可执行文件
add_executable(infer_libtorch tools/infer_libtorch.cc ${COMMONSRC} src/model_libtorch.cc)
add_executable(infer_tensorrt tools/infer_tensorrt.cc ${COMMONSRC} src/model_tensorrt.cc)

target_link_libraries(infer_libtorch ${TORCH_LIBRARIES})
target_link_libraries(infer_tensorrt ${CUDA_LIBRARIES} nvinfer nvonnxparser nvparsers)
# target_link_libraries(infer_tensorrt )

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:${TENSORRT_DIR}/lib:${CUDA_TOOLKIT_DIR}/lib64:${TORCH_LIBRARY_DIRS}")
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
