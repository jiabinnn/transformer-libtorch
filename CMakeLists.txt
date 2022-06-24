# 指定CMake的最小版本要求
cmake_minimum_required(VERSION 3.0)

# 定义工程名称
project(TRANSFORMER_LIBTORCH)

set(INCLUDE ${CMAKE_SOURCE_DIR}/include)

file(GLOB SRC src/*.cc)

# 向工程添加多个特定的头文件搜索路径
include_directories(${INCLUDE})

# 可执行文件输出的存放路径
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug)

# 生成可执行文件
add_executable(debug tools/debug.cc ${SRC})
add_executable(main tools/main.cc ${SRC})
