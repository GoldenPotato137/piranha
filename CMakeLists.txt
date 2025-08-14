cmake_minimum_required(VERSION 3.10)

project(piranha LANGUAGES CXX CUDA)

# 可配置项
set(CUTLASS_PATH "ext/cutlass" CACHE PATH "Path to CUTLASS")
set(CONFIG_FILE "config.json" CACHE FILEPATH "Config file path used by run helpers")
set(TEST "" CACHE STRING "gtest filter passed via --gtest_filter")
set(PARTY_NUM "" CACHE STRING "Party number for 'party' helper target")
set(PIRANHA_FLAGS "" CACHE STRING "Extra flags to pass to both C++ and CUDA compilers")

# C++/CUDA 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

# 源码收集
file(GLOB_RECURSE SRC_CPP_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE SRC_CU_FILES  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
file(GLOB_RECURSE HEADER_FILES  CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"    "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"  "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.cuh"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.inl"  "${CMAKE_CURRENT_SOURCE_DIR}/src/**/*.inl")

add_executable(piranha
  ${SRC_CPP_FILES}
  ${SRC_CU_FILES}
  ${HEADER_FILES}
)

# nvcc 的可分离编译（等价于 -dc）
set_target_properties(piranha PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

# ============== CUDA（无 find_package）=============
# 优先用 CUDA_HOME，其次用 /usr/local/cuda
set(CUDA_ROOT "$ENV{CUDA_HOME}")
if(NOT CUDA_ROOT)
  set(CUDA_ROOT "/usr/local/cuda")
endif()

if(NOT EXISTS "${CUDA_ROOT}/include/cuda.h")
  message(FATAL_ERROR "未找到 CUDA: ${CUDA_ROOT}。请设置环境变量 CUDA_HOME 指向你的 CUDA 根目录（例如 /usr/local/cuda-11.1），或建立 /usr/local/cuda 的符号链接。")
endif()

set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")
if(EXISTS "${CUDA_ROOT}/lib64")
  set(CUDA_LIB_DIR "${CUDA_ROOT}/lib64")
elseif(EXISTS "${CUDA_ROOT}/lib")
  set(CUDA_LIB_DIR "${CUDA_ROOT}/lib")
else()
  message(FATAL_ERROR "未找到 CUDA 库目录（在 ${CUDA_ROOT}/lib64 或 ${CUDA_ROOT}/lib 下）。")
endif()

include_directories("${CUDA_INCLUDE_DIR}")
# 为兼容老 CMake，使用全局 link_directories
link_directories("${CUDA_LIB_DIR}")

# 头文件路径（包括 CUTLASS 与本工程 include）
target_include_directories(piranha PRIVATE
  "${CUTLASS_PATH}/include"
  "${CUTLASS_PATH}/tools/util/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# 编译选项
target_compile_options(piranha PRIVATE
  # C++
  $<$<COMPILE_LANGUAGE:CXX>:
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:RelWithDebInfo>:-O3>
    $<$<CONFIG:MinSizeRel>:-Os>
    $<$<CONFIG:Debug>:-O0;-g>
    -w
    -fpermissive
    -fPIC
    -pthread
    -msse4.1
    -maes
    -msse2
    -mpclmul
  >
  # CUDA（通过 nvcc 传递宿主编译器选项）
  $<$<COMPILE_LANGUAGE:CUDA>:
    $<$<CONFIG:Release>:-Xcompiler=-O3>
    $<$<CONFIG:RelWithDebInfo>:-Xcompiler=-O3>
    $<$<CONFIG:MinSizeRel>:-Xcompiler=-Os>
    $<$<CONFIG:Debug>:-Xcompiler=-O0,-g>
    -Xcompiler=-w
    -Xcompiler=-fpermissive
    -Xcompiler=-fPIC
    -Xcompiler=-pthread
    -Xcompiler=-msse4.1
    -Xcompiler=-maes
    -Xcompiler=-msse2
    -Xcompiler=-mpclmul
    -Xcudafe;--diag_suppress=declared_but_not_referenced
  >
)

# 允许附加自定义编译开关
if(PIRANHA_FLAGS)
  separate_arguments(_PIRANHA_FLAGS UNIX_COMMAND "${PIRANHA_FLAGS}")
  target_compile_options(piranha PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${_PIRANHA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${PIRANHA_FLAGS}>
  )
endif()

# 链接库（按库名直接链接，路径由 link_directories 提供）
# 原 Makefile: -lcrypto -lssl -lcudart -lcuda -lgtest -lcublas
# 同时添加 pthread（如果没找到 Threads 包）。
find_package(Threads QUIET)
find_package(OpenSSL QUIET)
find_package(GTest QUIET)

# CUDA 库
target_link_libraries(piranha PRIVATE cudart cuda cublas)

# OpenSSL
if(OpenSSL_FOUND)
  target_link_libraries(piranha PRIVATE OpenSSL::SSL OpenSSL::Crypto)
else()
  target_link_libraries(piranha PRIVATE ssl crypto)
endif()

# GTest（如系统未装 gtest，可先 sudo apt-get install -y libgtest-dev 并编译安装）
if(GTest_FOUND)
  target_link_libraries(piranha PRIVATE GTest::gtest)
else()
  target_link_libraries(piranha PRIVATE gtest)
endif()

# 线程库
if(Threads_FOUND)
  target_link_libraries(piranha PRIVATE Threads::Threads)
else()
  target_link_libraries(piranha PRIVATE pthread)
endif()

# 运行时 rpath，确保运行时能找到 CUDA 动态库
set_target_properties(piranha PROPERTIES
  BUILD_RPATH "${CUDA_LIB_DIR}"
  INSTALL_RPATH "${CUDA_LIB_DIR}"
)

# ============== 便捷运行/调试目标 ==============
add_custom_target(run
  COMMAND $<TARGET_FILE:piranha> 3 ${CONFIG_FILE} --gtest_filter=${TEST}
  COMMAND $<TARGET_FILE:piranha> 2 ${CONFIG_FILE} --gtest_filter=${TEST}
  COMMAND $<TARGET_FILE:piranha> 1 ${CONFIG_FILE} --gtest_filter=${TEST}
  COMMAND $<TARGET_FILE:piranha> 0 ${CONFIG_FILE} --gtest_filter=${TEST}
  DEPENDS piranha
  USES_TERMINAL
  COMMENT "Running parties 3,2,1,0 sequentially"
)

add_custom_target(gdb
  COMMAND cuda-gdb --args $<TARGET_FILE:piranha> 0 ${CONFIG_FILE} --gtest_filter=${TEST}
  DEPENDS piranha
  USES_TERMINAL
)

add_custom_target(gdb-one
  COMMAND cuda-gdb --args $<TARGET_FILE:piranha> 1 ${CONFIG_FILE} --gtest_filter=${TEST}
  DEPENDS piranha
  USES_TERMINAL
)
add_custom_target(gdb-two
  COMMAND cuda-gdb --args $<TARGET_FILE:piranha> 2 ${CONFIG_FILE} --gtest_filter=${TEST}
  DEPENDS piranha
  USES_TERMINAL
)
add_custom_target(gdb-three
  COMMAND cuda-gdb --args $<TARGET_FILE:piranha> 3 ${CONFIG_FILE} --gtest_filter=${TEST}
  DEPENDS piranha
  USES_TERMINAL
)
add_custom_target(memcheck
  COMMAND cuda-memcheck $<TARGET_FILE:piranha> 0 ${CONFIG_FILE} --gtest_filter=${TEST}
  DEPENDS piranha
  USES_TERMINAL
)

add_custom_target(party
  COMMAND $<TARGET_FILE:piranha> ${PARTY_NUM} ${CONFIG_FILE} --gtest_filter=${TEST}
  DEPENDS piranha
  USES_TERMINAL
)
add_custom_target(zero  COMMAND $<TARGET_FILE:piranha> 0 ${CONFIG_FILE} --gtest_filter=${TEST} DEPENDS piranha USES_TERMINAL)
add_custom_target(one   COMMAND $<TARGET_FILE:piranha> 1 ${CONFIG_FILE} --gtest_filter=${TEST} DEPENDS piranha USES_TERMINAL)
add_custom_target(two   COMMAND $<TARGET_FILE:piranha> 2 ${CONFIG_FILE} --gtest_filter=${TEST} DEPENDS piranha USES_TERMINAL)
add_custom_target(three COMMAND $<TARGET_FILE:piranha> 3 ${CONFIG_FILE} --gtest_filter=${TEST} DEPENDS piranha USES_TERMINAL)

message(STATUS "Using manual CUDA ROOT: ${CUDA_ROOT}")
message(STATUS "CUDA include: ${CUDA_INCLUDE_DIR}")
message(STATUS "CUDA libdir : ${CUDA_LIB_DIR}")
message(STATUS "CUTLASS_PATH = ${CUTLASS_PATH}")
message(STATUS "CONFIG_FILE  = ${CONFIG_FILE}")
message(STATUS "TEST filter  = ${TEST}")
