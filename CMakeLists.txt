cmake_minimum_required(VERSION 3.5)
project(ppeng_cuda)

# Assign env variables
set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
# Assign cuda variables
set(GENCODE "-gencode arch=compute_30,code=sm_30")
set(GENCODE "${GENCODE} -gencode arch=compute_35,code=sm_35")
set(GENCODE "${GENCODE} -gencode arch=compute_50,code=sm_50")
set(GENCODE "${GENCODE} -gencode arch=compute_52,code=sm_52")
set(GENCODE "${GENCODE} -gencode arch=compute_60,code=sm_60")
set(GENCODE "${GENCODE} -gencode arch=compute_61,code=sm_61")
message(STATUS "CUDA_ARCH: ${GENCODE}")
list(APPEND CMAKE_CXX_FLAGS -std=c++11)
list(APPEND CUDA_NVCC_FLAGS ${GENCODE})

# find cuda
find_package(CUDA REQUIRED)
# set(CUDA_VERBOSE_BUILD ON)
set(CUDA_COMMON_INCLUDE_DIR $ENV{HOME}/Downloads/nv/cuda10.1_sample/common/inc)
message(STATUS "CUDA_HEADER: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_COMMON_HEADER: ${CUDA_COMMON_INCLUDE_DIR}")
message(STATUS "CUDA_LIB: ${CUDA_LIBRARIES}")

# set PKG_CONFIG_PATH for pkg-config find
find_package(PkgConfig)

# find protobuf3.x
pkg_search_module(PROTOBUF REQUIRED protobuf)
message(STATUS "Protobuf_HEADER: ${PROTOBUF_INCLUDE_DIRS}")
message(STATUS "Protobuf_LIB: ${PROTOBUF_LIBRARIES}")
link_directories("/usr/local/protobuf3.7/lib") # find version 3 lib first

# find opencv4.x
set(OpenCV_DIR /usr/local/opencv3.4/share/OpenCV)
find_package(OpenCV 3 REQUIRED)
# pkg_search_module(OpenCV REQUIRED opencv)
message(STATUS "OpenCV_HEADER: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIB: ${OpenCV_LIBRARIES}")

# add common header
set(COMMON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/common/include
                       ${CMAKE_SOURCE_DIR}/common/proto)
# aux_source_directory(${CMAKE_SOURCE_DIR}/common/src COMMON_SRC)
# aux_source_directory(${CMAKE_SOURCE_DIR}/common/proto PROTO_SRCS)
file(GLOB_RECURSE COMMON_SRCS ${CMAKE_SOURCE_DIR}/common/*.c*)
message(STATUS "common srcs: ${COMMON_SRCS}")

# add deviceQuery
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/deviceQuery)
add_subdirectory(deviceQuery)

# add learn_deviceQuery
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/learn_deviceQuery)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/learn_deviceQuery)
add_subdirectory(learn_deviceQuery)

# add learn_basicCalculation
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/learn_basicCalculation)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/learn_basicCalculation)
add_subdirectory(learn_basicCalculation)

# add learn_matrixCalculation
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/learn_matrixCalculation)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/learn_matrixCalculation)
add_subdirectory(learn_matrixCalculation)

# add learn_performanceanalysis
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/learn_performanceanalysis)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib/learn_performanceanalysis)
add_subdirectory(learn_performanceanalysis)