cmake_minimum_required(VERSION 3.5)
project(happen_cuda)

# Set macro switch
option(WITH_CUDA   "Set switch to build learn_cuda" OFF)
option(WITH_TRT    "Set switch to build learn_trt"  ON)

# Assign env variables
set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# find cuda
find_package(CUDA REQUIRED)
# Assign cuda variables
set(GENCODE "-O3 -D_FORCE_INLINES -Xcompiler -fPIC -gencode arch=compute_30,code=sm_30")
set(GENCODE "${GENCODE} -gencode arch=compute_35,code=sm_35")
set(GENCODE "${GENCODE} -gencode arch=compute_50,code=sm_50")
set(GENCODE "${GENCODE} -gencode arch=compute_52,code=sm_52")
set(GENCODE "${GENCODE} -gencode arch=compute_60,code=sm_60")
set(GENCODE "${GENCODE} -gencode arch=compute_61,code=sm_61")
message(STATUS "CUDA_ARCH: ${GENCODE}")
list(APPEND CMAKE_CXX_FLAGS "-std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
list(APPEND CUDA_NVCC_FLAGS ${GENCODE})
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
# set(CUDA_VERBOSE_BUILD ON)
set(CUDA_COMMON_INCLUDE_DIR $ENV{HOME}/Downloads/nv/cuda10.1_sample/common/inc)
message(STATUS "CUDA_include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_common_include: ${CUDA_COMMON_INCLUDE_DIR}")
message(STATUS "CUDA_lib: ${CUDA_LIBRARIES}")

# set PKG_CONFIG_PATH for pkg-config find
find_package(PkgConfig)

# find protobuf3.x
pkg_search_module(PROTOBUF REQUIRED protobuf)
message(STATUS "Protobuf_include: ${PROTOBUF_INCLUDE_DIRS}")
message(STATUS "Protobuf_lib: ${PROTOBUF_LIBRARIES}")
link_directories("/usr/local/protobuf3.7/lib") # find version 3 lib first

# find opencv4.x
set(OpenCV_DIR /usr/local/opencv3.4/share/OpenCV)
find_package(OpenCV 3 REQUIRED)
# pkg_search_module(OpenCV REQUIRED opencv)
message(STATUS "OpenCV_include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_lib: ${OpenCV_LIBRARIES}")

# add common header
set(COMMON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/common/include
                       ${CMAKE_SOURCE_DIR}/common/proto)
# aux_source_directory(${CMAKE_SOURCE_DIR}/common/src COMMON_SRC)
# aux_source_directory(${CMAKE_SOURCE_DIR}/common/proto PROTO_SRCS)
file(GLOB_RECURSE COMMON_SRCS ${CMAKE_SOURCE_DIR}/common/*.c*)
message(STATUS "COMMON SRCS: ${COMMON_SRCS}")

# cmake subdir macro
macro(SUBDIRLIST output input)
  file(GLOB children RELATIVE ${input} ${input}/*)
  set(DIRLIST "")
  foreach(child ${children})
    if(IS_DIRECTORY ${input}/${child})
      list(APPEND DIRLIST ${child})
    endif()
  endforeach()
  set(${output} ${DIRLIST})
endmacro()

# add deviceQuery
# add_subdirectory(deviceQuery)

# add learn_cuda
if(WITH_CUDA)
  add_subdirectory(learn_cuda)
endif()
# add learn_trt
if(WITH_TRT)
  set(TensorRT_ROOT "/home/haipeng.wan/Downloads/nv/TensorRT-5.1.5.0")
  find_package(TENSORRT REQUIRED)
  add_subdirectory(learn_trt)
endif()
