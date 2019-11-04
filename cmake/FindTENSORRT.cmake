message(STATUS "Finding local TensorRT")

find_path(TENSORRT_INCLUDE_DIRS NvInfer.h
  HINTS ${TensorRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TensorRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${TensorRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS ${TensorRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARIES ${TENSORRT_LIBRARY_INFER}
                       ${TENSORRT_LIBRARY_INFER_PLUGIN}
                       ${TENSORRT_LIBRARY_PARSER})
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIRS TENSORRT_LIBRARIES)
if(NOT TENSORRT_FOUND)
  message(ERROR "Cannot find TensorRT library.")
else()
  message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIRS}")
  message(STATUS "TensorRT libs: ${TENSORRT_LIBRARIES}")
endif()

