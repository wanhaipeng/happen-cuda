#pragma once
#include <cuda_runtime.h>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"

#define CHECK_CUDA(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      spdlog::error("ERROR: {}:{},",__FILE__,__LINE__);\
      spdlog::error("code:{},reason:{}", error, cudaGetErrorString(error));\
      exit(1);\
  }\
}
