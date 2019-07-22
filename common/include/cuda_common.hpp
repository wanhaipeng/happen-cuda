#pragma once
#include <cuda_runtime.h>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#define EPS 1e-5

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

#define CHECK_RESULT(a, b, n)\
{\
  for (int i = 0; i < n; i++)\
  {\
    if (std::fabs(a[i]-b[i]) > EPS)\
    {\
      spdlog::error("cmp error at {}: cpu({}) vs gpu({})", i, a[i], b[i]);\
      exit(1);\
    }\
  }\
  spdlog::info("cmp success!");\
}
