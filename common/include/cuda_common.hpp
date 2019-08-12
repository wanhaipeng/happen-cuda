#pragma once
#include <sys/time.h>
#include <cuda_runtime.h>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include <google/protobuf/message.h>
#define SEPS 1e-5
#define LEPS 1e-1

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
    if (abs(a[i]) < 1.0 && std::fabs(a[i]-b[i]) > SEPS)\
    {\
      spdlog::error("cmp error at {}: cpu({}) vs gpu({}) ---> {}", i, a[i], b[i], std::fabs(a[i] - b[i]));\
      exit(1);\
    }\
    if (abs(a[i]) > 1.0 && std::fabs(a[i]-b[i]) > LEPS)\
    {\
      spdlog::error("cmp error at {}: cpu({}) vs gpu({}) ---> {}", i, a[i], b[i], std::fabs(a[i] - b[i]));\
      exit(1);\
    }\
  }\
  spdlog::info("cmp success!");\
}

/**
 * @brief this inline function calculate cuda kernel cost time
 */
inline double calcuTime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return((double)tp.tv_sec * 1e3+(double)tp.tv_usec * 1e-3);
}

/**
 * @brief this function parse protobuf message from proto text
 * 
 * @param filename
 *    input text file path
 * @param proto
 *    parse protobuf message
 */
void ReadProtoFromTextFile(const char* filename, google::protobuf::Message& proto);