#include "deviceQuery.hpp"

int get_deviceNum() {
  int deviceCount = 0;
  cudaError error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }
  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    spdlog::error("There are no available device(s) that support CUDA");
  } else {
    spdlog::info("Detected {} CUDA Capable device(s)", deviceCount);
  }
  return deviceCount;
}