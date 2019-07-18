#pragma once
#include <stdio.h>
#include <iostream>
#include <memory>
#include <string>
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"

#if CUDART_VERSION < 5000
// CUDA-C includes
#include <cuda.h>

/**
 * @brief This function wraps the CUDA Driver API into a template function.
 */
template <typename T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
                             int device) {
  CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);
  if (CUDA_SUCCESS != error) {
    fprintf(
        stderr,
        "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
        error, __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}
#endif /* CUDART_VERSION < 5000 */

/**
 * @brief this function return the CUDA Capable device number
 */
extern "C"
int get_deviceNum();