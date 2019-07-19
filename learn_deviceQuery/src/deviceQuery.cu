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

void get_deviceInfo(int dev) {
  int driverVersion = 0, runtimeVersion = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  spdlog::critical("Device {}: {}", dev, deviceProp.name);
  // Console log
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  spdlog::info("CUDA Driver Version / Runtime Version: {}.{} / {}.{}",
    driverVersion / 1000, (driverVersion % 100) / 10,
    runtimeVersion / 1000, (runtimeVersion % 100) / 10);
  spdlog::info("CUDA Capability Major/Minor version number: {}.{}",
    deviceProp.major, deviceProp.minor);

  char msg[256];
  snprintf(msg, sizeof(msg),
           "Total amount of global memory: %.0f MBytes "
           "(%llu bytes)",
           static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
           (unsigned long long)deviceProp.totalGlobalMem);
  spdlog::info("{}", msg);
  spdlog::info("({}) Multiprocessors, ({}) CUDA Cores/MP: {} CUDA Cores",
    deviceProp.multiProcessorCount,
    _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
    _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
  spdlog::info("GPU Max Clock rate: {:.0f} MHz ({:.2f} GHz)",
    deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

  // This is supported in CUDA 5.0 (runtime API device properties)
  spdlog::info("Memory Clock rate: {:.0f} Mhz", deviceProp.memoryClockRate * 1e-3f);
  spdlog::info("Memory Bus Width: {}-bit", deviceProp.memoryBusWidth);
  if (deviceProp.l2CacheSize) {
    spdlog::info("L2 Cache Size: {} bytes", deviceProp.l2CacheSize);
  }
  spdlog::info("Maximum Texture Dimension Size (x,y,z): 1D={}, 2D=({}, {}), 3D=({}, {}, {})",
    deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
    deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
    deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
  spdlog::info("Maximum Layered 1D Texture Size, (num) layers:  1D=({}), {} layers",
    deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
  spdlog::info("Maximum Layered 2D Texture Size, (num) layers:  2D=({}, {}), {} layers",
    deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
    deviceProp.maxTexture2DLayered[2]);
  spdlog::info("Total amount of constant memory: {} bytes",
    deviceProp.totalConstMem);
  spdlog::info("Total amount of shared memory per block: {} bytes",
    deviceProp.sharedMemPerBlock);
  spdlog::info("Total number of registers available per block: {}",
    deviceProp.regsPerBlock);
  spdlog::info("Warp size: {}", deviceProp.warpSize);
  spdlog::info("Maximum number of threads per multiprocessor: {}",
    deviceProp.maxThreadsPerMultiProcessor);
  spdlog::info("Maximum number of threads per block: {}",
    deviceProp.maxThreadsPerBlock);
  spdlog::info("Max dimension size of a thread block (x,y,z): ({}, {}, {})",
    deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
    deviceProp.maxThreadsDim[2]);
  spdlog::info("Max dimension size of a grid size    (x,y,z): ({}, {}, {})",
    deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
    deviceProp.maxGridSize[2]);
  spdlog::info("Maximum memory pitch: {} bytes", deviceProp.memPitch);
  spdlog::info("Texture alignment: {} bytes", deviceProp.textureAlignment);
  spdlog::info("Device PCI Domain ID / Bus ID / location ID: {} / {} / {}",
    deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
}