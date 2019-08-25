#include "basicCalculation.hpp"

void sumArraysCPU(float * a,float * b,float * res,const int size) {
  for(int i = 0; i < size; i += 4) {
    res[i] = a[i] + b[i];
    res[i+1] = a[i+1] + b[i+1];
    res[i+2] = a[i+2] + b[i+2];
    res[i+3] = a[i+3] + b[i+3];
  }
}

__global__ void sumArrayGPU(float* a,float* b,float* res)
{
  // printf("blockIdx: (%d, %d) threadIdx: (%d, %d)\n",
  //   blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
  int i = blockIdx.x * blockDim.x * 
    blockDim.y + threadIdx.x * blockDim.y + threadIdx.y;
  res[i] = a[i] + b[i];
}

void sumArraysGPU(float* a,float* b,float* res,const int size) {
  // init arrays device mem
  int dev_id = 0;
  cudaSetDevice(dev_id);
  float* array1_dev = nullptr;
  float* array2_dev = nullptr;
  float* result_dev = nullptr;
  int byte_size = sizeof(float) * size;
  cudaError_t status = cudaMalloc(reinterpret_cast<float**>(&array1_dev), byte_size);
  spdlog::info("array1 malloc: {}", cudaGetErrorString(status));
  CHECK_CUDA(status);
  status = cudaMalloc(reinterpret_cast<float**>(&array2_dev), byte_size);
  spdlog::info("array2 malloc: {}", cudaGetErrorString(status));
  CHECK_CUDA(status);
  status = cudaMalloc(reinterpret_cast<float**>(&result_dev), byte_size);
  spdlog::info("result malloc: {}", cudaGetErrorString(status));
  CHECK_CUDA(status);

  // move data host2device
  status = cudaMemcpy(array1_dev, a, byte_size, cudaMemcpyHostToDevice);
  spdlog::info("array1 s2d: {}", cudaGetErrorString(status));
  status = cudaMemcpy(array2_dev, b, byte_size, cudaMemcpyHostToDevice);
  spdlog::info("array2 s2d: {}", cudaGetErrorString(status));

  // create kernel function
  dim3 block(4, 4);
  dim3 grid(size / (block.x * block.y));
  sumArrayGPU<<<grid, block>>>(array1_dev, array2_dev, result_dev);

  // move data decvice2host
  status = cudaMemcpy(res, result_dev, byte_size, cudaMemcpyDeviceToHost);
  spdlog::info("result d2s: {}", cudaGetErrorString(status));

  // free device mem
  status = cudaFree(array1_dev);
  spdlog::info("free array1: {}", cudaGetErrorString(status));
  status = cudaFree(array2_dev);
  spdlog::info("free array2: {}", cudaGetErrorString(status));
  status = cudaFree(result_dev);
  spdlog::info("free result: {}", cudaGetErrorString(status));
}