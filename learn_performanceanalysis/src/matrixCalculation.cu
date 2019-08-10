#include "matrixCalculation.hpp"

float getIndexValue(Matrix& a, Matrix&b, int row, int col) {
  int ah_stride = a.col, bh_stride = b.col;
  float output = 0.0;
  for (int i = 0; i < ah_stride; i++) {
    output += a.data[row * a.col + i] * b.data[col + i * bh_stride];
  }
  return output;
}

void productMatrixCPU(Matrix& a, Matrix& b, Matrix& res) {
  int resh_stride = b.col;
  for (int i = 0; i < a.row; i++) {
    for (int j = 0; j < b.col; j++) {
      res.data[i * resh_stride + j] = getIndexValue(a, b, i, j);
      // spdlog::info("({},{})： {}", i, j, res.data[i * resh_stride + j]);
    }
  }
}

// 获取矩阵A的(row, col)元素
__device__ float getElement(Matrix* input, int row, int col) {
  return input->data[row * input->col + col];
}

// 为矩阵A的(row, col)元素赋值
__device__ void setElement(Matrix* input, int row, int col, float value) {
  input->data[row * input->col + col] = value;
}

__global__ void matrixKernel(Matrix* a, Matrix* b, Matrix* res) {
  float resValue = 0.0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < res->row && col < res->col) {
    for (int i = 0; i < a->col; i++) {
      resValue += getElement(a, row, i) * getElement(b, i, col);
    }
    // printf("(%d, %d): %f \n", col, row, resValue);
    setElement(res, row, col, resValue);
  }
}

void productMatrixGPU(Matrix& a, Matrix& b, Matrix& res) {
  // init arrays device mem
  int dev_id = 0;
  cudaSetDevice(dev_id);
  // mem agent
  Matrix* A_dev;
  Matrix* B_dev;
  Matrix* Res_dev;
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&A_dev), sizeof(Matrix)));
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&B_dev), sizeof(Matrix)));
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&Res_dev), sizeof(Matrix)));
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&A_dev->data), a.size * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&B_dev->data), b.size * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&Res_dev->data), res.size * sizeof(float)));
  memcpy(A_dev->data, a.data, sizeof(float) * a.size);
  A_dev->col = a.col;
  A_dev->row = a.row;
  memcpy(B_dev->data, b.data, sizeof(float) * b.size);
  B_dev->col = b.col;
  B_dev->row = b.row;
  Res_dev->col = res.col;
  Res_dev->row = res.row;
  // create kernel function
  int blockx = 16, blocky = 16;
  dim3 blockSize(blockx, blocky);
  dim3 gridSize((res.col + blockSize.x - 1) / blockSize.x, (res.row + blockSize.y - 1) / blockSize.y);
  spdlog::info("blockSize: ({},{})", blockx, blocky);
  spdlog::info("gridSize: ({},{})", (res.col + blockSize.x - 1) / blockSize.x, (res.row + blockSize.y - 1) / blockSize.y);
  double begin_time, end_time;
  begin_time = calcuTime();
  matrixKernel<<<gridSize, blockSize>>>(A_dev, B_dev, Res_dev);
  // sync
  cudaDeviceSynchronize();
  end_time = calcuTime();
  spdlog::info("cost time: {} ms", end_time - begin_time);
  memcpy(res.data, Res_dev->data, sizeof(float) * res.size);
}