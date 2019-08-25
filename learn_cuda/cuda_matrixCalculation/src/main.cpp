#include "matrixCalculation.hpp"

int main(int argc, char* argv[]) {
  spdlog::set_pattern("[%^%l%$] %v");
  // init matrix host mem
  Matrix A(16, 15);
  int A_size = A.size, A_bytesie = A.size * sizeof(float);
  A.data = reinterpret_cast<float*>(malloc(A_bytesie));
  Matrix B(15, 17);
  int B_size = B.size, B_bytesie = B.size * sizeof(float);
  B.data = reinterpret_cast<float*>(malloc(B_bytesie));
  Matrix cpu_RESULT(16, 17);
  int cpu_size = cpu_RESULT.size, cpu_bytesize = cpu_RESULT.size * sizeof(float);
  cpu_RESULT.data = reinterpret_cast<float*>(malloc(cpu_bytesize));
  Matrix gpu_RESULT(16, 17);
  int gpu_size = gpu_RESULT.size, gpu_bytesize = gpu_RESULT.size * sizeof(float);
  gpu_RESULT.data = reinterpret_cast<float*>(malloc(gpu_bytesize));

  // init A B value
  for (int i = 0; i < A.size; i++) {
    A.data[i] = i + 1.2;
    B.data[i] = i + 1.2;
  }

  // calculation matrix product with cpu
  productMatrixCPU(A, B, cpu_RESULT);
  spdlog::info("cpu_result: [{:.4f}]",
    fmt::join(std::vector<float>(cpu_RESULT.data, cpu_RESULT.data + cpu_RESULT.size), ","));
  
  // calculation matrix product with gpu
  productMatrixGPU(A, B, gpu_RESULT);
  spdlog::info("gpu_result: [{:.4f}]",
    fmt::join(std::vector<float>(gpu_RESULT.data, gpu_RESULT.data + gpu_RESULT.size), ","));
  
  CHECK_RESULT(cpu_RESULT.data, gpu_RESULT.data, cpu_RESULT.size);
  // free mem
  free(A.data);
  free(B.data);
  free(cpu_RESULT.data);
  free(gpu_RESULT.data);
  return 0;
}