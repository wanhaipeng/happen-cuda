#include "basicCalculation.hpp"

int main(int argc, char* argv[]) {
  spdlog::set_pattern("[%^%l%$] %v");
  // init array host mem
  float* array1 = nullptr;
  float* array2 = nullptr;
  float* cpu_result = nullptr;
  float* gpu_result = nullptr;
  int elementNUM = 1 << 8;
  int elementBytes = elementNUM * sizeof(float);
  spdlog::info("array element size: {} byte size: {}",
    elementNUM, elementBytes);
  array1 = reinterpret_cast<float*>(malloc(elementBytes));
  array2 = reinterpret_cast<float*>(malloc(elementBytes));
  cpu_result = reinterpret_cast<float*>(malloc(elementBytes));
  gpu_result = reinterpret_cast<float*>(malloc(elementBytes));
  for (int i = 0; i < elementNUM; i++) {
    array1[i] = i + 0.1;
    array2[i] = i + 0.1;
  }
  // calculate with cpu
  sumArraysCPU(array1, array2, cpu_result, elementNUM);
  spdlog::info("[{}]", fmt::join(std::vector<float>(cpu_result, cpu_result + elementNUM), ","));

  // calculate with gpu
  sumArraysGPU(array1, array2, gpu_result, elementNUM);
  spdlog::info("[{}]", fmt::join(std::vector<float>(gpu_result, gpu_result + elementNUM), ","));

  // check result
  CHECK_RESULT(cpu_result, gpu_result, elementNUM);

  // free mem
  free(array1);
  free(array2);
  free(cpu_result);
  free(gpu_result);
  return 0;
}