#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "cuda_common.hpp"

/**
 * @brief this function calculate the sum of two arrays with cpu
 * 
 * @param a
 *    the first input array
 * @param b
 *    the second input array
 * @param res
 *    the calculation result
 * @param size
 *    input data size
 */
void sumArraysCPU(float* a, float* b, float* res, const int size);

/**
 * @brief this function calculate the sum of two arrays with gpu
 * 
 * @param a
 *    the first input array
 * @param b
 *    the second input array
 * @param res
 *    the calculation result
 * @param size
 *    input data size
 */
void sumArraysGPU(float* a, float* b, float* res, const int size);