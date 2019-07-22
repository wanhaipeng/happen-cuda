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
 * @brief 2-dims matrix struct
 */
typedef struct matrix {
  int row; /* matrix row */
  int col; /* matrix col */
  int size; /* matrix element size */
  float* data; /* matrix element data ptr */
  matrix(int h, int w, float* input = nullptr) {
    row = h;
    col = w;
    size = row * col;
    data = input;
  }
}Matrix;

/**
 * @brief this function calculate the product of two matrix with cpu
 * 
 * @param a
 *    the first input matrix
 * 
 * @param b
 *    the second input matrix
 * @param res
 *    the calculation result matrix
 */
void productMatrixCPU(Matrix& a, Matrix& b, Matrix& res);

/**
 * @brief this function calculate the product of two matrix with gpu
 * 
 * @param a
 *    the first input matrix
 * @param b
 *    the second input matrix
 * @param res
 *    the calculation result matrix
 */
void productMatrixGPU(Matrix& a, Matrix& b, Matrix& res);

/**
 * @brief this function calculate the value of output matrix(row, col)
 * 
 * @param a
 *    the left input matrix
 * @param b
 *    the right input matrix
 * @param row
 *    output matrix row
 * @param col
 *    output matrix col
 */
float getIndexValue(Matrix& a, Matrix&b, int row, int col);