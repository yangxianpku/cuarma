#pragma once

/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/** @file  cuarma/blas/cuda/matrix_operations_prod.hpp
 *  @encoding:UTF-8 文档编码
    @brief Dense matrix-matrix product CUDA kernels reside here.
*/

#include "cuarma/forwards.h"

namespace cuarma
{
namespace blas
{
namespace cuda
{

// matrix-matrix multiplication C = A * B
// matrix layouts: C...col_major, A...col_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_col_col_col_prod_AA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;
  arma_size_t aStep = block_size * A_col_inc * A_internal_rows;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;
  arma_size_t bStep = block_size * B_row_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_col_size) && (row_block_id * block_size + row_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_row_size) && (col_block_id * block_size + col_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A * B^T
// matrix layouts: C...col_major, A...col_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_col_col_col_prod_AT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;
  arma_size_t aStep = block_size * A_col_inc * A_internal_rows;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;
  arma_size_t bStep = block_size * B_internal_rows * B_col_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_col_size) && (row_block_id * block_size + row_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_col_size) && (col_block_id * block_size + row_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A^T * B
// matrix layouts: C...col_major, A...col_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_col_col_col_prod_TA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;
  arma_size_t aStep = block_size * A_row_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;
  arma_size_t bStep = block_size * B_row_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_row_size) && (row_block_id * block_size + col_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_row_size) && (col_block_id * block_size + col_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A^T * B^T
// matrix layouts: C...col_major, A...col_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_col_col_col_prod_TT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;
  arma_size_t aStep = block_size * A_row_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;
  arma_size_t bStep = block_size * B_internal_rows * B_col_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_row_size) && (row_block_id * block_size + col_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_col_size) && (col_block_id * block_size + row_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}



////////////////////////////////////////////////////////////////////////////




// matrix-matrix multiplication C = A * B
// matrix layouts: C...row_major, A...col_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_row_col_col_prod_AA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;
  arma_size_t aStep = block_size * A_col_inc * A_internal_rows;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;
  arma_size_t bStep = block_size * B_row_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_col_size) && (row_block_id * block_size + row_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_row_size) && (col_block_id * block_size + col_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A * B^T
// matrix layouts: C...row_major, A...col_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_row_col_col_prod_AT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;
  arma_size_t aStep = block_size * A_col_inc * A_internal_rows;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;
  arma_size_t bStep = block_size * B_internal_rows * B_col_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_col_size) && (row_block_id * block_size + row_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_col_size) && (col_block_id * block_size + row_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A^T * B
// matrix layouts: C...row_major, A...col_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_row_col_col_prod_TA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;
  arma_size_t aStep = block_size * A_row_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;
  arma_size_t bStep = block_size * B_row_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_row_size) && (row_block_id * block_size + col_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_row_size) && (col_block_id * block_size + col_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A^T * B^T
// matrix layouts: C...row_major, A...col_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_row_col_col_prod_TT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;
  arma_size_t aStep = block_size * A_row_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;
  arma_size_t bStep = block_size * B_internal_rows * B_col_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_row_size) && (row_block_id * block_size + col_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_col_size) && (col_block_id * block_size + row_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}




////////////////////////////////////////////////////////////////////////////




// matrix-matrix multiplication C = A * B
// matrix layouts: C...col_major, A...col_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_col_col_row_prod_AA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;
  arma_size_t aStep = block_size * A_col_inc * A_internal_rows;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;
  arma_size_t bStep = block_size * B_internal_cols * B_row_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_col_size) && (row_block_id * block_size + row_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A * B^T
// matrix layouts: C...col_major, A...col_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_col_col_row_prod_AT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;
  arma_size_t aStep = block_size * A_col_inc * A_internal_rows;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;
  arma_size_t bStep = block_size * B_col_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_col_size) && (row_block_id * block_size + row_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A^T * B
// matrix layouts: C...col_major, A...col_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_col_col_row_prod_TA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;
  arma_size_t aStep = block_size * A_row_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;
  arma_size_t bStep = block_size * B_internal_cols * B_row_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_row_size) && (row_block_id * block_size + col_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A^T * B^T
// matrix layouts: C...col_major, A...col_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_col_col_row_prod_TT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;
  arma_size_t aStep = block_size * A_row_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;
  arma_size_t bStep = block_size * B_col_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_row_size) && (row_block_id * block_size + col_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}



////////////////////////////////////////////////////////////////////////////




// matrix-matrix multiplication C = A * B
// matrix layouts: C...row_major, A...col_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_row_col_row_prod_AA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;
  arma_size_t aStep = block_size * A_col_inc * A_internal_rows;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;
  arma_size_t bStep = block_size * B_internal_cols * B_row_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_col_size) && (row_block_id * block_size + row_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A * B^T
// matrix layouts: C...row_major, A...col_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_row_col_row_prod_AT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) + A_col_start * A_internal_rows;
  arma_size_t aStep = block_size * A_col_inc * A_internal_rows;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;
  arma_size_t bStep = block_size * B_col_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_col_size) && (row_block_id * block_size + row_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A^T * B
// matrix layouts: C...row_major, A...col_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_row_col_row_prod_TA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;
  arma_size_t aStep = block_size * A_row_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;
  arma_size_t bStep = block_size * B_internal_cols * B_row_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_row_size) && (row_block_id * block_size + col_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A^T * B^T
// matrix layouts: C...row_major, A...col_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_row_col_row_prod_TT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) * A_internal_rows + A_row_start;
  arma_size_t aStep = block_size * A_row_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;
  arma_size_t bStep = block_size * B_col_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_row_inc + col_thread_id * A_col_inc * A_internal_rows;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_row_size) && (row_block_id * block_size + col_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}





////////////////////////////////////////////////////////////////////////////






// matrix-matrix multiplication C = A * B
// matrix layouts: C...col_major, A...row_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_col_row_col_prod_AA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;
  arma_size_t aStep = block_size * A_col_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;
  arma_size_t bStep = block_size * B_row_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_row_size) && (col_block_id * block_size + col_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A * B^T
// matrix layouts: C...col_major, A...row_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_col_row_col_prod_AT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;
  arma_size_t aStep = block_size * A_col_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;
  arma_size_t bStep = block_size * B_internal_rows * B_col_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_col_size) && (col_block_id * block_size + row_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A^T * B
// matrix layouts: C...col_major, A...row_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_col_row_col_prod_TA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;
  arma_size_t aStep = block_size * A_row_inc * A_internal_cols;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;
  arma_size_t bStep = block_size * B_row_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_row_size) && (col_block_id * block_size + col_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A^T * B^T
// matrix layouts: C...col_major, A...row_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_col_row_col_prod_TT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;
  arma_size_t aStep = block_size * A_row_inc * A_internal_cols;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;
  arma_size_t bStep = block_size * B_internal_rows * B_col_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_col_size) && (col_block_id * block_size + row_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}




////////////////////////////////////////////////////////////////////////////




// matrix-matrix multiplication C = A * B
// matrix layouts: C...row_major, A...row_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_row_row_col_prod_AA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;
  arma_size_t aStep = block_size * A_col_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;
  arma_size_t bStep = block_size * B_row_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_row_size) && (col_block_id * block_size + col_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A * B^T
// matrix layouts: C...row_major, A...row_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_row_row_col_prod_AT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;
  arma_size_t aStep = block_size * A_col_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;
  arma_size_t bStep = block_size * B_internal_rows * B_col_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_col_size) && (col_block_id * block_size + row_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A^T * B
// matrix layouts: C...row_major, A...row_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_row_row_col_prod_TA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;
  arma_size_t aStep = block_size * A_row_inc * A_internal_cols;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) * B_internal_rows + B_row_start;
  arma_size_t bStep = block_size * B_row_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_row_size) && (col_block_id * block_size + col_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A^T * B^T
// matrix layouts: C...row_major, A...row_major, B...col_major
template<typename NumericT>
__global__ void matrix_matrix_row_row_col_prod_TT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;
  arma_size_t aStep = block_size * A_row_inc * A_internal_cols;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) + B_col_start * B_internal_rows;
  arma_size_t bStep = block_size * B_internal_rows * B_col_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_row_inc + col_thread_id * B_col_inc *  B_internal_rows;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_col_size) && (col_block_id * block_size + row_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}





////////////////////////////////////////////////////////////////////////////






// matrix-matrix multiplication C = A * B
// matrix layouts: C...col_major, A...row_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_col_row_row_prod_AA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;
  arma_size_t aStep = block_size * A_col_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;
  arma_size_t bStep = block_size * B_internal_cols * B_row_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A * B^T
// matrix layouts: C...col_major, A...row_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_col_row_row_prod_AT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;
  arma_size_t aStep = block_size * A_col_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;
  arma_size_t bStep = block_size * B_col_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A^T * B
// matrix layouts: C...col_major, A...row_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_col_row_row_prod_TA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;
  arma_size_t aStep = block_size * A_row_inc * A_internal_cols;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;
  arma_size_t bStep = block_size * B_internal_cols * B_row_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}

// matrix-matrix multiplication C = A^T * B^T
// matrix layouts: C...col_major, A...row_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_col_row_row_prod_TT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;
  arma_size_t aStep = block_size * A_row_inc * A_internal_cols;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;
  arma_size_t bStep = block_size * B_col_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[(blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start + ((blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start) * C_internal_rows];
}





////////////////////////////////////////////////////////////////////////////




// matrix-matrix multiplication C = A * B
// matrix layouts: C...row_major, A...row_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_row_row_row_prod_AA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;
  arma_size_t aStep = block_size * A_col_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;
  arma_size_t bStep = block_size * B_internal_cols * B_row_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A * B^T
// matrix layouts: C...row_major, A...row_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_row_row_row_prod_AT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_row_inc + A_row_start) * A_internal_cols + A_col_start;
  arma_size_t aStep = block_size * A_col_inc;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;
  arma_size_t bStep = block_size * B_col_inc;
  arma_size_t block_num = (A_col_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < A_col_size) && (row_block_id * block_size + col_thread_id < A_row_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_row_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A^T * B
// matrix layouts: C...row_major, A...row_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_row_row_row_prod_TA_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;
  arma_size_t aStep = block_size * A_row_inc * A_internal_cols;
  arma_size_t bBegin = (col_block_id * block_size * B_col_inc + B_col_start) + B_row_start * B_internal_cols;
  arma_size_t bStep = block_size * B_internal_cols * B_row_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < B_row_size) && (col_block_id * block_size + row_thread_id < B_col_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_col_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}

// matrix-matrix multiplication C = A^T * B^T
// matrix layouts: C...row_major, A...row_major, B...row_major
template<typename NumericT>
__global__ void matrix_matrix_row_row_row_prod_TT_kernel(
          NumericT alpha,
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * B,
          unsigned int B_row_start,
          unsigned int B_col_start,
          unsigned int B_row_inc,
          unsigned int B_col_inc,
          unsigned int B_row_size,
          unsigned int B_col_size,
          unsigned int B_internal_rows,
          unsigned int B_internal_cols,
          NumericT beta,
          NumericT * C,
          unsigned int C_row_start,
          unsigned int C_col_start,
          unsigned int C_row_inc,
          unsigned int C_col_inc,
          unsigned int C_row_size,
          unsigned int C_col_size,
          unsigned int C_internal_rows,
          unsigned int C_internal_cols)
{

  __shared__ NumericT bufA[272];
  __shared__ NumericT bufB[272];

  arma_size_t block_size = 16;//get_local_size(0);
  arma_size_t row_block_id = blockIdx.x;
  arma_size_t col_block_id = blockIdx.y;
  arma_size_t row_thread_id = threadIdx.x;
  arma_size_t col_thread_id = threadIdx.y;
  arma_size_t aBegin = (row_block_id * block_size * A_col_inc + A_col_start) + A_row_start * A_internal_cols;
  arma_size_t aStep = block_size * A_row_inc * A_internal_cols;
  arma_size_t bBegin = (col_block_id * block_size * B_row_inc + B_row_start) * B_internal_cols + B_col_start;
  arma_size_t bStep = block_size * B_col_inc;
  arma_size_t block_num = (A_row_size + block_size - 1) / block_size;
  NumericT Csub = 0;
  arma_size_t aOffset = row_thread_id * A_col_inc + col_thread_id * A_row_inc * A_internal_cols;
  arma_size_t bOffset = row_thread_id * B_col_inc + col_thread_id * B_row_inc * B_internal_cols;

  arma_size_t row_thread_id_times_block_size = row_thread_id * (block_size + 1);
  arma_size_t col_thread_id_times_block_size = col_thread_id * (block_size + 1);
  for (arma_size_t block = 0;
          block < block_num;
          ++block)
  {
    bufA[row_thread_id_times_block_size + col_thread_id] = ((block * block_size + col_thread_id < A_row_size) && (row_block_id * block_size + row_thread_id < A_col_size)) ? A[aBegin + aOffset] : 0;
    bufB[col_thread_id_times_block_size + row_thread_id] = ((block * block_size + row_thread_id < B_col_size) && (col_block_id * block_size + col_thread_id < B_row_size)) ? B[bBegin + bOffset] : 0;
    __syncthreads();
    NumericT * bufAptr = bufA + row_thread_id_times_block_size;
    NumericT * bufBptr = bufB + col_thread_id_times_block_size;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
      Csub += (*bufAptr) * (*bufBptr); ++bufAptr; ++bufBptr;
    __syncthreads();
    aBegin += aStep;
    bBegin += bStep;
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < A_col_size && (blockIdx.y * blockDim.y + threadIdx.y) < B_row_size)
    C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start] = (beta == 0) ? alpha * Csub : alpha * Csub + beta * C[((blockIdx.x * blockDim.x + threadIdx.x) * C_row_inc + C_row_start) * C_internal_cols + (blockIdx.y * blockDim.y + threadIdx.y) * C_col_inc + C_col_start];
}


} // namespace cuda
} //namespace blas
} //namespace cuarma