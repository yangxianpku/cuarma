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


/** @file cuarma/blas/cuda/bisect_kernel_calls.hpp
 *  @encoding:UTF-8 文档编码
    @brief CUDA kernel calls for the bisection algorithm

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/

#include "cuarma/blas/detail/bisect/structs.hpp"

// includes, kernels
#include "cuarma/blas/cuda/bisect_kernel_small.hpp"
#include "cuarma/blas/cuda/bisect_kernel_large.hpp"
#include "cuarma/blas/cuda/bisect_kernel_large_onei.hpp"
#include "cuarma/blas/cuda/bisect_kernel_large_multi.hpp"


namespace cuarma
{
namespace blas
{
namespace cuda
{
template<typename NumericT>
void bisectSmall(const cuarma::blas::detail::InputData<NumericT> &input, cuarma::blas::detail::ResultDataSmall<NumericT> &result,
                       const unsigned int mat_size,
                       const NumericT lg, const NumericT ug,
                       const NumericT precision)
{


  dim3  blocks(1, 1, 1);
  dim3  threads(CUARMA_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX, 1, 1);

  bisectKernelSmall<<< blocks, threads >>>(
    cuarma::cuda_arg(input.g_a),
    cuarma::cuda_arg(input.g_b) + 1,
    mat_size,
    cuarma::cuda_arg(result.arma_g_left),
    cuarma::cuda_arg(result.arma_g_right),
    cuarma::cuda_arg(result.arma_g_left_count),
    cuarma::cuda_arg(result.arma_g_right_count),
    lg, ug, 0, mat_size,
    precision
    );
  cuarma::blas::cuda::CUARMA_CUDA_LAST_ERROR_CHECK("Kernel launch failed");
}


template<typename NumericT>
void bisectLarge(const cuarma::blas::detail::InputData<NumericT> &input, cuarma::blas::detail::ResultDataLarge<NumericT> &result,
                   const unsigned int mat_size,
                   const NumericT lg, const NumericT ug,
                   const NumericT precision)
 {

  dim3  blocks(1, 1, 1);
  dim3  threads(mat_size > 512 ? CUARMA_BISECT_MAX_THREADS_BLOCK : CUARMA_BISECT_MAX_THREADS_BLOCK / 2 , 1, 1);
  bisectKernelLarge<<< blocks, threads >>>
    (cuarma::cuda_arg(input.g_a),
     cuarma::cuda_arg(input.g_b) + 1,
     mat_size,
     lg, ug, static_cast<unsigned int>(0), mat_size, precision,
     cuarma::cuda_arg(result.g_num_one),
     cuarma::cuda_arg(result.g_num_blocks_mult),
     cuarma::cuda_arg(result.g_left_one),
     cuarma::cuda_arg(result.g_right_one),
     cuarma::cuda_arg(result.g_pos_one),
     cuarma::cuda_arg(result.g_left_mult),
     cuarma::cuda_arg(result.g_right_mult),
     cuarma::cuda_arg(result.g_left_count_mult),
     cuarma::cuda_arg(result.g_right_count_mult),
     cuarma::cuda_arg(result.g_blocks_mult),
     cuarma::cuda_arg(result.g_blocks_mult_sum)
     );
  cuarma::blas::cuda::CUARMA_CUDA_LAST_ERROR_CHECK("Kernel launch failed.");
}


// compute eigenvalues for intervals that contained only one eigenvalue
// after the first processing step
template<typename NumericT>
void bisectLarge_OneIntervals(const cuarma::blas::detail::InputData<NumericT> &input, cuarma::blas::detail::ResultDataLarge<NumericT> &result,
                   const unsigned int mat_size,
                   const NumericT precision)
 {

  unsigned int num_one_intervals = result.g_num_one;
  unsigned int num_blocks = cuarma::blas::detail::getNumBlocksLinear(num_one_intervals,
                                                                         mat_size > 512 ? CUARMA_BISECT_MAX_THREADS_BLOCK : CUARMA_BISECT_MAX_THREADS_BLOCK / 2);
  dim3 grid_onei;
  grid_onei.x = num_blocks;
  grid_onei.y = 1, grid_onei.z = 1;
  dim3 threads_onei(mat_size > 512 ? CUARMA_BISECT_MAX_THREADS_BLOCK : CUARMA_BISECT_MAX_THREADS_BLOCK / 2, 1, 1);


  bisectKernelLarge_OneIntervals<<< grid_onei , threads_onei >>>
    (cuarma::cuda_arg(input.g_a),
     cuarma::cuda_arg(input.g_b) + 1,
     mat_size, num_one_intervals,
     cuarma::cuda_arg(result.g_left_one),
     cuarma::cuda_arg(result.g_right_one),
     cuarma::cuda_arg(result.g_pos_one),
     precision
     );
  cuarma::blas::cuda::CUARMA_CUDA_LAST_ERROR_CHECK("bisectKernelLarge_OneIntervals() FAILED.");
}


// process intervals that contained more than one eigenvalue after
// the first processing step
template<typename NumericT>
void bisectLarge_MultIntervals(const cuarma::blas::detail::InputData<NumericT> &input, cuarma::blas::detail::ResultDataLarge<NumericT> &result,
                   const unsigned int mat_size,
                   const NumericT precision)
 {
    // get the number of blocks of intervals that contain, in total when
    // each interval contains only one eigenvalue, not more than
    // MAX_THREADS_BLOCK threads
    unsigned int  num_blocks_mult = result.g_num_blocks_mult;

    // setup the execution environment
    dim3  grid_mult(num_blocks_mult, 1, 1);
    dim3  threads_mult(mat_size > 512 ? CUARMA_BISECT_MAX_THREADS_BLOCK : CUARMA_BISECT_MAX_THREADS_BLOCK / 2, 1, 1);

    bisectKernelLarge_MultIntervals<<< grid_mult, threads_mult >>>
      (cuarma::cuda_arg(input.g_a),
       cuarma::cuda_arg(input.g_b) + 1,
       mat_size,
       cuarma::cuda_arg(result.g_blocks_mult),
       cuarma::cuda_arg(result.g_blocks_mult_sum),
       cuarma::cuda_arg(result.g_left_mult),
       cuarma::cuda_arg(result.g_right_mult),
       cuarma::cuda_arg(result.g_left_count_mult),
       cuarma::cuda_arg(result.g_right_count_mult),
       cuarma::cuda_arg(result.g_lambda_mult),
       cuarma::cuda_arg(result.g_pos_mult),
       precision
      );
    cuarma::blas::cuda::CUARMA_CUDA_LAST_ERROR_CHECK("bisectKernelLarge_MultIntervals() FAILED.");
}
}
}
}
