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


/** @file cuarma/blas/detail//bisect/bisect_large.hpp
 *  @encoding:UTF-8 文档编码
    @brief 大型对称、对角矩阵的特征值计算
    @brief Computation of eigenvalues of a large symmetric, tridiagonal matrix

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/


// includes, system
#include <iostream>
#include <iomanip>  
#include <stdlib.h>
#include <stdio.h>

// includes, project
#include "cuarma/blas/detail/bisect/config.hpp"
#include "cuarma/blas/detail/bisect/structs.hpp"
#include "cuarma/blas/detail/bisect/bisect_kernel_calls.hpp"

namespace cuarma
{
namespace blas
{
namespace detail
{

////////////////////////////////////////////////////////////////////////////////
//! Run the kernels to compute the eigenvalues for large matrices
//! @param  input   handles to input data
//! @param  result  handles to result data
//! @param  mat_size  matrix size
//! @param  precision  desired precision of eigenvalues
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
////////////////////////////////////////////////////////////////////////////////
template<typename NumericT>
void computeEigenvaluesLargeMatrix(InputData<NumericT> &input, ResultDataLarge<NumericT> &result,
                const unsigned int mat_size, const NumericT lg, const NumericT ug,  const NumericT precision)
{

  // First kernel call: decide on which intervals bisect_Large_OneIntervals/
  // bisect_Large_MultIntervals is executed
  cuarma::blas::detail::bisectLarge(input, result, mat_size, lg, ug, precision);

  // compute eigenvalues for intervals that contained only one eigenvalue
  // after the first processing step
  cuarma::blas::detail::bisectLarge_OneIntervals(input, result, mat_size, precision);

  // process intervals that contained more than one eigenvalue after
  // the first processing step
  cuarma::blas::detail::bisectLarge_MultIntervals(input, result, mat_size, precision);

}

////////////////////////////////////////////////////////////////////////////////
//! Process the result, that is obtain result from device and do simple sanity
//! checking
//! @param  result  handles to result data
//! @param  mat_size  matrix size
////////////////////////////////////////////////////////////////////////////////
template<typename NumericT>
bool processResultDataLargeMatrix(ResultDataLarge<NumericT> &result, const unsigned int mat_size)
{
    bool bCompareResult = true;
    // copy data from intervals that contained more than one eigenvalue after
    // the first processing step
    std::vector<NumericT> lambda_mult(mat_size);
    cuarma::copy(result.g_lambda_mult, lambda_mult);

    std::vector<unsigned int> pos_mult(mat_size);
    cuarma::copy(result.g_pos_mult, pos_mult);

    std::vector<unsigned int> blocks_mult_sum(mat_size);
    cuarma::copy(result.g_blocks_mult_sum, blocks_mult_sum);

    unsigned int num_one_intervals = result.g_num_one;
    unsigned int sum_blocks_mult = mat_size - num_one_intervals;

    // copy data for intervals that contained one eigenvalue after the first
    // processing step
    std::vector<NumericT> left_one(mat_size);
    std::vector<NumericT> right_one(mat_size);
    std::vector<unsigned int> pos_one(mat_size);

    cuarma::copy(result.g_left_one, left_one);
    cuarma::copy(result.g_right_one, right_one);
    cuarma::copy(result.g_pos_one, pos_one);

    // singleton intervals generated in the second step
    for (unsigned int i = 0; i < sum_blocks_mult; ++i)
    {
      if (pos_mult[i] != 0)
        result.std_eigenvalues[pos_mult[i] - 1] = lambda_mult[i];

      else
      {
        throw memory_exception("Invalid array index! Are there more than 256 equal eigenvalues?");
      }
    }

    // singleton intervals generated in the first step
    unsigned int index = 0;

    for (unsigned int i = 0; i < num_one_intervals; ++i, ++index)
    {
        result.std_eigenvalues[pos_one[i] - 1] = left_one[i];
    }
    return bCompareResult;
}
} // namespace detail
}  // namespace blas
}  // namespace cuarma
