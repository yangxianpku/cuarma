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


/** @file cuarma/blas/bisect_gpu.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementation of an bisection algorithm for eigenvalues

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"

// includes, project
#include "cuarma/blas/detail/bisect/structs.hpp"
#include "cuarma/blas/detail/bisect/gerschgorin.hpp"
#include "cuarma/blas/detail/bisect/bisect_large.hpp"
#include "cuarma/blas/detail/bisect/bisect_small.hpp"


namespace cuarma
{
namespace blas
{
///////////////////////////////////////////////////////////////////////////
//! @brief bisect           The bisection algorithm computes the eigevalues
//!                         of a symmetric tridiagonal matrix.
//! @param diagonal         diagonal elements of the matrix
//! @param superdiagonal    superdiagonal elements of the matrix
//! @param eigenvalues      Vectors with the eigenvalues in ascending order
//! @return                 return false if any errors occured
///
//! overloaded function template: std::vectors as parameters
template<typename NumericT>
bool
bisect(const std::vector<NumericT> & diagonal, const std::vector<NumericT> & superdiagonal, std::vector<NumericT> & eigenvalues)
{
  assert(diagonal.size() == superdiagonal.size() &&
         diagonal.size() == eigenvalues.size()   &&
         bool("Input vectors do not have the same sizes!"));
  bool bResult = false;
  // flag if the matrix size is due to explicit user request
  // desired precision of eigenvalues
  NumericT  precision = static_cast<NumericT>(0.00001);
  const unsigned int mat_size = static_cast<unsigned int>(diagonal.size());

  // set up input
  cuarma::blas::detail::InputData<NumericT> input(diagonal, superdiagonal, mat_size);

  NumericT lg =  FLT_MAX;
  NumericT ug = -FLT_MAX;
  // compute Gerschgorin interval
  cuarma::blas::detail::computeGerschgorin(input.std_a, input.std_b, mat_size, lg, ug);

  // decide wheter the algorithm for small or for large matrices will be started
  if (mat_size <= CUARMA_BISECT_MAX_SMALL_MATRIX)
  {
    // initialize memory for result
    cuarma::blas::detail::ResultDataSmall<NumericT> result(mat_size);

    // run the kernel
    cuarma::blas::detail::computeEigenvaluesSmallMatrix(input, result, mat_size, lg, ug, precision);

    // get the result from the device and do some sanity checks,
    cuarma::blas::detail::processResultSmallMatrix(result, mat_size);
    eigenvalues = result.std_eigenvalues;
    bResult = true;
  }

  else
  {
    // initialize memory for result
    cuarma::blas::detail::ResultDataLarge<NumericT> result(mat_size);

    // run the kernel
    cuarma::blas::detail::computeEigenvaluesLargeMatrix(input, result, mat_size, lg, ug, precision);

    // get the result from the device and do some sanity checks
    bResult = cuarma::blas::detail::processResultDataLargeMatrix(result, mat_size);

    eigenvalues = result.std_eigenvalues;
  }
  return bResult;
}


///////////////////////////////////////////////////////////////////////////
//! @brief bisect           The bisection algorithm computes the eigevalues
//!                         of a symmetric tridiagonal matrix.
//! @param diagonal         diagonal elements of the matrix
//! @param superdiagonal    superdiagonal elements of the matrix
//! @param eigenvalues      Vectors with the eigenvalues in ascending order
//! @return                 return false if any errors occured
///
//! overloaded function template: cuarma::vectors as parameters
template<typename NumericT>
bool
bisect(const cuarma::vector<NumericT> & diagonal, const cuarma::vector<NumericT> & superdiagonal, cuarma::vector<NumericT> & eigenvalues)
{
  assert(diagonal.size() == superdiagonal.size() &&
         diagonal.size() == eigenvalues.size()   &&
         bool("Input vectors do not have the same sizes!"));
  bool bResult = false;
  // flag if the matrix size is due to explicit user request
  // desired precision of eigenvalues
  NumericT  precision = static_cast<NumericT>(0.00001);
  const unsigned int mat_size = static_cast<unsigned int>(diagonal.size());

  // set up input
  cuarma::blas::detail::InputData<NumericT> input(diagonal, superdiagonal, mat_size);

  NumericT lg =  FLT_MAX;
  NumericT ug = -FLT_MAX;
  // compute Gerschgorin interval
  cuarma::blas::detail::computeGerschgorin(input.std_a, input.std_b, mat_size, lg, ug);

  // decide wheter the algorithm for small or for large matrices will be started
  if (mat_size <= CUARMA_BISECT_MAX_SMALL_MATRIX)
  {
    // initialize memory for result
    cuarma::blas::detail::ResultDataSmall<NumericT> result(mat_size);

    // run the kernel
    cuarma::blas::detail::computeEigenvaluesSmallMatrix(input, result, mat_size, lg, ug, precision);

    // get the result from the device and do some sanity checks,
    cuarma::blas::detail::processResultSmallMatrix(result, mat_size);
    copy(result.std_eigenvalues, eigenvalues);
    bResult = true;
  }

  else
  {
    // initialize memory for result
    cuarma::blas::detail::ResultDataLarge<NumericT> result(mat_size);

    // run the kernel
    cuarma::blas::detail::computeEigenvaluesLargeMatrix(input, result, mat_size, lg, ug, precision);

    // get the result from the device and do some sanity checks
    bResult = cuarma::blas::detail::processResultDataLargeMatrix(result, mat_size);

    copy(result.std_eigenvalues, eigenvalues);
  }
  return bResult;
}
} // namespace blas
} // namespace cuarma
