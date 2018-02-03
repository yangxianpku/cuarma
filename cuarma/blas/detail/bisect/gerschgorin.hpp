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


/** @file cuarma/blas/detail//bisect/gerschgorin.hpp
 *  @encoding:UTF-8 文档编码
    @brief  Computation of Gerschgorin interval for symmetric, tridiagonal matrix

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include "cuarma/blas/detail/bisect/util.hpp"
#include "cuarma/vector.hpp"

namespace cuarma
{
namespace blas
{
namespace detail
{
  ////////////////////////////////////////////////////////////////////////////////
  //! Compute Gerschgorin interval for symmetric, tridiagonal matrix
  //! @param  d  diagonal elements
  //! @param  s  superdiagonal elements
  //! @param  n  size of matrix
  //! @param  lg  lower limit of Gerschgorin interval
  //! @param  ug  upper limit of Gerschgorin interval
  ////////////////////////////////////////////////////////////////////////////////
  template<typename NumericT>
  void
  computeGerschgorin(std::vector<NumericT> & d, std::vector<NumericT> & s, unsigned int n, NumericT &lg, NumericT &ug)
  {
      // compute bounds
      for (unsigned int i = 1; i < (n - 1); ++i)
      {

          // sum over the absolute values of all elements of row i
          NumericT sum_abs_ni = fabsf(s[i]) + fabsf(s[i + 1]);

          lg = min(lg, d[i] - sum_abs_ni);
          ug = max(ug, d[i] + sum_abs_ni);
      }

      // first and last row, only one superdiagonal element

      // first row
      lg = min(lg, d[0] - fabsf(s[1]));
      ug = max(ug, d[0] + fabsf(s[1]));

      // last row
      lg = min(lg, d[n-1] - fabsf(s[n-1]));
      ug = max(ug, d[n-1] + fabsf(s[n-1]));

      // increase interval to avoid side effects of fp arithmetic
      NumericT bnorm = max(fabsf(ug), fabsf(lg));

      // these values depend on the implmentation of floating count that is
      // employed in the following
      NumericT psi_0 = 11 * FLT_EPSILON * bnorm;
      NumericT psi_n = 11 * FLT_EPSILON * bnorm;

      lg = lg - bnorm * 2 * static_cast<NumericT>(n) * FLT_EPSILON - psi_0;
      ug = ug + bnorm * 2 * static_cast<NumericT>(n) * FLT_EPSILON + psi_n;

      ug = max(lg, ug);
  }
}  // namespace detail
}  // namespace blas
} // namespace cuarma
