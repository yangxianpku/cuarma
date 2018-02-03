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

/** @file cuarma/blas/cuda/vector_operations.hpp
 *  @encoding:UTF-8 文档编码
 @brief Implementations of NMF operations using CUDA
 */

#include "cuarma/blas/host_based/nmf_operations.hpp"

#include "cuarma/blas/cuda/common.hpp"

namespace cuarma
{
namespace blas
{
namespace cuda
{

/** @brief Main CUDA kernel for nonnegative matrix factorization of a dense matrices. */
template<typename NumericT>
__global__ void el_wise_mul_div(NumericT       * matrix1,
                                NumericT const * matrix2,
                                NumericT const * matrix3,
                                unsigned int size)
{
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i +=gridDim.x * blockDim.x)
  {
    NumericT val = matrix1[i] * matrix2[i];
    NumericT divisor = matrix3[i];
    matrix1[i] = (divisor > (NumericT) 0.00001) ? (val / divisor) : NumericT(0);
  }
}

/** @brief The nonnegative matrix factorization (approximation) algorithm as suggested by Lee and Seung. Factorizes a matrix V with nonnegative entries into matrices W and H such that ||V - W*H|| is minimized.
 *
 * @param V     Input matrix
 * @param W     First factor
 * @param H     Second factor
 * @param conf  A configuration object holding tolerances and the like
 */
template<typename NumericT>
void nmf(cuarma::matrix_base<NumericT> const & V,
         cuarma::matrix_base<NumericT> & W,
         cuarma::matrix_base<NumericT> & H,
         cuarma::blas::nmf_config const & conf)
{
  arma_size_t k = W.size2();
  conf.iters_ = 0;

  if (!cuarma::blas::norm_frobenius(W))
    W = cuarma::scalar_matrix<NumericT>(W.size1(), W.size2(), NumericT(1.0));

  if (!cuarma::blas::norm_frobenius(H))
    H = cuarma::scalar_matrix<NumericT>(H.size1(), H.size2(), NumericT(1.0));

  cuarma::matrix_base<NumericT> wn(V.size1(), k, W.row_major());
  cuarma::matrix_base<NumericT> wd(V.size1(), k, W.row_major());
  cuarma::matrix_base<NumericT> wtmp(V.size1(), V.size2(), W.row_major());

  cuarma::matrix_base<NumericT> hn(k, V.size2(), H.row_major());
  cuarma::matrix_base<NumericT> hd(k, V.size2(), H.row_major());
  cuarma::matrix_base<NumericT> htmp(k, k, H.row_major());

  cuarma::matrix_base<NumericT> appr(V.size1(), V.size2(), V.row_major());

  cuarma::vector<NumericT> diff(V.size1() * V.size2());

  NumericT last_diff = 0;
  NumericT diff_init = 0;
  bool stagnation_flag = false;

  for (arma_size_t i = 0; i < conf.max_iterations(); i++)
  {
    conf.iters_ = i + 1;

    hn = cuarma::blas::prod(trans(W), V);
    htmp = cuarma::blas::prod(trans(W), W);
    hd = cuarma::blas::prod(htmp, H);

    el_wise_mul_div<<<128, 128>>>(cuarma::cuda_arg<NumericT>(H),
                                  cuarma::cuda_arg<NumericT>(hn),
                                  cuarma::cuda_arg<NumericT>(hd),
                                  static_cast<unsigned int>(H.internal_size1() * H.internal_size2()));
    CUARMA_CUDA_LAST_ERROR_CHECK("el_wise_mul_div");

    wn   = cuarma::blas::prod(V, trans(H));
    wtmp = cuarma::blas::prod(W, H);
    wd   = cuarma::blas::prod(wtmp, trans(H));

    el_wise_mul_div<<<128, 128>>>(cuarma::cuda_arg<NumericT>(W),
                                  cuarma::cuda_arg<NumericT>(wn),
                                  cuarma::cuda_arg<NumericT>(wd),
                                  static_cast<unsigned int>( W.internal_size1() * W.internal_size2()));
    CUARMA_CUDA_LAST_ERROR_CHECK("el_wise_mul_div");

    if (i % conf.check_after_steps() == 0)  //check for convergence
    {
      appr = cuarma::blas::prod(W, H);

      appr -= V;
      NumericT diff_val = cuarma::blas::norm_frobenius(appr);

      if (i == 0)
        diff_init = diff_val;

      if (conf.print_relative_error())
        std::cout << diff_val / diff_init << std::endl;

      // Approximation check
      if (diff_val / diff_init < conf.tolerance())
        break;

      // Stagnation check
      if (std::fabs(diff_val - last_diff) / (diff_val * conf.check_after_steps()) < conf.stagnation_tolerance()) //avoid situations where convergence stagnates
      {
        if (stagnation_flag)  // iteration stagnates (two iterates with no notable progress)
          break;
        else
          // record stagnation in this iteration
          stagnation_flag = true;
      } else
        // good progress in this iteration, so unset stagnation flag
        stagnation_flag = false;

      // prepare for next iterate:
      last_diff = diff_val;
    }
  }
}

} //namespace cuda
} //namespace blas
} //namespace cuarma
