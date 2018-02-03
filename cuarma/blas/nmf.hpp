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

/** @file cuarma/blas/nmf.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief 非负矩阵分解
    @brief Provides a nonnegative matrix factorization implementation.  Experimental.
 */

#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/norm_frobenius.hpp"
#include "cuarma/blas/host_based/nmf_operations.hpp"

#ifdef CUARMA_WITH_CUDA
#include "cuarma/blas/cuda/nmf_operations.hpp"
#endif

namespace cuarma
{
  namespace blas
  {
    /** @brief The nonnegative matrix factorization (approximation) algorithm as suggested by Lee and Seung. Factorizes a matrix V with nonnegative entries into matrices W and H such that ||V - W*H|| is minimized.
     *
     * @param V     Input matrix
     * @param W     First factor
     * @param H     Second factor
     * @param conf  A configuration object holding tolerances and the like
     */
    template<typename ScalarType>
    void nmf(cuarma::matrix_base<ScalarType> const & V, cuarma::matrix_base<ScalarType> & W, cuarma::matrix_base<ScalarType> & H, cuarma::blas::nmf_config const & conf)
    {
      assert(V.size1() == W.size1() && V.size2() == H.size2() && bool("Dimensions of W and H don't allow for V = W * H"));
      assert(W.size2() == H.size1() && bool("Dimensions of W and H don't match, prod(W, H) impossible"));

      switch (cuarma::traits::handle(V).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::nmf(V, W, H, conf);
          break;
#ifdef CUARMA_WITH_CUDA
          case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::nmf(V,W,H,conf);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");

      }

    }
  }
}
