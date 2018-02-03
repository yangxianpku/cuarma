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

// include necessary system headers
#include <iostream>

#include "cuarma.hpp"
#include "cuarma_private.hpp"

//include basic scalar and vector types of cuarma
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"

#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/blas/prod.hpp"

namespace detail
{
  template <typename ScalarType, typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
  void gemm_dispatch(ScalarType alpha,
                     MatrixTypeA const & A, cuarmaTranspose transA,
                     MatrixTypeB const & B, cuarmaTranspose transB,
                     ScalarType beta,
                     MatrixTypeC & C)
  {

    if (transA == cuarmaTrans && transB == cuarmaTrans)
      cuarma::blas::prod_impl(cuarma::trans(A), cuarma::trans(B), C, alpha, beta);
    else if (transA == cuarmaTrans && transB == cuarmaNoTrans)
      cuarma::blas::prod_impl(cuarma::trans(A), B, C, alpha, beta);
    else if (transA == cuarmaNoTrans && transB == cuarmaTrans)
      cuarma::blas::prod_impl(A, cuarma::trans(B), C, alpha, beta);
    else if (transA == cuarmaNoTrans && transB == cuarmaNoTrans)
      cuarma::blas::prod_impl(A, B, C, alpha, beta);
    //else
    //  return cuarmaGenericFailure;
  }
}
