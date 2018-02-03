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

#include "blas3.hpp"

//include basic scalar and vector types of cuarma
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/blas/prod.hpp"


#ifdef CUARMA_WITH_CUDA

//
// xGEMV
//

namespace detail
{
  template <typename NumericT>
  cuarmaStatus cuarmaCUDAgemm_impl(cuarmaBackend /*backend*/,
                                       cuarmaOrder orderA, cuarmaTranspose transA,
                                       cuarmaOrder orderB, cuarmaTranspose transB,
                                       cuarmaOrder orderC,
                                       cuarmaInt m, cuarmaInt n, cuarmaInt k,
                                       NumericT alpha,
                                       NumericT *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                       NumericT *B, cuarmaInt offB_row, cuarmaInt offB_col, cuarmaInt incB_row, cuarmaInt incB_col, cuarmaInt ldb,
                                       NumericT beta,
                                       NumericT *C, cuarmaInt offC_row, cuarmaInt offC_col, cuarmaInt incC_row, cuarmaInt incC_col, cuarmaInt ldc)
  {
    cuarmaInt A_size1 = (transA == cuarmaTrans) ? k : m;
    cuarmaInt A_size2 = (transA == cuarmaTrans) ? m : k;

    cuarmaInt B_size1 = (transB == cuarmaTrans) ? n : k;
    cuarmaInt B_size2 = (transB == cuarmaTrans) ? k : n;

    bool A_row_major = (orderA == cuarmaRowMajor);
    bool B_row_major = (orderB == cuarmaRowMajor);
    bool C_row_major = (orderC == cuarmaRowMajor);

    cuarma::matrix_base<NumericT> matA(A, cuarma::CUDA_MEMORY,
                                         A_size1, offA_row, incA_row, A_row_major ? m : lda,
                                         A_size2, offA_col, incA_col, A_row_major ? lda : k, A_row_major);

    cuarma::matrix_base<NumericT> matB(B, cuarma::CUDA_MEMORY,
                                         B_size1, offB_row, incB_row, B_row_major ? k : ldb,
                                         B_size2, offB_col, incB_col, B_row_major ? ldb : n, B_row_major);

    cuarma::matrix_base<NumericT> matC(C, cuarma::CUDA_MEMORY,
                                         m, offC_row, incC_row, C_row_major ? m : ldc,
                                         n, offC_col, incC_col, C_row_major ? ldc : n, C_row_major);

    detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);

    return cuarmaSuccess;
  }

}


CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASgemm(cuarmaBackend backend,
                                                            cuarmaOrder orderA, cuarmaTranspose transA,
                                                            cuarmaOrder orderB, cuarmaTranspose transB,
                                                            cuarmaOrder orderC,
                                                            cuarmaInt m, cuarmaInt n, cuarmaInt k,
                                                            float alpha,
                                                            float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            float *B, cuarmaInt offB_row, cuarmaInt offB_col, cuarmaInt incB_row, cuarmaInt incB_col, cuarmaInt ldb,
                                                            float beta,
                                                            float *C, cuarmaInt offC_row, cuarmaInt offC_col, cuarmaInt incC_row, cuarmaInt incC_col, cuarmaInt ldc)
{
  return detail::cuarmaCUDAgemm_impl<float>(backend,
                                              orderA, transA,
                                              orderB, transB,
                                              orderC,
                                              m, n, k,
                                              alpha,
                                              A, offA_row, offA_col, incA_row, incA_col, lda,
                                              B, offB_row, offB_col, incB_row, incB_col, ldb,
                                              beta,
                                              C, offC_row, offC_col, incC_row, incC_col, ldc);
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADgemm(cuarmaBackend backend,
                                                            cuarmaOrder orderA, cuarmaTranspose transA,
                                                            cuarmaOrder orderB, cuarmaTranspose transB,
                                                            cuarmaOrder orderC,
                                                            cuarmaInt m, cuarmaInt n, cuarmaInt k,
                                                            double alpha,
                                                            double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            double *B, cuarmaInt offB_row, cuarmaInt offB_col, cuarmaInt incB_row, cuarmaInt incB_col, cuarmaInt ldb,
                                                            double beta,
                                                            double *C, cuarmaInt offC_row, cuarmaInt offC_col, cuarmaInt incC_row, cuarmaInt incC_col, cuarmaInt ldc)
{
  return detail::cuarmaCUDAgemm_impl<double>(backend,
                                               orderA, transA,
                                               orderB, transB,
                                               orderC,
                                               m, n, k,
                                               alpha,
                                               A, offA_row, offA_col, incA_row, incA_col, lda,
                                               B, offB_row, offB_col, incB_row, incB_col, ldb,
                                               beta,
                                               C, offC_row, offC_col, incC_row, incC_col, ldc);
}


#endif
