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


#ifdef CUARMA_WITH_CUDA

// xGEMV

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASgemv(cuarmaBackend /*backend*/,
                                                            cuarmaOrder order, cuarmaTranspose transA,
                                                            cuarmaInt m, cuarmaInt n, float alpha, float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float beta,
                                                            float *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<float> v2(y, cuarma::CUDA_MEMORY, m, offy, incy);
  cuarma::matrix_base<float> mat(A, cuarma::CUDA_MEMORY,
                                   m, offA_row, incA_row, m,
                                   n, offA_col, incA_col, lda, order == cuarmaRowMajor);
  v2 *= beta;
  if (transA == cuarmaTrans)
    v2 += alpha * cuarma::blas::prod(cuarma::trans(mat), v1);
  else
    v2 += alpha * cuarma::blas::prod(mat, v1);

  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADgemv(cuarmaBackend /*backend*/,
                                                            cuarmaOrder order, cuarmaTranspose transA,
                                                            cuarmaInt m, cuarmaInt n, double alpha, double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double beta,
                                                            double *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<double> v2(y, cuarma::CUDA_MEMORY, m, offy, incy);
  cuarma::matrix_base<double> mat(A, cuarma::CUDA_MEMORY,
                                    m, offA_row, incA_row, m,
                                    n, offA_col, incA_col, lda, order == cuarmaRowMajor);
  v2 *= beta;
  if (transA == cuarmaTrans)
    v2 += alpha * cuarma::blas::prod(cuarma::trans(mat), v1);
  else
    v2 += alpha * cuarma::blas::prod(mat, v1);

  return cuarmaSuccess;
}



// xTRSV

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDAStrsv(cuarmaBackend /*backend*/,
                                                            cuarmaUplo uplo, cuarmaOrder order, cuarmaTranspose transA, cuarmaDiag diag,
                                                            cuarmaInt n, float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            float *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<float> v(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::matrix_base<float> mat(A, cuarma::CUDA_MEMORY,
                                   n, offA_row, incA_row, n,
                                   n, offA_col, incA_col, lda, order == cuarmaRowMajor);
  if (transA == cuarmaTrans)
  {
    if (uplo == cuarmaUpper)
      if (diag == cuarmaUnit)
        cuarma::blas::inplace_solve(cuarma::trans(mat), v, cuarma::blas::unit_upper_tag());
      else
        cuarma::blas::inplace_solve(cuarma::trans(mat), v, cuarma::blas::upper_tag());
    else
      if (diag == cuarmaUnit)
        cuarma::blas::inplace_solve(cuarma::trans(mat), v, cuarma::blas::unit_lower_tag());
      else
        cuarma::blas::inplace_solve(cuarma::trans(mat), v, cuarma::blas::lower_tag());
  }
  else
  {
    if (uplo == cuarmaUpper)
      if (diag == cuarmaUnit)
        cuarma::blas::inplace_solve(mat, v, cuarma::blas::unit_upper_tag());
      else
        cuarma::blas::inplace_solve(mat, v, cuarma::blas::upper_tag());
    else
      if (diag == cuarmaUnit)
        cuarma::blas::inplace_solve(mat, v, cuarma::blas::unit_lower_tag());
      else
        cuarma::blas::inplace_solve(mat, v, cuarma::blas::lower_tag());
  }

  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADtrsv(cuarmaBackend /*backend*/,
                                                            cuarmaUplo uplo, cuarmaOrder order, cuarmaTranspose transA, cuarmaDiag diag,
                                                            cuarmaInt n, double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda,
                                                            double *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<double> v(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::matrix_base<double> mat(A, cuarma::CUDA_MEMORY,
                                    n, offA_row, incA_row, n,
                                    n, offA_col, incA_col, lda, order == cuarmaRowMajor);
  if (transA == cuarmaTrans)
  {
    if (uplo == cuarmaUpper)
      if (diag == cuarmaUnit)
        cuarma::blas::inplace_solve(cuarma::trans(mat), v, cuarma::blas::unit_upper_tag());
      else
        cuarma::blas::inplace_solve(cuarma::trans(mat), v, cuarma::blas::upper_tag());
    else
      if (diag == cuarmaUnit)
        cuarma::blas::inplace_solve(cuarma::trans(mat), v, cuarma::blas::unit_lower_tag());
      else
        cuarma::blas::inplace_solve(cuarma::trans(mat), v, cuarma::blas::lower_tag());
  }
  else
  {
    if (uplo == cuarmaUpper)
      if (diag == cuarmaUnit)
        cuarma::blas::inplace_solve(mat, v, cuarma::blas::unit_upper_tag());
      else
        cuarma::blas::inplace_solve(mat, v, cuarma::blas::upper_tag());
    else
      if (diag == cuarmaUnit)
        cuarma::blas::inplace_solve(mat, v, cuarma::blas::unit_lower_tag());
      else
        cuarma::blas::inplace_solve(mat, v, cuarma::blas::lower_tag());
  }

  return cuarmaSuccess;
}



// xGER

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASger(cuarmaBackend /*backend*/,
                                                           cuarmaOrder order,
                                                           cuarmaInt m, cuarmaInt n,
                                                           float alpha,
                                                           float *x, cuarmaInt offx, cuarmaInt incx,
                                                           float *y, cuarmaInt offy, cuarmaInt incy,
                                                           float *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<float> v2(y, cuarma::CUDA_MEMORY, m, offy, incy);
  cuarma::matrix_base<float> mat(A, cuarma::CUDA_MEMORY,
                                   m, offA_row, incA_row, m,
                                   n, offA_col, incA_col, lda, order == cuarmaRowMajor);

  mat += alpha * cuarma::blas::outer_prod(v1, v2);

  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADger(cuarmaBackend /*backend*/,
                                                           cuarmaOrder order,
                                                           cuarmaInt m,  cuarmaInt n,
                                                           double alpha,
                                                           double *x, cuarmaInt offx, cuarmaInt incx,
                                                           double *y, cuarmaInt offy, cuarmaInt incy,
                                                           double *A, cuarmaInt offA_row, cuarmaInt offA_col, cuarmaInt incA_row, cuarmaInt incA_col, cuarmaInt lda)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<double> v2(y, cuarma::CUDA_MEMORY, m, offy, incy);
  cuarma::matrix_base<double> mat(A, cuarma::CUDA_MEMORY,
                                    m, offA_row, incA_row, m,
                                    n, offA_col, incA_col, lda, order == cuarmaRowMajor);

  mat += alpha * cuarma::blas::outer_prod(v1, v2);

  return cuarmaSuccess;
}

#endif
