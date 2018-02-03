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

#include "init_matrix.hpp"

//include basic scalar and vector types of cuarma
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/blas/prod.hpp"

// GEMV

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmagemm(cuarmaHostScalar alpha, cuarmaMatrix A, cuarmaMatrix B, cuarmaHostScalar beta, cuarmaMatrix C)
{
  cuarma::backend::mem_handle A_handle;
  cuarma::backend::mem_handle B_handle;
  cuarma::backend::mem_handle C_handle;

  if (init_matrix(A_handle, A) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_matrix(B_handle, B) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_matrix(C_handle, C) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (A->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::matrix_base<float>::size_type           size_type;
      typedef cuarma::matrix_base<float>::size_type           difference_type;

      cuarma::matrix_base<float> mat_A(A_handle,
                                         size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                         size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);
      cuarma::matrix_base<float> mat_B(B_handle,
                                         size_type(B->size1), size_type(B->start1), difference_type(B->stride1), size_type(B->internal_size1),
                                         size_type(B->size2), size_type(B->start2), difference_type(B->stride2), size_type(B->internal_size2), B->order == cuarmaRowMajor);
      cuarma::matrix_base<float> mat_C(C_handle,
                                         size_type(C->size1), size_type(C->start1), difference_type(C->stride1), size_type(C->internal_size1),
                                         size_type(C->size2), size_type(C->start2), difference_type(C->stride2), size_type(C->internal_size2), C->order == cuarmaRowMajor);

      if (A->trans == cuarmaTrans && B->trans == cuarmaTrans)
        cuarma::blas::prod_impl(cuarma::trans(mat_A), cuarma::trans(mat_B), mat_C, alpha->value_float, beta->value_float);
      else if (A->trans == cuarmaTrans && B->trans == cuarmaNoTrans)
        cuarma::blas::prod_impl(cuarma::trans(mat_A), mat_B, mat_C, alpha->value_float, beta->value_float);
      else if (A->trans == cuarmaNoTrans && B->trans == cuarmaTrans)
        cuarma::blas::prod_impl(mat_A, cuarma::trans(mat_B), mat_C, alpha->value_float, beta->value_float);
      else if (A->trans == cuarmaNoTrans && B->trans == cuarmaNoTrans)
        cuarma::blas::prod_impl(mat_A, mat_B, mat_C, alpha->value_float, beta->value_float);
      else
        return cuarmaGenericFailure;

      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::matrix_base<double>::size_type           size_type;
      typedef cuarma::matrix_base<double>::size_type           difference_type;

      cuarma::matrix_base<double> mat_A(A_handle,
                                          size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                          size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);
      cuarma::matrix_base<double> mat_B(B_handle,
                                          size_type(B->size1), size_type(B->start1), difference_type(B->stride1), size_type(B->internal_size1),
                                          size_type(B->size2), size_type(B->start2), difference_type(B->stride2), size_type(B->internal_size2), B->order == cuarmaRowMajor);
      cuarma::matrix_base<double> mat_C(C_handle,
                                          size_type(C->size1), size_type(C->start1), difference_type(C->stride1), size_type(C->internal_size1),
                                          size_type(C->size2), size_type(C->start2), difference_type(C->stride2), size_type(C->internal_size2), C->order == cuarmaRowMajor);

      if (A->trans == cuarmaTrans && B->trans == cuarmaTrans)
        cuarma::blas::prod_impl(cuarma::trans(mat_A), cuarma::trans(mat_B), mat_C, alpha->value_double, beta->value_double);
      else if (A->trans == cuarmaTrans && B->trans == cuarmaNoTrans)
        cuarma::blas::prod_impl(cuarma::trans(mat_A), mat_B, mat_C, alpha->value_double, beta->value_double);
      else if (A->trans == cuarmaNoTrans && B->trans == cuarmaTrans)
        cuarma::blas::prod_impl(mat_A, cuarma::trans(mat_B), mat_C, alpha->value_double, beta->value_double);
      else if (A->trans == cuarmaNoTrans && B->trans == cuarmaNoTrans)
        cuarma::blas::prod_impl(mat_A, mat_B, mat_C, alpha->value_double, beta->value_double);
      else
        return cuarmaGenericFailure;

      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}


// xTRSV

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmatrsm(cuarmaMatrix A, cuarmaUplo uplo, cuarmaDiag diag, cuarmaMatrix B)
{
  cuarma::backend::mem_handle A_handle;
  cuarma::backend::mem_handle B_handle;

  if (init_matrix(A_handle, A) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_matrix(B_handle, B) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (A->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::matrix_base<float>::size_type           size_type;
      typedef cuarma::matrix_base<float>::size_type           difference_type;

      cuarma::matrix_base<float> mat_A(A_handle,
                                         size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                         size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);
      cuarma::matrix_base<float> mat_B(B_handle,
                                         size_type(B->size1), size_type(B->start1), difference_type(B->stride1), size_type(B->internal_size1),
                                         size_type(B->size2), size_type(B->start2), difference_type(B->stride2), size_type(B->internal_size2), B->order == cuarmaRowMajor);

      if (A->trans == cuarmaTrans && B->trans == cuarmaTrans)
      {
        if (uplo == cuarmaUpper && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::upper_tag());
        else if (uplo == cuarmaUpper && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::unit_upper_tag());
        else if (uplo == cuarmaLower && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::lower_tag());
        else if (uplo == cuarmaLower && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::unit_lower_tag());
        else
          return cuarmaGenericFailure;
      }
      else if (A->trans == cuarmaTrans && B->trans == cuarmaNoTrans)
      {
        if (uplo == cuarmaUpper && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), mat_B, cuarma::blas::upper_tag());
        else if (uplo == cuarmaUpper && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), mat_B, cuarma::blas::unit_upper_tag());
        else if (uplo == cuarmaLower && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), mat_B, cuarma::blas::lower_tag());
        else if (uplo == cuarmaLower && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), mat_B, cuarma::blas::unit_lower_tag());
        else
          return cuarmaGenericFailure;
      }
      else if (A->trans == cuarmaNoTrans && B->trans == cuarmaTrans)
      {
        if (uplo == cuarmaUpper && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::upper_tag());
        else if (uplo == cuarmaUpper && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::unit_upper_tag());
        else if (uplo == cuarmaLower && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::lower_tag());
        else if (uplo == cuarmaLower && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::unit_lower_tag());
        else
          return cuarmaGenericFailure;
      }
      else if (A->trans == cuarmaNoTrans && B->trans == cuarmaNoTrans)
      {
        if (uplo == cuarmaUpper && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(mat_A, mat_B, cuarma::blas::upper_tag());
        else if (uplo == cuarmaUpper && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(mat_A, mat_B, cuarma::blas::unit_upper_tag());
        else if (uplo == cuarmaLower && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(mat_A, mat_B, cuarma::blas::lower_tag());
        else if (uplo == cuarmaLower && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(mat_A, mat_B, cuarma::blas::unit_lower_tag());
        else
          return cuarmaGenericFailure;
      }

      return cuarmaSuccess;
    }
    case cuarmaDouble:
    {
      typedef cuarma::matrix_base<double>::size_type           size_type;
      typedef cuarma::matrix_base<double>::size_type           difference_type;

      cuarma::matrix_base<double> mat_A(A_handle,
                                          size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                          size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);
      cuarma::matrix_base<double> mat_B(B_handle,
                                          size_type(B->size1), size_type(B->start1), difference_type(B->stride1), size_type(B->internal_size1),
                                          size_type(B->size2), size_type(B->start2), difference_type(B->stride2), size_type(B->internal_size2), B->order == cuarmaRowMajor);

      if (A->trans == cuarmaTrans && B->trans == cuarmaTrans)
      {
        if (uplo == cuarmaUpper && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::upper_tag());
        else if (uplo == cuarmaUpper && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::unit_upper_tag());
        else if (uplo == cuarmaLower && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::lower_tag());
        else if (uplo == cuarmaLower && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::unit_lower_tag());
        else
          return cuarmaGenericFailure;
      }
      else if (A->trans == cuarmaTrans && B->trans == cuarmaNoTrans)
      {
        if (uplo == cuarmaUpper && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), mat_B, cuarma::blas::upper_tag());
        else if (uplo == cuarmaUpper && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), mat_B, cuarma::blas::unit_upper_tag());
        else if (uplo == cuarmaLower && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), mat_B, cuarma::blas::lower_tag());
        else if (uplo == cuarmaLower && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), mat_B, cuarma::blas::unit_lower_tag());
        else
          return cuarmaGenericFailure;
      }
      else if (A->trans == cuarmaNoTrans && B->trans == cuarmaTrans)
      {
        if (uplo == cuarmaUpper && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::upper_tag());
        else if (uplo == cuarmaUpper && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::unit_upper_tag());
        else if (uplo == cuarmaLower && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::lower_tag());
        else if (uplo == cuarmaLower && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(cuarma::trans(mat_A), cuarma::trans(mat_B), cuarma::blas::unit_lower_tag());
        else
          return cuarmaGenericFailure;
      }
      else if (A->trans == cuarmaNoTrans && B->trans == cuarmaNoTrans)
      {
        if (uplo == cuarmaUpper && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(mat_A, mat_B, cuarma::blas::upper_tag());
        else if (uplo == cuarmaUpper && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(mat_A, mat_B, cuarma::blas::unit_upper_tag());
        else if (uplo == cuarmaLower && diag == cuarmaNonUnit)
          cuarma::blas::inplace_solve(mat_A, mat_B, cuarma::blas::lower_tag());
        else if (uplo == cuarmaLower && diag == cuarmaUnit)
          cuarma::blas::inplace_solve(mat_A, mat_B, cuarma::blas::unit_lower_tag());
        else
          return cuarmaGenericFailure;
      }

      return cuarmaSuccess;
    }

    default:
      return  cuarmaGenericFailure;
  }
}



