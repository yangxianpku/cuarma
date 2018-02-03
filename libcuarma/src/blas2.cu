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

#include "init_vector.hpp"
#include "init_matrix.hpp"

//include basic scalar and vector types of cuarma
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/blas/prod.hpp"

// GEMV

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmagemv(cuarmaHostScalar alpha, cuarmaMatrix A, cuarmaVector x, cuarmaHostScalar beta, cuarmaVector y)
{
  cuarma::backend::mem_handle v1_handle;
  cuarma::backend::mem_handle v2_handle;
  cuarma::backend::mem_handle A_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_vector(v2_handle, y) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_matrix(A_handle, A) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type           size_type;
      typedef cuarma::vector_base<float>::size_type           difference_type;

      cuarma::vector_base<float> v1(v1_handle, size_type(x->size), size_type(x->offset), difference_type(x->inc));
      cuarma::vector_base<float> v2(v2_handle, size_type(y->size), size_type(y->offset), difference_type(y->inc));

      cuarma::matrix_base<float> mat(A_handle,
                                       size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                       size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);
      v2 *= beta->value_float;
      if (A->trans == cuarmaTrans)
        v2 += alpha->value_float * cuarma::blas::prod(cuarma::trans(mat), v1);
      else
        v2 += alpha->value_float * cuarma::blas::prod(mat, v1);

      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type           size_type;
      typedef cuarma::vector_base<double>::size_type           difference_type;

      cuarma::vector_base<double> v1(v1_handle, size_type(x->size), size_type(x->offset), difference_type(x->inc));
      cuarma::vector_base<double> v2(v2_handle, size_type(y->size), size_type(y->offset), difference_type(y->inc));

      cuarma::matrix_base<double> mat(A_handle,
                                        size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                        size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);
      v2 *= beta->value_double;
      if (A->trans == cuarmaTrans)
        v2 += alpha->value_double * cuarma::blas::prod(cuarma::trans(mat), v1);
      else
        v2 += alpha->value_double * cuarma::blas::prod(mat, v1);

      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}


// xTRSV

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmatrsv(cuarmaMatrix A, cuarmaVector x, cuarmaUplo uplo)
{
  cuarma::backend::mem_handle v1_handle;
  cuarma::backend::mem_handle A_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_matrix(A_handle, A) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type           size_type;
      typedef cuarma::vector_base<float>::size_type           difference_type;

      cuarma::vector_base<float> v1(v1_handle, size_type(x->size), size_type(x->offset), difference_type(x->inc));

      cuarma::matrix_base<float> mat(A_handle,
                                       size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                       size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);
      if (A->trans == cuarmaTrans)
      {
        if (uplo == cuarmaUpper)
          cuarma::blas::inplace_solve(cuarma::trans(mat), v1, cuarma::blas::upper_tag());
        else
          cuarma::blas::inplace_solve(cuarma::trans(mat), v1, cuarma::blas::lower_tag());
      }
      else
      {
        if (uplo == cuarmaUpper)
          cuarma::blas::inplace_solve(mat, v1, cuarma::blas::upper_tag());
        else
          cuarma::blas::inplace_solve(mat, v1, cuarma::blas::lower_tag());
      }

      return cuarmaSuccess;
    }
    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type           size_type;
      typedef cuarma::vector_base<double>::size_type           difference_type;

      cuarma::vector_base<double> v1(v1_handle, size_type(x->size), size_type(x->offset), difference_type(x->inc));

      cuarma::matrix_base<double> mat(A_handle,
                                        size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                        size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);
      if (A->trans == cuarmaTrans)
      {
        if (uplo == cuarmaUpper)
          cuarma::blas::inplace_solve(cuarma::trans(mat), v1, cuarma::blas::upper_tag());
        else
          cuarma::blas::inplace_solve(cuarma::trans(mat), v1, cuarma::blas::lower_tag());
      }
      else
      {
        if (uplo == cuarmaUpper)
          cuarma::blas::inplace_solve(mat, v1, cuarma::blas::upper_tag());
        else
          cuarma::blas::inplace_solve(mat, v1, cuarma::blas::lower_tag());
      }

      return cuarmaSuccess;
    }

    default:
      return  cuarmaGenericFailure;
  }
}


// xGER

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmager(cuarmaHostScalar alpha, cuarmaVector x, cuarmaVector y, cuarmaMatrix A)
{
  cuarma::backend::mem_handle v1_handle;
  cuarma::backend::mem_handle v2_handle;
  cuarma::backend::mem_handle A_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_vector(v2_handle, y) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_matrix(A_handle, A) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type           size_type;
      typedef cuarma::vector_base<float>::size_type           difference_type;

      cuarma::vector_base<float> v1(v1_handle, size_type(x->size), size_type(x->offset), difference_type(x->inc));
      cuarma::vector_base<float> v2(v2_handle, size_type(y->size), size_type(y->offset), difference_type(y->inc));

      cuarma::matrix_base<float> mat(A_handle,
                                       size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                       size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);

      mat += alpha->value_float * cuarma::blas::outer_prod(v1, v2);

      return cuarmaSuccess;
    }
    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type           size_type;
      typedef cuarma::vector_base<double>::size_type           difference_type;

      cuarma::vector_base<double> v1(v1_handle, size_type(x->size), size_type(x->offset), difference_type(x->inc));
      cuarma::vector_base<double> v2(v2_handle, size_type(y->size), size_type(y->offset), difference_type(y->inc));

      cuarma::matrix_base<double> mat(A_handle,
                                        size_type(A->size1), size_type(A->start1), difference_type(A->stride1), size_type(A->internal_size1),
                                        size_type(A->size2), size_type(A->start2), difference_type(A->stride2), size_type(A->internal_size2), A->order == cuarmaRowMajor);

      mat += alpha->value_double * cuarma::blas::outer_prod(v1, v2);

      return cuarmaSuccess;
    }
    default:
      return  cuarmaGenericFailure;
  }
}


