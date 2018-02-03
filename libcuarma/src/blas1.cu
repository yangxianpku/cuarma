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

//include basic scalar and vector types of cuarma
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"

//include the generic inner product functions of cuarma
#include "cuarma/blas/inner_prod.hpp"

//include the generic norm functions of cuarma
#include "cuarma/blas/norm_1.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/norm_inf.hpp"

// IxAMAX

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaiamax(cuarmaInt *index, cuarmaVector x)
{
  cuarma::backend::mem_handle v1_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type     difference_type;
      cuarma::vector_base<float> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));

      *index = static_cast<cuarmaInt>(cuarma::blas::index_norm_inf(v1));
      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type     difference_type;
      cuarma::vector_base<double> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));

      *index = static_cast<cuarmaInt>(cuarma::blas::index_norm_inf(v1));
      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}




// xASUM

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaasum(cuarmaHostScalar *alpha, cuarmaVector x)
{
  if ((*alpha)->precision != x->precision)
    return cuarmaGenericFailure;

  cuarma::backend::mem_handle v1_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type     difference_type;
      cuarma::vector_base<float> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));

      (*alpha)->value_float = cuarma::blas::norm_1(v1);
      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type     difference_type;
      cuarma::vector_base<double> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));

      (*alpha)->value_double = cuarma::blas::norm_1(v1);
      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}



// xAXPY

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaaxpy(cuarmaHostScalar alpha, cuarmaVector x, cuarmaVector y)
{
  if (alpha->precision != x->precision)
    return cuarmaGenericFailure;

  if (x->precision != y->precision)
    return cuarmaGenericFailure;

  cuarma::backend::mem_handle v1_handle;
  cuarma::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_vector(v2_handle, y) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type     difference_type;
      cuarma::vector_base<float> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<float> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      v2 += alpha->value_float * v1;
      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type     difference_type;
      cuarma::vector_base<double> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<double> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      v2 += alpha->value_double * v1;
      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}


// xCOPY

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmacopy(cuarmaVector x, cuarmaVector y)
{
  if (x->precision != y->precision)
    return cuarmaGenericFailure;

  cuarma::backend::mem_handle v1_handle;
  cuarma::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_vector(v2_handle, y) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type     difference_type;
      cuarma::vector_base<float> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<float> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      v2 = v1;
      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type     difference_type;
      cuarma::vector_base<double> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<double> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      v2 = v1;
      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}

// xDOT

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmadot(cuarmaHostScalar *alpha, cuarmaVector x, cuarmaVector y)
{
  if ((*alpha)->precision != x->precision)
    return cuarmaGenericFailure;

  if (x->precision != y->precision)
    return cuarmaGenericFailure;

  cuarma::backend::mem_handle v1_handle;
  cuarma::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_vector(v2_handle, y) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type     difference_type;
      cuarma::vector_base<float> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<float> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      (*alpha)->value_float = cuarma::blas::inner_prod(v1, v2);
      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type     difference_type;
      cuarma::vector_base<double> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<double> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      (*alpha)->value_double = cuarma::blas::inner_prod(v1, v2);
      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}

// xNRM2

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmanrm2(cuarmaHostScalar *alpha, cuarmaVector x)
{
  if ((*alpha)->precision != x->precision)
    return cuarmaGenericFailure;

  cuarma::backend::mem_handle v1_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type     difference_type;
      cuarma::vector_base<float> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));

      (*alpha)->value_float = cuarma::blas::norm_2(v1);
      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type     difference_type;
      cuarma::vector_base<double> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));

      (*alpha)->value_double = cuarma::blas::norm_2(v1);
      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}



// xROT

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmarot(cuarmaVector     x, cuarmaVector     y,
                                                      cuarmaHostScalar c, cuarmaHostScalar s)
{
  if (c->precision != x->precision)
    return cuarmaGenericFailure;

  if (s->precision != x->precision)
    return cuarmaGenericFailure;

  if (x->precision != y->precision)
    return cuarmaGenericFailure;

  cuarma::backend::mem_handle v1_handle;
  cuarma::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_vector(v2_handle, y) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type     difference_type;
      cuarma::vector_base<float> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<float> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      cuarma::blas::plane_rotation(v1, v2, c->value_float, s->value_float);
      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type     difference_type;
      cuarma::vector_base<double> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<double> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      cuarma::blas::plane_rotation(v1, v2, c->value_double, s->value_double);
      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}

// xSCAL

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmascal(cuarmaHostScalar alpha, cuarmaVector x)
{
  if (alpha->precision != x->precision)
    return cuarmaGenericFailure;

  cuarma::backend::mem_handle v1_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type     difference_type;
      cuarma::vector_base<float> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));

      v1 *= alpha->value_float;
      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type     difference_type;
      cuarma::vector_base<double> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));

      v1 *= alpha->value_double;
      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}


// xSWAP

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaswap(cuarmaVector x, cuarmaVector y)
{
  if (x->precision != y->precision)
    return cuarmaGenericFailure;

  cuarma::backend::mem_handle v1_handle;
  cuarma::backend::mem_handle v2_handle;

  if (init_vector(v1_handle, x) != cuarmaSuccess)
    return cuarmaGenericFailure;

  if (init_vector(v2_handle, y) != cuarmaSuccess)
    return cuarmaGenericFailure;

  switch (x->precision)
  {
    case cuarmaFloat:
    {
      typedef cuarma::vector_base<float>::size_type     difference_type;
      cuarma::vector_base<float> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<float> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      cuarma::swap(v1, v2);
      return cuarmaSuccess;
    }

    case cuarmaDouble:
    {
      typedef cuarma::vector_base<double>::size_type     difference_type;
      cuarma::vector_base<double> v1(v1_handle, static_cast<cuarma::arma_size_t>(x->size), static_cast<cuarma::arma_size_t>(x->offset), static_cast<difference_type>(x->inc));
      cuarma::vector_base<double> v2(v2_handle, static_cast<cuarma::arma_size_t>(y->size), static_cast<cuarma::arma_size_t>(y->offset), static_cast<difference_type>(y->inc));

      cuarma::swap(v1, v2);
      return cuarmaSuccess;
    }

    default:
      return cuarmaGenericFailure;
  }
}


