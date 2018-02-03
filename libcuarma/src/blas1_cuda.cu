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

//include the generic inner product functions of cuarma
#include "cuarma/blas/inner_prod.hpp"

//include the generic norm functions of cuarma
#include "cuarma/blas/norm_1.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/norm_inf.hpp"


#ifdef CUARMA_WITH_CUDA


// IxAMAX

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDAiSamax(cuarmaBackend /*backend*/, cuarmaInt n,
                                                             cuarmaInt *index,
                                                             float *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);

  *index = static_cast<cuarmaInt>(cuarma::blas::index_norm_inf(v1));
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDAiDamax(cuarmaBackend /*backend*/, cuarmaInt n,
                                                             cuarmaInt *index,
                                                             double *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);

  *index = static_cast<cuarmaInt>(cuarma::blas::index_norm_inf(v1));
  return cuarmaSuccess;
}



// xASUM

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASasum(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float *alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);

  *alpha = cuarma::blas::norm_1(v1);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADasum(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double *alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);

  *alpha = cuarma::blas::norm_1(v1);
  return cuarmaSuccess;
}


// xAXPY

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASaxpy(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<float> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  v2 += alpha * v1;
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADaxpy(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<double> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  v2 += alpha * v1;
  return cuarmaSuccess;
}


// xCOPY

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDAScopy(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<float> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  v2 = v1;
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADcopy(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<double> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  v2 = v1;
  return cuarmaSuccess;
}

// xDOT

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASdot(cuarmaBackend /*backend*/, cuarmaInt n,
                                                           float *alpha,
                                                           float *x, cuarmaInt offx, cuarmaInt incx,
                                                           float *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<float> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  *alpha = cuarma::blas::inner_prod(v1, v2);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADdot(cuarmaBackend /*backend*/, cuarmaInt n,
                                                           double *alpha,
                                                           double *x, cuarmaInt offx, cuarmaInt incx,
                                                           double *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<double> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  *alpha = cuarma::blas::inner_prod(v1, v2);
  return cuarmaSuccess;
}

// xNRM2

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASnrm2(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float *alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);

  *alpha = cuarma::blas::norm_2(v1);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADnrm2(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double *alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);

  *alpha = cuarma::blas::norm_2(v1);
  return cuarmaSuccess;
}



// xROT

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASrot(cuarmaBackend /*backend*/, cuarmaInt n,
                                                           float *x, cuarmaInt offx, cuarmaInt incx,
                                                           float *y, cuarmaInt offy, cuarmaInt incy,
                                                           float c, float s)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<float> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  cuarma::blas::plane_rotation(v1, v2, c, s);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADrot(cuarmaBackend /*backend*/, cuarmaInt n,
                                                           double *x, cuarmaInt offx, cuarmaInt incx,
                                                           double *y, cuarmaInt offy, cuarmaInt incy,
                                                           double c, double s)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<double> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  cuarma::blas::plane_rotation(v1, v2, c, s);
  return cuarmaSuccess;
}



// xSCAL

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASscal(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float alpha,
                                                            float *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);

  v1 *= alpha;
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADscal(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double alpha,
                                                            double *x, cuarmaInt offx, cuarmaInt incx)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);

  v1 *= alpha;
  return cuarmaSuccess;
}


// xSWAP

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDASswap(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float *x, cuarmaInt offx, cuarmaInt incx,
                                                            float *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<float> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<float> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  cuarma::swap(v1, v2);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaCUDADswap(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double *x, cuarmaInt offx, cuarmaInt incx,
                                                            double *y, cuarmaInt offy, cuarmaInt incy)
{
  cuarma::vector_base<double> v1(x, cuarma::CUDA_MEMORY, n, offx, incx);
  cuarma::vector_base<double> v2(y, cuarma::CUDA_MEMORY, n, offy, incy);

  cuarma::swap(v1, v2);
  return cuarmaSuccess;
}
#endif


