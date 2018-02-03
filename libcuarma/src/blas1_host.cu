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


// IxAMAX

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostiSamax(cuarmaBackend /*backend*/, cuarmaInt n,
                                                             cuarmaInt *index,
                                                             float *x, cuarmaInt offx, int incx)
{
  typedef cuarma::vector_base<float>::size_type           size_type;
  typedef cuarma::vector_base<float>::size_type           difference_type;
  cuarma::vector_base<float> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *index = static_cast<cuarmaInt>(cuarma::blas::index_norm_inf(v1));
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostiDamax(cuarmaBackend /*backend*/, cuarmaInt n,
                                                             cuarmaInt *index,
                                                             double *x, cuarmaInt offx, int incx)
{
  typedef cuarma::vector_base<double>::size_type           size_type;
  typedef cuarma::vector_base<double>::size_type           difference_type;
  cuarma::vector_base<double> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *index = static_cast<cuarmaInt>(cuarma::blas::index_norm_inf(v1));
  return cuarmaSuccess;
}



// xASUM

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSasum(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float *alpha,
                                                            float *x, cuarmaInt offx, int incx)
{
  typedef cuarma::vector_base<float>::size_type           size_type;
  typedef cuarma::vector_base<float>::size_type           difference_type;
  cuarma::vector_base<float> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *alpha = cuarma::blas::norm_1(v1);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDasum(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double *alpha,
                                                            double *x, cuarmaInt offx, int incx)
{
  typedef cuarma::vector_base<double>::size_type           size_type;
  typedef cuarma::vector_base<double>::size_type           difference_type;
  cuarma::vector_base<double> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *alpha = cuarma::blas::norm_1(v1);
  return cuarmaSuccess;
}



// xAXPY

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSaxpy(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float alpha,
                                                            float *x, cuarmaInt offx, int incx,
                                                            float *y, cuarmaInt offy, int incy)
{
  typedef cuarma::vector_base<float>::size_type           size_type;
  typedef cuarma::vector_base<float>::size_type           difference_type;
  cuarma::vector_base<float> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<float> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  v2 += alpha * v1;
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDaxpy(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double alpha,
                                                            double *x, cuarmaInt offx, int incx,
                                                            double *y, cuarmaInt offy, int incy)
{
  typedef cuarma::vector_base<double>::size_type           size_type;
  typedef cuarma::vector_base<double>::size_type           difference_type;
  cuarma::vector_base<double> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<double> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  v2 += alpha * v1;
  return cuarmaSuccess;
}


// xCOPY

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostScopy(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float *x, cuarmaInt offx, int incx,
                                                            float *y, cuarmaInt offy, int incy)
{
  typedef cuarma::vector_base<float>::size_type           size_type;
  typedef cuarma::vector_base<float>::size_type           difference_type;
  cuarma::vector_base<float> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<float> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  v2 = v1;
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDcopy(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double *x, cuarmaInt offx, int incx,
                                                            double *y, cuarmaInt offy, int incy)
{
  typedef cuarma::vector_base<double>::size_type           size_type;
  typedef cuarma::vector_base<double>::size_type           difference_type;
  cuarma::vector_base<double> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<double> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  v2 = v1;
  return cuarmaSuccess;
}

// xAXPY

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSdot(cuarmaBackend /*backend*/, cuarmaInt n,
                                                           float *alpha,
                                                           float *x, cuarmaInt offx, int incx,
                                                           float *y, cuarmaInt offy, int incy)
{
  typedef cuarma::vector_base<float>::size_type           size_type;
  typedef cuarma::vector_base<float>::size_type           difference_type;
  cuarma::vector_base<float> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<float> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  *alpha = cuarma::blas::inner_prod(v1, v2);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDdot(cuarmaBackend /*backend*/, cuarmaInt n,
                                                           double *alpha,
                                                           double *x, cuarmaInt offx, int incx,
                                                           double *y, cuarmaInt offy, int incy)
{
  typedef cuarma::vector_base<double>::size_type           size_type;
  typedef cuarma::vector_base<double>::size_type           difference_type;
  cuarma::vector_base<double> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<double> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  *alpha = cuarma::blas::inner_prod(v1, v2);
  return cuarmaSuccess;
}

// xNRM2

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSnrm2(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float *alpha,
                                                            float *x, cuarmaInt offx, int incx)
{
  typedef cuarma::vector_base<float>::size_type           size_type;
  typedef cuarma::vector_base<float>::size_type           difference_type;
  cuarma::vector_base<float> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *alpha = cuarma::blas::norm_2(v1);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDnrm2(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double *alpha,
                                                            double *x, cuarmaInt offx, int incx)
{
  typedef cuarma::vector_base<double>::size_type           size_type;
  typedef cuarma::vector_base<double>::size_type           difference_type;
  cuarma::vector_base<double> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  *alpha = cuarma::blas::norm_2(v1);
  return cuarmaSuccess;
}


// xROT

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSrot(cuarmaBackend /*backend*/, cuarmaInt n,
                                                           float *x, cuarmaInt offx, int incx,
                                                           float *y, cuarmaInt offy, int incy,
                                                           float c, float s)
{
  typedef cuarma::vector_base<float>::size_type           size_type;
  typedef cuarma::vector_base<float>::size_type           difference_type;
  cuarma::vector_base<float> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<float> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  cuarma::blas::plane_rotation(v1, v2, c, s);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDrot(cuarmaBackend /*backend*/, cuarmaInt n,
                                                           double *x, cuarmaInt offx, int incx,
                                                           double *y, cuarmaInt offy, int incy,
                                                           double c, double s)
{
  typedef cuarma::vector_base<double>::size_type           size_type;
  typedef cuarma::vector_base<double>::size_type           difference_type;
  cuarma::vector_base<double> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<double> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  cuarma::blas::plane_rotation(v1, v2, c, s);
  return cuarmaSuccess;
}



// xSCAL

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSscal(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float alpha,
                                                            float *x, cuarmaInt offx, int incx)
{
  typedef cuarma::vector_base<float>::size_type           size_type;
  typedef cuarma::vector_base<float>::size_type           difference_type;
  cuarma::vector_base<float> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  v1 *= alpha;
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDscal(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double alpha,
                                                            double *x, cuarmaInt offx, int incx)
{
  typedef cuarma::vector_base<double>::size_type           size_type;
  typedef cuarma::vector_base<double>::size_type           difference_type;
  cuarma::vector_base<double> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));

  v1 *= alpha;
  return cuarmaSuccess;
}

// xSWAP

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostSswap(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            float *x, cuarmaInt offx, int incx,
                                                            float *y, cuarmaInt offy, int incy)
{
  typedef cuarma::vector_base<float>::size_type           size_type;
  typedef cuarma::vector_base<float>::size_type           difference_type;
  cuarma::vector_base<float> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<float> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  cuarma::swap(v1, v2);
  return cuarmaSuccess;
}

CUARMA_EXPORTED_FUNCTION cuarmaStatus cuarmaHostDswap(cuarmaBackend /*backend*/, cuarmaInt n,
                                                            double *x, cuarmaInt offx, int incx,
                                                            double *y, cuarmaInt offy, int incy)
{
  typedef cuarma::vector_base<double>::size_type           size_type;
  typedef cuarma::vector_base<double>::size_type           difference_type;
  cuarma::vector_base<double> v1(x, cuarma::MAIN_MEMORY, size_type(n), size_type(offx), difference_type(incx));
  cuarma::vector_base<double> v2(y, cuarma::MAIN_MEMORY, size_type(n), size_type(offy), difference_type(incy));

  cuarma::swap(v1, v2);
  return cuarmaSuccess;
}
