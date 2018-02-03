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

#include <stdlib.h>


#endif

#include "cuarma.hpp"


/************* Backend Management ******************/

struct cuarmaCUDABackend_impl
{
    //TODO: Add stream and/or device descriptors here
};

struct cuarmaHostBackend_impl
{
  // Nothing to specify *at the moment*
};


/** @brief Generic backend for CUDA host-based stuff */
struct cuarmaBackend_impl
{
  cuarmaBackendTypes backend_type;
  cuarmaCUDABackend_impl     cuda_backend;
  cuarmaHostBackend_impl     host_backend;
};



/******** User Types **********/

struct cuarmaHostScalar_impl
{
  cuarmaPrecision  precision;

  union {
    float  value_float;
    double value_double;
  };
};

struct cuarmaScalar_impl
{
  cuarmaBackend    backend;
  cuarmaPrecision  precision;

  // buffer:
#ifdef CUARMA_WITH_CUDA
  char * cuda_mem;
#endif

  char * host_mem;

  cuarmaInt   offset;
};

struct cuarmaVector_impl
{
  cuarmaBackend    backend;
  cuarmaPrecision  precision;

  // buffer:
#ifdef CUARMA_WITH_CUDA
  char * cuda_mem;
#endif

  char * host_mem;

  cuarmaInt   offset;
  cuarmaInt   inc;
  cuarmaInt   size;
};

struct cuarmaMatrix_impl
{
  cuarmaBackend    backend;
  cuarmaPrecision  precision;
  cuarmaOrder      order;
  cuarmaTranspose  trans;

  // buffer:
#ifdef CUARMA_WITH_CUDA
  char * cuda_mem;
#endif

  char * host_mem;

  cuarmaInt   size1;
  cuarmaInt   start1;
  cuarmaInt   stride1;
  cuarmaInt   internal_size1;

  cuarmaInt   size2;
  cuarmaInt   start2;
  cuarmaInt   stride2;
  cuarmaInt   internal_size2;
};