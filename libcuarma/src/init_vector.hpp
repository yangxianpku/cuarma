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

#include "cuarma.hpp"
#include "cuarma/backend/mem_handle.hpp"



static cuarmaStatus init_cuda_vector(cuarma::backend::mem_handle & h, cuarmaVector x)
{
#ifdef CUARMA_WITH_CUDA
  h.switch_active_handle_id(cuarma::CUDA_MEMORY);
  h.cuda_handle().reset(x->cuda_mem);
  h.cuda_handle().inc();
  if (x->precision == cuarmaFloat)
    h.raw_size(static_cast<cuarma::arma_size_t>(x->inc) * x->size * sizeof(float)); // not necessary, but still set for conciseness
  else if (x->precision == cuarmaDouble)
    h.raw_size(static_cast<cuarma::arma_size_t>(x->inc) * x->size * sizeof(double)); // not necessary, but still set for conciseness
  else
    return cuarmaGenericFailure;

  return cuarmaSuccess;
#else
  (void)h;
  (void)x;
  return cuarmaGenericFailure;
#endif
}

static cuarmaStatus init_host_vector(cuarma::backend::mem_handle & h, cuarmaVector x)
{
  h.switch_active_handle_id(cuarma::MAIN_MEMORY);
  h.ram_handle().reset(x->host_mem);
  h.ram_handle().inc();
  if (x->precision == cuarmaFloat)
    h.raw_size(static_cast<cuarma::arma_size_t>(x->inc) * static_cast<cuarma::arma_size_t>(x->size) * sizeof(float)); // not necessary, but still set for conciseness
  else if (x->precision == cuarmaDouble)
    h.raw_size(static_cast<cuarma::arma_size_t>(x->inc) * static_cast<cuarma::arma_size_t>(x->size) * sizeof(double)); // not necessary, but still set for conciseness
  else
    return cuarmaGenericFailure;

  return cuarmaSuccess;
}


static cuarmaStatus init_vector(cuarma::backend::mem_handle & h, cuarmaVector x)
{
  switch (x->backend->backend_type)
  {
    case cuarmaCUDA:
      return init_cuda_vector(h, x);
    case cuarmaHost:
      return init_host_vector(h, x);

    default:
      return cuarmaGenericFailure;
  }
}