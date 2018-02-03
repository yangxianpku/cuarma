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



static cuarmaStatus init_cuda_matrix(cuarma::backend::mem_handle & h, cuarmaMatrix A)
{
#ifdef CUARMA_WITH_CUDA
  h.switch_active_handle_id(cuarma::CUDA_MEMORY);
  h.cuda_handle().reset(A->cuda_mem);
  h.cuda_handle().inc();
  if (A->precision == cuarmaFloat)
    h.raw_size(static_cast<cuarma::arma_size_t>(A->internal_size1) * static_cast<cuarma::arma_size_t>(A->internal_size2) * sizeof(float)); // not necessary, but still set for conciseness
  else if (A->precision == cuarmaDouble)
    h.raw_size(static_cast<cuarma::arma_size_t>(A->internal_size1) * static_cast<cuarma::arma_size_t>(A->internal_size2) * sizeof(double)); // not necessary, but still set for conciseness
  else
    return cuarmaGenericFailure;

  return cuarmaSuccess;
#else
  (void)h;
  (void)A;
  return cuarmaGenericFailure;
#endif
}


static cuarmaStatus init_host_matrix(cuarma::backend::mem_handle & h, cuarmaMatrix A)
{
  h.switch_active_handle_id(cuarma::MAIN_MEMORY);
  h.ram_handle().reset(A->host_mem);
  h.ram_handle().inc();
  if (A->precision == cuarmaFloat)
    h.raw_size(static_cast<cuarma::arma_size_t>(A->internal_size1) * static_cast<cuarma::arma_size_t>(A->internal_size2) * sizeof(float)); // not necessary, but still set for conciseness
  else if (A->precision == cuarmaDouble)
    h.raw_size(static_cast<cuarma::arma_size_t>(A->internal_size1) * static_cast<cuarma::arma_size_t>(A->internal_size2) * sizeof(double)); // not necessary, but still set for conciseness
  else
    return cuarmaGenericFailure;

  return cuarmaSuccess;
}


static cuarmaStatus init_matrix(cuarma::backend::mem_handle & h, cuarmaMatrix A)
{
  switch (A->backend->backend_type)
  {
    case cuarmaCUDA:
      return init_cuda_matrix(h, A);
    case cuarmaHost:
      return init_host_matrix(h, A);

    default:
      return cuarmaGenericFailure;
  }
}