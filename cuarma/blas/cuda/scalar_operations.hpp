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

/** @file cuarma/blas/cuda/scalar_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of scalar operations using CUDA
*/

#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/start.hpp"
#include "cuarma/traits/stride.hpp"
#include "cuarma/blas/cuda/common.hpp"

// includes CUDA
#include <cuda_runtime.h>


namespace cuarma
{
namespace blas
{
namespace cuda
{

/////////////////// as /////////////////////////////

template<typename NumericT>
__global__ void as_kernel(NumericT * s1, const NumericT * fac2, unsigned int options2, const NumericT * s2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = NumericT(1) / alpha;

  *s1 = *s2 * alpha;
}

template<typename NumericT>
__global__ void as_kernel(NumericT * s1, NumericT fac2, unsigned int options2, const NumericT * s2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = NumericT(1) / alpha;

  *s1 = *s2 * alpha;
}

template <typename ScalarT1, typename ScalarT2, typename NumericT>
typename cuarma::enable_if< cuarma::is_scalar<ScalarT1>::value
                              && cuarma::is_scalar<ScalarT2>::value
                              && cuarma::is_any_scalar<NumericT>::value
                            >::type
as(ScalarT1 & s1,
   ScalarT2 const & s2, NumericT const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  typedef typename cuarma::result_of::cpu_value_type<ScalarT1>::type        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  value_type temporary_alpha = 0;
  if (cuarma::is_cpu_scalar<NumericT>::value)
    temporary_alpha = alpha;

  as_kernel<<<1, 1>>>(cuarma::cuda_arg(s1),
                      cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                      options_alpha,
                      cuarma::cuda_arg(s2));
  CUARMA_CUDA_LAST_ERROR_CHECK("as_kernel");
}

//////////////////// asbs ////////////////////////////

// alpha and beta on GPU
template<typename NumericT>
__global__ void asbs_kernel(NumericT * s1,
                            const NumericT * fac2, unsigned int options2, const NumericT * s2,
                            const NumericT * fac3, unsigned int options3, const NumericT * s3)
{
    NumericT alpha = *fac2;
    if (options2 & (1 << 0))
      alpha = -alpha;
    if (options2 & (1 << 1))
      alpha = NumericT(1) / alpha;

    NumericT beta = *fac3;
    if (options3 & (1 << 0))
      beta = -beta;
    if (options3 & (1 << 1))
      beta = NumericT(1) / beta;

    *s1 = *s2 * alpha + *s3 * beta;
}

// alpha on CPU, beta on GPU
template<typename NumericT>
__global__ void asbs_kernel(NumericT * s1,
                            NumericT fac2,         unsigned int options2, const NumericT * s2,
                            NumericT const * fac3, unsigned int options3, const NumericT * s3)
{
    NumericT alpha = fac2;
    if (options2 & (1 << 0))
      alpha = -alpha;
    if (options2 & (1 << 1))
      alpha = NumericT(1) / alpha;

    NumericT beta = *fac3;
    if (options3 & (1 << 0))
      beta = -beta;
    if (options3 & (1 << 1))
      beta = NumericT(1) / beta;

    *s1 = *s2 * alpha + *s3 * beta;
}

// alpha on GPU, beta on CPU
template<typename NumericT>
__global__ void asbs_kernel(NumericT * s1,
                            NumericT const * fac2, unsigned int options2, const NumericT * s2,
                            NumericT         fac3, unsigned int options3, const NumericT * s3)
{
    NumericT alpha = *fac2;
    if (options2 & (1 << 0))
      alpha = -alpha;
    if (options2 & (1 << 1))
      alpha = NumericT(1) / alpha;

    NumericT beta = fac3;
    if (options3 & (1 << 0))
      beta = -beta;
    if (options3 & (1 << 1))
      beta = NumericT(1) / beta;

    *s1 = *s2 * alpha + *s3 * beta;
}

// alpha and beta on CPU
template<typename NumericT>
__global__ void asbs_kernel(NumericT * s1,
                            NumericT fac2, unsigned int options2, const NumericT * s2,
                            NumericT fac3, unsigned int options3, const NumericT * s3)
{
    NumericT alpha = fac2;
    if (options2 & (1 << 0))
      alpha = -alpha;
    if (options2 & (1 << 1))
      alpha = NumericT(1) / alpha;

    NumericT beta = fac3;
    if (options3 & (1 << 0))
      beta = -beta;
    if (options3 & (1 << 1))
      beta = NumericT(1) / beta;

    *s1 = *s2 * alpha + *s3 * beta;
}


template<typename ScalarT1,
         typename ScalarT2, typename NumericT1,
         typename ScalarT3, typename NumericT2>
typename cuarma::enable_if< cuarma::is_scalar<ScalarT1>::value
                              && cuarma::is_scalar<ScalarT2>::value
                              && cuarma::is_scalar<ScalarT3>::value
                              && cuarma::is_any_scalar<NumericT1>::value
                              && cuarma::is_any_scalar<NumericT2>::value
                            >::type
asbs(ScalarT1 & s1,
     ScalarT2 const & s2, NumericT1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
     ScalarT3 const & s3, NumericT2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef typename cuarma::result_of::cpu_value_type<ScalarT1>::type        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  unsigned int options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  value_type temporary_alpha = 0;
  if (cuarma::is_cpu_scalar<NumericT1>::value)
    temporary_alpha = alpha;

  value_type temporary_beta = 0;
  if (cuarma::is_cpu_scalar<NumericT2>::value)
    temporary_beta = beta;

  asbs_kernel<<<1, 1>>>(cuarma::cuda_arg(s1),
                        cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                        options_alpha,
                        cuarma::cuda_arg(s2),
                        cuarma::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                        options_beta,
                        cuarma::cuda_arg(s3) );
  CUARMA_CUDA_LAST_ERROR_CHECK("asbs_kernel");
}

//////////////////// asbs_s ////////////////////

// alpha and beta on GPU
template<typename NumericT>
__global__ void asbs_s_kernel(NumericT * s1,
                              const NumericT * fac2, unsigned int options2, const NumericT * s2,
                              const NumericT * fac3, unsigned int options3, const NumericT * s3)
{
    NumericT alpha = *fac2;
    if (options2 & (1 << 0))
      alpha = -alpha;
    if (options2 & (1 << 1))
      alpha = NumericT(1) / alpha;

    NumericT beta = *fac3;
    if (options3 & (1 << 0))
      beta = -beta;
    if (options3 & (1 << 1))
      beta = NumericT(1) / beta;

    *s1 += *s2 * alpha + *s3 * beta;
}

// alpha on CPU, beta on GPU
template<typename NumericT>
__global__ void asbs_s_kernel(NumericT * s1,
                              NumericT         fac2, unsigned int options2, const NumericT * s2,
                              NumericT const * fac3, unsigned int options3, const NumericT * s3)
{
    NumericT alpha = fac2;
    if (options2 & (1 << 0))
      alpha = -alpha;
    if (options2 & (1 << 1))
      alpha = NumericT(1) / alpha;

    NumericT beta = *fac3;
    if (options3 & (1 << 0))
      beta = -beta;
    if (options3 & (1 << 1))
      beta = NumericT(1) / beta;

    *s1 += *s2 * alpha + *s3 * beta;
}

// alpha on GPU, beta on CPU
template<typename NumericT>
__global__ void asbs_s_kernel(NumericT * s1,
                              NumericT const * fac2, unsigned int options2, const NumericT * s2,
                              NumericT         fac3, unsigned int options3, const NumericT * s3)
{
    NumericT alpha = *fac2;
    if (options2 & (1 << 0))
      alpha = -alpha;
    if (options2 & (1 << 1))
      alpha = NumericT(1) / alpha;

    NumericT beta = fac3;
    if (options3 & (1 << 0))
      beta = -beta;
    if (options3 & (1 << 1))
      beta = NumericT(1) / beta;

    *s1 += *s2 * alpha + *s3 * beta;
}

// alpha and beta on CPU
template<typename NumericT>
__global__ void asbs_s_kernel(NumericT * s1,
                              NumericT fac2, unsigned int options2, const NumericT * s2,
                              NumericT fac3, unsigned int options3, const NumericT * s3)
{
    NumericT alpha = fac2;
    if (options2 & (1 << 0))
      alpha = -alpha;
    if (options2 & (1 << 1))
      alpha = NumericT(1) / alpha;

    NumericT beta = fac3;
    if (options3 & (1 << 0))
      beta = -beta;
    if (options3 & (1 << 1))
      beta = NumericT(1) / beta;

    *s1 += *s2 * alpha + *s3 * beta;
}


template<typename ScalarT1,
         typename ScalarT2, typename NumericT1,
         typename ScalarT3, typename NumericT2>
typename cuarma::enable_if< cuarma::is_scalar<ScalarT1>::value
                              && cuarma::is_scalar<ScalarT2>::value
                              && cuarma::is_scalar<ScalarT3>::value
                              && cuarma::is_any_scalar<NumericT1>::value
                              && cuarma::is_any_scalar<NumericT2>::value
                            >::type
asbs_s(ScalarT1 & s1,
       ScalarT2 const & s2, NumericT1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
       ScalarT3 const & s3, NumericT2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef typename cuarma::result_of::cpu_value_type<ScalarT1>::type        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  unsigned int options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  value_type temporary_alpha = 0;
  if (cuarma::is_cpu_scalar<NumericT1>::value)
    temporary_alpha = alpha;

  value_type temporary_beta = 0;
  if (cuarma::is_cpu_scalar<NumericT2>::value)
    temporary_beta = beta;

  std::cout << "Launching asbs_s_kernel..." << std::endl;
  asbs_s_kernel<<<1, 1>>>(cuarma::cuda_arg(s1),
                          cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                          options_alpha,
                          cuarma::cuda_arg(s2),
                          cuarma::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                          options_beta,
                          cuarma::cuda_arg(s3) );
  CUARMA_CUDA_LAST_ERROR_CHECK("asbs_s_kernel");
}

///////////////// swap //////////////////

template<typename NumericT>
__global__ void scalar_swap_kernel(NumericT * s1, NumericT * s2)
{
  NumericT tmp = *s2;
  *s2 = *s1;
  *s1 = tmp;
}

/** @brief Swaps the contents of two scalars, data is copied
*
* @param s1   The first scalar
* @param s2   The second scalar
*/
template<typename ScalarT1, typename ScalarT2>
typename cuarma::enable_if<    cuarma::is_scalar<ScalarT1>::value
                              && cuarma::is_scalar<ScalarT2>::value
                            >::type
swap(ScalarT1 & s1, ScalarT2 & s2)
{
  typedef typename cuarma::result_of::cpu_value_type<ScalarT1>::type        value_type;

  scalar_swap_kernel<<<1, 1>>>(cuarma::cuda_arg(s1), cuarma::cuda_arg(s2));
}



} //namespace single_threaded
} //namespace blas
} //namespace cuarma
