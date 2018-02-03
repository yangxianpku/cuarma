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

/** @file cuarma/blas/host_based/scalar_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of scalar operations using a plain single-threaded execution on CPU
*/

#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/start.hpp"
#include "cuarma/traits/stride.hpp"
#include "cuarma/blas/host_based/common.hpp"

namespace cuarma
{
namespace blas
{
namespace host_based
{
template<typename ScalarT1,
         typename ScalarT2, typename FactorT>
typename cuarma::enable_if< cuarma::is_scalar<ScalarT1>::value
                              && cuarma::is_scalar<ScalarT2>::value
                              && cuarma::is_any_scalar<FactorT>::value
                            >::type
as(ScalarT1       & s1,
   ScalarT2 const & s2, FactorT const & alpha, arma_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha)
{
  typedef typename cuarma::result_of::cpu_value_type<ScalarT1>::type        value_type;

  value_type       * data_s1 = detail::extract_raw_pointer<value_type>(s1);
  value_type const * data_s2 = detail::extract_raw_pointer<value_type>(s2);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;
  if (reciprocal_alpha)
    data_alpha = static_cast<value_type>(1) / data_alpha;

  *data_s1 = *data_s2 * data_alpha;
}


template<typename ScalarT1,
         typename ScalarT2, typename FactorT2,
         typename ScalarT3, typename FactorT3>
typename cuarma::enable_if< cuarma::is_scalar<ScalarT1>::value
                              && cuarma::is_scalar<ScalarT2>::value
                              && cuarma::is_scalar<ScalarT3>::value
                              && cuarma::is_any_scalar<FactorT2>::value
                              && cuarma::is_any_scalar<FactorT3>::value
                            >::type
asbs(ScalarT1       & s1,
     ScalarT2 const & s2, FactorT2 const & alpha, arma_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
     ScalarT3 const & s3, FactorT3 const & beta,  arma_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef typename cuarma::result_of::cpu_value_type<ScalarT1>::type        value_type;

  value_type       * data_s1 = detail::extract_raw_pointer<value_type>(s1);
  value_type const * data_s2 = detail::extract_raw_pointer<value_type>(s2);
  value_type const * data_s3 = detail::extract_raw_pointer<value_type>(s3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;
  if (reciprocal_alpha)
    data_alpha = static_cast<value_type>(1) / data_alpha;

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = -data_beta;
  if (reciprocal_beta)
    data_beta = static_cast<value_type>(1) / data_beta;

  *data_s1 = *data_s2 * data_alpha + *data_s3 * data_beta;
}


template<typename ScalarT1,
         typename ScalarT2, typename FactorT2,
         typename ScalarT3, typename FactorT3>
typename cuarma::enable_if< cuarma::is_scalar<ScalarT1>::value
                              && cuarma::is_scalar<ScalarT2>::value
                              && cuarma::is_scalar<ScalarT3>::value
                              && cuarma::is_any_scalar<FactorT2>::value
                              && cuarma::is_any_scalar<FactorT3>::value
                            >::type
asbs_s(ScalarT1       & s1,
       ScalarT2 const & s2, FactorT2 const & alpha, arma_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
       ScalarT3 const & s3, FactorT3 const & beta,  arma_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef typename cuarma::result_of::cpu_value_type<ScalarT1>::type        value_type;

  value_type       * data_s1 = detail::extract_raw_pointer<value_type>(s1);
  value_type const * data_s2 = detail::extract_raw_pointer<value_type>(s2);
  value_type const * data_s3 = detail::extract_raw_pointer<value_type>(s3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;
  if (reciprocal_alpha)
    data_alpha = static_cast<value_type>(1) / data_alpha;

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = -data_beta;
  if (reciprocal_beta)
    data_beta = static_cast<value_type>(1) / data_beta;

  *data_s1 += *data_s2 * data_alpha + *data_s3 * data_beta;
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

  value_type * data_s1 = detail::extract_raw_pointer<value_type>(s1);
  value_type * data_s2 = detail::extract_raw_pointer<value_type>(s2);

  value_type temp = *data_s2;
  *data_s2 = *data_s1;
  *data_s1 = temp;
}



} //namespace host_based
} //namespace blas
} //namespace cuarma