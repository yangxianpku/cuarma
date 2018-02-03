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


/** @file cuarma/scheduler/execute_scalar_dispatcher.hpp
 *  @encoding:UTF-8 文档编码
    @brief Provides wrappers for as(), asbs(), asbs_s(), etc. in cuarma/blas/scalar_operations.hpp such that scheduler logic is not cluttered with numeric type decutions
*/

#include <assert.h>
#include "cuarma/forwards.h"
#include "cuarma/scheduler/forwards.h"
#include "cuarma/scheduler/execute_util.hpp"
#include "cuarma/blas/scalar_operations.hpp"

namespace cuarma
{
namespace scheduler
{
namespace detail
{

/** @brief Wrapper for cuarma::blas::av(), taking care of the argument unwrapping */
template<typename ScalarType1>
void as(lhs_rhs_element & s1,
        lhs_rhs_element const & s2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(   s1.type_family == SCALAR_TYPE_FAMILY && (s1.subtype == HOST_SCALAR_TYPE || s1.subtype == DEVICE_SCALAR_TYPE)
            && s2.type_family == SCALAR_TYPE_FAMILY && (s2.subtype == HOST_SCALAR_TYPE || s2.subtype == DEVICE_SCALAR_TYPE)
            && bool("Arguments are not vector types!"));

  switch (s1.numeric_type)
  {
  case FLOAT_TYPE:
    assert(s2.numeric_type == FLOAT_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::av(*s1.vector_float,
                         *s2.vector_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  case DOUBLE_TYPE:
    assert(s2.numeric_type == DOUBLE_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::av(*s1.vector_double,
                         *s2.vector_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  default:
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling av()");
  }
}

/** @brief Wrapper for cuarma::blas::avbv(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void asbs(lhs_rhs_element & s1,
          lhs_rhs_element const & s2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          lhs_rhs_element const & s3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   s1.type_family == SCALAR_TYPE_FAMILY && (s1.subtype == HOST_SCALAR_TYPE || s1.subtype == DEVICE_SCALAR_TYPE)
            && s2.type_family == SCALAR_TYPE_FAMILY && (s2.subtype == HOST_SCALAR_TYPE || s2.subtype == DEVICE_SCALAR_TYPE)
            && s3.type_family == SCALAR_TYPE_FAMILY && (s3.subtype == HOST_SCALAR_TYPE || s3.subtype == DEVICE_SCALAR_TYPE)
            && bool("Arguments are not vector types!"));

  switch (s1.numeric_type)
  {
  case FLOAT_TYPE:
    assert(s2.numeric_type == FLOAT_TYPE && s3.numeric_type == FLOAT_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::avbv(*s1.vector_float,
                           *s2.vector_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                           *s3.vector_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case DOUBLE_TYPE:
    assert(s2.numeric_type == DOUBLE_TYPE && s3.numeric_type == DOUBLE_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::avbv(*s1.vector_double,
                           *s2.vector_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                           *s3.vector_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  default:
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling avbv()");
  }
}

/** @brief Wrapper for cuarma::blas::avbv_v(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void asbs_s(lhs_rhs_element & s1,
            lhs_rhs_element const & s2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            lhs_rhs_element const & s3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   s1.type_family == SCALAR_TYPE_FAMILY && (s1.subtype == HOST_SCALAR_TYPE || s1.subtype == DEVICE_SCALAR_TYPE)
            && s2.type_family == SCALAR_TYPE_FAMILY && (s2.subtype == HOST_SCALAR_TYPE || s2.subtype == DEVICE_SCALAR_TYPE)
            && s3.type_family == SCALAR_TYPE_FAMILY && (s3.subtype == HOST_SCALAR_TYPE || s3.subtype == DEVICE_SCALAR_TYPE)
            && bool("Arguments are not vector types!"));

  switch (s1.numeric_type)
  {
  case FLOAT_TYPE:
    assert(s2.numeric_type == FLOAT_TYPE && s3.numeric_type == FLOAT_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::avbv_v(*s1.vector_float,
                             *s2.vector_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                             *s3.vector_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case DOUBLE_TYPE:
    assert(s2.numeric_type == DOUBLE_TYPE && s3.numeric_type == DOUBLE_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::avbv_v(*s1.vector_double,
                             *s2.vector_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                             *s3.vector_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  default:
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling avbv_v()");
  }
}

} // namespace detail
} // namespace scheduler
} // namespace cuarma