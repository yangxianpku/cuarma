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


/** @file cuarma/scheduler/execute_vector_dispatcher.hpp
 *  @encoding:UTF-8 文档编码
    @brief Provides wrappers for av(), avbv(), avbv_v(), etc. in cuarma/blas/vector_operations.hpp such that scheduler logic is not cluttered with numeric type decutions
*/

#include <assert.h>
#include "cuarma/forwards.h"
#include "cuarma/scheduler/forwards.h"
#include "cuarma/scheduler/execute_util.hpp"
#include "cuarma/blas/vector_operations.hpp"

namespace cuarma
{
namespace scheduler
{
namespace detail
{

/** @brief Wrapper for cuarma::blas::av(), taking care of the argument unwrapping */
template<typename ScalarType1>
void av(lhs_rhs_element & vec1,
        lhs_rhs_element const & vec2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(   vec1.type_family == VECTOR_TYPE_FAMILY && vec1.subtype == DENSE_VECTOR_TYPE
            && vec2.type_family == VECTOR_TYPE_FAMILY && vec2.subtype == DENSE_VECTOR_TYPE
            && bool("Arguments are not vector types!"));

  switch (vec1.numeric_type)
  {
  case FLOAT_TYPE:
    assert(vec2.numeric_type == FLOAT_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::av(*vec1.vector_float,
                         *vec2.vector_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  case DOUBLE_TYPE:
    assert(vec2.numeric_type == DOUBLE_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::av(*vec1.vector_double,
                         *vec2.vector_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  default:
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling av()");
  }
}

/** @brief Wrapper for cuarma::blas::avbv(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void avbv(lhs_rhs_element & vec1,
          lhs_rhs_element const & vec2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          lhs_rhs_element const & vec3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   vec1.type_family == VECTOR_TYPE_FAMILY && vec1.subtype == DENSE_VECTOR_TYPE
            && vec2.type_family == VECTOR_TYPE_FAMILY && vec2.subtype == DENSE_VECTOR_TYPE
            && vec3.type_family == VECTOR_TYPE_FAMILY && vec3.subtype == DENSE_VECTOR_TYPE
            && bool("Arguments are not vector types!"));

  switch (vec1.numeric_type)
  {
  case FLOAT_TYPE:
    assert(vec2.numeric_type == FLOAT_TYPE && vec3.numeric_type == FLOAT_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::avbv(*vec1.vector_float,
                           *vec2.vector_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                           *vec3.vector_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case DOUBLE_TYPE:
    assert(vec2.numeric_type == DOUBLE_TYPE && vec3.numeric_type == DOUBLE_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::avbv(*vec1.vector_double,
                           *vec2.vector_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                           *vec3.vector_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  default:
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling avbv()");
  }
}

/** @brief Wrapper for cuarma::blas::avbv_v(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void avbv_v(lhs_rhs_element & vec1,
            lhs_rhs_element const & vec2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            lhs_rhs_element const & vec3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   vec1.type_family == VECTOR_TYPE_FAMILY && vec1.subtype == DENSE_VECTOR_TYPE
            && vec2.type_family == VECTOR_TYPE_FAMILY && vec2.subtype == DENSE_VECTOR_TYPE
            && vec3.type_family == VECTOR_TYPE_FAMILY && vec3.subtype == DENSE_VECTOR_TYPE
            && bool("Arguments are not vector types!"));

  switch (vec1.numeric_type)
  {
  case FLOAT_TYPE:
    assert(vec2.numeric_type == FLOAT_TYPE && vec3.numeric_type == FLOAT_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::avbv_v(*vec1.vector_float,
                             *vec2.vector_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                             *vec3.vector_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case DOUBLE_TYPE:
    assert(vec2.numeric_type == DOUBLE_TYPE && vec3.numeric_type == DOUBLE_TYPE && bool("Vectors do not have the same scalar type"));
    cuarma::blas::avbv_v(*vec1.vector_double,
                             *vec2.vector_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                             *vec3.vector_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  default:
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling avbv_v()");
  }
}


/** @brief Dispatcher interface for computing s = norm_1(x) */
inline void norm_impl(lhs_rhs_element const & x,
                      lhs_rhs_element const & s,
                      operation_node_type op_type)
{
  assert( x.type_family == VECTOR_TYPE_FAMILY && x.subtype == DENSE_VECTOR_TYPE && bool("Argument is not a dense vector type!"));
  assert( s.type_family == SCALAR_TYPE_FAMILY && s.subtype == DEVICE_SCALAR_TYPE && bool("Argument is not a scalar type!"));

  switch (x.numeric_type)
  {
  case FLOAT_TYPE:
    assert(s.numeric_type == FLOAT_TYPE && bool("Vector and scalar do not have the same numeric type"));
    if (op_type == OPERATION_UNARY_NORM_1_TYPE)
      cuarma::blas::norm_1_impl(*x.vector_float, *s.scalar_float);
    else if (op_type == OPERATION_UNARY_NORM_2_TYPE)
      cuarma::blas::norm_2_impl(*x.vector_float, *s.scalar_float);
    else if (op_type == OPERATION_UNARY_NORM_INF_TYPE)
      cuarma::blas::norm_inf_impl(*x.vector_float, *s.scalar_float);
    else if (op_type == OPERATION_UNARY_MAX_TYPE)
      cuarma::blas::max_impl(*x.vector_float, *s.scalar_float);
    else if (op_type == OPERATION_UNARY_MIN_TYPE)
      cuarma::blas::min_impl(*x.vector_float, *s.scalar_float);
    else
      throw statement_not_supported_exception("Invalid norm type in scheduler::detail::norm_impl()");
    break;
  case DOUBLE_TYPE:
    if (op_type == OPERATION_UNARY_NORM_1_TYPE)
      cuarma::blas::norm_1_impl(*x.vector_double, *s.scalar_double);
    else if (op_type == OPERATION_UNARY_NORM_2_TYPE)
      cuarma::blas::norm_2_impl(*x.vector_double, *s.scalar_double);
    else if (op_type == OPERATION_UNARY_NORM_INF_TYPE)
      cuarma::blas::norm_inf_impl(*x.vector_double, *s.scalar_double);
    else if (op_type == OPERATION_UNARY_MAX_TYPE)
      cuarma::blas::max_impl(*x.vector_double, *s.scalar_double);
    else if (op_type == OPERATION_UNARY_MIN_TYPE)
      cuarma::blas::min_impl(*x.vector_double, *s.scalar_double);
    else
      throw statement_not_supported_exception("Invalid norm type in scheduler::detail::norm_impl()");
    break;
  default:
    throw statement_not_supported_exception("Invalid numeric type in scheduler when calling norm_impl()");
  }
}

/** @brief Dispatcher interface for computing s = inner_prod(x, y) */
inline void inner_prod_impl(lhs_rhs_element const & x,
                            lhs_rhs_element const & y,
                            lhs_rhs_element const & s)
{
  assert( x.type_family == VECTOR_TYPE_FAMILY && x.subtype == DENSE_VECTOR_TYPE && bool("Argument is not a dense vector type!"));
  assert( y.type_family == VECTOR_TYPE_FAMILY && y.subtype == DENSE_VECTOR_TYPE && bool("Argument is not a dense vector type!"));
  assert( s.type_family == SCALAR_TYPE_FAMILY && s.subtype == DEVICE_SCALAR_TYPE && bool("Argument is not a scalar type!"));

  switch (x.numeric_type)
  {
  case FLOAT_TYPE:
    assert(y.numeric_type == FLOAT_TYPE && s.numeric_type == FLOAT_TYPE && bool("Vector and scalar do not have the same numeric type"));
    cuarma::blas::inner_prod_impl(*x.vector_float, *y.vector_float, *s.scalar_float);
    break;
  case DOUBLE_TYPE:
    assert(y.numeric_type == DOUBLE_TYPE && s.numeric_type == DOUBLE_TYPE && bool("Vector and scalar do not have the same numeric type"));
    cuarma::blas::inner_prod_impl(*x.vector_double, *y.vector_double, *s.scalar_double);
    break;
  default:
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling av()");
  }
}

} // namespace detail
} // namespace scheduler
} // namespace cuarma