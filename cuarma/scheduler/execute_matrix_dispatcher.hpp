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


/** @file cuarma/scheduler/execute_matrix_dispatcher.hpp
 *  @encoding:UTF-8 文档编码
    @brief Provides wrappers for am(), ambm(), ambm_m(), etc. in cuarma/blas/matrix_operations.hpp such that scheduler logic is not cluttered with numeric type decutions
*/

#include <assert.h>

#include "cuarma/forwards.h"
#include "cuarma/scheduler/forwards.h"
#include "cuarma/scheduler/execute_util.hpp"
#include "cuarma/blas/matrix_operations.hpp"

namespace cuarma
{
namespace scheduler
{
namespace detail
{

/** @brief Wrapper for cuarma::blas::av(), taking care of the argument unwrapping */
template<typename ScalarType1>
void am(lhs_rhs_element & mat1,
        lhs_rhs_element const & mat2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(   mat1.type_family == MATRIX_TYPE_FAMILY && mat2.type_family == MATRIX_TYPE_FAMILY
            && bool("Arguments are not matrix types!"));

  assert(mat1.numeric_type == mat2.numeric_type && bool("Matrices do not have the same scalar type"));

  if (mat1.subtype == DENSE_MATRIX_TYPE)
  {
    switch (mat1.numeric_type)
    {
    case FLOAT_TYPE:
      cuarma::blas::am(*mat1.matrix_float,
                           *mat2.matrix_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
      break;
    case DOUBLE_TYPE:
      cuarma::blas::am(*mat1.matrix_double,
                           *mat2.matrix_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
      break;

    default:
      throw statement_not_supported_exception("Invalid arguments in scheduler when calling am()");
    }
  }
  else
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling am()");
}

/** @brief Wrapper for cuarma::blas::avbv(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void ambm(lhs_rhs_element & mat1,
          lhs_rhs_element const & mat2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          lhs_rhs_element const & mat3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   mat1.type_family == MATRIX_TYPE_FAMILY
            && mat2.type_family == MATRIX_TYPE_FAMILY
            && mat3.type_family == MATRIX_TYPE_FAMILY
            && bool("Arguments are not matrix types!"));

  assert(   (mat1.subtype == mat2.subtype)
            && (mat2.subtype == mat3.subtype)
            && bool("Matrices do not have the same layout"));

  assert(   (mat1.numeric_type == mat2.numeric_type)
            && (mat2.numeric_type == mat3.numeric_type)
            && bool("Matrices do not have the same scalar type"));

  if (mat1.subtype == DENSE_MATRIX_TYPE)
  {
    switch (mat1.numeric_type)
    {
    case FLOAT_TYPE:
      cuarma::blas::ambm(*mat1.matrix_float,
                             *mat2.matrix_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                             *mat3.matrix_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
      break;
    case DOUBLE_TYPE:
      cuarma::blas::ambm(*mat1.matrix_double,
                             *mat2.matrix_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                             *mat3.matrix_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
      break;
    default:
      throw statement_not_supported_exception("Invalid arguments in scheduler when calling ambm()");
    }
  }
  else
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling ambm()");
}

/** @brief Wrapper for cuarma::blas::avbv_v(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void ambm_m(lhs_rhs_element & mat1,
            lhs_rhs_element const & mat2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            lhs_rhs_element const & mat3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   mat1.type_family == MATRIX_TYPE_FAMILY
            && mat2.type_family == MATRIX_TYPE_FAMILY
            && mat3.type_family == MATRIX_TYPE_FAMILY
            && bool("Arguments are not matrix types!"));

  assert(   (mat1.subtype == mat2.subtype)
            && (mat2.subtype == mat3.subtype)
            && bool("Matrices do not have the same layout"));

  assert(   (mat1.numeric_type == mat2.numeric_type)
            && (mat2.numeric_type == mat3.numeric_type)
            && bool("Matrices do not have the same scalar type"));

  if (mat1.subtype == DENSE_MATRIX_TYPE)
  {
    switch (mat1.numeric_type)
    {
    case FLOAT_TYPE:
      cuarma::blas::ambm_m(*mat1.matrix_float,
                               *mat2.matrix_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                               *mat3.matrix_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
      break;
    case DOUBLE_TYPE:
      cuarma::blas::ambm_m(*mat1.matrix_double,
                               *mat2.matrix_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                               *mat3.matrix_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
      break;
    default:
      throw statement_not_supported_exception("Invalid arguments in scheduler when calling ambm_m()");
    }
  }
  else
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling ambm_m()");
}

/** @brief Scheduler unwrapper for A =/+=/-= trans(B) */
inline void assign_trans(lhs_rhs_element const & A,
                         lhs_rhs_element const & B)
{
  assert(   A.type_family == MATRIX_TYPE_FAMILY && B.type_family == MATRIX_TYPE_FAMILY
            && bool("Arguments are not matrix types!"));

  assert(A.numeric_type == B.numeric_type && bool("Matrices do not have the same scalar type"));

  if (A.subtype == DENSE_MATRIX_TYPE)
  {
    switch (A.numeric_type)
    {
    case FLOAT_TYPE:
      *A.matrix_float = cuarma::trans(*B.matrix_float);
      break;
    case DOUBLE_TYPE:
      *A.matrix_double = cuarma::trans(*B.matrix_double);
      break;
    default:
      throw statement_not_supported_exception("Invalid arguments in scheduler when calling assign_trans()");
    }
  }
  else
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling assign_trans()");
}

} // namespace detail
} // namespace scheduler
} // namespace cuarma