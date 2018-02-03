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


/** @file cuarma/scheduler/execute_generic_dispatcher.hpp
 *  @encoding:UTF-8 文档编码
    @brief Provides unified wrappers for the common routines {as(), asbs(), asbs_s()}, {av(), avbv(), avbv_v()}, and {am(), ambm(), ambm_m()} such that scheduler logic is not cluttered with numeric type decutions
*/

#include <assert.h>
#include "cuarma/forwards.h"
#include "cuarma/scheduler/forwards.h"
#include "cuarma/scheduler/execute_util.hpp"
#include "cuarma/scheduler/execute_scalar_dispatcher.hpp"
#include "cuarma/scheduler/execute_vector_dispatcher.hpp"
#include "cuarma/scheduler/execute_matrix_dispatcher.hpp"

namespace cuarma
{
namespace scheduler
{
namespace detail
{

/** @brief Wrapper for cuarma::blas::av(), taking care of the argument unwrapping */
template<typename ScalarType1>
void ax(lhs_rhs_element & x1,
        lhs_rhs_element const & x2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(x1.type_family == x2.type_family && bool("Arguments are not of the same type family!"));

  switch (x1.type_family)
  {
  case SCALAR_TYPE_FAMILY:
    detail::as(x1, x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  case VECTOR_TYPE_FAMILY:
    detail::av(x1, x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  case MATRIX_TYPE_FAMILY:
    detail::am(x1, x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  default:
    throw statement_not_supported_exception("Invalid argument in scheduler ax() while dispatching.");
  }
}

/** @brief Wrapper for cuarma::blas::avbv(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void axbx(lhs_rhs_element & x1,
          lhs_rhs_element const & x2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          lhs_rhs_element const & x3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   x1.type_family == x2.type_family
            && x2.type_family == x3.type_family
            && bool("Arguments are not of the same type family!"));

  switch (x1.type_family)
  {
  case SCALAR_TYPE_FAMILY:
    detail::asbs(x1,
                 x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                 x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case VECTOR_TYPE_FAMILY:
    detail::avbv(x1,
                 x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                 x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case MATRIX_TYPE_FAMILY:
    detail::ambm(x1,
                 x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                 x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  default:
    throw statement_not_supported_exception("Invalid argument in scheduler ax() while dispatching.");
  }
}

/** @brief Wrapper for cuarma::blas::avbv_v(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void axbx_x(lhs_rhs_element & x1,
            lhs_rhs_element const & x2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            lhs_rhs_element const & x3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   x1.type_family == x2.type_family
            && x2.type_family == x3.type_family
            && bool("Arguments are not of the same type family!"));

  switch (x1.type_family)
  {
  case SCALAR_TYPE_FAMILY:
    detail::asbs_s(x1,
                   x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                   x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case VECTOR_TYPE_FAMILY:
    detail::avbv_v(x1,
                   x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                   x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case MATRIX_TYPE_FAMILY:
    detail::ambm_m(x1,
                   x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                   x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  default:
    throw statement_not_supported_exception("Invalid argument in scheduler ax() while dispatching.");
  }
}

} // namespace detail
} // namespace scheduler
} // namespace cuarma