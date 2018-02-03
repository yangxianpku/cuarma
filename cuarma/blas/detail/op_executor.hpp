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
/** @file cuarma/blas/detail/op_executor.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief Defines the worker class for decomposing an expression tree into small chunks, which can be processed by the predefined operations in cuarma.
*/

#include "cuarma/forwards.h"

namespace cuarma
{
namespace blas
{
namespace detail
{

template<typename NumericT, typename B>
bool op_aliasing(vector_base<NumericT> const & /*lhs*/, B const & /*b*/)
{
  return false;
}

template<typename NumericT>
bool op_aliasing(vector_base<NumericT> const & lhs, vector_base<NumericT> const & b)
{
  return lhs.handle() == b.handle();
}

template<typename NumericT, typename LhsT, typename RhsT, typename OpT>
bool op_aliasing(vector_base<NumericT> const & lhs, vector_expression<const LhsT, const RhsT, OpT> const & rhs)
{
  return op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs());
}


template<typename NumericT, typename B>
bool op_aliasing(matrix_base<NumericT> const & /*lhs*/, B const & /*b*/)
{
  return false;
}

template<typename NumericT>
bool op_aliasing(matrix_base<NumericT> const & lhs, matrix_base<NumericT> const & b)
{
  return lhs.handle() == b.handle();
}

template<typename NumericT, typename LhsT, typename RhsT, typename OpT>
bool op_aliasing(matrix_base<NumericT> const & lhs, matrix_expression<const LhsT, const RhsT, OpT> const & rhs)
{
  return op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs());
}


/** @brief Worker class for decomposing expression templates.
  *
  * @tparam A    Type to which is assigned to
  * @tparam OP   One out of {op_assign, op_inplace_add, op_inplace_sub}
  @ @tparam T    Right hand side of the assignment
*/
template<typename A, typename OP, typename T>
struct op_executor {};

}
}
}
