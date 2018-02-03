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

/** @file cuarma/blas/detail/op_applier.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief Defines the action of certain unary and binary operators and its arguments (for host execution).
*/

#include "cuarma/forwards.h"
#include <cmath>

namespace cuarma
{
namespace blas
{
namespace detail
{

/** @brief Worker class for decomposing expression templates.
  *
  * @tparam A    Type to which is assigned to
  * @tparam OP   One out of {op_assign, op_inplace_add, op_inplace_sub}
  @ @tparam T    Right hand side of the assignment
*/
template<typename OpT>
struct op_applier
{
  typedef typename OpT::ERROR_UNKNOWN_OP_TAG_PROVIDED    error_type;
};

/** \cond */
template<>
struct op_applier<op_element_binary<op_prod> >
{
  template<typename T>
  static void apply(T & result, T const & x, T const & y) { result = x * y; }
};

template<>
struct op_applier<op_element_binary<op_div> >
{
  template<typename T>
  static void apply(T & result, T const & x, T const & y) { result = x / y; }
};

template<>
struct op_applier<op_element_binary<op_pow> >
{
  template<typename T>
  static void apply(T & result, T const & x, T const & y) { result = std::pow(x, y); }
};

#define CUARMA_MAKE_UNARY_OP_APPLIER(funcname)  \
template<> \
struct op_applier<op_element_unary<op_##funcname> > \
{ \
  template<typename T> \
  static void apply(T & result, T const & x) { using namespace std; result = funcname(x); } \
}

CUARMA_MAKE_UNARY_OP_APPLIER(abs);
CUARMA_MAKE_UNARY_OP_APPLIER(acos);
CUARMA_MAKE_UNARY_OP_APPLIER(asin);
CUARMA_MAKE_UNARY_OP_APPLIER(atan);
CUARMA_MAKE_UNARY_OP_APPLIER(ceil);
CUARMA_MAKE_UNARY_OP_APPLIER(cos);
CUARMA_MAKE_UNARY_OP_APPLIER(cosh);
CUARMA_MAKE_UNARY_OP_APPLIER(exp);
CUARMA_MAKE_UNARY_OP_APPLIER(fabs);
CUARMA_MAKE_UNARY_OP_APPLIER(floor);
CUARMA_MAKE_UNARY_OP_APPLIER(log);
CUARMA_MAKE_UNARY_OP_APPLIER(log10);
CUARMA_MAKE_UNARY_OP_APPLIER(sin);
CUARMA_MAKE_UNARY_OP_APPLIER(sinh);
CUARMA_MAKE_UNARY_OP_APPLIER(sqrt);
CUARMA_MAKE_UNARY_OP_APPLIER(tan);
CUARMA_MAKE_UNARY_OP_APPLIER(tanh);

#undef CUARMA_MAKE_UNARY_OP_APPLIER
/** \endcond */

}
}
}