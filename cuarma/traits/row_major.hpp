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

/** @file cuarma/traits/row_major.hpp
 *  @encoding:UTF-8 文档编码
    @brief Determines whether a given expression has a row-major matrix layout
*/

#include <string>
#include <fstream>
#include <sstream>
#include "cuarma/forwards.h"
#include "cuarma/meta/result_of.hpp"

namespace cuarma
{
namespace traits
{

template<typename T>
bool row_major(T const &) { return true; } //default implementation: If there is no underlying matrix type, we take the result to be row-major

template<typename NumericT>
bool row_major(matrix_base<NumericT> const & A) { return A.row_major(); }

template<typename LHS, typename RHS, typename OP>
bool row_major(matrix_expression<LHS, RHS, OP> const & proxy) { return cuarma::traits::row_major(proxy.lhs()); }

} //namespace traits
} //namespace cuarma