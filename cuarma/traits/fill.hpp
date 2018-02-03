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

/** @file cuarma/traits/fill.hpp
 *  @encoding:UTF-8 文档编码
    @brief Generic fill functionality for different matrix types
*/

#include <string>
#include <fstream>
#include <sstream>
#include "cuarma/forwards.h"
#include "cuarma/meta/result_of.hpp"
#include <vector>
#include <map>

namespace cuarma
{
namespace traits
{

/** @brief Generic filler routine for setting an entry of a matrix to a particular value */
template<typename MatrixType, typename NumericT>
void fill(MatrixType & matrix, arma_size_t row_index, arma_size_t col_index, NumericT value)
{
  matrix(row_index, col_index) = value;
}



} //namespace traits
} //namespace cuarma