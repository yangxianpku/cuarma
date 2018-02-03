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

/** @file cuarma/traits/stride.hpp
 *  @encoding:UTF-8 文档编码
    @brief Determines row and column increments for matrices and matrix proxies
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

//
// inc: Increment for vectors. Defaults to 1
//
template<typename T>
typename result_of::size_type< cuarma::vector_base<T> >::type
stride(cuarma::vector_base<T> const & s) { return s.stride(); }

//
// inc1: Row increment for matrices. Defaults to 1
//
//template<typename MatrixType>
//typename result_of::size_type<MatrixType>::type
//stride1(MatrixType const &) { return 1; }
template<typename NumericT>
typename result_of::size_type< matrix_base<NumericT> >::type
stride1(matrix_base<NumericT> const & s) { return s.stride1(); }

//
// inc2: Column increment for matrices. Defaults to 1
//
//template<typename MatrixType>
//typename result_of::size_type<MatrixType>::type
//stride2(MatrixType const &) { return 1; }
template<typename NumericT>
typename result_of::size_type< matrix_base<NumericT> >::type
stride2(matrix_base<NumericT> const & s) { return s.stride2(); }


} //namespace traits
} //namespace cuarma