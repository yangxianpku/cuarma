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

/** @file cuarma/traits/start.hpp
 *  @encoding:UTF-8 文档编码
    @brief Extracts the underlying start index handle from a vector, a matrix, an expression etc.
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

//
// start: Mostly for vectors
//

// Default: Try to get the start index from the .start() member function
template<typename T>
typename result_of::size_type<T>::type
start(T const & obj)
{
  return obj.start();
}

//cuarma vector leads to start index 0:
template<typename ScalarType, unsigned int AlignmentV>
typename result_of::size_type<cuarma::vector<ScalarType, AlignmentV> >::type
start(cuarma::vector<ScalarType, AlignmentV> const &)
{
  return 0;
}


//
// start1: Row start index
//

// Default: Try to get the start index from the .start1() member function
template<typename T>
typename result_of::size_type<T>::type
start1(T const & obj)
{
  return obj.start1();
}

//cuarma matrix leads to start index 0:
template<typename ScalarType, typename F, unsigned int AlignmentV>
typename result_of::size_type<cuarma::matrix<ScalarType, F, AlignmentV> >::type
start1(cuarma::matrix<ScalarType, F, AlignmentV> const &)
{
  return 0;
}


//
// start2: Column start index
//
template<typename T>
typename result_of::size_type<T>::type
start2(T const & obj)
{
  return obj.start2();
}

//cuarma matrix leads to start index 0:
template<typename ScalarType, typename F, unsigned int AlignmentV>
typename result_of::size_type<cuarma::matrix<ScalarType, F, AlignmentV> >::type
start2(cuarma::matrix<ScalarType, F, AlignmentV> const &)
{
  return 0;
}


} //namespace traits
} //namespace cuarma