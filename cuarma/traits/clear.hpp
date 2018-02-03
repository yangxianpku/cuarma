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

/** @file cuarma/traits/clear.hpp
 *  @encoding:UTF-8 文档编码
    @brief Generic clear functionality for different vector and matrix types
*/

#include <string>
#include <fstream>
#include <sstream>
#include "cuarma/forwards.h"
#include "cuarma/traits/size.hpp"
#include <vector>
#include <map>

namespace cuarma
{
namespace traits
{

//clear:
/** @brief Generic routine for setting all entries of a vector to zero. This is the version for non-cuarma objects. */
template<typename VectorType>
void clear(VectorType & vec)
{
  typedef typename cuarma::result_of::size_type<VectorType>::type  size_type;

  for (size_type i=0; i<cuarma::traits::size(vec); ++i)
    vec[i] = 0;  //TODO: Quantity access can also be wrapped...
}

/** @brief Generic routine for setting all entries of a vector to zero. This is the version for cuarma objects. */
template<typename ScalarType, unsigned int AlignmentV>
void clear(cuarma::vector<ScalarType, AlignmentV> & vec)
{
  vec.clear();
}

} //namespace traits
} //namespace cuarma