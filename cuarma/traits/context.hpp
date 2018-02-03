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

/** @file cuarma/traits/context.hpp
 *  @encoding:UTF-8 文档编码
    @brief Extracts the underlying context from objects
*/

#include <string>
#include <fstream>
#include <sstream>
#include "cuarma/forwards.h"
#include "cuarma/context.hpp"
#include "cuarma/traits/handle.hpp"

namespace cuarma
{
namespace traits
{

// Context
/** @brief Returns an ID for the currently active memory domain of an object */
template<typename T>
cuarma::context context(T const & t)
{


  return cuarma::context(traits::active_handle_id(t));
}

/** @brief Returns an ID for the currently active memory domain of an object */
inline cuarma::context context(cuarma::backend::mem_handle const & h)
{


  return cuarma::context(h.get_active_handle_id());
}

} //namespace traits
} //namespace cuarma