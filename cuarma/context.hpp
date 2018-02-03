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

/** @file cuarma/context.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief Implementation of a CUDA-like context, which serves as a unification of {OpenMP, CUDA} at the user API.
*/

#include <vector>
#include <stddef.h>
#include <assert.h>
#include "cuarma/forwards.h"
#include "cuarma/backend/mem_handle.hpp"

namespace cuarma
{
/** @brief Represents a generic 'context' similar to an CUDA context, but is backend-agnostic and thus also suitable for CUDA and OpenMP
  *
  * Context objects are used to distinguish between different memory domains. One context may refer to an CUDA device, another context may refer to a CUDA device, and a third context to main RAM.
  * Thus, operations are only defined on objects residing on the same context.
  */
class context
{
public:
  context() : mem_type_(cuarma::backend::default_memory_type())
  {

  }

  explicit context(cuarma::memory_types mtype) : mem_type_(mtype)
  {
    if (mem_type_ == MEMORY_NOT_INITIALIZED)
      mem_type_ = cuarma::backend::default_memory_type();

  }

  // TODO: Add CUDA and OpenMP contexts
  cuarma::memory_types  memory_type() const { return mem_type_; }

private:
  cuarma::memory_types   mem_type_;

};


}

