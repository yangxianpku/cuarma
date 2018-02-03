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

/** @file cuarma/blas/detail/bisect/bisect_kernel_calls.hpp
 *  @encoding:UTF-8 文档编码
    @brief Kernel calls for the bisection algorithm

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/


#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/vector_proxy.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/meta/result_of.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/start.hpp"
#include "cuarma/traits/handle.hpp"
#include "cuarma/traits/stride.hpp"
#include "cuarma/blas/detail/bisect/structs.hpp"

#ifdef CUARMA_WITH_CUDA
  #include "cuarma/blas/cuda/bisect_kernel_calls.hpp"
#endif

namespace cuarma
{
namespace blas
{
namespace detail
{
 template<typename NumericT>
 void bisectSmall(const InputData<NumericT> &input, ResultDataSmall<NumericT> &result,
                  const unsigned int mat_size,const NumericT lg, const NumericT ug,const NumericT precision)
  {
    switch (cuarma::traits::handle(input.g_a).get_active_handle_id())
    {
#ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::bisectSmall(input, result, mat_size,lg,ug,precision);
        break;
#endif
      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }

 template<typename NumericT>
 void bisectLarge(const InputData<NumericT> &input, ResultDataLarge<NumericT> &result,
                  const unsigned int mat_size, const NumericT lg, const NumericT ug, const NumericT precision)
  {
    switch (cuarma::traits::handle(input.g_a).get_active_handle_id())
    {
#ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::bisectLarge(input, result, mat_size,lg,ug,precision);
        break;
#endif
      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }

 template<typename NumericT>
 void bisectLarge_OneIntervals(const InputData<NumericT> &input, ResultDataLarge<NumericT> &result,
                    const unsigned int mat_size, const NumericT precision)
  {
    switch (cuarma::traits::handle(input.g_a).get_active_handle_id())
    {
#ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::bisectLarge_OneIntervals(input, result, mat_size, precision);
        break;
#endif
      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }

 template<typename NumericT>
 void bisectLarge_MultIntervals(const InputData<NumericT> &input, ResultDataLarge<NumericT> &result,
                    const unsigned int mat_size, const NumericT precision)
  {
    switch (cuarma::traits::handle(input.g_a).get_active_handle_id())
    {

#ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::bisectLarge_MultIntervals(input, result,mat_size,precision);
        break;
#endif
      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }
} // namespace detail
} // namespace blas
} //namespace cuarma