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


/** @file cuarma/blas/detail//bisect/util.hpp
 *  @encoding:UTF-8 文档编码
    @brief Utility functions

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/

namespace cuarma
{
namespace blas
{
namespace detail
{

////////////////////////////////////////////////////////////////////////////////
//! Minimum
////////////////////////////////////////////////////////////////////////////////
template<class T>
#ifdef __CUDACC__
__host__  __device__
#endif
T
min(const T &lhs, const T &rhs)
{

    return (lhs < rhs) ? lhs : rhs;
}

////////////////////////////////////////////////////////////////////////////////
//! Maximum
////////////////////////////////////////////////////////////////////////////////
template<class T>
#ifdef __CUDACC__
__host__  __device__
#endif
T
max(const T &lhs, const T &rhs)
{

    return (lhs < rhs) ? rhs : lhs;
}

////////////////////////////////////////////////////////////////////////////////
//! Sign of number (float)
////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
__host__  __device__
#endif
inline float
sign_f(const float &val)
{
    return (val < 0.0f) ? -1.0f : 1.0f;
}

////////////////////////////////////////////////////////////////////////////////
//! Sign of number (double)
////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
__host__  __device__
#endif
inline double
sign_d(const double &val)
{
    return (val < 0.0) ? -1.0 : 1.0;
}

///////////////////////////////////////////////////////////////////////////////
//! Get the number of blocks that are required to process \a num_threads with
//! \a num_threads_blocks threads per block
///////////////////////////////////////////////////////////////////////////////
extern "C"
inline
unsigned int
getNumBlocksLinear(const unsigned int num_threads,
                   const unsigned int num_threads_block)
{
    const unsigned int block_rem =
        ((num_threads % num_threads_block) != 0) ? 1 : 0;
    return (num_threads / num_threads_block) + block_rem;
}
} // namespace detail
} // namespace blas
} // namespace cuarma
