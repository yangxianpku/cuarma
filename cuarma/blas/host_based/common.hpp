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

/** @file cuarma/blas/host_based/common.hpp
 *  @encoding:UTF-8 文档编码
    @brief Common routines for single-threaded execution on CPU
*/

#include "cuarma/traits/handle.hpp"

namespace cuarma
{
namespace blas
{
namespace host_based
{
namespace detail
{

template<typename ResultT, typename VectorT>
ResultT * extract_raw_pointer(VectorT & vec)
{
  return reinterpret_cast<ResultT *>(cuarma::traits::ram_handle(vec).get());
}

template<typename ResultT, typename VectorT>
ResultT const * extract_raw_pointer(VectorT const & vec)
{
  return reinterpret_cast<ResultT const *>(cuarma::traits::ram_handle(vec).get());
}

/** @brief Helper class for accessing a strided subvector of a larger vector. */
template<typename NumericT>
class vector_array_wrapper
{
public:
  typedef NumericT   value_type;

  vector_array_wrapper(value_type * A,
                       arma_size_t start,
                       arma_size_t inc)
   : A_(A),
     start_(start),
     inc_(inc) {}

  value_type & operator()(arma_size_t i) { return A_[i * inc_ + start_]; }

private:
  value_type * A_;
  arma_size_t start_;
  arma_size_t inc_;
};


/** @brief Helper array for accessing a strided submatrix embedded in a larger matrix. */
template<typename NumericT, typename LayoutT, bool is_transposed>
class matrix_array_wrapper
{
  public:
    typedef NumericT   value_type;

    matrix_array_wrapper(value_type * A,
                         arma_size_t start1, arma_size_t start2,
                         arma_size_t inc1,   arma_size_t inc2,
                         arma_size_t internal_size1, arma_size_t internal_size2)
     : A_(A),
       start1_(start1), start2_(start2),
       inc1_(inc1), inc2_(inc2),
       internal_size1_(internal_size1), internal_size2_(internal_size2) {}

    value_type & operator()(arma_size_t i, arma_size_t j)
    {
      return A_[LayoutT::mem_index(i * inc1_ + start1_,
                                   j * inc2_ + start2_,
                                   internal_size1_, internal_size2_)];
    }

    // convenience overloads to address signed index types for OpenMP:
    value_type & operator()(arma_size_t i, long j) { return operator()(i, static_cast<arma_size_t>(j)); }
    value_type & operator()(long i, arma_size_t j) { return operator()(static_cast<arma_size_t>(i), j); }
    value_type & operator()(long i, long j)       { return operator()(static_cast<arma_size_t>(i), static_cast<arma_size_t>(j)); }

  private:
    value_type * A_;
    arma_size_t start1_, start2_;
    arma_size_t inc1_, inc2_;
    arma_size_t internal_size1_, internal_size2_;
};

/** \cond */
template<typename NumericT, typename LayoutT>
class matrix_array_wrapper<NumericT, LayoutT, true>
{
public:
  typedef NumericT   value_type;

  matrix_array_wrapper(value_type * A,
                       arma_size_t start1, arma_size_t start2,
                       arma_size_t inc1,   arma_size_t inc2,
                       arma_size_t internal_size1, arma_size_t internal_size2)
   : A_(A),
     start1_(start1), start2_(start2),
     inc1_(inc1), inc2_(inc2),
     internal_size1_(internal_size1), internal_size2_(internal_size2) {}

  value_type & operator()(arma_size_t i, arma_size_t j)
  {
    //swapping row and column indices here
    return A_[LayoutT::mem_index(j * inc1_ + start1_,
                                 i * inc2_ + start2_,
                                 internal_size1_, internal_size2_)];
  }

  // convenience overloads to address signed index types for OpenMP:
  value_type & operator()(arma_size_t i, long j) { return operator()(i, static_cast<arma_size_t>(j)); }
  value_type & operator()(long i, arma_size_t j) { return operator()(static_cast<arma_size_t>(i), j); }
  value_type & operator()(long i, long j) { return operator()(static_cast<arma_size_t>(i), static_cast<arma_size_t>(j)); }

private:
  value_type * A_;
  arma_size_t start1_, start2_;
  arma_size_t inc1_, inc2_;
  arma_size_t internal_size1_, internal_size2_;
};
/** \endcond */

} //namespace detail
} //namespace host_based
} //namespace blas
} //namespace cuarma