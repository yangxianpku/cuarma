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

/** @file cuarma/blas/host_based/vector_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of vector operations using a plain single-threaded execution on CPU
*/

#include <cmath>
#include <algorithm>  //for std::max and std::min

#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/start.hpp"
#include "cuarma/blas/host_based/common.hpp"
#include "cuarma/blas/detail/op_applier.hpp"
#include "cuarma/traits/stride.hpp"

namespace cuarma
{
namespace blas
{
namespace host_based
{
namespace detail
{
  template<typename NumericT>
  NumericT flip_sign(NumericT val) { return -val; }
  inline unsigned long  flip_sign(unsigned long  val) { return val; }
  inline unsigned int   flip_sign(unsigned int   val) { return val; }
  inline unsigned short flip_sign(unsigned short val) { return val; }
  inline unsigned char  flip_sign(unsigned char  val) { return val; }
}

//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//
template<typename DestNumericT, typename SrcNumericT>
void convert(vector_base<DestNumericT> & dest, vector_base<SrcNumericT> const & src)
{
  DestNumericT      * data_dest = detail::extract_raw_pointer<DestNumericT>(dest);
  SrcNumericT const * data_src  = detail::extract_raw_pointer<SrcNumericT>(src);

  arma_size_t start_dest = cuarma::traits::start(dest);
  arma_size_t inc_dest   = cuarma::traits::stride(dest);
  arma_size_t size_dest  = cuarma::traits::size(dest);

  arma_size_t start_src = cuarma::traits::start(src);
  arma_size_t inc_src   = cuarma::traits::stride(src);


  for (long i = 0; i < static_cast<long>(size_dest); ++i)
    data_dest[static_cast<arma_size_t>(i)*inc_dest+start_dest] = static_cast<DestNumericT>(data_src[static_cast<arma_size_t>(i)*inc_src+start_src]);
}

template<typename NumericT, typename ScalarT1>
void av(vector_base<NumericT> & vec1,
        vector_base<NumericT> const & vec2, ScalarT1 const & alpha, arma_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha)
{
  typedef NumericT        value_type;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = detail::flip_sign(data_alpha);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t start2 = cuarma::traits::start(vec2);
  arma_size_t inc2   = cuarma::traits::stride(vec2);

  if (reciprocal_alpha)
  {

    for (long i = 0; i < static_cast<long>(size1); ++i)
      data_vec1[static_cast<arma_size_t>(i)*inc1+start1] = data_vec2[static_cast<arma_size_t>(i)*inc2+start2] / data_alpha;
  }
  else
  {

    for (long i = 0; i < static_cast<long>(size1); ++i)
      data_vec1[static_cast<arma_size_t>(i)*inc1+start1] = data_vec2[static_cast<arma_size_t>(i)*inc2+start2] * data_alpha;
  }
}


template<typename NumericT, typename ScalarT1, typename ScalarT2>
void avbv(vector_base<NumericT> & vec1,
          vector_base<NumericT> const & vec2, ScalarT1 const & alpha, arma_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
          vector_base<NumericT> const & vec3, ScalarT2 const & beta,  arma_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef NumericT      value_type;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
  value_type const * data_vec3 = detail::extract_raw_pointer<value_type>(vec3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = detail::flip_sign(data_alpha);

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = detail::flip_sign(data_beta);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t start2 = cuarma::traits::start(vec2);
  arma_size_t inc2   = cuarma::traits::stride(vec2);

  arma_size_t start3 = cuarma::traits::start(vec3);
  arma_size_t inc3   = cuarma::traits::stride(vec3);

  if (reciprocal_alpha)
  {
    if (reciprocal_beta)
    {

      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<arma_size_t>(i)*inc1+start1] = data_vec2[static_cast<arma_size_t>(i)*inc2+start2] / data_alpha + data_vec3[static_cast<arma_size_t>(i)*inc3+start3] / data_beta;
    }
    else
    {

      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<arma_size_t>(i)*inc1+start1] = data_vec2[static_cast<arma_size_t>(i)*inc2+start2] / data_alpha + data_vec3[static_cast<arma_size_t>(i)*inc3+start3] * data_beta;
    }
  }
  else
  {
    if (reciprocal_beta)
    {

      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<arma_size_t>(i)*inc1+start1] = data_vec2[static_cast<arma_size_t>(i)*inc2+start2] * data_alpha + data_vec3[static_cast<arma_size_t>(i)*inc3+start3] / data_beta;
    }
    else
    {

      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<arma_size_t>(i)*inc1+start1] = data_vec2[static_cast<arma_size_t>(i)*inc2+start2] * data_alpha + data_vec3[static_cast<arma_size_t>(i)*inc3+start3] * data_beta;
    }
  }
}


template<typename NumericT, typename ScalarT1, typename ScalarT2>
void avbv_v(vector_base<NumericT> & vec1,
            vector_base<NumericT> const & vec2, ScalarT1 const & alpha, arma_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
            vector_base<NumericT> const & vec3, ScalarT2 const & beta,  arma_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef NumericT        value_type;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
  value_type const * data_vec3 = detail::extract_raw_pointer<value_type>(vec3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = detail::flip_sign(data_alpha);

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = detail::flip_sign(data_beta);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t start2 = cuarma::traits::start(vec2);
  arma_size_t inc2   = cuarma::traits::stride(vec2);

  arma_size_t start3 = cuarma::traits::start(vec3);
  arma_size_t inc3   = cuarma::traits::stride(vec3);

  if (reciprocal_alpha)
  {
    if (reciprocal_beta)
    {

      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<arma_size_t>(i)*inc1+start1] += data_vec2[static_cast<arma_size_t>(i)*inc2+start2] / data_alpha + data_vec3[static_cast<arma_size_t>(i)*inc3+start3] / data_beta;
    }
    else
    {

      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<arma_size_t>(i)*inc1+start1] += data_vec2[static_cast<arma_size_t>(i)*inc2+start2] / data_alpha + data_vec3[static_cast<arma_size_t>(i)*inc3+start3] * data_beta;
    }
  }
  else
  {
    if (reciprocal_beta)
    {

      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<arma_size_t>(i)*inc1+start1] += data_vec2[static_cast<arma_size_t>(i)*inc2+start2] * data_alpha + data_vec3[static_cast<arma_size_t>(i)*inc3+start3] / data_beta;
    }
    else
    {

      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<arma_size_t>(i)*inc1+start1] += data_vec2[static_cast<arma_size_t>(i)*inc2+start2] * data_alpha + data_vec3[static_cast<arma_size_t>(i)*inc3+start3] * data_beta;
    }
  }
}




/** @brief Assign a constant value to a vector (-range/-slice)
*
* @param vec1   The vector to which the value should be assigned
* @param alpha  The value to be assigned
* @param up_to_internal_size  Specifies whether alpha should also be written to padded memory (mostly used for clearing the whole buffer).
*/
template<typename NumericT>
void vector_assign(vector_base<NumericT> & vec1, const NumericT & alpha, bool up_to_internal_size = false)
{
  typedef NumericT       value_type;

  value_type * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);
  arma_size_t loop_bound  = up_to_internal_size ? vec1.internal_size() : size1;  //Note: Do NOT use traits::internal_size() here, because vector proxies don't require padding.

  value_type data_alpha = static_cast<value_type>(alpha);


  for (long i = 0; i < static_cast<long>(loop_bound); ++i)
    data_vec1[static_cast<arma_size_t>(i)*inc1+start1] = data_alpha;
}


/** @brief Swaps the contents of two vectors, data is copied
*
* @param vec1   The first vector (or -range, or -slice)
* @param vec2   The second vector (or -range, or -slice)
*/
template<typename NumericT>
void vector_swap(vector_base<NumericT> & vec1, vector_base<NumericT> & vec2)
{
  typedef NumericT      value_type;

  value_type * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t start2 = cuarma::traits::start(vec2);
  arma_size_t inc2   = cuarma::traits::stride(vec2);


  for (long i = 0; i < static_cast<long>(size1); ++i)
  {
    value_type temp = data_vec2[static_cast<arma_size_t>(i)*inc2+start2];
    data_vec2[static_cast<arma_size_t>(i)*inc2+start2] = data_vec1[static_cast<arma_size_t>(i)*inc1+start1];
    data_vec1[static_cast<arma_size_t>(i)*inc1+start1] = temp;
  }
}


///////////////////////// Elementwise operations /////////////

/** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
*
* @param vec1   The result vector (or -range, or -slice)
* @param proxy  The proxy object holding v2, v3 and the operation
*/
template<typename NumericT, typename OpT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_binary<OpT> > const & proxy)
{
  typedef NumericT                                           value_type;
  typedef cuarma::blas::detail::op_applier<op_element_binary<OpT> >    OpFunctor;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(proxy.lhs());
  value_type const * data_vec3 = detail::extract_raw_pointer<value_type>(proxy.rhs());

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t start2 = cuarma::traits::start(proxy.lhs());
  arma_size_t inc2   = cuarma::traits::stride(proxy.lhs());

  arma_size_t start3 = cuarma::traits::start(proxy.rhs());
  arma_size_t inc3   = cuarma::traits::stride(proxy.rhs());


  for (long i = 0; i < static_cast<long>(size1); ++i)
    OpFunctor::apply(data_vec1[static_cast<arma_size_t>(i)*inc1+start1], data_vec2[static_cast<arma_size_t>(i)*inc2+start2], data_vec3[static_cast<arma_size_t>(i)*inc3+start3]);
}

/** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
*
* @param vec1   The result vector (or -range, or -slice)
* @param proxy  The proxy object holding v2, v3 and the operation
*/
template<typename NumericT, typename OpT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<OpT> > const & proxy)
{
  typedef NumericT      value_type;
  typedef cuarma::blas::detail::op_applier<op_element_unary<OpT> >    OpFunctor;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(proxy.lhs());

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t start2 = cuarma::traits::start(proxy.lhs());
  arma_size_t inc2   = cuarma::traits::stride(proxy.lhs());


  for (long i = 0; i < static_cast<long>(size1); ++i)
    OpFunctor::apply(data_vec1[static_cast<arma_size_t>(i)*inc1+start1], data_vec2[static_cast<arma_size_t>(i)*inc2+start2]);
}


///////////////////////// Norms and inner product ///////////////////


//implementation of inner product:

namespace detail
{

// the following circumvents problems when trying to use a variable of template parameter type for a reduction.
// Such a behavior is not covered by the OpenMP standard, hence we manually apply some preprocessor magic to resolve the problem.
// See https://github.com/cuarma/cuarma-dev/issues/112 for a detailed explanation and discussion.

#define CUARMA_INNER_PROD_IMPL_1(RESULTSCALART, TEMPSCALART) \
  inline RESULTSCALART inner_prod_impl(RESULTSCALART const * data_vec1, arma_size_t start1, arma_size_t inc1, arma_size_t size1, \
                                       RESULTSCALART const * data_vec2, arma_size_t start2, arma_size_t inc2) { \
    TEMPSCALART temp = 0;

#define CUARMA_INNER_PROD_IMPL_2(RESULTSCALART) \
    for (long i = 0; i < static_cast<long>(size1); ++i) \
      temp += data_vec1[static_cast<arma_size_t>(i)*inc1+start1] * data_vec2[static_cast<arma_size_t>(i)*inc2+start2]; \
    return static_cast<RESULTSCALART>(temp); \
  }

// char
CUARMA_INNER_PROD_IMPL_1(char, int)

CUARMA_INNER_PROD_IMPL_2(char)

CUARMA_INNER_PROD_IMPL_1(unsigned char, int)

CUARMA_INNER_PROD_IMPL_2(unsigned char)


// short
CUARMA_INNER_PROD_IMPL_1(short, int)

CUARMA_INNER_PROD_IMPL_2(short)

CUARMA_INNER_PROD_IMPL_1(unsigned short, int)

CUARMA_INNER_PROD_IMPL_2(unsigned short)


// int
CUARMA_INNER_PROD_IMPL_1(int, int)

CUARMA_INNER_PROD_IMPL_2(int)

CUARMA_INNER_PROD_IMPL_1(unsigned int, unsigned int)

CUARMA_INNER_PROD_IMPL_2(unsigned int)


// long
CUARMA_INNER_PROD_IMPL_1(long, long)

CUARMA_INNER_PROD_IMPL_2(long)

CUARMA_INNER_PROD_IMPL_1(unsigned long, unsigned long)

CUARMA_INNER_PROD_IMPL_2(unsigned long)


// float
CUARMA_INNER_PROD_IMPL_1(float, float)

CUARMA_INNER_PROD_IMPL_2(float)

// double
CUARMA_INNER_PROD_IMPL_1(double, double)

CUARMA_INNER_PROD_IMPL_2(double)

#undef CUARMA_INNER_PROD_IMPL_1
#undef CUARMA_INNER_PROD_IMPL_2
}

/** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
*
* @param vec1 The first vector
* @param vec2 The second vector
* @param result The result scalar (on the gpu)
*/
template<typename NumericT, typename ScalarT>
void inner_prod_impl(vector_base<NumericT> const & vec1,
                     vector_base<NumericT> const & vec2,
                     ScalarT & result)
{
  typedef NumericT      value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t start2 = cuarma::traits::start(vec2);
  arma_size_t inc2   = cuarma::traits::stride(vec2);

  result = detail::inner_prod_impl(data_vec1, start1, inc1, size1,
                                   data_vec2, start2, inc2);  //Note: Assignment to result might be expensive, thus a temporary is introduced here
}

template<typename NumericT>
void inner_prod_impl(vector_base<NumericT> const & x,
                     vector_tuple<NumericT> const & vec_tuple,
                     vector_base<NumericT> & result)
{
  typedef NumericT        value_type;

  value_type const * data_x = detail::extract_raw_pointer<value_type>(x);

  arma_size_t start_x = cuarma::traits::start(x);
  arma_size_t inc_x   = cuarma::traits::stride(x);
  arma_size_t size_x  = cuarma::traits::size(x);

  std::vector<value_type> temp(vec_tuple.const_size());
  std::vector<value_type const *> data_y(vec_tuple.const_size());
  std::vector<arma_size_t> start_y(vec_tuple.const_size());
  std::vector<arma_size_t> stride_y(vec_tuple.const_size());

  for (arma_size_t j=0; j<vec_tuple.const_size(); ++j)
  {
    data_y[j] = detail::extract_raw_pointer<value_type>(vec_tuple.const_at(j));
    start_y[j] = cuarma::traits::start(vec_tuple.const_at(j));
    stride_y[j] = cuarma::traits::stride(vec_tuple.const_at(j));
  }

  // Note: No OpenMP here because it cannot perform a reduction on temp-array. Savings in memory bandwidth are expected to still justify this approach...
  for (arma_size_t i = 0; i < size_x; ++i)
  {
    value_type entry_x = data_x[i*inc_x+start_x];
    for (arma_size_t j=0; j < vec_tuple.const_size(); ++j)
      temp[j] += entry_x * data_y[j][i*stride_y[j]+start_y[j]];
  }

  for (arma_size_t j=0; j < vec_tuple.const_size(); ++j)
    result[j] = temp[j];  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}


namespace detail
{

#define CUARMA_NORM_1_IMPL_1(RESULTSCALART, TEMPSCALART) \
  inline RESULTSCALART norm_1_impl(RESULTSCALART const * data_vec1, arma_size_t start1, arma_size_t inc1, arma_size_t size1) { \
    TEMPSCALART temp = 0;

#define CUARMA_NORM_1_IMPL_2(RESULTSCALART, TEMPSCALART) \
    for (long i = 0; i < static_cast<long>(size1); ++i) \
      temp += static_cast<TEMPSCALART>(std::fabs(static_cast<double>(data_vec1[static_cast<arma_size_t>(i)*inc1+start1]))); \
    return static_cast<RESULTSCALART>(temp); \
  }

// char
CUARMA_NORM_1_IMPL_1(char, int)

CUARMA_NORM_1_IMPL_2(char, int)

CUARMA_NORM_1_IMPL_1(unsigned char, int)

CUARMA_NORM_1_IMPL_2(unsigned char, int)

// short
CUARMA_NORM_1_IMPL_1(short, int)

CUARMA_NORM_1_IMPL_2(short, int)

CUARMA_NORM_1_IMPL_1(unsigned short, int)

CUARMA_NORM_1_IMPL_2(unsigned short, int)


// int
CUARMA_NORM_1_IMPL_1(int, int)

CUARMA_NORM_1_IMPL_2(int, int)

CUARMA_NORM_1_IMPL_1(unsigned int, unsigned int)

CUARMA_NORM_1_IMPL_2(unsigned int, unsigned int)


// long
CUARMA_NORM_1_IMPL_1(long, long)

CUARMA_NORM_1_IMPL_2(long, long)

CUARMA_NORM_1_IMPL_1(unsigned long, unsigned long)

CUARMA_NORM_1_IMPL_2(unsigned long, unsigned long)


// float
CUARMA_NORM_1_IMPL_1(float, float)

CUARMA_NORM_1_IMPL_2(float, float)

// double
CUARMA_NORM_1_IMPL_1(double, double)

CUARMA_NORM_1_IMPL_2(double, double)

#undef CUARMA_NORM_1_IMPL_1
#undef CUARMA_NORM_1_IMPL_2

}

/** @brief Computes the l^1-norm of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void norm_1_impl(vector_base<NumericT> const & vec1,
                 ScalarT & result)
{
  typedef NumericT        value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  result = detail::norm_1_impl(data_vec1, start1, inc1, size1);  //Note: Assignment to result might be expensive, thus using a temporary for accumulation
}



namespace detail
{

#define CUARMA_NORM_2_IMPL_1(RESULTSCALART, TEMPSCALART) \
  inline RESULTSCALART norm_2_impl(RESULTSCALART const * data_vec1, arma_size_t start1, arma_size_t inc1, arma_size_t size1) { \
    TEMPSCALART temp = 0;

#define CUARMA_NORM_2_IMPL_2(RESULTSCALART, TEMPSCALART) \
    for (long i = 0; i < static_cast<long>(size1); ++i) { \
      RESULTSCALART data = data_vec1[static_cast<arma_size_t>(i)*inc1+start1]; \
      temp += static_cast<TEMPSCALART>(data * data); \
    } \
    return static_cast<RESULTSCALART>(temp); \
  }

// char
CUARMA_NORM_2_IMPL_1(char, int)

CUARMA_NORM_2_IMPL_2(char, int)

CUARMA_NORM_2_IMPL_1(unsigned char, int)

CUARMA_NORM_2_IMPL_2(unsigned char, int)


// short
CUARMA_NORM_2_IMPL_1(short, int)

CUARMA_NORM_2_IMPL_2(short, int)

CUARMA_NORM_2_IMPL_1(unsigned short, int)

CUARMA_NORM_2_IMPL_2(unsigned short, int)


// int
CUARMA_NORM_2_IMPL_1(int, int)

CUARMA_NORM_2_IMPL_2(int, int)

CUARMA_NORM_2_IMPL_1(unsigned int, unsigned int)

CUARMA_NORM_2_IMPL_2(unsigned int, unsigned int)


// long
CUARMA_NORM_2_IMPL_1(long, long)

CUARMA_NORM_2_IMPL_2(long, long)

CUARMA_NORM_2_IMPL_1(unsigned long, unsigned long)

CUARMA_NORM_2_IMPL_2(unsigned long, unsigned long)


// float
CUARMA_NORM_2_IMPL_1(float, float)

CUARMA_NORM_2_IMPL_2(float, float)

// double
CUARMA_NORM_2_IMPL_1(double, double)

CUARMA_NORM_2_IMPL_2(double, double)

#undef CUARMA_NORM_2_IMPL_1
#undef CUARMA_NORM_2_IMPL_2

}


/** @brief Computes the l^2-norm of a vector - implementation
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void norm_2_impl(vector_base<NumericT> const & vec1, ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  result = std::sqrt(detail::norm_2_impl(data_vec1, start1, inc1, size1));  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}

/** @brief Computes the supremum-norm of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void norm_inf_impl(vector_base<NumericT> const & vec1,
                   ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);
  arma_size_t thread_count=1;

  std::vector<value_type> temp(thread_count);

  {
    arma_size_t id = 0;


    arma_size_t begin = (size1 * id) / thread_count;
    arma_size_t end   = (size1 * (id + 1)) / thread_count;
    temp[id]         = 0;

    for (arma_size_t i = begin; i < end; ++i)
      temp[id] = std::max<value_type>(temp[id], static_cast<value_type>(std::fabs(static_cast<double>(data_vec1[i*inc1+start1]))));  //casting to double in order to avoid problems if T is an integer type
  }
  for (arma_size_t i = 1; i < thread_count; ++i)
    temp[0] = std::max<value_type>( temp[0], temp[i]);
  result  = temp[0];
}

//This function should return a CPU scalar, otherwise statements like
// arma_rhs[index_norm_inf(arma_rhs)]
// are ambiguous
/** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
*
* @param vec1 The vector
* @return The result. Note that the result must be a CPU scalar (unsigned int), since gpu scalars are floating point types.
*/
template<typename NumericT>
arma_size_t index_norm_inf(vector_base<NumericT> const & vec1)
{
  typedef NumericT      value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);
  arma_size_t thread_count=1;

  std::vector<value_type> temp(thread_count);
  std::vector<arma_size_t> index(thread_count);

  {
    arma_size_t id = 0;

    arma_size_t begin = (size1 * id) / thread_count;
    arma_size_t end   = (size1 * (id + 1)) / thread_count;
    index[id]        = start1;
    temp[id]         = 0;
    value_type data;

    for (arma_size_t i = begin; i < end; ++i)
    {
      data = static_cast<value_type>(std::fabs(static_cast<double>(data_vec1[i*inc1+start1])));  //casting to double in order to avoid problems if T is an integer type
      if (data > temp[id])
      {
        index[id] = i;
        temp[id]  = data;
      }
    }
  }
  for (arma_size_t i = 1; i < thread_count; ++i)
  {
    if (temp[i] > temp[0])
    {
      index[0] = index[i];
      temp[0] = temp[i];
    }
  }
  return index[0];
}

/** @brief Computes the maximum of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void max_impl(vector_base<NumericT> const & vec1, ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t thread_count=1;

  std::vector<value_type> temp(thread_count);

  {
    arma_size_t id = 0;

    arma_size_t begin = (size1 * id) / thread_count;
    arma_size_t end   = (size1 * (id + 1)) / thread_count;
    temp[id]         = data_vec1[start1];

    for (arma_size_t i = begin; i < end; ++i)
    {
      value_type v = data_vec1[i*inc1+start1];//Note: Assignment to 'vec1' in std::min might be expensive, thus 'v' is used for the function
      temp[id] = std::max<value_type>(temp[id],v);
    }
  }
  for (arma_size_t i = 1; i < thread_count; ++i)
    temp[0] = std::max<value_type>( temp[0], temp[i]);
  result  = temp[0];//Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}

/** @brief Computes the minimum of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void min_impl(vector_base<NumericT> const & vec1, ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t thread_count=1;

  std::vector<value_type> temp(thread_count);

  {
    arma_size_t id = 0;

    arma_size_t begin = (size1 * id) / thread_count;
    arma_size_t end   = (size1 * (id + 1)) / thread_count;
    temp[id]         = data_vec1[start1];

    for (arma_size_t i = begin; i < end; ++i)
    {
      value_type v = data_vec1[i*inc1+start1];//Note: Assignment to 'vec1' in std::min might be expensive, thus 'v' is used for the function
      temp[id] = std::min<value_type>(temp[id],v);
    }
  }
  for (arma_size_t i = 1; i < thread_count; ++i)
    temp[0] = std::min<value_type>( temp[0], temp[i]);
  result  = temp[0];//Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}

/** @brief Computes the sum of all elements from the vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void sum_impl(vector_base<NumericT> const & vec1,
              ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  value_type temp = 0;

  for (long i = 0; i < static_cast<long>(size1); ++i)
    temp += data_vec1[static_cast<arma_size_t>(i)*inc1+start1];

  result = temp;  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}

/** @brief Computes a plane rotation of two vectors.
*
* Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
*
* @param vec1   The first vector
* @param vec2   The second vector
* @param alpha  The first transformation coefficient
* @param beta   The second transformation coefficient
*/
template<typename NumericT>
void plane_rotation(vector_base<NumericT> & vec1,
                    vector_base<NumericT> & vec2,
                    NumericT alpha, NumericT beta)
{
  typedef NumericT  value_type;

  value_type * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);

  arma_size_t start1 = cuarma::traits::start(vec1);
  arma_size_t inc1   = cuarma::traits::stride(vec1);
  arma_size_t size1  = cuarma::traits::size(vec1);

  arma_size_t start2 = cuarma::traits::start(vec2);
  arma_size_t inc2   = cuarma::traits::stride(vec2);

  value_type data_alpha = alpha;
  value_type data_beta  = beta;


  for (long i = 0; i < static_cast<long>(size1); ++i)
  {
    value_type temp1 = data_vec1[static_cast<arma_size_t>(i)*inc1+start1];
    value_type temp2 = data_vec2[static_cast<arma_size_t>(i)*inc2+start2];

    data_vec1[static_cast<arma_size_t>(i)*inc1+start1] = data_alpha * temp1 + data_beta * temp2;
    data_vec2[static_cast<arma_size_t>(i)*inc2+start2] = data_alpha * temp2 - data_beta * temp1;
  }
}

namespace detail
{
  /** @brief Implementation of inclusive_scan and exclusive_scan for the host (OpenMP) backend. */
  template<typename NumericT>
  void vector_scan_impl(vector_base<NumericT> const & vec1,
                        vector_base<NumericT>       & vec2,
                        bool is_inclusive)
  {
    NumericT const * data_vec1 = detail::extract_raw_pointer<NumericT>(vec1);
    NumericT       * data_vec2 = detail::extract_raw_pointer<NumericT>(vec2);

    arma_size_t start1 = cuarma::traits::start(vec1);
    arma_size_t inc1   = cuarma::traits::stride(vec1);
    arma_size_t size1  = cuarma::traits::size(vec1);
    if (size1 < 1)
      return;

    arma_size_t start2 = cuarma::traits::start(vec2);
    arma_size_t inc2   = cuarma::traits::stride(vec2);


    {
      NumericT sum = 0;
      if (is_inclusive)
      {
        for(arma_size_t i = 0; i < size1; i++)
        {
          sum += data_vec1[i * inc1 + start1];
          data_vec2[i * inc2 + start2] = sum;
        }
      }
      else
      {
        for(arma_size_t i = 0; i < size1; i++)
        {
          NumericT tmp = data_vec1[i * inc1 + start1];
          data_vec2[i * inc2 + start2] = sum;
          sum += tmp;
        }
      }
    }

  }
}

/** @brief This function implements an inclusive scan on the host using OpenMP.
*
* Given an element vector (x_0, x_1, ..., x_{n-1}),
* this routine computes (x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-1})
*
* @param vec1       Input vector: Gets overwritten by the routine.
* @param vec2       The output vector. Either idential to vec1 or non-overlapping.
*/
template<typename NumericT>
void inclusive_scan(vector_base<NumericT> const & vec1,
                    vector_base<NumericT>       & vec2)
{
  detail::vector_scan_impl(vec1, vec2, true);
}

/** @brief This function implements an exclusive scan on the host using OpenMP.
*
* Given an element vector (x_0, x_1, ..., x_{n-1}),
* this routine computes (0, x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-2})
*
* @param vec1       Input vector: Gets overwritten by the routine.
* @param vec2       The output vector. Either idential to vec1 or non-overlapping.
*/
template<typename NumericT>
void exclusive_scan(vector_base<NumericT> const & vec1,
                    vector_base<NumericT>       & vec2)
{
  detail::vector_scan_impl(vec1, vec2, false);
}


} //namespace host_based
} //namespace blas
} //namespace cuarma