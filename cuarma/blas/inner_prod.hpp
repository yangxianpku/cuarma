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

/** @file cuarma/blas/inner_prod.hpp
 *  @encoding:UTF-8 文档编码
    @brief Generic interface for the computation of inner products. See cuarma/blas/vector_operations.hpp for implementations.
*/

#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/meta/tag_of.hpp"
#include "cuarma/meta/result_of.hpp"

namespace cuarma
{
//
// generic inner_prod function
//   uses tag dispatch to identify which algorithm
//   should be called
//
namespace blas
{

#ifdef CUARMA_WITH_UBLAS
// ----------------------------------------------------
// UBLAS
template<typename VectorT1, typename VectorT2>
typename cuarma::enable_if< cuarma::is_ublas< typename cuarma::traits::tag_of< VectorT1 >::type >::value, typename VectorT1::value_type>::type
inner_prod(VectorT1 const & v1, VectorT2 const & v2)
{
  //std::cout << "ublas .. " << std::endl;
  return boost::numeric::ublas::inner_prod(v1, v2);
}
#endif

// ----------------------------------------------------
// STL
//
template<typename VectorT1, typename VectorT2>
typename cuarma::enable_if< cuarma::is_stl< typename cuarma::traits::tag_of< VectorT1 >::type >::value, typename VectorT1::value_type>::type
inner_prod(VectorT1 const & v1, VectorT2 const & v2)
{
  assert(v1.size() == v2.size() && bool("Vector sizes mismatch"));
  //std::cout << "stl .. " << std::endl;
  typename VectorT1::value_type result = 0;
  for (typename VectorT1::size_type i=0; i<v1.size(); ++i)
    result += v1[i] * v2[i];

  return result;
}

// ----------------------------------------------------
// CUARMA
//
template<typename NumericT>
cuarma::scalar_expression< const vector_base<NumericT>, const vector_base<NumericT>, cuarma::op_inner_prod >
inner_prod(vector_base<NumericT> const & vector1, vector_base<NumericT> const & vector2)
{
  //std::cout << "cuarma .. " << std::endl;
  return cuarma::scalar_expression< const vector_base<NumericT>, const vector_base<NumericT>, cuarma::op_inner_prod >(vector1, vector2);
}


// expression on lhs:
template< typename LHS, typename RHS, typename OP, typename NumericT>
cuarma::scalar_expression< const cuarma::vector_expression<LHS, RHS, OP>,  const vector_base<NumericT>, cuarma::op_inner_prod >
inner_prod(cuarma::vector_expression<LHS, RHS, OP> const & vector1, vector_base<NumericT> const & vector2)
{
  //std::cout << "cuarma .. " << std::endl;
  return cuarma::scalar_expression< const cuarma::vector_expression<LHS, RHS, OP>,  const vector_base<NumericT>, cuarma::op_inner_prod >(vector1, vector2);
}

// expression on rhs:
template<typename NumericT, typename LHS, typename RHS, typename OP>
cuarma::scalar_expression< const vector_base<NumericT>, const cuarma::vector_expression<LHS, RHS, OP>, cuarma::op_inner_prod >
inner_prod(vector_base<NumericT> const & vector1, cuarma::vector_expression<LHS, RHS, OP> const & vector2)
{
  //std::cout << "cuarma .. " << std::endl;
  return cuarma::scalar_expression< const vector_base<NumericT>, const cuarma::vector_expression<LHS, RHS, OP>, cuarma::op_inner_prod >(vector1, vector2);
}

// expression on lhs and rhs:
template<typename LHS1, typename RHS1, typename OP1, typename LHS2, typename RHS2, typename OP2>
cuarma::scalar_expression< const cuarma::vector_expression<LHS1, RHS1, OP1>,
                             const cuarma::vector_expression<LHS2, RHS2, OP2>,
                             cuarma::op_inner_prod >
inner_prod(cuarma::vector_expression<LHS1, RHS1, OP1> const & vector1,
           cuarma::vector_expression<LHS2, RHS2, OP2> const & vector2)
{
  //std::cout << "cuarma .. " << std::endl;
  return cuarma::scalar_expression< const cuarma::vector_expression<LHS1, RHS1, OP1>,
                                      const cuarma::vector_expression<LHS2, RHS2, OP2>,
                                      cuarma::op_inner_prod >(vector1, vector2);
}


// Multiple inner products:
template<typename NumericT>
cuarma::vector_expression< const vector_base<NumericT>, const vector_tuple<NumericT>, cuarma::op_inner_prod >
inner_prod(vector_base<NumericT> const & x,
           vector_tuple<NumericT> const & y_tuple)
{
  return cuarma::vector_expression< const vector_base<NumericT>,
                                      const vector_tuple<NumericT>,
                                      cuarma::op_inner_prod >(x, y_tuple);
}


} // end namespace blas
} // end namespace cuarma


