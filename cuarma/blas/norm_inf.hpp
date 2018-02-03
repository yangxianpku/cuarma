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

/** @file norm_inf.hpp
 *  @encoding:UTF-8 文档编码
    @brief Generic interface for the l^infty-norm. See cuarma/blas/vector_operations.hpp for implementations.
*/

#include <cmath>
#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/meta/tag_of.hpp"

namespace cuarma
{
  //
  //   generic norm_inf function
  //   uses tag dispatch to identify which algorithm
  //   should be called
  //
  namespace blas
  {

    #ifdef CUARMA_WITH_UBLAS
    // ----------------------------------------------------
    // UBLAS
    //
    template< typename VectorT >
    typename cuarma::enable_if< cuarma::is_ublas< typename cuarma::traits::tag_of< VectorT >::type >::value,
                                  typename VectorT::value_type
                                >::type
    norm_inf(VectorT const& v1)
    {
      return boost::numeric::ublas::norm_inf(v1);
    }
    #endif


    // ----------------------------------------------------
    // STL
    //
    template< typename T, typename A >
    T norm_inf(std::vector<T, A> const & v1)
    {
      //std::cout << "stl .. " << std::endl;
      T result = 0;
      for (typename std::vector<T, A>::size_type i=0; i<v1.size(); ++i)
      {
        if (std::fabs(v1[i]) > result)
          result = std::fabs(v1[i]);
      }

      return result;
    }

    // ----------------------------------------------------
    // CUARMA
    //
    template< typename ScalarType>
    cuarma::scalar_expression< const cuarma::vector_base<ScalarType>, const cuarma::vector_base<ScalarType>, cuarma::op_norm_inf >
    norm_inf(cuarma::vector_base<ScalarType> const & v1)
    {
       //std::cout << "cuarma .. " << std::endl;
      return cuarma::scalar_expression< const cuarma::vector_base<ScalarType>, const cuarma::vector_base<ScalarType>, cuarma::op_norm_inf >(v1, v1);
    }

    // with vector expression:
    template<typename LHS, typename RHS, typename OP>
    cuarma::scalar_expression<const cuarma::vector_expression<const LHS, const RHS, OP>,
                  const cuarma::vector_expression<const LHS, const RHS, OP>, cuarma::op_norm_inf>
    norm_inf(cuarma::vector_expression<const LHS, const RHS, OP> const & vector)
    {
      return cuarma::scalar_expression< const cuarma::vector_expression<const LHS, const RHS, OP>,
                  const cuarma::vector_expression<const LHS, const RHS, OP>, cuarma::op_norm_inf >(vector, vector);
    }


  } // end namespace blas
} // end namespace cuarma