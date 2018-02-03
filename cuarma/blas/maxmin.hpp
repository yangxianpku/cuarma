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
#include "cuarma/meta/result_of.hpp"

namespace cuarma
{
  //
  // generic norm_inf function
  //   uses tag dispatch to identify which algorithm
  //   should be called
  //
  namespace blas
  {
    // ----------------------------------------------------
    // STL
    template< typename NumericT >
    NumericT max(std::vector<NumericT> const & v1)
    {
      //std::cout << "stl .. " << std::endl;
      NumericT result = v1[0];
      for (arma_size_t i=1; i<v1.size(); ++i)
      {
        if (v1[i] > result)
          result = v1[i];
      }
      return result;
    }

    // ----------------------------------------------------
    // CUARMA
    //
    template< typename ScalarType>
    cuarma::scalar_expression< const cuarma::vector_base<ScalarType>, const cuarma::vector_base<ScalarType>, cuarma::op_max >
    max(cuarma::vector_base<ScalarType> const & v1)
    {
       //std::cout << "cuarma .. " << std::endl;
      return cuarma::scalar_expression< const cuarma::vector_base<ScalarType>, const cuarma::vector_base<ScalarType>, cuarma::op_max >(v1, v1);
    }

    // with vector expression:
    template<typename LHS, typename RHS, typename OP>
    cuarma::scalar_expression<const cuarma::vector_expression<const LHS, const RHS, OP>,
                                const cuarma::vector_expression<const LHS, const RHS, OP>,
                                cuarma::op_max>
    max(cuarma::vector_expression<const LHS, const RHS, OP> const & vector)
    {
      return cuarma::scalar_expression< const cuarma::vector_expression<const LHS, const RHS, OP>,
                                          const cuarma::vector_expression<const LHS, const RHS, OP>,
                                          cuarma::op_max >(vector, vector);
    }

    // ----------------------------------------------------
    // STL
    //
    template< typename NumericT >
    NumericT min(std::vector<NumericT> const & v1)
    {
      //std::cout << "stl .. " << std::endl;
      NumericT result = v1[0];
      for (arma_size_t i=1; i<v1.size(); ++i)
      {
        if (v1[i] < result)
          result = v1[i];
      }
      return result;
    }

    // ----------------------------------------------------
    // CUARMA
    //
    template< typename ScalarType>
    cuarma::scalar_expression< const cuarma::vector_base<ScalarType>, const cuarma::vector_base<ScalarType>, cuarma::op_min >
    min(cuarma::vector_base<ScalarType> const & v1)
    {
       //std::cout << "cuarma .. " << std::endl;
      return cuarma::scalar_expression< const cuarma::vector_base<ScalarType>, const cuarma::vector_base<ScalarType>, cuarma::op_min >(v1, v1);
    }

    template< typename ScalarType>
    cuarma::scalar_expression< const cuarma::vector_base<ScalarType>, const cuarma::vector_base<ScalarType>, cuarma::op_min >
    min(cuarma::vector<ScalarType> const & v1)
    {
       //std::cout << "cuarma .. " << std::endl;
      return cuarma::scalar_expression< const cuarma::vector_base<ScalarType>, const cuarma::vector_base<ScalarType>, cuarma::op_min >(v1, v1);
    }

    // with vector expression:
    template<typename LHS, typename RHS, typename OP>
    cuarma::scalar_expression<const cuarma::vector_expression<const LHS, const RHS, OP>, const cuarma::vector_expression<const LHS, const RHS, OP>, cuarma::op_min>
    min(cuarma::vector_expression<const LHS, const RHS, OP> const & vector)
    {
      return cuarma::scalar_expression< const cuarma::vector_expression<const LHS, const RHS, OP>, const cuarma::vector_expression<const LHS, const RHS, OP>, cuarma::op_min >(vector, vector);
    }


  } // end namespace blas
} // end namespace cuarma