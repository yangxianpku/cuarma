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

/** @file cuarma/blas/norm_frobenius.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief Frobenius范数
    @brief Generic interface for the Frobenius norm.
*/

#include <cmath>
#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/meta/tag_of.hpp"

namespace cuarma
{
  //
  // generic norm_frobenius function
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
    typename cuarma::enable_if< cuarma::is_ublas< typename cuarma::traits::tag_of< VectorT >::type >::value, typename VectorT::value_type >::type norm_frobenius(VectorT const& v1)
    {
      return boost::numeric::ublas::norm_frobenius(v1);
    }
    #endif


    // ----------------------------------------------------
    // CUARMA
    //
    template<typename NumericT>
    scalar_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_norm_frobenius> norm_frobenius(const matrix_base<NumericT> & A)
    {
      return scalar_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_norm_frobenius>(A, A);
    }

  } // end namespace blas
} // end namespace cuarma





