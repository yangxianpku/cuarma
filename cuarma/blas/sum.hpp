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

/** @file cuarma/blas/sum.hpp
 *  @encoding:UTF-8 文档编码
    @brief Stub routines for the summation of elements in a vector, or all elements in either a row or column of a dense matrix.
*/

#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/meta/tag_of.hpp"
#include "cuarma/meta/result_of.hpp"

namespace cuarma
{
namespace blas
{

//
// Sum of vector entries
//

/** @brief User interface function for computing the sum of all elements of a vector */
template<typename NumericT>
cuarma::scalar_expression< const cuarma::vector_base<NumericT>,
                             const cuarma::vector_base<NumericT>,
                             cuarma::op_sum >
sum(cuarma::vector_base<NumericT> const & x)
{
  return cuarma::scalar_expression< const cuarma::vector_base<NumericT>,
                                      const cuarma::vector_base<NumericT>,
                                      cuarma::op_sum >(x, x);
}

/** @brief User interface function for computing the sum of all elements of a vector specified by a vector operation.
 *
 *  Typical use case:   double my_sum = cuarma::blas::sum(x + y);
 */
template<typename LHS, typename RHS, typename OP>
cuarma::scalar_expression<const cuarma::vector_expression<const LHS, const RHS, OP>,
                            const cuarma::vector_expression<const LHS, const RHS, OP>,
                            cuarma::op_sum>
sum(cuarma::vector_expression<const LHS, const RHS, OP> const & x)
{
  return cuarma::scalar_expression< const cuarma::vector_expression<const LHS, const RHS, OP>,
                                      const cuarma::vector_expression<const LHS, const RHS, OP>,
                                      cuarma::op_sum >(x, x);
}


//
// Sum of entries in rows of a matrix
//

/** @brief User interface function for computing the sum of all elements of each row of a matrix. */
template<typename NumericT>
cuarma::vector_expression< const cuarma::matrix_base<NumericT>,
                             const cuarma::matrix_base<NumericT>,
                             cuarma::op_row_sum >
row_sum(cuarma::matrix_base<NumericT> const & A)
{
  return cuarma::vector_expression< const cuarma::matrix_base<NumericT>,
                                      const cuarma::matrix_base<NumericT>,
                                      cuarma::op_row_sum >(A, A);
}

/** @brief User interface function for computing the sum of all elements of each row of a matrix specified by a matrix operation.
 *
 *  Typical use case:   vector<double> my_sums = cuarma::blas::row_sum(A + B);
 */
template<typename LHS, typename RHS, typename OP>
cuarma::vector_expression<const cuarma::matrix_expression<const LHS, const RHS, OP>,
                            const cuarma::matrix_expression<const LHS, const RHS, OP>,
                            cuarma::op_row_sum>
row_sum(cuarma::matrix_expression<const LHS, const RHS, OP> const & A)
{
  return cuarma::vector_expression< const cuarma::matrix_expression<const LHS, const RHS, OP>,
                                      const cuarma::matrix_expression<const LHS, const RHS, OP>,
                                      cuarma::op_row_sum >(A, A);
}


//
// Sum of entries in columns of a matrix
//

/** @brief User interface function for computing the sum of all elements of each column of a matrix. */
template<typename NumericT>
cuarma::vector_expression< const cuarma::matrix_base<NumericT>,
                             const cuarma::matrix_base<NumericT>,
                             cuarma::op_col_sum >
column_sum(cuarma::matrix_base<NumericT> const & A)
{
  return cuarma::vector_expression< const cuarma::matrix_base<NumericT>,
                                      const cuarma::matrix_base<NumericT>,
                                      cuarma::op_col_sum >(A, A);
}

/** @brief User interface function for computing the sum of all elements of each column of a matrix specified by a matrix operation.
 *
 *  Typical use case:   vector<double> my_sums = cuarma::blas::column_sum(A + B);
 */
template<typename LHS, typename RHS, typename OP>
cuarma::vector_expression<const cuarma::matrix_expression<const LHS, const RHS, OP>,
                            const cuarma::matrix_expression<const LHS, const RHS, OP>,
                            cuarma::op_col_sum>
column_sum(cuarma::matrix_expression<const LHS, const RHS, OP> const & A)
{
  return cuarma::vector_expression< const cuarma::matrix_expression<const LHS, const RHS, OP>,
                                      const cuarma::matrix_expression<const LHS, const RHS, OP>,
                                      cuarma::op_col_sum >(A, A);
}


} // end namespace blas
} // end namespace cuarma


