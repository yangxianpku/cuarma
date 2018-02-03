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

/** @file cuarma/traits/size.hpp
 *  @encoding:UTF-8 文档编码
    @brief Generic size and resize functionality for different vector and matrix types
*/

#include <string>
#include <fstream>
#include <sstream>
#include "cuarma/forwards.h"
#include "cuarma/meta/result_of.hpp"
#include "cuarma/meta/predicate.hpp"

#ifdef CUARMA_WITH_UBLAS
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#endif

#include <vector>
#include <map>

namespace cuarma
{
namespace traits
{

//
// Resize: Change the size of vectors and matrices
//
/** @brief Generic resize routine for resizing a matrix (cuarma, uBLAS, etc.) to a new size/dimension */
template<typename MatrixType>
void resize(MatrixType & matrix, arma_size_t rows, arma_size_t cols)
{
  matrix.resize(rows, cols);
}

/** @brief Generic resize routine for resizing a vector (cuarma, uBLAS, etc.) to a new size */
template<typename VectorType>
void resize(VectorType & vec, arma_size_t new_size)
{
  vec.resize(new_size);
}

/** \cond */
#ifdef CUARMA_WITH_UBLAS
//ublas needs separate treatment:
template<typename ScalarType>
void resize(boost::numeric::ublas::compressed_matrix<ScalarType> & matrix,
            arma_size_t rows,
            arma_size_t cols)
{
  matrix.resize(rows, cols, false); //Note: omitting third parameter leads to compile time error (not implemented in ublas <= 1.42)
}
#endif

//
// size1: No. of rows for matrices
//
/** @brief Generic routine for obtaining the number of rows of a matrix (cuarma, uBLAS, etc.) */
template<typename MatrixType>
arma_size_t
size1(MatrixType const & mat) { return mat.size1(); }

/** \cond */
template<typename RowType>
arma_size_t
size1(std::vector< RowType > const & mat) { return mat.size(); }

//
// size2: No. of columns for matrices
//
/** @brief Generic routine for obtaining the number of columns of a matrix (cuarma, uBLAS, etc.) */
template<typename MatrixType>
typename result_of::size_type<MatrixType>::type
size2(MatrixType const & mat) { return mat.size2(); }

/** \cond */
template<typename RowType>
arma_size_t
size2(std::vector< RowType > const & mat) { return mat[0].size(); }

//
// size: Returns the length of vectors
//
/** @brief Generic routine for obtaining the size of a vector (cuarma, uBLAS, etc.) */
template<typename VectorType>
arma_size_t size(VectorType const & vec)
{
  return vec.size();
}

/** \cond */
template<typename SparseMatrixType, typename VectorType>
arma_size_t size(vector_expression<const SparseMatrixType, const VectorType, op_prod> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename NumericT>
arma_size_t size(vector_expression<const matrix_base<NumericT>, const vector_base<NumericT>, op_prod> const & proxy)  //matrix-vector product
{
  return proxy.lhs().size1();
}

template<typename NumericT, typename LhsT, typename RhsT, typename OpT>
arma_size_t size(vector_expression<const matrix_base<NumericT>, const vector_expression<LhsT, RhsT, OpT>, op_prod> const & proxy)  //matrix-vector product
{
  return proxy.lhs().size1();
}

template<typename NumericT>
arma_size_t size(vector_expression<const matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>,
                const vector_base<NumericT>,
                op_prod> const & proxy)  //transposed matrix-vector product
{
  return proxy.lhs().lhs().size2();
}


template<typename LHS, typename RHS, typename OP>
arma_size_t size(vector_expression<LHS, RHS, OP> const & proxy)
{
  return size(proxy.lhs());
}

template<typename LHS, typename RHS>
arma_size_t size(vector_expression<LHS, const vector_tuple<RHS>, op_inner_prod> const & proxy)
{
  return proxy.rhs().const_size();
}

template<typename LhsT, typename RhsT, typename OpT, typename VectorT>
arma_size_t size(vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                  VectorT,
                                  op_prod> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename LhsT, typename RhsT, typename OpT, typename NumericT>
arma_size_t size(vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                  const vector_base<NumericT>,
                                  op_prod> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename LhsT1, typename RhsT1, typename OpT1,
         typename LhsT2, typename RhsT2, typename OpT2>
arma_size_t size(vector_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                  const vector_expression<const LhsT2, const RhsT2, OpT2>,
                                  op_prod> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename NumericT>
arma_size_t size(vector_expression<const matrix_base<NumericT>,
                                  const matrix_base<NumericT>,
                                  op_row_sum> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename LhsT, typename RhsT, typename OpT>
arma_size_t size(vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                  const matrix_expression<const LhsT, const RhsT, OpT>,
                                  op_row_sum> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename NumericT>
arma_size_t size(vector_expression<const matrix_base<NumericT>,
                                  const matrix_base<NumericT>,
                                  op_col_sum> const & proxy)
{
  return size2(proxy.lhs());
}

template<typename LhsT, typename RhsT, typename OpT>
arma_size_t size(vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                  const matrix_expression<const LhsT, const RhsT, OpT>,
                                  op_col_sum> const & proxy)
{
  return size2(proxy.lhs());
}

/** \endcond */

//
// internal_size: Returns the internal (padded) length of vectors
//
/** @brief Helper routine for obtaining the buffer length of a cuarma vector  */
template<typename NumericT>
arma_size_t internal_size(vector_base<NumericT> const & vec)
{
  return vec.internal_size();
}


//
// internal_size1: No. of internal (padded) rows for matrices
//
/** @brief Helper routine for obtaining the internal number of entries per row of a cuarma matrix  */
template<typename NumericT>
arma_size_t internal_size1(matrix_base<NumericT> const & mat) { return mat.internal_size1(); }


//
// internal_size2: No. of internal (padded) columns for matrices
//
/** @brief Helper routine for obtaining the internal number of entries per column of a cuarma matrix  */
template<typename NumericT>
arma_size_t internal_size2(matrix_base<NumericT> const & mat) { return mat.internal_size2(); }

/** @brief Helper routine for obtaining the internal number of entries per row of a cuarma matrix  */
template<typename NumericT>
arma_size_t ld(matrix_base<NumericT> const & mat)
{
  if (mat.row_major())
    return mat.internal_size2();
  return mat.internal_size1();
}

template<typename NumericT>
arma_size_t nld(matrix_base<NumericT> const & mat)
{
  if (mat.row_major())
    return mat.stride2();
  return mat.stride1();
}

template<typename LHS>
arma_size_t size(vector_expression<LHS, const int, op_matrix_diag> const & proxy)
{
  int k = proxy.rhs();
  int A_size1 = static_cast<int>(size1(proxy.lhs()));
  int A_size2 = static_cast<int>(size2(proxy.lhs()));

  int row_depth = std::min(A_size1, A_size1 + k);
  int col_depth = std::min(A_size2, A_size2 - k);

  return arma_size_t(std::min(row_depth, col_depth));
}

template<typename LHS>
arma_size_t size(vector_expression<LHS, const unsigned int, op_row> const & proxy)
{
  return size2(proxy.lhs());
}

template<typename LHS>
arma_size_t size(vector_expression<LHS, const unsigned int, op_column> const & proxy)
{
  return size1(proxy.lhs());
}

} //namespace traits
} //namespace cuarma