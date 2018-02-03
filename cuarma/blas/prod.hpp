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

/** @file cuarma/blas/prod.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief 矩阵-向量、矩阵-矩阵乘积运算
    @brief Generic interface for matrix-vector and matrix-matrix products.
           See cuarma/blas/vector_operations.hpp, cuarma/blas/matrix_operations.hpp, and
           cuarma/blas/sparse_matrix_operations.hpp for implementations.
*/

#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/meta/tag_of.hpp"
#include <vector>
#include <map>

namespace cuarma
{
  //
  //   generic prod function
  //   uses tag dispatch to identify which algorithm
  //   should be called
  //
  namespace blas
  {
    #ifdef CUARMA_WITH_UBLAS
    // ----------------------------------------------------
    // UBLAS
    //
    template< typename MatrixT, typename VectorT >
    typename cuarma::enable_if< cuarma::is_ublas< typename cuarma::traits::tag_of< MatrixT >::type >::value, VectorT>::type
    prod(MatrixT const& matrix, VectorT const& vector)
    {
      // std::cout << "ublas .. " << std::endl;
      return boost::numeric::ublas::prod(matrix, vector);
    }
    #endif

    // ----------------------------------------------------
    // STL type
    // dense matrix-vector product:
    template< typename T, typename A1, typename A2, typename VectorT >
    VectorT prod(std::vector< std::vector<T, A1>, A2 > const & matrix, VectorT const& vector)
    {
      VectorT result(matrix.size());
      for (typename std::vector<T, A1>::size_type i=0; i<matrix.size(); ++i)
      {
        result[i] = 0; //we will not assume that VectorT is initialized to zero
        for (typename std::vector<T, A1>::size_type j=0; j<matrix[i].size(); ++j)
          result[i] += matrix[i][j] * vector[j];
      }
      return result;
    }

    // sparse matrix-vector product:
    template< typename KEY, typename DATA, typename COMPARE, typename AMAP, typename AVEC, typename VectorT >
    VectorT prod(std::vector< std::map<KEY, DATA, COMPARE, AMAP>, AVEC > const& matrix, VectorT const& vector)
    {
      typedef std::vector< std::map<KEY, DATA, COMPARE, AMAP>, AVEC > MatrixType;

      VectorT result(matrix.size());
      for (typename MatrixType::size_type i=0; i<matrix.size(); ++i)
      {
        result[i] = 0; //we will not assume that VectorT is initialized to zero
        for (typename std::map<KEY, DATA, COMPARE, AMAP>::const_iterator row_entries = matrix[i].begin();
             row_entries != matrix[i].end();
             ++row_entries)
          result[i] += row_entries->second * vector[row_entries->first];
      }
      return result;
    }

    // ----------------------------------------------------
    // CUARMA

    // standard product:
    template<typename NumericT>
    cuarma::matrix_expression< const cuarma::matrix_base<NumericT>,
                                 const cuarma::matrix_base<NumericT>,
                                 cuarma::op_mat_mat_prod >
    prod(cuarma::matrix_base<NumericT> const & A,
         cuarma::matrix_base<NumericT> const & B)
    {
      return cuarma::matrix_expression< const cuarma::matrix_base<NumericT>,
                                          const cuarma::matrix_base<NumericT>,
                                          cuarma::op_mat_mat_prod >(A, B);
    }

    // right factor is a matrix expression:
    template<typename NumericT, typename LhsT, typename RhsT, typename OpT>
    cuarma::matrix_expression< const cuarma::matrix_base<NumericT>,
                                 const cuarma::matrix_expression<const LhsT, const RhsT, OpT>,
                                 cuarma::op_mat_mat_prod >
    prod(cuarma::matrix_base<NumericT> const & A,
         cuarma::matrix_expression<const LhsT, const RhsT, OpT> const & B)
    {
      return cuarma::matrix_expression< const cuarma::matrix_base<NumericT>,
                                          const cuarma::matrix_expression<const LhsT, const RhsT, OpT>,
                                          cuarma::op_mat_mat_prod >(A, B);
    }

    // left factor is a matrix expression:
    template<typename LhsT, typename RhsT, typename OpT, typename NumericT>
    cuarma::matrix_expression< const cuarma::matrix_expression<const LhsT, const RhsT, OpT>,
                                 const cuarma::matrix_base<NumericT>,
                                 cuarma::op_mat_mat_prod >
    prod(cuarma::matrix_expression<const LhsT, const RhsT, OpT> const & A,
         cuarma::matrix_base<NumericT> const & B)
    {
      return cuarma::matrix_expression< const cuarma::matrix_expression<const LhsT, const RhsT, OpT>,
                                          const cuarma::matrix_base<NumericT>,
                                          cuarma::op_mat_mat_prod >(A, B);
    }

    // both factors transposed:
    template<typename LhsT1, typename RhsT1, typename OpT1, typename LhsT2, typename RhsT2, typename OpT2>
    cuarma::matrix_expression< const cuarma::matrix_expression<const LhsT1, const RhsT1, OpT1>,
                               const cuarma::matrix_expression<const LhsT2, const RhsT2, OpT2>, cuarma::op_mat_mat_prod >
    prod(cuarma::matrix_expression<const LhsT1, const RhsT1, OpT1> const & A,
         cuarma::matrix_expression<const LhsT2, const RhsT2, OpT2> const & B)
    {
      return cuarma::matrix_expression< const cuarma::matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                          const cuarma::matrix_expression<const LhsT2, const RhsT2, OpT2>,
                                          cuarma::op_mat_mat_prod >(A, B);
    }



    // matrix-vector product
    template< typename NumericT>
    cuarma::vector_expression< const cuarma::matrix_base<NumericT>,
                                 const cuarma::vector_base<NumericT>,
                                 cuarma::op_prod >
    prod(cuarma::matrix_base<NumericT> const & A,
         cuarma::vector_base<NumericT> const & x)
    {
      return cuarma::vector_expression< const cuarma::matrix_base<NumericT>,
                                          const cuarma::vector_base<NumericT>,
                                          cuarma::op_prod >(A, x);
    }

    // matrix-vector product (resolve ambiguity)
    template<typename NumericT, typename F>
    cuarma::vector_expression< const cuarma::matrix_base<NumericT>,
                                 const cuarma::vector_base<NumericT>,
                                 cuarma::op_prod >
    prod(cuarma::matrix<NumericT, F> const & A,
         cuarma::vector_base<NumericT> const & x)
    {
      return cuarma::vector_expression< const cuarma::matrix_base<NumericT>,
                                          const cuarma::vector_base<NumericT>,
                                          cuarma::op_prod >(A, x);
    }

    // matrix-vector product (resolve ambiguity)
    template<typename MatrixT, typename NumericT>
    cuarma::vector_expression< const cuarma::matrix_base<NumericT>, const cuarma::vector_base<NumericT>, cuarma::op_prod >
    prod(cuarma::matrix_range<MatrixT> const & A,
         cuarma::vector_base<NumericT> const & x)
    {
      return cuarma::vector_expression< const cuarma::matrix_base<NumericT>, const cuarma::vector_base<NumericT>, cuarma::op_prod >(A, x);
    }

    // matrix-vector product (resolve ambiguity)
    template<typename MatrixT, typename NumericT>
    cuarma::vector_expression< const cuarma::matrix_base<NumericT>, const cuarma::vector_base<NumericT>, cuarma::op_prod >
    prod(cuarma::matrix_slice<MatrixT> const & A,
         cuarma::vector_base<NumericT> const & x)
    {
      return cuarma::vector_expression< const cuarma::matrix_base<NumericT>, const cuarma::vector_base<NumericT>, cuarma::op_prod >(A, x);
    }

    // matrix-vector product with matrix expression (including transpose)
    template< typename NumericT, typename LhsT, typename RhsT, typename OpT>
    cuarma::vector_expression< const cuarma::matrix_expression<const LhsT, const RhsT, OpT>,
                                 const cuarma::vector_base<NumericT>,
                                 cuarma::op_prod >
    prod(cuarma::matrix_expression<const LhsT, const RhsT, OpT> const & A,
         cuarma::vector_base<NumericT> const & x)
    {
      return cuarma::vector_expression< const cuarma::matrix_expression<const LhsT, const RhsT, OpT>,
                                          const cuarma::vector_base<NumericT>,
                                          cuarma::op_prod >(A, x);
    }


    // matrix-vector product with vector expression
    template< typename NumericT, typename LhsT, typename RhsT, typename OpT>
    cuarma::vector_expression< const cuarma::matrix_base<NumericT>,
                                 const cuarma::vector_expression<const LhsT, const RhsT, OpT>,
                                 cuarma::op_prod >
    prod(cuarma::matrix_base<NumericT> const & A,
         cuarma::vector_expression<const LhsT, const RhsT, OpT> const & x)
    {
      return cuarma::vector_expression< const cuarma::matrix_base<NumericT>, const cuarma::vector_expression<const LhsT, const RhsT, OpT>, cuarma::op_prod >(A, x);
    }


    // matrix-vector product with matrix expression (including transpose) and vector expression
    template<typename LhsT1, typename RhsT1, typename OpT1, typename LhsT2, typename RhsT2, typename OpT2>
    cuarma::vector_expression< const cuarma::matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                const cuarma::vector_expression<const LhsT2, const RhsT2, OpT2>, cuarma::op_prod >
    prod(cuarma::matrix_expression<const LhsT1, const RhsT1, OpT1> const & A,
         cuarma::vector_expression<const LhsT2, const RhsT2, OpT2> const & x)
    {
      return cuarma::vector_expression< const cuarma::matrix_expression<const LhsT1, const RhsT1, OpT1>,
                          const cuarma::vector_expression<const LhsT2, const RhsT2, OpT2>, cuarma::op_prod >(A, x);
    }

    template< typename SparseMatrixType, typename SCALARTYPE>
    typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value,
                                  cuarma::matrix_expression<const SparseMatrixType,const matrix_base <SCALARTYPE>,op_prod > >::type
    prod(const SparseMatrixType & sp_mat, const cuarma::matrix_base<SCALARTYPE> & d_mat)
    {
      return cuarma::matrix_expression<const SparseMatrixType, const cuarma::matrix_base<SCALARTYPE>,  op_prod >(sp_mat, d_mat);
    }

    // right factor is transposed
    template< typename SparseMatrixType, typename SCALARTYPE>
    typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value,
                                  cuarma::matrix_expression< const SparseMatrixType,
                                  const cuarma::matrix_expression<const cuarma::matrix_base<SCALARTYPE>,
                                  const cuarma::matrix_base<SCALARTYPE>, op_trans>, cuarma::op_prod > >::type
    prod(const SparseMatrixType & A,
         cuarma::matrix_expression<const cuarma::matrix_base<SCALARTYPE>, const cuarma::matrix_base<SCALARTYPE>,  op_trans> const & B)
    {
      return cuarma::matrix_expression< const SparseMatrixType,const cuarma::matrix_expression<const cuarma::matrix_base<SCALARTYPE>,
                                        const cuarma::matrix_base<SCALARTYPE>, op_trans>, cuarma::op_prod >(A, B);
    }


    /** @brief Sparse matrix-matrix product with compressed_matrix objects */
    template<typename NumericT>
    cuarma::matrix_expression<const compressed_matrix<NumericT>, const compressed_matrix<NumericT>, op_prod >
    prod(compressed_matrix<NumericT> const & A, compressed_matrix<NumericT> const & B)
    {
      return cuarma::matrix_expression<const compressed_matrix<NumericT>,  const compressed_matrix<NumericT>, op_prod >(A, B);
    }

    /** @brief Generic matrix-vector product with user-provided sparse matrix type */
    template<typename SparseMatrixType, typename NumericT>
    vector_expression<const SparseMatrixType, const vector_base<NumericT>, op_prod > prod(const SparseMatrixType & A, const vector_base<NumericT> & x)
    {
      return vector_expression<const SparseMatrixType,  const vector_base<NumericT>, op_prod >(A, x);
    }

  } // end namespace blas
} // end namespace cuarma