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

/** @file cuarma/blas/sparse_matrix_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of operations using sparse matrices
*/

#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/host_based/sparse_matrix_operations.hpp"

#ifdef CUARMA_WITH_CUDA
  #include "cuarma/blas/cuda/sparse_matrix_operations.hpp"
#endif

namespace cuarma
{
  namespace blas
  {
    namespace detail
    {

      template<typename SparseMatrixType, typename SCALARTYPE, unsigned int VEC_ALIGNMENT>
      typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value >::type
      row_info(SparseMatrixType const & mat,
               vector<SCALARTYPE, VEC_ALIGNMENT> & vec,
               row_info_types info_selector)
      {
        switch (cuarma::traits::handle(mat).get_active_handle_id())
        {
          case cuarma::MAIN_MEMORY:
            cuarma::blas::host_based::detail::row_info(mat, vec, info_selector);
            break;

#ifdef CUARMA_WITH_CUDA
          case cuarma::CUDA_MEMORY:
            cuarma::blas::cuda::detail::row_info(mat, vec, info_selector);
            break;
#endif
          case cuarma::MEMORY_NOT_INITIALIZED:
            throw memory_exception("not initialised!");
          default:
            throw memory_exception("not implemented");
        }
      }

    }

    // A * x

    /** @brief Carries out matrix-vector multiplication involving a sparse matrix type
    *
    * Implementation of the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    */
    template<typename SparseMatrixType, class ScalarType>
    typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value>::type
    prod_impl(const SparseMatrixType & mat,
              const cuarma::vector_base<ScalarType> & vec,
              ScalarType alpha,
                    cuarma::vector_base<ScalarType> & result,
              ScalarType beta)
    {
      assert( (mat.size1() == result.size()) && bool("Size check failed for compressed matrix-vector product: size1(mat) != size(result)"));
      assert( (mat.size2() == vec.size())    && bool("Size check failed for compressed matrix-vector product: size2(mat) != size(x)"));

      switch (cuarma::traits::handle(mat).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
        cuarma::blas::host_based::prod_impl(mat, vec, alpha, result, beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(mat, vec, alpha, result, beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    // A * B
    /** @brief Carries out matrix-matrix multiplication first matrix being sparse
    *
    * Implementation of the convenience expression result = prod(sp_mat, d_mat);
    *
    * @param sp_mat   The sparse matrix
    * @param d_mat    The dense matrix
    * @param result   The result matrix (dense)
    */
    template<typename SparseMatrixType, class ScalarType>
    typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value>::type
    prod_impl(const SparseMatrixType & sp_mat,
              const cuarma::matrix_base<ScalarType> & d_mat,
                    cuarma::matrix_base<ScalarType> & result)
    {
      assert( (sp_mat.size1() == result.size1()) && bool("Size check failed for compressed matrix - dense matrix product: size1(sp_mat) != size1(result)"));
      assert( (sp_mat.size2() == d_mat.size1()) && bool("Size check failed for compressed matrix - dense matrix product: size2(sp_mat) != size1(d_mat)"));

      switch (cuarma::traits::handle(sp_mat).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::prod_impl(sp_mat, d_mat, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(sp_mat, d_mat, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    // A * transpose(B)
    /** @brief Carries out matrix-matrix multiplication first matrix being sparse, and the second transposed
    *
    * Implementation of the convenience expression result = prod(sp_mat, d_mat);
    *
    * @param sp_mat   The sparse matrix
    * @param d_mat    The dense matrix (transposed)
    * @param result   The result matrix (dense)
    */
    template<typename SparseMatrixType, class ScalarType>
    typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value>::type
    prod_impl(const SparseMatrixType & sp_mat,
              const cuarma::matrix_expression<const cuarma::matrix_base<ScalarType>,
                                                const cuarma::matrix_base<ScalarType>,
                                                cuarma::op_trans>& d_mat,
                    cuarma::matrix_base<ScalarType> & result)
    {
      assert( (sp_mat.size1() == result.size1()) && bool("Size check failed for compressed matrix - dense matrix product: size1(sp_mat) != size1(result)"));
      assert( (sp_mat.size2() == d_mat.size1()) && bool("Size check failed for compressed matrix - dense matrix product: size2(sp_mat) != size1(d_mat)"));

      switch (cuarma::traits::handle(sp_mat).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::prod_impl(sp_mat, d_mat, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(sp_mat, d_mat, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    // A * B with both A and B sparse

    /** @brief Carries out sparse_matrix-sparse_matrix multiplication for CSR matrices
    *
    * Implementation of the convenience expression C = prod(A, B);
    * Based on computing C(i, :) = A(i, :) * B via merging the respective rows of B
    *
    * @param A     Left factor
    * @param B     Right factor
    * @param C     Result matrix
    */
    template<typename NumericT>
    void
    prod_impl(const cuarma::compressed_matrix<NumericT> & A,
              const cuarma::compressed_matrix<NumericT> & B,
                    cuarma::compressed_matrix<NumericT> & C)
    {
      assert( (A.size2() == B.size1())                    && bool("Size check failed for sparse matrix-matrix product: size2(A) != size1(B)"));
      assert( (C.size1() == 0 || C.size1() == A.size1())  && bool("Size check failed for sparse matrix-matrix product: size1(A) != size1(C)"));
      assert( (C.size2() == 0 || C.size2() == B.size2())  && bool("Size check failed for sparse matrix-matrix product: size2(B) != size2(B)"));

      switch (cuarma::traits::handle(A).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::prod_impl(A, B, C);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(A, B, C);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Carries out triangular inplace solves
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param tag    The solver tag (lower_tag, unit_lower_tag, unit_upper_tag, or upper_tag)
    */
    template<typename SparseMatrixType, class ScalarType, typename SOLVERTAG>
    typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value>::type
    inplace_solve(const SparseMatrixType & mat,
                  cuarma::vector_base<ScalarType> & vec,
                  SOLVERTAG tag)
    {
      assert( (mat.size1() == mat.size2()) && bool("Size check failed for triangular solve on compressed matrix: size1(mat) != size2(mat)"));
      assert( (mat.size2() == vec.size())  && bool("Size check failed for compressed matrix-vector product: size2(mat) != size(x)"));

      switch (cuarma::traits::handle(mat).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::inplace_solve(mat, vec, tag);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::inplace_solve(mat, vec, tag);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Carries out transposed triangular inplace solves
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param tag    The solver tag (lower_tag, unit_lower_tag, unit_upper_tag, or upper_tag)
    */
    template<typename SparseMatrixType, class ScalarType, typename SOLVERTAG>
    typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value>::type
    inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat,
                  cuarma::vector_base<ScalarType> & vec,
                  SOLVERTAG tag)
    {
      assert( (mat.size1() == mat.size2()) && bool("Size check failed for triangular solve on transposed compressed matrix: size1(mat) != size2(mat)"));
      assert( (mat.size1() == vec.size())    && bool("Size check failed for transposed compressed matrix triangular solve: size1(mat) != size(x)"));

      switch (cuarma::traits::handle(mat.lhs()).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::inplace_solve(mat, vec, tag);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::inplace_solve(mat, vec, tag);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }



    namespace detail
    {

      template<typename SparseMatrixType, class ScalarType, typename SOLVERTAG>
      typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value>::type
      block_inplace_solve(const matrix_expression<const SparseMatrixType, const SparseMatrixType, op_trans> & mat,
                          cuarma::backend::mem_handle const & block_index_array, arma_size_t num_blocks,
                          cuarma::vector_base<ScalarType> const & mat_diagonal,
                          cuarma::vector_base<ScalarType> & vec,
                          SOLVERTAG tag)
      {
        assert( (mat.size1() == mat.size2()) && bool("Size check failed for triangular solve on transposed compressed matrix: size1(mat) != size2(mat)"));
        assert( (mat.size1() == vec.size())  && bool("Size check failed for transposed compressed matrix triangular solve: size1(mat) != size(x)"));

        switch (cuarma::traits::handle(mat.lhs()).get_active_handle_id())
        {
          case cuarma::MAIN_MEMORY:
            cuarma::blas::host_based::detail::block_inplace_solve(mat, block_index_array, num_blocks, mat_diagonal, vec, tag);
            break;
  
  #ifdef CUARMA_WITH_CUDA
          case cuarma::CUDA_MEMORY:
            cuarma::blas::cuda::detail::block_inplace_solve(mat, block_index_array, num_blocks, mat_diagonal, vec, tag);
            break;
  #endif
          case cuarma::MEMORY_NOT_INITIALIZED:
            throw memory_exception("not initialised!");
          default:
            throw memory_exception("not implemented");
        }
      }

    }

  } //namespace blas


  /** @brief Returns an expression template class representing a transposed matrix */
  template<typename M1>
  typename cuarma::enable_if<cuarma::is_any_sparse_matrix<M1>::value,
                                matrix_expression< const M1, const M1, op_trans>
                              >::type
  trans(const M1 & mat)
  {
    return matrix_expression< const M1, const M1, op_trans>(mat, mat);
  }

  //free functions:
  /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
  *
  * @param result The vector the result is written to.
  * @param proxy  An expression template proxy class holding v1, A, and v2.
  */
  template<typename SCALARTYPE, typename SparseMatrixType>
  typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value,
                                cuarma::vector<SCALARTYPE> >::type
  operator+(cuarma::vector_base<SCALARTYPE> & result,
            const cuarma::vector_expression< const SparseMatrixType, const cuarma::vector_base<SCALARTYPE>, cuarma::op_prod> & proxy)
  {
    assert(proxy.lhs().size1() == result.size() && bool("Dimensions for addition of sparse matrix-vector product to vector don't match!"));
    vector<SCALARTYPE> temp(proxy.lhs().size1());
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), temp);
    result += temp;
    return result;
  }

  /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
  *
  * @param result The vector the result is written to.
  * @param proxy  An expression template proxy class.
  */
  template<typename SCALARTYPE, typename SparseMatrixType>
  typename cuarma::enable_if< cuarma::is_any_sparse_matrix<SparseMatrixType>::value,
                                cuarma::vector<SCALARTYPE> >::type
  operator-(cuarma::vector_base<SCALARTYPE> & result,
            const cuarma::vector_expression< const SparseMatrixType, const cuarma::vector_base<SCALARTYPE>, cuarma::op_prod> & proxy)
  {
    assert(proxy.lhs().size1() == result.size() && bool("Dimensions for addition of sparse matrix-vector product to vector don't match!"));
    vector<SCALARTYPE> temp(proxy.lhs().size1());
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), temp);
    result += temp;
    return result;
  }

} //namespace cuarma
