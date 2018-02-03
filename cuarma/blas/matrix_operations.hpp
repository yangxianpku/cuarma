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

/** @file cuarma/blas/matrix_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of dense matrix related operations including matrix-vector products.
*/

#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/vector_proxy.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/meta/result_of.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/start.hpp"
#include "cuarma/traits/handle.hpp"
#include "cuarma/traits/stride.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/blas/host_based/matrix_operations.hpp"

#ifdef CUARMA_WITH_CUDA
  #include "cuarma/blas/cuda/matrix_operations.hpp"
#endif

namespace cuarma
{
  namespace blas
  {

    template<typename DestNumericT, typename SrcNumericT>
    void convert(matrix_base<DestNumericT> & dest, matrix_base<SrcNumericT> const & src)
    {
      assert(cuarma::traits::size1(dest) == cuarma::traits::size1(src) && bool("Incompatible matrix sizes in m1 = m2 (convert): size1(m1) != size1(m2)"));
      assert(cuarma::traits::size2(dest) == cuarma::traits::size2(src) && bool("Incompatible matrix sizes in m1 = m2 (convert): size2(m1) != size2(m2)"));

      switch (cuarma::traits::handle(dest).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::convert(dest, src);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::convert(dest, src);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    template<typename NumericT, typename SizeT, typename DistanceT>
    void trans(const matrix_expression<const matrix_base<NumericT, SizeT, DistanceT>,const matrix_base<NumericT, SizeT, DistanceT>, op_trans> & proxy,
              matrix_base<NumericT> & temp_trans)
    {
      switch (cuarma::traits::handle(proxy).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::trans(proxy, temp_trans);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::trans(proxy,temp_trans);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT, typename ScalarType1>
    void am(matrix_base<NumericT> & mat1, matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
    {
      switch (cuarma::traits::handle(mat1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::am(mat1, mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::am(mat1, mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT, typename ScalarType1, typename ScalarType2>
    void ambm(matrix_base<NumericT> & mat1,
              matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
              matrix_base<NumericT> const & mat3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      switch (cuarma::traits::handle(mat1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::ambm(mat1,
                                             mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                             mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::ambm(mat1,
                                       mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                       mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT, typename ScalarType1, typename ScalarType2>
    void ambm_m(matrix_base<NumericT> & mat1,
                matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                matrix_base<NumericT> const & mat3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      switch (cuarma::traits::handle(mat1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::ambm_m(mat1,
                                               mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                               mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::ambm_m(mat1,
                                         mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT>
    void matrix_assign(matrix_base<NumericT> & mat, NumericT s, bool clear = false)
    {
      switch (cuarma::traits::handle(mat).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::matrix_assign(mat, s, clear);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::matrix_assign(mat, s, clear);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT>
    void matrix_diagonal_assign(matrix_base<NumericT> & mat, NumericT s)
    {
      switch (cuarma::traits::handle(mat).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::matrix_diagonal_assign(mat, s);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::matrix_diagonal_assign(mat, s);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Dispatcher interface for A = diag(v, k) */
    template<typename NumericT>
    void matrix_diag_from_vector(const vector_base<NumericT> & v, int k, matrix_base<NumericT> & A)
    {
      switch (cuarma::traits::handle(v).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::matrix_diag_from_vector(v, k, A);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::matrix_diag_from_vector(v, k, A);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Dispatcher interface for v = diag(A, k) */
    template<typename NumericT>
    void matrix_diag_to_vector(const matrix_base<NumericT> & A, int k, vector_base<NumericT> & v)
    {
      switch (cuarma::traits::handle(A).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::matrix_diag_to_vector(A, k, v);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::matrix_diag_to_vector(A, k, v);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    template<typename NumericT>
    void matrix_row(const matrix_base<NumericT> & A, unsigned int i, vector_base<NumericT> & v)
    {
      switch (cuarma::traits::handle(A).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::matrix_row(A, i, v);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::matrix_row(A, i, v);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    template<typename NumericT>
    void matrix_column(const matrix_base<NumericT> & A, unsigned int j, vector_base<NumericT> & v)
    {
      switch (cuarma::traits::handle(A).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::matrix_column(A, j, v);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::matrix_column(A, j, v);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the Frobenius norm of a matrix - dispatcher interface
    *
    * @param A      The matrix
    * @param result The result scalar
    *
    * Note that if A is strided or off-set, then a copy will be created.
    */
    template<typename T>
    void norm_frobenius_impl(matrix_base<T> const & A, scalar<T> & result)
    {
      typedef typename matrix_base<T>::handle_type  HandleType;

      if ((A.start1() > 0) || (A.start2() > 0) || (A.stride1() > 1) || (A.stride2() > 1)) {
        if (A.row_major()) {
          cuarma::matrix<T, cuarma::row_major> temp_A(A);
          cuarma::vector_base<T> temp(const_cast<HandleType &>(temp_A.handle()), temp_A.internal_size(), 0, 1);
          norm_2_impl(temp, result);
        } else {
          cuarma::matrix<T, cuarma::column_major> temp_A(A);
          cuarma::vector_base<T> temp(const_cast<HandleType &>(temp_A.handle()), temp_A.internal_size(), 0, 1);
          norm_2_impl(temp, result);
        }
      } else {
        cuarma::vector_base<T> temp(const_cast<HandleType &>(A.handle()), A.internal_size(), 0, 1);
        norm_2_impl(temp, result);
      }

    }

    /** @brief Computes the Frobenius norm of a vector with final reduction on the CPU
    *
    * @param A      The matrix
    * @param result The result scalar
    *
    * Note that if A is strided or off-set, then a copy will be created.
    */
    template<typename T>
    void norm_frobenius_cpu(matrix_base<T> const & A, T & result)
    {
      typedef typename matrix_base<T>::handle_type  HandleType;

      if ((A.start1() > 0) || (A.start2() > 0) || (A.stride1() > 1) || (A.stride2() > 1)) {
        if (A.row_major()) {
          cuarma::matrix<T, cuarma::row_major> temp_A(A);
          cuarma::vector_base<T> temp(const_cast<HandleType &>(temp_A.handle()), temp_A.internal_size(), 0, 1);
          norm_2_cpu(temp, result);
        } else {
          cuarma::matrix<T, cuarma::column_major> temp_A(A);
          cuarma::vector_base<T> temp(const_cast<HandleType &>(temp_A.handle()), temp_A.internal_size(), 0, 1);
          norm_2_cpu(temp, result);
        }
      } else {
        cuarma::vector_base<T> temp(const_cast<HandleType &>(A.handle()), A.internal_size(), 0, 1);
        norm_2_cpu(temp, result);
      }

    }

    //
    /////////////////////////   matrix-vector products /////////////////////////////////
    //
    // A * x

    /** @brief Carries out matrix-vector multiplication
    *
    * Implementation of the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    */
    template<typename NumericT>
    void prod_impl(const matrix_base<NumericT> & mat,
                   const vector_base<NumericT> & vec,
                         vector_base<NumericT> & result)
    {
      assert( (cuarma::traits::size1(mat) == cuarma::traits::size(result)) && bool("Size check failed at v1 = prod(A, v2): size1(A) != size(v1)"));
      assert( (cuarma::traits::size2(mat) == cuarma::traits::size(vec))    && bool("Size check failed at v1 = prod(A, v2): size2(A) != size(v2)"));

      switch (cuarma::traits::handle(mat).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::prod_impl(mat, false, vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(mat, false, vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    // trans(A) * x

    /** @brief Carries out matrix-vector multiplication with a transposed matrix
    *
    * Implementation of the convenience expression result = trans(mat) * vec;
    *
    * @param mat_trans  The transposed matrix proxy
    * @param vec        The vector
    * @param result     The result vector
    */
    template<typename NumericT>
    void prod_impl(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & mat_trans,
                   const vector_base<NumericT> & vec,
                         vector_base<NumericT> & result)
    {
      assert( (cuarma::traits::size1(mat_trans.lhs()) == cuarma::traits::size(vec))    && bool("Size check failed at v1 = trans(A) * v2: size1(A) != size(v2)"));
      assert( (cuarma::traits::size2(mat_trans.lhs()) == cuarma::traits::size(result)) && bool("Size check failed at v1 = trans(A) * v2: size2(A) != size(v1)"));

      switch (cuarma::traits::handle(mat_trans.lhs()).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::prod_impl(mat_trans.lhs(), true, vec, result);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(mat_trans.lhs(), true, vec, result);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    //
    /////////////////////////   matrix-matrix products /////////////////////////////////
    //

    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(A, B);
    *
    */
    template<typename NumericT, typename ScalarType >
    void prod_impl(const matrix_base<NumericT> & A,
                   const matrix_base<NumericT> & B,
                         matrix_base<NumericT> & C,
                   ScalarType alpha,
                   ScalarType beta)
    {
      assert( (cuarma::traits::size1(A) == cuarma::traits::size1(C)) && bool("Size check failed at C = prod(A, B): size1(A) != size1(C)"));
      assert( (cuarma::traits::size2(A) == cuarma::traits::size1(B)) && bool("Size check failed at C = prod(A, B): size2(A) != size1(B)"));
      assert( (cuarma::traits::size2(B) == cuarma::traits::size2(C)) && bool("Size check failed at C = prod(A, B): size2(B) != size2(C)"));


      switch (cuarma::traits::handle(A).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::prod_impl(A, false, B, false, C, alpha, beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(A, false, B, false, C, alpha, beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }



    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(trans(A), B);
    *
    */
    template<typename NumericT, typename ScalarType >
    void prod_impl(const cuarma::matrix_expression< const matrix_base<NumericT>,
                                                      const matrix_base<NumericT>,
                                                      op_trans> & A,
                   const matrix_base<NumericT> & B,
                         matrix_base<NumericT> & C,
                   ScalarType alpha,
                   ScalarType beta)
    {
      assert(cuarma::traits::size2(A.lhs()) == cuarma::traits::size1(C) && bool("Size check failed at C = prod(trans(A), B): size2(A) != size1(C)"));
      assert(cuarma::traits::size1(A.lhs()) == cuarma::traits::size1(B) && bool("Size check failed at C = prod(trans(A), B): size1(A) != size1(B)"));
      assert(cuarma::traits::size2(B)       == cuarma::traits::size2(C) && bool("Size check failed at C = prod(trans(A), B): size2(B) != size2(C)"));

      switch (cuarma::traits::handle(A.lhs()).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::prod_impl(A.lhs(), true, B, false, C, alpha, beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(A.lhs(), true, B, false, C, alpha, beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(A, trans(B));
    *
    */
    template<typename NumericT, typename ScalarType >
    void prod_impl(const matrix_base<NumericT> & A,
                   const cuarma::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & B,
                         matrix_base<NumericT> & C,
                         ScalarType alpha,
                         ScalarType beta)
    {
      assert(cuarma::traits::size1(A)       == cuarma::traits::size1(C)       && bool("Size check failed at C = prod(A, trans(B)): size1(A) != size1(C)"));
      assert(cuarma::traits::size2(A)       == cuarma::traits::size2(B.lhs()) && bool("Size check failed at C = prod(A, trans(B)): size2(A) != size2(B)"));
      assert(cuarma::traits::size1(B.lhs()) == cuarma::traits::size2(C)       && bool("Size check failed at C = prod(A, trans(B)): size1(B) != size2(C)"));

      switch (cuarma::traits::handle(A).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::prod_impl(A, false, B.lhs(), true, C, alpha, beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(A, false, B.lhs(), true, C, alpha, beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }



    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(trans(A), trans(B));
    *
    */
    template<typename NumericT, typename ScalarType >
    void prod_impl(const cuarma::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & A,
                   const cuarma::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & B,
                   matrix_base<NumericT> & C,
                   ScalarType alpha,
                   ScalarType beta)
    {
      assert(cuarma::traits::size2(A.lhs()) == cuarma::traits::size1(C)       && bool("Size check failed at C = prod(trans(A), trans(B)): size2(A) != size1(C)"));
      assert(cuarma::traits::size1(A.lhs()) == cuarma::traits::size2(B.lhs()) && bool("Size check failed at C = prod(trans(A), trans(B)): size1(A) != size2(B)"));
      assert(cuarma::traits::size1(B.lhs()) == cuarma::traits::size2(C)       && bool("Size check failed at C = prod(trans(A), trans(B)): size1(B) != size2(C)"));

      switch (cuarma::traits::handle(A.lhs()).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::prod_impl(A.lhs(), true, B.lhs(), true, C, alpha, beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::prod_impl(A.lhs(), true, B.lhs(), true, C, alpha, beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    ///////////////////////// summation operations /////////////

    template<typename NumericT>
    void row_sum_impl(matrix_base<NumericT> const & A, vector_base<NumericT> & result)
    {
      cuarma::vector<NumericT> all_ones = cuarma::scalar_vector<NumericT>(A.size2(), NumericT(1), cuarma::traits::context(A));
      cuarma::blas::prod_impl(A, all_ones, result);
    }

    template<typename NumericT>
    void column_sum_impl(matrix_base<NumericT> const & A, vector_base<NumericT> & result)
    {
      cuarma::vector<NumericT> all_ones = cuarma::scalar_vector<NumericT>(A.size1(), NumericT(1), cuarma::traits::context(A));
      cuarma::blas::prod_impl(matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>(A, A), all_ones, result);
    }

    ///////////////////////// Elementwise operations /////////////



    /** @brief Implementation of the element-wise operation A = B .* C and A = B ./ C for matrices (using MATLAB syntax). Don't use this function directly, use element_prod() and element_div().
    *
    * @param A      The result matrix (or -range, or -slice)
    * @param proxy  The proxy object holding B, C, and the operation
    */
    template<typename T, typename OP>
    void element_op(matrix_base<T> & A, matrix_expression<const matrix_base<T>, const matrix_base<T>, OP> const & proxy)
    {
      assert( (cuarma::traits::size1(A) == cuarma::traits::size1(proxy)) && bool("Size check failed at A = element_op(B): size1(A) != size1(B)"));
      assert( (cuarma::traits::size2(A) == cuarma::traits::size2(proxy)) && bool("Size check failed at A = element_op(B): size2(A) != size2(B)"));

      switch (cuarma::traits::handle(A).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::element_op(A, proxy);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::element_op(A, proxy);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


#define CUARMA_MAKE_BINARY_OP(OPNAME)\
    template<typename T>\
    cuarma::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<op_##OPNAME> >\
    element_##OPNAME(matrix_base<T> const & A, matrix_base<T> const & B)\
    {\
      return cuarma::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<op_##OPNAME> >(A, B);\
    }\
\
    template<typename M1, typename M2, typename OP, typename T>\
    cuarma::matrix_expression<const matrix_expression<const M1, const M2, OP>,\
                                const matrix_base<T>,\
                                op_element_binary<op_##OPNAME> >\
    element_##OPNAME(matrix_expression<const M1, const M2, OP> const & proxy, matrix_base<T> const & B)\
    {\
      return cuarma::matrix_expression<const matrix_expression<const M1, const M2, OP>,\
                                         const matrix_base<T>,\
                                         op_element_binary<op_##OPNAME> >(proxy, B);\
    }\
\
    template<typename T, typename M2, typename M3, typename OP>\
    cuarma::matrix_expression<const matrix_base<T>,\
                                const matrix_expression<const M2, const M3, OP>,\
                                op_element_binary<op_##OPNAME> >\
    element_##OPNAME(matrix_base<T> const & A, matrix_expression<const M2, const M3, OP> const & proxy)\
    {\
      return cuarma::matrix_expression<const matrix_base<T>,\
                                         const matrix_expression<const M2, const M3, OP>,\
                                         op_element_binary<op_##OPNAME> >(A, proxy);\
    }\
\
    template<typename M1, typename M2, typename OP1,\
              typename M3, typename M4, typename OP2>\
    cuarma::matrix_expression<const matrix_expression<const M1, const M2, OP1>,\
                                const matrix_expression<const M3, const M4, OP2>,\
                                op_element_binary<op_##OPNAME> >\
    element_##OPNAME(matrix_expression<const M1, const M2, OP1> const & proxy1,\
                 matrix_expression<const M3, const M4, OP2> const & proxy2)\
    {\
      return cuarma::matrix_expression<const matrix_expression<const M1, const M2, OP1>,\
                                         const matrix_expression<const M3, const M4, OP2>,\
                                         op_element_binary<op_##OPNAME> >(proxy1, proxy2);\
    }

    CUARMA_MAKE_BINARY_OP(prod)
    CUARMA_MAKE_BINARY_OP(div)
    CUARMA_MAKE_BINARY_OP(pow)

    CUARMA_MAKE_BINARY_OP(eq)
    CUARMA_MAKE_BINARY_OP(neq)
    CUARMA_MAKE_BINARY_OP(greater)
    CUARMA_MAKE_BINARY_OP(less)
    CUARMA_MAKE_BINARY_OP(geq)
    CUARMA_MAKE_BINARY_OP(leq)

#undef CUARMA_GENERATE_BINARY_OP_OVERLOADS



#define CUARMA_MAKE_UNARY_ELEMENT_OP(funcname) \
    template<typename T> \
    cuarma::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<op_##funcname> > \
    element_##funcname(matrix_base<T> const & A) \
    { \
      return cuarma::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<op_##funcname> >(A, A); \
    } \
    template<typename LHS, typename RHS, typename OP> \
    cuarma::matrix_expression<const matrix_expression<const LHS, const RHS, OP>, \
                                const matrix_expression<const LHS, const RHS, OP>, \
                                op_element_unary<op_##funcname> > \
    element_##funcname(matrix_expression<const LHS, const RHS, OP> const & proxy) \
    { \
      return cuarma::matrix_expression<const matrix_expression<const LHS, const RHS, OP>, \
                                         const matrix_expression<const LHS, const RHS, OP>, \
                                         op_element_unary<op_##funcname> >(proxy, proxy); \
    } \

    CUARMA_MAKE_UNARY_ELEMENT_OP(abs)
    CUARMA_MAKE_UNARY_ELEMENT_OP(acos)
    CUARMA_MAKE_UNARY_ELEMENT_OP(asin)
    CUARMA_MAKE_UNARY_ELEMENT_OP(atan)
    CUARMA_MAKE_UNARY_ELEMENT_OP(ceil)
    CUARMA_MAKE_UNARY_ELEMENT_OP(cos)
    CUARMA_MAKE_UNARY_ELEMENT_OP(cosh)
    CUARMA_MAKE_UNARY_ELEMENT_OP(exp)
    CUARMA_MAKE_UNARY_ELEMENT_OP(fabs)
    CUARMA_MAKE_UNARY_ELEMENT_OP(floor)
    CUARMA_MAKE_UNARY_ELEMENT_OP(log)
    CUARMA_MAKE_UNARY_ELEMENT_OP(log10)
    CUARMA_MAKE_UNARY_ELEMENT_OP(sin)
    CUARMA_MAKE_UNARY_ELEMENT_OP(sinh)
    CUARMA_MAKE_UNARY_ELEMENT_OP(sqrt)
    CUARMA_MAKE_UNARY_ELEMENT_OP(tan)
    CUARMA_MAKE_UNARY_ELEMENT_OP(tanh)

#undef CUARMA_MAKE_UNARY_ELEMENT_OP


    //
    /////////////////////////   miscellaneous operations /////////////////////////////////
    //


    /** @brief Returns a proxy class for the operation mat += vec1 * vec2^T, i.e. a rank 1 update
    *
    * @param vec1    The first vector
    * @param vec2    The second vector
    */
    template<typename NumericT>
    cuarma::matrix_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_prod>
    outer_prod(const vector_base<NumericT> & vec1, const vector_base<NumericT> & vec2)
    {
      return cuarma::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>(vec1, vec2);
    }


    /** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
    *
    * Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
    *
    * @param mat1             The matrix to be updated
    * @param alpha            The scaling factor (either a cuarma::scalar<>, float, or double)
    * @param len_alpha        Length of the buffer for an eventual final reduction step (currently always '1')
    * @param reciprocal_alpha Use 1/alpha instead of alpha
    * @param flip_sign_alpha  Use -alpha instead of alpha
    * @param vec1             The first vector
    * @param vec2             The second vector
    */
    template<typename NumericT, typename S1>
    void scaled_rank_1_update(matrix_base<NumericT> & mat1,
                              S1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                              const vector_base<NumericT> & vec1,
                              const vector_base<NumericT> & vec2)
    {
      switch (cuarma::traits::handle(mat1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::scaled_rank_1_update(mat1,
                                                             alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                             vec1, vec2);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::scaled_rank_1_update(mat1,
                                                       alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                       vec1, vec2);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief This function stores the diagonal and the superdiagonal of a matrix in two vectors.
    *
    *
    * @param A     The matrix from which the vectors will be extracted of.
    * @param dh    The vector in which the diagonal of the matrix will be stored in.
    * @param sh    The vector in which the superdiagonal of the matrix will be stored in.
    */
    template <typename NumericT, typename VectorType>
    void bidiag_pack(matrix_base<NumericT> & A,
                     VectorType & dh,
                     VectorType & sh
                    )
    {
      switch (cuarma::traits::handle(A).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::bidiag_pack(A, dh, sh);
          break;


#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::bidiag_pack(A, dh, sh);
          break;
#endif

        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }


    }
    /** @brief This function copies a row or a column from a matrix to a vector.
    *
    *
    * @param A          The matrix where to copy from.
    * @param V          The vector to fill with data.
    * @param row_start  The number of the first row to copy.
    * @param col_start  The number of the first column to copy.
    * @param copy_col   Set to TRUE to copy a column, FALSE to copy a row.
    */

    template <typename SCALARTYPE>
    void copy_vec(matrix_base<SCALARTYPE>& A,
                  vector_base<SCALARTYPE>& V,
                  arma_size_t row_start,
                  arma_size_t col_start,
                  bool copy_col
    )
    {
      switch (cuarma::traits::handle(A).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::copy_vec(A, V, row_start, col_start, copy_col);
          break;


#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::copy_vec(A, V, row_start, col_start, copy_col);
          break;
#endif

        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }

    }

    /** @brief This function applies a householder transformation to a matrix. A <- P * A with a householder reflection P
    *
    * @param A       The matrix to be updated.
    * @param D       The normalized householder vector.
    * @param start   The repetition counter.
    */
  template <typename NumericT>
  void house_update_A_left(matrix_base<NumericT> & A,
                           vector_base<NumericT>    & D,
                           arma_size_t start)
  {
    switch (cuarma::traits::handle(A).get_active_handle_id())
    {
      case cuarma::MAIN_MEMORY:
        cuarma::blas::host_based::house_update_A_left(A, D, start);
        break;


#ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::house_update_A_left(A, D, start);
        break;
#endif

      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }


  /** @brief This function applies a householder transformation to a matrix: A <- A * P with a householder reflection P
  *
  *
  * @param A        The matrix to be updated.
  * @param D        The normalized householder vector.
  */

  template <typename NumericT>
  void house_update_A_right(matrix_base<NumericT>& A, vector_base<NumericT>   & D)
  {
    switch (cuarma::traits::handle(A).get_active_handle_id())
    {
      case cuarma::MAIN_MEMORY:
        cuarma::blas::host_based::house_update_A_right(A, D);
        break;
#ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::house_update_A_right(A, D);
        break;
#endif
      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }

  /** @brief This function updates the matrix Q, which is needed for the computation of the eigenvectors.
  *
  * @param Q        The matrix to be updated.
  * @param D        The householder vector.
  * @param A_size1  size1 of matrix A
  */

  template <typename NumericT>
  void house_update_QL(matrix_base<NumericT> & Q,
                       vector_base<NumericT>    & D,
                       arma_size_t A_size1)
  {
    switch (cuarma::traits::handle(Q).get_active_handle_id())
    {
      case cuarma::MAIN_MEMORY:
        cuarma::blas::host_based::house_update_QL(Q, D, A_size1);
        break;


#ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::house_update_QL(Q, D, A_size1);
        break;
#endif

      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }


  /** @brief This function updates the matrix Q. It is part of the tql2 algorithm.
  * @param Q       The matrix to be updated.
  * @param tmp1    Vector with data from the tql2 algorithm.
  * @param tmp2    Vector with data from the tql2 algorithm.
  * @param l       Data from the tql2 algorithm.
  * @param m       Data from the tql2 algorithm.
  */
  template<typename NumericT>
  void givens_next(matrix_base<NumericT> & Q,
                   vector_base<NumericT> & tmp1,
                   vector_base<NumericT> & tmp2,
                   int l,
                   int m)
  {
    switch (cuarma::traits::handle(Q).get_active_handle_id())
    {
      case cuarma::MAIN_MEMORY:
        cuarma::blas::host_based::givens_next(Q, tmp1, tmp2, l, m);
        break;
#ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::givens_next(Q, tmp1, tmp2, l, m);
        break;
#endif
      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }

  } //namespace blas

  //
  /////////////////////////  Operator overloads /////////////////////////////////
  //
  //v += A * x
  /** @brief Implementation of the operation v1 += A * v2, where A is a matrix
  *
  * @param v1     The result vector v1 where A * v2 is added to
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator+=(vector_base<NumericT> & v1,
             const cuarma::vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, cuarma::op_prod> & proxy)
  {
    assert(cuarma::traits::size1(proxy.lhs()) == v1.size() && bool("Size check failed for v1 += A * v2: size1(A) != size(v1)"));

    vector<NumericT> result(cuarma::traits::size1(proxy.lhs()));
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 += result;
    return v1;
  }

  /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
  *
  * @param v1     The result vector v1 where A * v2 is subtracted from
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator-=(vector_base<NumericT> & v1,
             const cuarma::vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, cuarma::op_prod> & proxy)
  {
    assert(cuarma::traits::size1(proxy.lhs()) == v1.size() && bool("Size check failed for v1 -= A * v2: size1(A) != size(v1)"));

    vector<NumericT> result(cuarma::traits::size1(proxy.lhs()));
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 -= result;
    return v1;
  }


  //free functions:
  /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  cuarma::vector<NumericT>
  operator+(const vector_base<NumericT> & v1,
            const vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, op_prod> & proxy)
  {
    assert(cuarma::traits::size1(proxy.lhs()) == cuarma::traits::size(v1) && bool("Size check failed for v1 + A * v2: size1(A) != size(v1)"));

    vector<NumericT> result(cuarma::traits::size(v1));
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result += v1;
    return result;
  }

  /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  cuarma::vector<NumericT>
  operator-(const vector_base<NumericT> & v1,
            const vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, op_prod> & proxy)
  {
    assert(cuarma::traits::size1(proxy.lhs()) == cuarma::traits::size(v1) && bool("Size check failed for v1 - A * v2: size1(A) != size(v1)"));

    vector<NumericT> result(cuarma::traits::size(v1));
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result = v1 - result;
    return result;
  }


  ////////// transposed_matrix_proxy


  //v += A^T * x
  /** @brief Implementation of the operation v1 += A * v2, where A is a matrix
  *
  * @param v1     The addend vector where the result is written to.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator+=(vector_base<NumericT> & v1,
             const vector_expression< const matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>,
                                                              const vector_base<NumericT>,
                                                              op_prod> & proxy)
  {
    assert(cuarma::traits::size2(proxy.lhs()) == v1.size() && bool("Size check failed in v1 += trans(A) * v2: size2(A) != size(v1)"));

    vector<NumericT> result(cuarma::traits::size2(proxy.lhs()));
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 += result;
    return v1;
  }

  //v -= A^T * x
  /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
  *
  * @param v1     The addend vector where the result is written to.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT> operator-=(vector_base<NumericT> & v1, const vector_expression< const matrix_expression<const matrix_base<NumericT>, 
             const matrix_base<NumericT>, op_trans>, const vector_base<NumericT>,  op_prod> & proxy)
  {
    assert(cuarma::traits::size2(proxy.lhs()) == v1.size() && bool("Size check failed in v1 += trans(A) * v2: size2(A) != size(v1)"));

    vector<NumericT> result(cuarma::traits::size2(proxy.lhs()));
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 -= result;
    return v1;
  }


  //free functions:
  /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator+(const vector_base<NumericT> & v1,
            const vector_expression< const matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>,
                                     const vector_base<NumericT>, op_prod> & proxy)
  {
    assert(cuarma::traits::size2(proxy.lhs()) == cuarma::traits::size(v1) && bool("Size check failed in v1 + trans(A) * v2: size2(A) != size(v1)"));

    vector<NumericT> result(cuarma::traits::size(v1));
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result += v1;
    return result;
  }

  /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator-(const vector_base<NumericT> & v1, const vector_expression< const matrix_expression<const matrix_base<NumericT>, 
            const matrix_base<NumericT>, op_trans>, const vector_base<NumericT>,  op_prod> & proxy)
  {
    assert(cuarma::traits::size2(proxy.lhs()) == cuarma::traits::size(v1) && bool("Size check failed in v1 - trans(A) * v2: size2(A) != size(v1)"));
    vector<NumericT> result(cuarma::traits::size(v1));
    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result = v1 - result;
    return result;
  }


} //namespace cuarma
