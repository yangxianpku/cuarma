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

/** @file cuarma/blas/ilu_operations.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief Implementations of specialized routines for the Chow-Patel parallel ILU preconditioner
*/

#include "cuarma/forwards.h"
#include "cuarma/range.hpp"
#include "cuarma/scalar.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/start.hpp"
#include "cuarma/traits/handle.hpp"
#include "cuarma/traits/stride.hpp"
#include "cuarma/blas/host_based/ilu_operations.hpp"

#ifdef CUARMA_WITH_CUDA
  #include "cuarma/blas/cuda/ilu_operations.hpp"
#endif

namespace cuarma
{
namespace blas
{

/** @brief Extracts the lower triangular part L from A.
  *
  * Diagonal of L is stored explicitly in order to enable better code reuse.
  *
  */
template<typename NumericT>
void extract_L(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  switch (cuarma::traits::handle(A).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::extract_L(A, L);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::extract_L(A, L);
    break;
#endif
  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L accordingly.
  *
  * Since A should not be modified (const-correctness), updates are in L.
  *
  */
template<typename NumericT>
void icc_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  switch (cuarma::traits::handle(A).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::icc_scale(A, L);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::icc_scale(A, L);
    break;
#endif
  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ICC (cf. Algorithm 3 in paper, but for L rather than U)
  *
  * We use a fully synchronous (Jacobi-like) variant, because asynchronous methods as described in the paper are a nightmare to debug
  * (and particularly funny if they sometimes fail, sometimes not)
  *
  * @param L       Factor L to be updated for the incomplete Cholesky factorization
  * @param aij_L   Lower triangular potion from system matrix
  */
template<typename NumericT>
void icc_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>                  & aij_L)
{
  switch (cuarma::traits::handle(L).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::icc_chow_patel_sweep(L, aij_L);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::icc_chow_patel_sweep(L, aij_L);
    break;
#endif
  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}



//////////////////////// ILU ////////////////////

/** @brief Extracts the lower triangular part L and the upper triangular part U from A.
  *
  * Diagonals of L and U are stored explicitly in order to enable better code reuse.
  *
  */
template<typename NumericT>
void extract_LU(compressed_matrix<NumericT> const & A,
                compressed_matrix<NumericT>       & L,
                compressed_matrix<NumericT>       & U)
{
  switch (cuarma::traits::handle(A).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::extract_LU(A, L, U);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::extract_LU(A, L, U);
    break;
#endif
  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly.
  * Since A should not be modified (const-correctness), updates are in L and U.
  */
template<typename NumericT>
void ilu_scale(compressed_matrix<NumericT> const & A, compressed_matrix<NumericT> & L, compressed_matrix<NumericT> & U)
{
  switch (cuarma::traits::handle(A).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::ilu_scale(A, L, U);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::ilu_scale(A, L, U);
    break;
#endif
  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Transposition B <- A^T, where the aij-vector is permuted in the same way as the value array in A when assigned to B
  * @param A     Input matrix to be transposed
  * @param B     Output matrix containing the transposed matrix
  */
template<typename NumericT>
void ilu_transpose(compressed_matrix<NumericT> const & A,
                   compressed_matrix<NumericT>       & B)
{
  cuarma::context orig_ctx = cuarma::traits::context(A);
  cuarma::context cpu_ctx(cuarma::MAIN_MEMORY);
  (void)orig_ctx;
  (void)cpu_ctx;

  cuarma::compressed_matrix<NumericT> A_host(0, 0, 0, cpu_ctx);
  (void)A_host;

  switch (cuarma::traits::handle(A).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::ilu_transpose(A, B);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    A_host = A;
    B.switch_memory_context(cpu_ctx);
    cuarma::blas::host_based::ilu_transpose(A_host, B);
    B.switch_memory_context(orig_ctx);
    break;
#endif
  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}


/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ILU (cf. Algorithm 2 in paper)
  *
  * We use a fully synchronous (Jacobi-like) variant, because asynchronous methods as described in the paper are a nightmare to debug
  * (and particularly funny if they sometimes fail, sometimes not)
  *
  * @param L            Lower-triangular matrix L in LU factorization
  * @param aij_L        Lower-triangular matrix L from A
  * @param U_trans      Upper-triangular matrix U in CSC-storage, which is the same as U^trans in CSR-storage
  * @param aij_U_trans  Upper-triangular matrix from A in CSC-storage, which is the same as U^trans in CSR-storage
  */
template<typename NumericT>
void ilu_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>            const & aij_L,
                          compressed_matrix<NumericT>       & U_trans,
                          vector<NumericT>            const & aij_U_trans)
{
  switch (cuarma::traits::handle(L).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::ilu_chow_patel_sweep(L, aij_L, U_trans, aij_U_trans);
    break;
#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::ilu_chow_patel_sweep(L, aij_L, U_trans, aij_U_trans);
    break;
#endif
  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Extracts the lower triangular part L and the upper triangular part U from A.
  *
  * Diagonals of L and U are stored explicitly in order to enable better code reuse.
  *
  */
template<typename NumericT>
void ilu_form_neumann_matrix(compressed_matrix<NumericT> & R, vector<NumericT> & diag_R)
{
  switch (cuarma::traits::handle(R).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::ilu_form_neumann_matrix(R, diag_R);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::ilu_form_neumann_matrix(R, diag_R);
    break;
#endif
  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

} //namespace blas
} //namespace cuarma
