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

/** @file cuarma/blas/detail/ilu/chow_patel_ilu.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of incomplete factorization preconditioners with fine-grained parallelism.
  Based on "Fine-Grained Parallel Incomplete LU Factorization" by Chow and Patel, SIAM J. Sci. Comput., vol. 37, no. 2, pp. C169-C193
*/

#include <vector>
#include <cmath>
#include <iostream>
#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/detail/ilu/common.hpp"
#include "cuarma/blas/ilu_operations.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/backend/memory.hpp"

namespace cuarma
{
namespace blas
{

/** @brief A tag for incomplete LU and incomplete Cholesky factorization with static pattern (Parallel-ILU0, Parallel ICC0)
*/
class chow_patel_tag
{
public:
  /** @brief Constructor allowing to set the number of sweeps and Jacobi iterations.
    *
    * @param num_sweeps        Number of sweeps in setup phase
    * @param num_jacobi_iters  Number of Jacobi iterations for each triangular 'solve' when applying the preconditioner to a vector
    */
  chow_patel_tag(arma_size_t num_sweeps = 3, arma_size_t num_jacobi_iters = 2) : sweeps_(num_sweeps), jacobi_iters_(num_jacobi_iters) {}

  /** @brief Returns the number of sweeps (i.e. number of nonlinear iterations) in the solver setup stage */
  arma_size_t sweeps() const { return sweeps_; }
  /** @brief Sets the number of sweeps (i.e. number of nonlinear iterations) in the solver setup stage */
  void       sweeps(arma_size_t num) { sweeps_ = num; }

  /** @brief Returns the number of Jacobi iterations (i.e. applications of x_{k+1} = (I - D^{-1}R)x_k + D^{-1} b) for each of the solves y = U^{-1} x and z = L^{-1} y) for each preconditioner application. */
  arma_size_t jacobi_iters() const { return jacobi_iters_; }
  /** @brief Sets the number of Jacobi iterations for each triangular 'solve' when applying the preconditioner to a vector. */
  void       jacobi_iters(arma_size_t num) { jacobi_iters_ = num; }

private:
  arma_size_t sweeps_;
  arma_size_t jacobi_iters_;
};

namespace detail
{
  /** @brief Implementation of the parallel ICC0 factorization, Algorithm 3 in Chow-Patel paper.
   *
   *  Rather than dealing with a column-major upper triangular matrix U, we use the lower-triangular matrix L such that A is approximately given by LL^T.
   *  The advantage is that L is readily available in row-major format.
   */
  template<typename NumericT>
  void precondition(cuarma::compressed_matrix<NumericT> const & A,
                    cuarma::compressed_matrix<NumericT>       & L,
                    cuarma::vector<NumericT>                  & diag_L,
                    cuarma::compressed_matrix<NumericT>       & L_trans,
                    chow_patel_tag const & tag)
  {
    // make sure L and U have correct dimensions:
    L.resize(A.size1(), A.size2(), false);

    // initialize L and U from values in A:
    cuarma::blas::extract_L(A, L);

    // diagonally scale values from A in L:
    cuarma::blas::icc_scale(A, L);

    cuarma::vector<NumericT> aij_L(L.nnz(), cuarma::traits::context(A));
    cuarma::backend::memory_copy(L.handle(), aij_L.handle(), 0, 0, sizeof(NumericT) * L.nnz());

    // run sweeps:
    for (arma_size_t i=0; i<tag.sweeps(); ++i)
      cuarma::blas::icc_chow_patel_sweep(L, aij_L);

    // transpose L to obtain L_trans:
    cuarma::blas::ilu_transpose(L, L_trans);

    // form (I - D_L^{-1}L) and (I - D_U^{-1} U), with U := L_trans
    cuarma::blas::ilu_form_neumann_matrix(L,       diag_L);
    cuarma::blas::ilu_form_neumann_matrix(L_trans, diag_L);
  }


  /** @brief Implementation of the parallel ILU0 factorization, Algorithm 2 in Chow-Patel paper. */
  template<typename NumericT>
  void precondition(cuarma::compressed_matrix<NumericT> const & A,
                    cuarma::compressed_matrix<NumericT>       & L,
                    cuarma::vector<NumericT>                  & diag_L,
                    cuarma::compressed_matrix<NumericT>       & U,
                    cuarma::vector<NumericT>                  & diag_U,
                    chow_patel_tag const & tag)
  {
    // make sure L and U have correct dimensions:
    L.resize(A.size1(), A.size2(), false);
    U.resize(A.size1(), A.size2(), false);

    // initialize L and U from values in A:
    cuarma::blas::extract_LU(A, L, U);

    // diagonally scale values from A in L and U:
    cuarma::blas::ilu_scale(A, L, U);

    // transpose storage layout of U from CSR to CSC via transposition
    cuarma::compressed_matrix<NumericT> U_trans;
    cuarma::blas::ilu_transpose(U, U_trans);

    // keep entries of a_ij for the sweeps
    cuarma::vector<NumericT> aij_L      (L.nnz(),       cuarma::traits::context(A));
    cuarma::vector<NumericT> aij_U_trans(U_trans.nnz(), cuarma::traits::context(A));

    cuarma::backend::memory_copy(      L.handle(), aij_L.handle(),       0, 0, sizeof(NumericT) * L.nnz());
    cuarma::backend::memory_copy(U_trans.handle(), aij_U_trans.handle(), 0, 0, sizeof(NumericT) * U_trans.nnz());

    // run sweeps:
    for (arma_size_t i=0; i<tag.sweeps(); ++i)
      cuarma::blas::ilu_chow_patel_sweep(L, aij_L, U_trans, aij_U_trans);

    // transpose U_trans back:
    cuarma::blas::ilu_transpose(U_trans, U);

    // form (I - D_L^{-1}L) and (I - D_U^{-1} U)
    cuarma::blas::ilu_form_neumann_matrix(L, diag_L);
    cuarma::blas::ilu_form_neumann_matrix(U, diag_U);
  }

}




/** @brief Parallel Chow-Patel ILU preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class chow_patel_icc_precond
{
  // only works with compressed_matrix!
  typedef typename MatrixT::CHOW_PATEL_ICC_ONLY_WORKS_WITH_COMPRESSED_MATRIX  error_type;
};


/** @brief Parallel Chow-Patel ILU preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class chow_patel_icc_precond< cuarma::compressed_matrix<NumericT, AlignmentV> >
{

public:
  chow_patel_icc_precond(cuarma::compressed_matrix<NumericT, AlignmentV> const & A, chow_patel_tag const & tag)
    : tag_(tag),
      L_(0, 0, 0, cuarma::traits::context(A)),
      diag_L_(A.size1(), cuarma::traits::context(A)),
      L_trans_(0, 0, 0, cuarma::traits::context(A)),
      x_k_(A.size1(), cuarma::traits::context(A)),
      b_(A.size1(), cuarma::traits::context(A))
  {
    cuarma::blas::detail::precondition(A, L_, diag_L_, L_trans_, tag_);
  }

  /** @brief Preconditioner application: LL^Tx = b, computed via Ly = b, L^Tx = y using Jacobi iterations.
    *
    * L contains (I - D_L^{-1}L), L_trans contains (I - D_L^{-1}L^T) where D denotes the respective diagonal matrix
    */
  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    //
    // y = L^{-1} b through Jacobi iteration y_{k+1} = (I - D^{-1}L)y_k + D^{-1}x
    //
    b_ = cuarma::blas::element_div(vec, diag_L_);
    x_k_ = b_;
    for (unsigned int i=0; i<tag_.jacobi_iters(); ++i)
    {
      vec = cuarma::blas::prod(L_, x_k_);
      x_k_ = vec + b_;
    }

    //
    // x = U^{-1} y through Jacobi iteration x_{k+1} = (I - D^{-1}L^T)x_k + D^{-1}b
    //
    b_ = cuarma::blas::element_div(x_k_, diag_L_);
    x_k_ = b_; // x_1 if x_0 \equiv 0
    for (unsigned int i=0; i<tag_.jacobi_iters(); ++i)
    {
      vec = cuarma::blas::prod(L_trans_, x_k_);
      x_k_ = vec + b_;
    }

    // return result:
    vec = x_k_;
  }

private:
  chow_patel_tag                          tag_;
  cuarma::compressed_matrix<NumericT>   L_;
  cuarma::vector<NumericT>              diag_L_;
  cuarma::compressed_matrix<NumericT>   L_trans_;

  mutable cuarma::vector<NumericT>      x_k_;
  mutable cuarma::vector<NumericT>      b_;
};






/** @brief Parallel Chow-Patel ILU preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class chow_patel_ilu_precond
{
  // only works with compressed_matrix!
  typedef typename MatrixT::CHOW_PATEL_ILU_ONLY_WORKS_WITH_COMPRESSED_MATRIX  error_type;
};


/** @brief Parallel Chow-Patel ILU preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class chow_patel_ilu_precond< cuarma::compressed_matrix<NumericT, AlignmentV> >
{

public:
  chow_patel_ilu_precond(cuarma::compressed_matrix<NumericT, AlignmentV> const & A, chow_patel_tag const & tag)
    : tag_(tag),
      L_(0, 0, 0, cuarma::traits::context(A)),
      diag_L_(A.size1(), cuarma::traits::context(A)),
      U_(0, 0, 0, cuarma::traits::context(A)),
      diag_U_(A.size1(), cuarma::traits::context(A)),
      x_k_(A.size1(), cuarma::traits::context(A)),
      b_(A.size1(), cuarma::traits::context(A))
  {
    cuarma::blas::detail::precondition(A, L_, diag_L_, U_, diag_U_, tag_);
  }

  /** @brief Preconditioner application: LUx = b, computed via Ly = b, Ux = y using Jacobi iterations.
    *
    * L_ contains (I - D_L^{-1}L), U_ contains (I - D_U^{-1}U) where D denotes the respective diagonal matrix
    */
  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    //
    // y = L^{-1} b through Jacobi iteration y_{k+1} = (I - D^{-1}L)y_k + D^{-1}x
    //
    b_ = cuarma::blas::element_div(vec, diag_L_);
    x_k_ = b_;
    for (unsigned int i=0; i<tag_.jacobi_iters(); ++i)
    {
      vec = cuarma::blas::prod(L_, x_k_);
      x_k_ = vec + b_;
    }

    //
    // x = U^{-1} y through Jacobi iteration x_{k+1} = (I - D^{-1}U)x_k + D^{-1}b
    //
    b_ = cuarma::blas::element_div(x_k_, diag_U_);
    x_k_ = b_; // x_1 if x_0 \equiv 0
    for (unsigned int i=0; i<tag_.jacobi_iters(); ++i)
    {
      vec = cuarma::blas::prod(U_, x_k_);
      x_k_ = vec + b_;
    }

    // return result:
    vec = x_k_;
  }

private:
  chow_patel_tag                        tag_;
  cuarma::compressed_matrix<NumericT>   L_;
  cuarma::vector<NumericT>              diag_L_;
  cuarma::compressed_matrix<NumericT>   U_;
  cuarma::vector<NumericT>              diag_U_;

  mutable cuarma::vector<NumericT>      x_k_;
  mutable cuarma::vector<NumericT>      b_;
};


} // namespace blas
} // namespace cuarma