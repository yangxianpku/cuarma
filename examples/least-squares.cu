/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file  least-squares.cu
 *   @coding UTF-8
 *   @brief  This tutorial shows how least Squares problems for matrices from cuarma or Boost.uBLAS can be solved solved.
 *   @brief  测试：最小二乘问题
 */
 

#define CUARMA_WITH_UBLAS

#include <iostream>

#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "head_define.h"

#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/matrix_proxy.hpp"
#include "cuarma/blas/qr.hpp"
#include "cuarma/blas/lu.hpp"
#include "cuarma/blas/direct_solve.hpp"


/**
*  The minimization problem of finding x such that ||Ax-b|| is solved as follows:
*   - Compute the QR-factorization of A = QR.
*   - Compute \f$ b' = Q^{\mathrm{T}} b \f$ for the equivalent minimization problem \f$ \Vert Rx - Q^{\mathrm{T}} b \f$.
*   - Solve the triangular system \f$ \tilde{R} x = b' \f$, where \f$ \tilde{R} \f$ is the upper square matrix of R.
**/

int main (int, const char **)
{

  typedef boost::numeric::ublas::matrix<ScalarType>              MatrixType;
  typedef boost::numeric::ublas::vector<ScalarType>              VectorType;

  typedef cuarma::matrix<ScalarType, cuarma::column_major>       ARMAMatrixType;
  typedef cuarma::vector<ScalarType>                             ARMAVectorType;

  /** Create vectors and matrices with data: **/
  VectorType ublas_b(4);

  ublas_b(0) = -4;
  ublas_b(1) =  2;
  ublas_b(2) =  5;
  ublas_b(3) = -1;

  MatrixType ublas_A(4, 3);

  ublas_A(0, 0) =  2; ublas_A(0, 1) = -1; ublas_A(0, 2) =  1;
  ublas_A(1, 0) =  1; ublas_A(1, 1) = -5; ublas_A(1, 2) =  2;
  ublas_A(2, 0) = -3; ublas_A(2, 1) =  1; ublas_A(2, 2) = -4;
  ublas_A(3, 0) =  1; ublas_A(3, 1) = -1; ublas_A(3, 2) =  1;

  /** Setup the matrix and vector with cuarma objects and copy the data from the uBLAS objects: **/
  ARMAVectorType arma_b(ublas_b.size());
  ARMAMatrixType arma_A(ublas_A.size1(), ublas_A.size2());

  cuarma::copy(ublas_b, arma_b);
  cuarma::copy(ublas_A, arma_A);


  std::cout << "--- Boost.uBLAS ---" << std::endl;
  /**
  * The first (and computationally most expensive) step is to compute the QR factorization of A.
  * Since we do not need A later, we directly overwrite A with the householder reflectors and the upper triangular matrix R.
  * The returned vector holds the scalar coefficients (betas) for the Householder reflections \f$ I - \beta v v^{\mathrm{T}} \f$
  **/
  std::vector<ScalarType> ublas_betas = cuarma::blas::inplace_qr(ublas_A);

  /**
  * Compute the modified RHS of the minimization problem from the QR factorization, but do not form \f$ Q^{\mathrm{T}} \f$ explicitly:
  * b' := Q^T b
  **/
  cuarma::blas::inplace_qr_apply_trans_Q(ublas_A, ublas_betas, ublas_b);

  /**
  * Final step: triangular solve: Rx = b'', where b'' are the first three entries in b'
  * We only need the upper left square part of A, which defines the upper triangular matrix R
  **/
  boost::numeric::ublas::range ublas_range(0, 3);
  boost::numeric::ublas::matrix_range<MatrixType> ublas_R(ublas_A, ublas_range, ublas_range);
  boost::numeric::ublas::vector_range<VectorType> ublas_b2(ublas_b, ublas_range);
  boost::numeric::ublas::inplace_solve(ublas_R, ublas_b2, boost::numeric::ublas::upper_tag());

  std::cout << "Result: " << ublas_b2 << std::endl;

  std::cout << "--- cuarma (hybrid implementation)  ---" << std::endl;
  std::vector<ScalarType> hybrid_betas = cuarma::blas::inplace_qr(arma_A);

  /** compute modified RHS of the minimization problem: \f$ b' := Q^T b \f$ **/
  cuarma::blas::inplace_qr_apply_trans_Q(arma_A, hybrid_betas, arma_b);

  /**
  * Final step: triangular solve: Rx = b'.
  * We only need the upper part of A such that R is a square matrix
  **/
  cuarma::range arma_range(0, 3);
  cuarma::matrix_range<ARMAMatrixType> arma_R(arma_A, arma_range, arma_range);
  cuarma::vector_range<ARMAVectorType> arma_b2(arma_b, arma_range);
  cuarma::blas::inplace_solve(arma_R, arma_b2, cuarma::blas::upper_tag());

  std::cout << "Result: " << arma_b2 << std::endl;

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

