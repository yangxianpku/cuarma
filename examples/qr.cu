/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   qr.cu
 *   @coding UTF-8
 *   @brief  This tutorial shows how the QR factorization of matrices from cuarma or Boost.uBLAS can be computed.
 *   @brief  测试：qr分解
 */

#define CUARMA_WITH_UBLAS
#include <iostream>

#include "head_define.h"

#include "cuarma/matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/qr.hpp"

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

/** 辅助函数：比较两个矩阵元素的最大相对误差 **/
template<typename MatrixType>
double check(MatrixType const & qr, MatrixType const & ref)
{
  bool do_break = false;
  double max_error = 0;
  for (std::size_t i=0; i<ref.size1(); ++i)
  {
    for (std::size_t j=0; j<ref.size2(); ++j)
    {
      if (qr(i,j) != 0.0 && ref(i,j) != 0.0)
      {
        double rel_err = fabs(qr(i,j) - ref(i,j)) / fabs(ref(i,j) );

        if (rel_err > max_error)
          max_error = rel_err;
      }
    }
    if (do_break)
      break;
  }
  return max_error;
}


int main (int, const char **)
{
  typedef boost::numeric::ublas::matrix<ScalarType>          MatrixType;
  typedef cuarma::matrix<ScalarType, cuarma::column_major>   ARMAMatrixType;

  std::size_t rows = 113;   // number of rows in the matrix
  std::size_t cols = 54;    // number of columns

  MatrixType ublas_A(rows, cols);
  MatrixType Q(rows, rows);
  MatrixType R(rows, cols);

  for (std::size_t i=0; i<rows; ++i)
  {
    for (std::size_t j=0; j<cols; ++j)
    {
      ublas_A(i,j) = ScalarType(-1.0) + ScalarType((i+1)*(j+1)) + ScalarType( (rand() % 1000) - 500.0) / ScalarType(1000.0);

      if (i == j)
        ublas_A(i,j) += ScalarType(10.0);

      R(i,j) = 0.0;
    }

    for (std::size_t j=0; j<rows; ++j)
      Q(i,j) = ScalarType(0.0);
  }

  // keep initial input matrix for comparison
  MatrixType ublas_A_backup(ublas_A);


  /**
  *   Setup the matrix in cuarma and copy the data from the uBLAS matrix:
  **/
  ARMAMatrixType arma_A(ublas_A.size1(), ublas_A.size2());
  cuarma::copy(ublas_A, arma_A);


  std::cout << "--- Boost.uBLAS ---" << std::endl;
  std::vector<ScalarType> ublas_betas = cuarma::blas::inplace_qr(ublas_A);  //computes the QR factorization

  /**
  *  Let us check for the correct result:
  **/
  cuarma::blas::recoverQ(ublas_A, ublas_betas, Q, R);
  MatrixType ublas_QR = prod(Q, R);
  double ublas_error = check(ublas_QR, ublas_A_backup);
  std::cout << "Maximum relative error (ublas): " << ublas_error << std::endl;

  std::cout << "--- Hybrid (default) ---" << std::endl;
  cuarma::copy(ublas_A_backup, arma_A);
  std::vector<ScalarType> hybrid_betas = cuarma::blas::inplace_qr(arma_A);

  /**
  *  Let us check for the correct result:
  **/
  cuarma::copy(arma_A, ublas_A);
  Q.clear(); R.clear();
  cuarma::blas::recoverQ(ublas_A, hybrid_betas, Q, R);
  double hybrid_error = check(ublas_QR, ublas_A_backup);
  std::cout << "Maximum relative error (hybrid): " << hybrid_error << std::endl;

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  return EXIT_SUCCESS;
}

