/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   blas3range.cu
 *   @coding UTF-8
 *   @brief  Tutorial: BLAS level 3 functionality on sub-matrices.
 *           Operator overloading in C++ is used extensively to provide an intuitive syntax.
 *   @brief  测试：BLAS3 子矩阵相关功能
 */

#ifndef NDEBUG
 #define NDEBUG
#endif

#include <iostream>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

#define CUARMA_WITH_UBLAS 1

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/matrix_proxy.hpp"
#include "cuarma/tools/random.hpp"
#include "cuarma/tools/timer.hpp"

#define MATRIX_SIZE   1500

using namespace boost::numeric;

int main()
{
  typedef float     ScalarType;

  cuarma::tools::timer timer;
  double exec_time;

  cuarma::tools::uniform_random_numbers<ScalarType> randomNumber;

  // Set up some ublas objects
  ublas::matrix<ScalarType> ublas_A(MATRIX_SIZE, MATRIX_SIZE);
  ublas::matrix<ScalarType, ublas::column_major> ublas_B(MATRIX_SIZE, MATRIX_SIZE);
  ublas::matrix<ScalarType> ublas_C(MATRIX_SIZE, MATRIX_SIZE);
  ublas::matrix<ScalarType> ublas_C1(MATRIX_SIZE, MATRIX_SIZE);
  ublas::matrix<ScalarType> ublas_C2(MATRIX_SIZE, MATRIX_SIZE);

  // One alternative: Put the matrices into a contiguous block of memory (allows to use cuarma::fast_copy(), avoiding temporary memory)
  std::vector<ScalarType> stl_A(MATRIX_SIZE * MATRIX_SIZE);
  std::vector<ScalarType> stl_B(MATRIX_SIZE * MATRIX_SIZE);
  std::vector<ScalarType> stl_C(MATRIX_SIZE * MATRIX_SIZE);

  // Fill the matrix
  for (unsigned int i = 0; i < ublas_A.size1(); ++i)
    for (unsigned int j = 0; j < ublas_A.size2(); ++j)
    {
      ublas_A(i,j) = randomNumber();
      stl_A[i*ublas_A.size2() + j] = ublas_A(i,j);
    }

  for (unsigned int i = 0; i < ublas_B.size1(); ++i)
    for (unsigned int j = 0; j < ublas_B.size2(); ++j)
    {
      ublas_B(i,j) = randomNumber();
      stl_B[i + j*ublas_B.size1()] = ublas_B(i,j);
    }

  ublas::range ublas_r1(1, MATRIX_SIZE-1);
  ublas::range ublas_r2(2, MATRIX_SIZE-2);
  ublas::matrix_range< ublas::matrix<ScalarType> >  ublas_A_sub(ublas_A, ublas_r1, ublas_r2);
  ublas::matrix_range< ublas::matrix<ScalarType, ublas::column_major> >  ublas_B_sub(ublas_B, ublas_r2, ublas_r1);
  ublas::matrix_range< ublas::matrix<ScalarType> >  ublas_C_sub(ublas_C, ublas_r1, ublas_r1);

  // Set up some cuarma objects
  cuarma::matrix<ScalarType> arma_A(MATRIX_SIZE, MATRIX_SIZE);
  cuarma::matrix<ScalarType, cuarma::column_major> arma_B(MATRIX_SIZE, MATRIX_SIZE);
  cuarma::matrix<ScalarType> arma_C(MATRIX_SIZE, MATRIX_SIZE);

  cuarma::range arma_r1(1, MATRIX_SIZE-1);
  cuarma::range arma_r2(2, MATRIX_SIZE-2);
  cuarma::matrix_range< cuarma::matrix<ScalarType> >  arma_A_sub(arma_A, arma_r1, arma_r2);
  cuarma::matrix_range< cuarma::matrix<ScalarType, cuarma::column_major> >  arma_B_sub(arma_B, arma_r2, arma_r1);
  cuarma::matrix_range< cuarma::matrix<ScalarType> >  arma_C_sub(arma_C, arma_r1, arma_r1);

  ublas_C.clear();
  cuarma::copy(ublas_C, arma_C);

  //////////// Matrix-matrix products /////////////
  std::cout << "--- Computing matrix-matrix product using ublas ---" << std::endl;
  timer.start();
  ublas_C_sub = ublas::prod(ublas_A_sub, ublas_B_sub);
  exec_time = timer.get();
  std::cout << " - Execution time: " << exec_time << std::endl;


    cuarma::fast_copy(&(stl_A[0]),
                        &(stl_A[0]) + stl_A.size(),
                        arma_A);
    cuarma::fast_copy(&(stl_B[0]),
                        &(stl_B[0]) + stl_B.size(),
                        arma_B);
    arma_C_sub = cuarma::blas::prod(arma_A_sub, arma_B_sub);
 
    // Verify the result
    cuarma::fast_copy(arma_C, &(stl_C[0]));
    for (unsigned int i = 0; i < ublas_C1.size1(); ++i)
      for (unsigned int j = 0; j < ublas_C1.size2(); ++j)
        ublas_C1(i,j) = stl_C[i * ublas_C1.size2() + j];

    std::cout << " - Checking result... ";
    bool check_ok = true;
    for (unsigned int i = 0; i < ublas_A.size1(); ++i)
    {
      for (unsigned int j = 0; j < ublas_A.size2(); ++j)
      {
        if ( fabs(ublas_C1(i,j) - ublas_C(i,j)) / ublas_C(i,j) > 1e-4 )
        {
          check_ok = false;
          break;
        }
      }
      if (!check_ok)
        break;
    }
    if (check_ok)
      std::cout << "[OK]" << std::endl << std::endl;
    else
      std::cout << "[FAILED]" << std::endl << std::endl;


  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
  return EXIT_SUCCESS;
}

