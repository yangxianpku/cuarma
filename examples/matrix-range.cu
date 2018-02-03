/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   matrix-range.cu
 *   @coding UTF-8
 *   @brief  This tutorial explains the use of matrix ranges with simple BLAS level 1 and 2 operations.
 *   @brief  测试：矩阵range操作
 */
 
#define CUARMA_WITH_UBLAS

#include <iostream>
#include <string>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/matrix_proxy.hpp"


#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"


int main (int, const char **)
{
  typedef boost::numeric::ublas::matrix<ScalarType>       MatrixType;
  typedef cuarma::matrix<ScalarType, cuarma::row_major>    ARMAMatrixType;

  /**  创建uBLAS对象并赋值 **/
  std::size_t dim_large = 5;
  std::size_t dim_small = 3;

  MatrixType ublas_A(dim_large, dim_large);
  MatrixType ublas_B(dim_small, dim_small);
  MatrixType ublas_C(dim_large, dim_small);
  MatrixType ublas_D(dim_small, dim_large);


  for (std::size_t i=0; i<ublas_A.size1(); ++i)
    for (std::size_t j=0; j<ublas_A.size2(); ++j)
      ublas_A(i,j) = static_cast<ScalarType>((i+1) + (j+1)*(i+1));

  for (std::size_t i=0; i<ublas_B.size1(); ++i)
    for (std::size_t j=0; j<ublas_B.size2(); ++j)
      ublas_B(i,j) = static_cast<ScalarType>((i+1) + (j+1)*(i+1));

  for (std::size_t i=0; i<ublas_C.size1(); ++i)
    for (std::size_t j=0; j<ublas_C.size2(); ++j)
      ublas_C(i,j) = static_cast<ScalarType>((j+2) + (j+1)*(i+1));

  for (std::size_t i=0; i<ublas_D.size1(); ++i)
    for (std::size_t j=0; j<ublas_D.size2(); ++j)
      ublas_D(i,j) = static_cast<ScalarType>((j+2) + (j+1)*(i+1));


  /**  使用range提取子矩阵 **/
  boost::numeric::ublas::range ublas_r1(0, dim_small);                      //the first 'dim_small' entries
  boost::numeric::ublas::range ublas_r2(dim_large - dim_small, dim_large);  //the last 'dim_small' entries
  boost::numeric::ublas::matrix_range<MatrixType> ublas_A_sub1(ublas_A, ublas_r1, ublas_r1); //upper left part of A
  boost::numeric::ublas::matrix_range<MatrixType> ublas_A_sub2(ublas_A, ublas_r2, ublas_r2); //lower right part of A

  boost::numeric::ublas::matrix_range<MatrixType> ublas_C_sub(ublas_C, ublas_r1, ublas_r1); //upper left part of C
  boost::numeric::ublas::matrix_range<MatrixType> ublas_D_sub(ublas_D, ublas_r1, ublas_r1); //upper left part of D

  /** Setup cuarma objects and copy data from uBLAS objects **/
  ARMAMatrixType arma_A(dim_large, dim_large);
  ARMAMatrixType arma_B(dim_small, dim_small);
  ARMAMatrixType arma_C(dim_large, dim_small);
  ARMAMatrixType arma_D(dim_small, dim_large);

  cuarma::copy(ublas_A, arma_A);
  cuarma::copy(ublas_B, arma_B);
  cuarma::copy(ublas_C, arma_C);
  cuarma::copy(ublas_D, arma_D);

  std::cout << "Result cuarma:   " << arma_A << std::endl;

  /**
  * Extract submatrices using the ranges in cuarma. Similar to the code above for uBLAS.
  **/
  cuarma::range arma_r1(0, dim_small);   //the first 'dim_small' entries
  cuarma::range arma_r2(dim_large - dim_small, dim_large); //the last 'dim_small' entries
  cuarma::matrix_range<ARMAMatrixType>   arma_A_sub1(arma_A, arma_r1, arma_r1); //upper left part of A
  cuarma::matrix_range<ARMAMatrixType>   arma_A_sub2(arma_A, arma_r2, arma_r2); //lower right part of A

  std::cout << "Result cuarma:   " << arma_A_sub1 << std::endl;
  std::cout << "Result cuarma:   " << arma_A_sub2 << std::endl;

  cuarma::matrix_range<ARMAMatrixType>   arma_C_sub(arma_C, arma_r1, arma_r1);  //upper left part of C
  cuarma::matrix_range<ARMAMatrixType>   arma_D_sub(arma_D, arma_r1, arma_r1);  //upper left part of D


  ublas_A_sub1 = ublas_B;
  cuarma::copy(ublas_B, arma_A_sub1);
  cuarma::copy(arma_A_sub1, ublas_B);


  // range to range:
  ublas_A_sub2 += ublas_A_sub2;
  arma_A_sub2 += arma_A_sub2;

  // range to matrix:
  ublas_B += ublas_A_sub2;
  arma_B  += arma_A_sub2;


  ublas_A_sub1 += prod(ublas_C_sub, ublas_D_sub);
  arma_A_sub1 += cuarma::blas::prod(arma_C_sub, arma_D_sub);


  std::cout << "Result ublas:    " << ublas_A << std::endl;
  std::cout << "Result cuarma:   " << arma_A << std::endl;

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

