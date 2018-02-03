/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   power-iter.cu
 *   @coding UTF-8
 *   @brief  This tutorial demonstrates the calculation of the eigenvalue with largest modulus using the power iteration method.
 *   @brief  测试：幂迭代发计算矩阵特征值
 */
 
/** uBLAS中的稀疏矩阵运算在debug模式下非常慢速，这里禁用debug **/
#ifndef NDEBUG
  #define BOOST_UBLAS_NDEBUG
#endif

#include <iostream>
#include <fstream>
#include <limits>
#include <string>

#define CUARMA_WITH_UBLAS

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/blas/power_iter.hpp"
#include "cuarma/io/matrix_market.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector_expression.hpp>


int main()
{

	/** uBLAS创建稀疏矩阵 **/
  boost::numeric::ublas::compressed_matrix<ScalarType> ublas_A;

  if (!cuarma::io::read_matrix_market_file(ublas_A, "data/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  /** 数据传输 **/
  cuarma::compressed_matrix<ScalarType>  arma_A(ublas_A.size1(), ublas_A.size2());
  cuarma::copy(ublas_A, arma_A);

  /**
  *  Run the power iteration up until the largest eigenvalue changes by less than the specified tolerance.
  *  Print the results of running with uBLAS as well as cuarma and exit.
  **/
  cuarma::blas::power_iter_tag ptag(1e-6);

  std::cout << "Starting computation of eigenvalue with largest modulus (might take about a minute)..." << std::endl;
  std::cout << "Result of power iteration with ublas matrix (single-threaded): " << cuarma::blas::eig(ublas_A, ptag) << std::endl;
  std::cout << "Result of power iteration with cuarma (OpenCL accelerated): " << cuarma::blas::eig(arma_A, ptag) << std::endl;

  /**
   *  You can also obtain the associated *approximated* eigenvector by passing it as a third argument to eig()
   *  Tighten the tolerance passed to ptag above in order to obtain more accurate results.
   **/
  cuarma::vector<ScalarType> eigenvector(arma_A.size1());
  cuarma::blas::eig(arma_A, ptag, eigenvector);
  std::cout << "First three entries in eigenvector: " << eigenvector[0] << " " << eigenvector[1] << " " << eigenvector[2] << std::endl;
  cuarma::vector<ScalarType> Ax = cuarma::blas::prod(arma_A, eigenvector);
  std::cout << "First three entries in A*eigenvector: " << Ax[0] << " " << Ax[1] << " " << Ax[2] << std::endl;

  return EXIT_SUCCESS;
}

