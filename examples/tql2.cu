/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   tql2.cu
 *   @coding UTF-8
 *   @brief  This tutorial explains how one can use the tql-algorithm to compute the eigenvalues of tridiagonal matrices.
 *   @brief  测试：tql2算法计算三对角矩阵的特征值和特征向量
 */

#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <iomanip>

#define CUARMA_WITH_UBLAS

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/io/matrix_market.hpp"

#include "cuarma/blas/qr-method.hpp"
#include "cuarma/blas/qr-method-common.hpp"
#include "cuarma/blas/host_based/matrix_operations.hpp"

namespace ublas = boost::numeric::ublas;

int main()
{
  std::size_t sz = 10;
  std::cout << "Compute eigenvalues and eigenvectors of matrix of size " << sz << "-by-" << sz << std::endl << std::endl;

  std::vector<ScalarType> d(sz), e(sz);
  // Initialize diagonal and superdiagonal elements of the tridiagonal matrix
  d[0] = 1; e[0] = 0;
  d[1] = 2; e[1] = 4;
  d[2] =-4; e[2] = 5;
  d[3] = 6; e[3] = 1;
  d[4] = 3; e[4] = 2;
  d[5] = 4; e[5] =-3;
  d[6] = 7; e[6] = 5;
  d[7] = 9; e[7] = 1;
  d[8] = 3; e[8] = 5;
  d[9] = 8; e[9] = 2;

  /**
  * Initialize the matrix Q as the identity matrix. It will hold the eigenvectors.
  **/
  cuarma::matrix<ScalarType> Q = cuarma::identity_matrix<ScalarType>(sz);

  /**
  * Compute the eigenvalues and eigenvectors
  **/
  cuarma::blas::tql2(Q, d, e);

  /**
  * Print the results:
  **/
  std::cout << "Eigenvalues: " << std::endl;
  for (unsigned int i = 0; i < d.size(); i++)
    std::cout << std::setprecision(6) << std::fixed << d[i] << " ";
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "Eigenvectors corresponding to the eigenvalues above are the columns: " << std::endl << std::endl;
  std::cout << Q << std::endl;

  std::cout << std::endl <<"--------TUTORIAL COMPLETED----------" << std::endl;

  return EXIT_SUCCESS;
}
