/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   blas2.cu
 *   @coding UTF-8
 *   @brief  In this tutorial the eigenvalues and eigenvectors of a symmetric 9-by-9 matrix are calculated using the QR-method.
 *   @brief  测试：QR分解计算特征值和特征向量
 */
 
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "head_define.h"

#include "cuarma/matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/qr-method.hpp"

template <typename ScalarType>
void initialize(cuarma::matrix<ScalarType> & A, std::vector<ScalarType> & v)
{
  ScalarType M[9][9] = {{ 4,  1, -2, 2, -7,  3,  9, -6, -2},
                        { 1, -2,  0, 1, -1,  5,  4,  7,  3},
                        {-2,  0,  3, 2,  0,  3,  6,  1, -1},
                        { 2,  1,  2, 1,  4,  5,  6,  7,  8},
                        {-7, -1,  0, 4,  5,  4,  9,  1, -8},
                        { 3,  5,  3, 5,  4,  9, -3,  3,  3},
                        { 9,  4,  6, 6,  9, -3,  3,  6, -7},
                        {-6,  7,  1, 7,  1,  3,  6,  2,  6},
                        {-2,  3, -1, 8, -8,  3, -7,  6,  1}};

  for(std::size_t i = 0; i < 9; i++)
    for(std::size_t j = 0; j < 9; j++)
      A(i, j) = M[i][j];

  // Known eigenvalues:
  ScalarType V[9] = {ScalarType(12.6005), ScalarType(19.5905), ScalarType(8.06067), ScalarType(2.95074), ScalarType(0.223506),
                     ScalarType(24.3642), ScalarType(-9.62084), ScalarType(-13.8374), ScalarType(-18.3319)};

  for(std::size_t i = 0; i < 9; i++)
    v[i] = V[i];
}

/** 打印STL向量 **/
template <typename ScalarType>
void vector_print(std::vector<ScalarType>& v )
{
  for (unsigned int i = 0; i < v.size(); i++)
    std::cout << std::setprecision(6) << std::fixed << v[i] << "\t";
  std::cout << std::endl;
}

int main()
{
  std::cout << "Testing matrix of size " << 9 << "-by-" << 9 << std::endl;

  cuarma::matrix<ScalarType> A_input(9,9);
  cuarma::matrix<ScalarType> Q(9, 9);

  std::vector<ScalarType> eigenvalues_ref(9);   /** 参考特征值 **/
  std::vector<ScalarType> eigenvalues(9);

  cuarma::vector<ScalarType> arma_eigenvalues(9);

  initialize(A_input, eigenvalues_ref);

  std::cout << std::endl <<"Input matrix: " << std::endl;
  std::cout << A_input << std::endl;

  cuarma::matrix<ScalarType> A_input2(A_input); 


  /**
  * Call the function qr_method_sym() to calculate eigenvalues and eigenvectors
  * Parameters:
  *  -     A_input      - input matrix to find eigenvalues and eigenvectors from
  *  -     Q            - matrix, where the calculated eigenvectors will be stored in
  *  -     eigenvalues  - vector, where the calculated eigenvalues will be stored in
  **/

  std::cout << "Calculation..." << std::endl;
  cuarma::blas::qr_method_sym(A_input, Q, eigenvalues);

  /** Same as before, but writing the eigenvalues to a cuarma-vector: **/
  cuarma::blas::qr_method_sym(A_input2, Q, arma_eigenvalues);

  /** 打印计算结果 **/
  std::cout << std::endl << "Eigenvalues with std::vector<T>:" << std::endl;
  vector_print(eigenvalues);
  std::cout << "Eigenvalues with cuarma::vector<T>: " << std::endl << arma_eigenvalues << std::endl;
  std::cout << std::endl << "Reference eigenvalues:" << std::endl;
  vector_print(eigenvalues_ref);
  std::cout << std::endl;
  std::cout << "Eigenvectors - each column is an eigenvector" << std::endl;
  std::cout << Q << std::endl;

  std::cout << std::endl;
  std::cout << "------- Tutorial completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

