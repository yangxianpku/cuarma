/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   nmf.cu
 *   @coding UTF-8
 *   @brief  This tutorial explains how to use the nonnegative matrix factorization (NMF) functionality in cuarma.
 *   @brief  测试：非负矩阵分解
 */
 
#include "head_define.h"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/nmf.hpp"

// [0,1]区间均匀分布的随机数填充矩阵
template<typename MajorT>
void fill_random(cuarma::matrix<ScalarType, MajorT> & A)
{
  for (std::size_t i = 0; i < A.size1(); i++)
    for (std::size_t j = 0; j < A.size2(); ++j)
      A(i, j) = static_cast<ScalarType>(rand()) / ScalarType(RAND_MAX);
}


/**  V≈WH 其中所有的矩阵都只含有非负项 **/
int main()
{
  std::cout << std::endl;
  std::cout << "------- Tutorial NMF --------" << std::endl;
  std::cout << std::endl;

  /** Approximate the 7-by-6-matrix V by a 7-by-3-matrix W and a 3-by-6-matrix H **/
  unsigned int m = 7; //size1 of W and size1 of V
  unsigned int n = 6; //size2 of V and size2 of H
  unsigned int k = 3; //size2 of W and size1 of H

  cuarma::matrix<ScalarType, cuarma::column_major> V(m, n);
  cuarma::matrix<ScalarType, cuarma::column_major> W(m, k);
  cuarma::matrix<ScalarType, cuarma::column_major> H(k, n);

  /** Fill the matrices randomly. Initial guesses for W and H consisting of only zeros won't work. **/
  fill_random(V);
  fill_random(W);
  fill_random(H);

  std::cout << "Input matrices:" << std::endl;
  std::cout << "V" << V << std::endl;
  std::cout << "W" << W << std::endl;
  std::cout << "H" << H << "\n" << std::endl;

  /**
  *  Create configuration object to hold (and adjust) the respective parameters.
  **/
  cuarma::blas::nmf_config conf;
  conf.print_relative_error(false);
  conf.max_iterations(50); // 50 iterations are enough here


  std::cout << "Computing NMF" << std::endl;
  cuarma::blas::nmf(V, W, H, conf);

  std::cout << "RESULT:" << std::endl;
  std::cout << "V" << V << std::endl;
  std::cout << "W" << W << std::endl;
  std::cout << "H" << H << "\n" << std::endl;

  /**  Print the product W*H approximating V for comparison and exit: **/
  std::cout << "W*H:" << std::endl;
  cuarma::matrix<ScalarType> resultCorrect = cuarma::blas::prod(W, H);
  std::cout << resultCorrect << std::endl;

  std::cout << std::endl;
  std::cout << "------- Tutorial completed --------" << std::endl;
  std::cout << std::endl;

}
