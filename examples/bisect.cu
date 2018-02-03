/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   bisect.cu
 *   @coding UTF-8
 *   @brief  This tutorial shows how the eigenvalues of a symmetric, tridiagonal matrix can be computed using bisection.
 *           Operator overloading in C++ is used extensively to provide an intuitive syntax.
 *   @brief  测试：二分法计算对称-三角矩阵的特征值
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"

#include "cuarma/blas/bisect_gpu.hpp"


// 功能函数：生成合适的三角矩阵
template <typename NumericT>
void initInputData(std::vector<NumericT> &diagonal, std::vector<NumericT> &superdiagonal, const unsigned int mat_size)
{

  srand(278217421);
  bool randomValues = false;


  if(randomValues == true)
  {
    // 随机数填充对角和超对角线
    for (unsigned int i = 0; i < mat_size; ++i)
    {
        diagonal[i] =      static_cast<NumericT>(2.0 * (((double)rand() / (double) RAND_MAX) - 0.5));
        superdiagonal[i] = static_cast<NumericT>(2.0 * (((double)rand() / (double) RAND_MAX) - 0.5));
    }
  }

  else
  {
    for(unsigned int i = 0; i < mat_size; ++i)
    {
       diagonal[i] = ((NumericT)(i % 8)) - 4.5f;
       superdiagonal[i] = ((NumericT)(i % 5)) - 4.5f;
    }
  }
  superdiagonal[0] = 0.0f;
}


int main()
{
    typedef float NumericT;

    bool bResult = false;
    unsigned int mat_size = 30;

    std::vector<NumericT> diagonal(mat_size);
    std::vector<NumericT> superdiagonal(mat_size);
    std::vector<NumericT> eigenvalues_bisect(mat_size);

    initInputData(diagonal, superdiagonal, mat_size);

    std::cout << "Start the bisection algorithm" << std::endl;
    bResult = cuarma::blas::bisect(diagonal, superdiagonal, eigenvalues_bisect);
    std::cout << std::endl << "---TUTORIAL COMPLETED---" << std::endl;


    // ------------Print the results---------------
    std::cout << "mat_size = " << mat_size << std::endl;
    for (unsigned int i = 0; i < mat_size; ++i)
    {
      std::cout << "Eigenvalue " << i << ": " << std::setprecision(8) << eigenvalues_bisect[i] << std::endl;
    }

    exit(bResult == true ? EXIT_SUCCESS : EXIT_FAILURE);
}
