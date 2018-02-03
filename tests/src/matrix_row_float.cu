/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */


/**  @file   matrix_col_double.cu
*    @coding UTF-8
*    @brief  Tests routines for dense matrices, row-major, single precision.
*    @brief  稠密矩阵，行存储，单精度
*/


#include "matrix_float_double.hpp"

int main (int, const char **)
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Matrix operations, row-major, single precision " << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  double epsilon = 1e-4;
  std::cout << "# Testing setup:" << std::endl;
  std::cout << "  eps:     " << epsilon << std::endl;
  std::cout << "  numeric: float" << std::endl;
  std::cout << " --- row-major ---" << std::endl;
  if (run_test<cuarma::row_major, float>(epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

