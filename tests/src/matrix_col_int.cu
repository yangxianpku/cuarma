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
*    @brief  Tests routines for dense matrices, column-major, signed precision.
*    @brief  ≥Ì√‹æÿ’Û£¨¡–¥Ê¥¢£¨”–∑˚∫≈’˚–Õ
*/

#include "matrix_int.hpp"

int main (int, const char **)
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Matrix operations, column-major, integers " << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  std::cout << "# Testing setup:" << std::endl;
  std::cout << "  numeric: int" << std::endl;
  std::cout << " --- column-major ---" << std::endl;
  if (run_test<cuarma::column_major, int>() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "# Testing setup:" << std::endl;
  std::cout << "  numeric: long" << std::endl;
  std::cout << " --- column-major ---" << std::endl;
  if (run_test<cuarma::column_major, long>() != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}

