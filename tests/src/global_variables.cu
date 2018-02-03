/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   global_variables.cu
 *   @coding UTF-8
 *   @brief  Ensures that cuarma works properly when objects are used as global variables.
 */


#include <iostream>
#include <algorithm>
#include <cmath>

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/coordinate_matrix.hpp"
#include "cuarma/ell_matrix.hpp"
#include "cuarma/hyb_matrix.hpp"

// forward declarations of global variables:
extern cuarma::scalar<float> s1;
extern cuarma::scalar<int>   s2;

extern cuarma::vector<float> v1;
extern cuarma::vector<int>   v2;

extern cuarma::matrix<float> m1;

// instantiation of global variables:
cuarma::scalar<float>  s1;
cuarma::scalar<int> s2;

cuarma::vector<float>  v1;
cuarma::vector<int> v2;

cuarma::matrix<float>  m1;
//cuarma::matrix<int> m2;

int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Instantiation of global variables" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  s1 = cuarma::scalar<float>(1.0f);
  s2 = cuarma::scalar<int>(1);

  v1 = cuarma::vector<float>(5);
  v2 = cuarma::vector<int>(5);

  m1 = cuarma::matrix<float>(5, 4);
  //m2 = cuarma::matrix<int>(5, 4);

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return EXIT_SUCCESS;
}