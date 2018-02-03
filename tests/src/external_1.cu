/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   external_2.cu
 *   @coding UTF-8
 *   @brief  boost linking
 *   @brief  测试：外部库连接，boost
 */


#define CUARMA_WITH_UBLAS

#include <iostream>
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/coordinate_matrix.hpp"
#include "cuarma/ell_matrix.hpp"
#include "cuarma/fft.hpp"
#include "cuarma/hyb_matrix.hpp"
#include "cuarma/sliced_ell_matrix.hpp"

#include "cuarma/blas/bicgstab.hpp"
#include "cuarma/blas/bisect.hpp"
#include "cuarma/blas/bisect_gpu.hpp"
#include "cuarma/blas/cg.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/blas/gmres.hpp"
#include "cuarma/blas/ichol.hpp"
#include "cuarma/blas/ilu.hpp"
#include "cuarma/blas/inner_prod.hpp"
#include "cuarma/blas/jacobi_precond.hpp"
#include "cuarma/blas/norm_1.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/norm_inf.hpp"
#include "cuarma/blas/norm_frobenius.hpp"
#include "cuarma/blas/qr.hpp"
#include "cuarma/blas/qr-method.hpp"
#include "cuarma/blas/row_scaling.hpp"
#include "cuarma/blas/sum.hpp"
#include "cuarma/blas/tql2.hpp"

#include "cuarma/misc/bandwidth_reduction.hpp"

#include "cuarma/io/matrix_market.hpp"
#include "cuarma/scheduler/execute.hpp"


//defined in external_2.cpp
void other_func();

int main()
{
  typedef float   NumericType;

  //doing nothing but instantiating a few types
  cuarma::scalar<NumericType>  s;
  cuarma::vector<NumericType>  v(10);
  cuarma::matrix<NumericType>  m(10, 10);
  cuarma::compressed_matrix<NumericType>  compr(10, 10);
  cuarma::coordinate_matrix<NumericType>  coord(10, 10);

  //this is the external linkage check:
  other_func();

   std::cout << std::endl;
   std::cout << "------- Test completed --------" << std::endl;
   std::cout << std::endl;


  return EXIT_SUCCESS;
}
