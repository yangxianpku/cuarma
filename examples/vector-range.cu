/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   vector-range.cu
 *   @coding UTF-8
 *   @brief  This tutorial explains the use of vector ranges with simple BLAS level 1 and 2 operations.
 *           Vector slices are used similarly and not further considered in this tutorial.
 *   @brief  测试：向量分片
 */

#define CUARMA_WITH_UBLAS

#include <iostream>
#include <string>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/vector_proxy.hpp"

#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"

int main (int, const char **)
{

  typedef boost::numeric::ublas::vector<ScalarType>     VectorType;
  typedef cuarma::vector<ScalarType>                    ARMAVectorType;

  std::size_t dim_large = 7;
  std::size_t dim_small = 3;


  VectorType ublas_v1(dim_large);
  VectorType ublas_v2(dim_small);

  for (std::size_t i=0; i<ublas_v1.size(); ++i)
    ublas_v1(i) = ScalarType(i+1);

  for (std::size_t i=0; i<ublas_v2.size(); ++i)
    ublas_v2(i) = ScalarType(dim_large + i);


  /** Extract submatrices using the ranges in ublas **/
  boost::numeric::ublas::range ublas_r1(0, dim_small);                     //the first 'dim_small' entries
  boost::numeric::ublas::range ublas_r2(dim_small - 1, 2*dim_small - 1);   // 'dim_small' entries somewhere from the middle
  boost::numeric::ublas::range ublas_r3(dim_large - dim_small, dim_large); // the last 'dim_small' entries
  boost::numeric::ublas::vector_range<VectorType> ublas_v1_sub1(ublas_v1, ublas_r1); // front part of vector v_1
  boost::numeric::ublas::vector_range<VectorType> ublas_v1_sub2(ublas_v1, ublas_r2); // center part of vector v_1
  boost::numeric::ublas::vector_range<VectorType> ublas_v1_sub3(ublas_v1, ublas_r3); // tail of vector v_1

  /** Create cuarma objects and copy data over from uBLAS objects. **/
  ARMAVectorType arma_v1(dim_large);
  ARMAVectorType arma_v2(dim_small);

  cuarma::copy(ublas_v1, arma_v1);
  cuarma::copy(ublas_v2, arma_v2);


  /** Extract submatrices using the range-functionality in cuarma. This works exactly the same way as for uBLAS. **/
  cuarma::range arma_r1(0, dim_small);                      //the first 'dim_small' entries
  cuarma::range arma_r2(dim_small - 1, 2*dim_small - 1);    // 'dim_small' entries somewhere from the middle
  cuarma::range arma_r3(dim_large - dim_small, dim_large);  // the last 'dim_small' entries
  cuarma::vector_range<ARMAVectorType>   arma_v1_sub1(arma_v1, arma_r1); // front part of vector v_1
  cuarma::vector_range<ARMAVectorType>   arma_v1_sub2(arma_v1, arma_r2); // center part of vector v_1
  cuarma::vector_range<ARMAVectorType>   arma_v1_sub3(arma_v1, arma_r3); // tail of vector v_1

  ublas_v1_sub1 = ublas_v2;
  cuarma::copy(ublas_v2, arma_v1_sub1);
  cuarma::copy(arma_v1_sub1, ublas_v2);

  ublas_v1_sub1 += ublas_v1_sub1;
  arma_v1_sub1 += arma_v1_sub1;

  ublas_v1_sub2 += ublas_v1_sub2;
  arma_v1_sub2 += arma_v1_sub2;

  ublas_v1_sub3 += ublas_v1_sub3;
  arma_v1_sub3 += arma_v1_sub3;

  std::cout << "ublas:    " << ublas_v1 << std::endl;
  std::cout << "cuarma: " << arma_v1 << std::endl;

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

