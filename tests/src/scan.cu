/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   scan.cu
 *   @coding UTF-8
 *   @brief  Tests inclusive and exclusive scan operations.
 *   @brief  测试：非负矩阵分解
 */


#include "head_define.h"
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"

#include <iostream>
#include <limits>
#include <string>
#include <iomanip>


typedef int     ScalarType;

static void init_vector(cuarma::vector<ScalarType>& arma_v)
{
  std::vector<ScalarType> v(arma_v.size());
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = ScalarType(i % 7 + 1);
  cuarma::copy(v, arma_v);
}

static void test_scan_values(cuarma::vector<ScalarType> const & input,  cuarma::vector<ScalarType> & result, bool is_inclusive_scan)
{
  std::vector<ScalarType> host_input(input.size());
  std::vector<ScalarType> host_result(result.size());

  cuarma::copy(input, host_input);
  cuarma::copy(result, host_result);

  ScalarType sum = 0;
  if (is_inclusive_scan)
  {
    for(cuarma::arma_size_t i = 0; i < input.size(); i++)
    {
      sum += host_input[i];
      host_input[i] = sum;
    }
  }
  else
  {
    for(cuarma::arma_size_t i = 0; i < input.size(); i++)
    {
      ScalarType tmp = host_input[i];
      host_input[i] = sum;
      sum += tmp;
    }
  }


  for(cuarma::arma_size_t i = 0; i < input.size(); i++)
  {
    if (host_input[i] != host_result[i])
    {
      std::cout << "Fail at vector index " << i << std::endl;
      std::cout << " result[" << i << "] = " << host_result[i] << std::endl;
      std::cout << " Reference = " << host_input[i] << std::endl;
      if (i > 0)
      {
        std::cout << " previous result[" << i-1 << "] = " << host_result[i-1] << std::endl;
        std::cout << " previous Reference = " << host_input[i-1] << std::endl;
      }
      exit(EXIT_FAILURE);
    }
  }
  std::cout << "PASSED!" << std::endl;

}


static void test_scans(unsigned int sz)
{
  cuarma::vector<ScalarType> vec1(sz), vec2(sz);

  std::cout << "Initialize vector..." << std::endl;
  init_vector(vec1);


  // INCLUSIVE SCAN
  std::cout << " --- Inclusive scan ---" << std::endl;
  std::cout << "Separate vectors: ";
  cuarma::blas::inclusive_scan(vec1, vec2);
  test_scan_values(vec1, vec2, true);

  std::cout << "In-place: ";
  vec2 = vec1;
  cuarma::blas::inclusive_scan(vec2);
  test_scan_values(vec1, vec2, true);
  std::cout << "Inclusive scan tested successfully!" << std::endl << std::endl;

  std::cout << "Initialize vector..." << std::endl;
  init_vector(vec1);

  // EXCLUSIVE SCAN
  std::cout << " --- Exclusive scan ---" << std::endl;
  std::cout << "Separate vectors: ";
  cuarma::blas::exclusive_scan(vec1, vec2);
  test_scan_values(vec1, vec2, false);

  std::cout << "In-place: ";
  vec2 = vec1;
  cuarma::blas::exclusive_scan(vec2);
  test_scan_values(vec1, vec2, false);
  std::cout << "Exclusive scan tested successfully!" << std::endl << std::endl;

}

int main()
{

  std::cout << std::endl << "----TEST INCLUSIVE and EXCLUSIVE SCAN----" << std::endl << std::endl;
  std::cout << " //// Tiny vectors ////" << std::endl;
  test_scans(27);
  std::cout << " //// Small vectors ////" << std::endl;
  test_scans(298);
  std::cout << " //// Medium vectors ////" << std::endl;
  test_scans(12345);
  std::cout << " //// Large vectors ////" << std::endl;
  test_scans(123456);

  std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
}
