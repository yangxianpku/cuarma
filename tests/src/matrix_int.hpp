#pragma  once

/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <vector>

#include "cuarma/scalar.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/matrix_proxy.hpp"
#include "cuarma/vector_proxy.hpp"

template<typename NumericT, typename VCLMatrixType>
bool check_for_equality(std::vector<std::vector<NumericT> > const & std_A, VCLMatrixType const & arma_A)
{
  std::vector<std::vector<NumericT> > arma_A_cpu(arma_A.size1(), std::vector<NumericT>(arma_A.size2()));
  cuarma::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
  cuarma::copy(arma_A, arma_A_cpu);

  for (std::size_t i=0; i<std_A.size(); ++i)
  {
    for (std::size_t j=0; j<std_A[i].size(); ++j)
    {
      if (std_A[i][j] != arma_A_cpu[i][j])
      {
        std::cout << "Error at index (" << i << ", " << j << "): " << std_A[i][j] << " vs " << arma_A_cpu[i][j] << std::endl;
        std::cout << std::endl << "TEST failed!" << std::endl;
        return false;
      }
    }
  }

  std::cout << "PASSED!" << std::endl;
  return true;
}




template<typename STLMatrixType,
          typename cuarmaMatrixType1, typename cuarmaMatrixType2, typename cuarmaMatrixType3>
int run_test(STLMatrixType & std_A, STLMatrixType & std_B, STLMatrixType & std_C,
             cuarmaMatrixType1 & arma_A, cuarmaMatrixType2 & arma_B, cuarmaMatrixType3 arma_C)
{

  typedef typename cuarma::result_of::cpu_value_type<typename cuarmaMatrixType1::value_type>::type  cpu_value_type;

  cpu_value_type alpha = 3;
  cuarma::scalar<cpu_value_type>   gpu_alpha = alpha;

  cpu_value_type beta = 2;
  cuarma::scalar<cpu_value_type>   gpu_beta = beta;


  //
  // Initializer:
  //
  std::cout << "Checking for zero_matrix initializer..." << std::endl;
  std_A = std::vector<std::vector<cpu_value_type> >(std_A.size(), std::vector<cpu_value_type>(std_A[0].size()));
  arma_A = cuarma::zero_matrix<cpu_value_type>(arma_A.size1(), arma_A.size2());
  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  std::cout << "Checking for scalar_matrix initializer..." << std::endl;
  std_A = std::vector<std::vector<cpu_value_type> >(std_A.size(), std::vector<cpu_value_type>(std_A[0].size(), alpha));
  arma_A = cuarma::scalar_matrix<cpu_value_type>(arma_A.size1(), arma_A.size2(), alpha);
  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  std_A = std::vector<std::vector<cpu_value_type> >(std_A.size(), std::vector<cpu_value_type>(std_A[0].size(), gpu_beta));
  arma_A = cuarma::scalar_matrix<cpu_value_type>(  arma_A.size1(),   arma_A.size2(), gpu_beta);
  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  /*
  std::cout << "Checking for identity initializer..." << std::endl;
  std_A = boost::numeric::std::identity_matrix<cpu_value_type>(std_A.size1());
  arma_A = cuarma::identity_matrix<cpu_value_type>(arma_A.size1());
  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE; */


  std::cout << std::endl;
  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test: Assignments //////////" << std::endl;
  //std::cout << "//" << std::endl;

  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;

  std::cout << "Testing matrix assignment... ";
  //std::cout << std_B(0,0) << " vs. " << arma_B(0,0) << std::endl;
  std_A = std_B;
  arma_A = arma_B;
  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;



  //std::cout << std::endl;
  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test 1: Copy to GPU //////////" << std::endl;
  //std::cout << "//" << std::endl;

  std_A = std_B;
  cuarma::copy(std_B, arma_A);
  std::cout << "Testing upper left copy to GPU... ";
  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;


  std_C = std_B;
  cuarma::copy(std_B, arma_C);
  std::cout << "Testing lower right copy to GPU... ";
  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;


  //std::cout << std::endl;
  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test 2: Copy from GPU //////////" << std::endl;
  //std::cout << "//" << std::endl;

  std::cout << "Testing upper left copy to A... ";
  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  std::cout << "Testing lower right copy to C... ";
  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;



  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test 3: Addition //////////" << std::endl;
  //std::cout << "//" << std::endl;
  cuarma::copy(std_C, arma_C);

  std::cout << "Inplace add: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] += std_C[i][j];
  arma_C   +=   arma_C;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  std::cout << "Scaled inplace add: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] += beta * std_A[i][j];
  arma_C   += gpu_beta * arma_A;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  std::cout << "Add: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] = std_A[i][j] + std_B[i][j];
  arma_C   =   arma_A +   arma_B;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  std::cout << "Add with flipsign: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] = -std_A[i][j] + std_B[i][j];
  arma_C   = -   arma_A +   arma_B;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;


  std::cout << "Scaled add (left): ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] = alpha * std_A[i][j] + std_B[i][j];
  arma_C   = alpha *   arma_A +   arma_B;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  std::cout << "Scaled add (left): ";
  arma_C = gpu_alpha * arma_A + arma_B;
  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;


  std::cout << "Scaled add (right): ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] = std_A[i][j] + beta * std_B[i][j];
  arma_C   =   arma_A + beta *   arma_B;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  std::cout << "Scaled add (right): ";
  arma_C = arma_A + gpu_beta * arma_B;
  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;



  std::cout << "Scaled add (both): ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] = alpha * std_A[i][j] + beta * std_B[i][j];
  arma_C   = alpha *   arma_A + beta *   arma_B;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  std::cout << "Scaled add (both): ";
  arma_C = gpu_alpha * arma_A + gpu_beta * arma_B;
  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test 4: Subtraction //////////" << std::endl;
  //std::cout << "//" << std::endl;
  cuarma::copy(std_C, arma_C);

  std::cout << "Inplace sub: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] -= std_B[i][j];
  arma_C -= arma_B;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  std::cout << "Scaled Inplace sub: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] -= alpha * std_B[i][j];
  arma_C -= alpha * arma_B;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  std::cout << "Sub: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] = std_A[i][j] - std_B[i][j];
  arma_C = arma_A - arma_B;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;

  std::cout << "Scaled sub (left): ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_B[i][j] = alpha * std_A[i][j] - std_C[i][j];
  arma_B   = alpha *   arma_A - arma_C;

  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;

  std::cout << "Scaled sub (left): ";
  arma_B = gpu_alpha * arma_A - arma_C;
  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;


  std::cout << "Scaled sub (right): ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_B[i][j] = std_A[i][j] - beta * std_C[i][j];
  arma_B   =   arma_A - arma_C * beta;

  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;

  std::cout << "Scaled sub (right): ";
  arma_B = arma_A - arma_C * gpu_beta;
  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;


  std::cout << "Scaled sub (both): ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_B[i][j] = alpha * std_A[i][j] - beta * std_C[i][j];
  arma_B   = alpha * arma_A - arma_C * beta;

  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;

  std::cout << "Scaled sub (both): ";
  arma_B = gpu_alpha * arma_A - arma_C * gpu_beta;
  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;


  std::cout << "Unary operator-: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_C[i][j] = - std_A[i][j];
  arma_C   = -   arma_A;

  if (!check_for_equality(std_C, arma_C))
    return EXIT_FAILURE;



  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test 5: Scaling //////////" << std::endl;
  //std::cout << "//" << std::endl;
  cuarma::copy(std_A, arma_A);

  std::cout << "Multiplication with CPU scalar: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_A[i][j] *= alpha;
  arma_A   *= alpha;

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  std::cout << "Multiplication with GPU scalar: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_A[i][j] *= beta;
  arma_A *= gpu_beta;

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;


  std::cout << "Division with CPU scalar: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_A[i][j] /= alpha;
  arma_A /= alpha;

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  std::cout << "Division with GPU scalar: ";
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_A[i][j] /= beta;
  arma_A /= gpu_beta;

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;



  std::cout << "Testing elementwise multiplication..." << std::endl;
  std_B = std::vector<std::vector<cpu_value_type> >(std_B.size(), std::vector<cpu_value_type>(std_B[0].size(), 2));
  for (std::size_t i=0; i<std_C.size(); ++i)
    for (std::size_t j=0; j<std_C[i].size(); ++j)
      std_A[i][j] = 3 * std_B[i][j];
  cuarma::copy(std_A, arma_A);
  cuarma::copy(std_B, arma_B);
  cuarma::copy(std_B, arma_B);
  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] = std_A[i][j] * std_B[i][j];
  arma_A = cuarma::blas::element_prod(arma_A, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] += std_A[i][j] * std_B[i][j];
  arma_A += cuarma::blas::element_prod(arma_A, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] -= std_A[i][j] * std_B[i][j];
  arma_A -= cuarma::blas::element_prod(arma_A, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] = (std_A[i][j] + std_B[i][j]) * std_B[i][j];
  arma_A = cuarma::blas::element_prod(arma_A + arma_B, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] += (std_A[i][j] + std_B[i][j]) * std_B[i][j];
  arma_A += cuarma::blas::element_prod(arma_A + arma_B, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] -= (std_A[i][j] + std_B[i][j]) * std_B[i][j];
  arma_A -= cuarma::blas::element_prod(arma_A + arma_B, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] = std_A[i][j] * (std_B[i][j] + std_A[i][j]);
  arma_A = cuarma::blas::element_prod(arma_A, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] += std_A[i][j] * (std_B[i][j] + std_A[i][j]);
  arma_A += cuarma::blas::element_prod(arma_A, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] -= std_A[i][j] * (std_B[i][j] + std_A[i][j]);
  arma_A -= cuarma::blas::element_prod(arma_A, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] = (std_A[i][j] + std_B[i][j]) * (std_B[i][j] + std_A[i][j]);
  arma_A = cuarma::blas::element_prod(arma_A + arma_B, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] += (std_A[i][j] + std_B[i][j]) * (std_B[i][j] + std_A[i][j]);
  arma_A += cuarma::blas::element_prod(arma_A + arma_B, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] -= (std_A[i][j] + std_B[i][j]) * (std_B[i][j] + std_A[i][j]);
  arma_A -= cuarma::blas::element_prod(arma_A + arma_B, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;


  std_B = std::vector<std::vector<cpu_value_type> >(std_B.size(), std::vector<cpu_value_type>(std_B[0].size(), 2));
  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] =  3 * std_B[i][j];
  cuarma::copy(std_A, arma_A);
  cuarma::copy(std_B, arma_B);
  cuarma::copy(std_B, arma_B);

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] = std_A[i][j] / std_B[i][j];
  arma_A = cuarma::blas::element_div(arma_A, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] += std_A[i][j] / std_B[i][j];
  arma_A += cuarma::blas::element_div(arma_A, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] -= std_A[i][j] / std_B[i][j];
  arma_A -= cuarma::blas::element_div(arma_A, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] = (std_A[i][j] + std_B[i][j]) / std_B[i][j];
  arma_A = cuarma::blas::element_div(arma_A + arma_B, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] += (std_A[i][j] + std_B[i][j]) / std_B[i][j];
  arma_A += cuarma::blas::element_div(arma_A + arma_B, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] -= (std_A[i][j] + std_B[i][j]) / std_B[i][j];
  arma_A -= cuarma::blas::element_div(arma_A + arma_B, arma_B);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] = std_A[i][j] / (std_B[i][j] + std_A[i][j]);
  arma_A = cuarma::blas::element_div(arma_A, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] += std_A[i][j] / (std_B[i][j] + std_A[i][j]);
  arma_A += cuarma::blas::element_div(arma_A, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] -= std_A[i][j] / (std_B[i][j] + std_A[i][j]);
  arma_A -= cuarma::blas::element_div(arma_A, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] = (std_A[i][j] + std_B[i][j]) / (std_B[i][j] + std_A[i][j]);
  arma_A = cuarma::blas::element_div(arma_A + arma_B, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] += (std_A[i][j] + std_B[i][j]) / (std_B[i][j] + std_A[i][j]);
  arma_A += cuarma::blas::element_div(arma_A + arma_B, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_A[i][j] -= (std_A[i][j] + std_B[i][j]) / (std_B[i][j] + std_A[i][j]);
  arma_A -= cuarma::blas::element_div(arma_A + arma_B, arma_B + arma_A);

  if (!check_for_equality(std_A, arma_A))
    return EXIT_FAILURE;

  std::cout << "Testing unary elementwise operations..." << std::endl;

#define GENERATE_UNARY_OP_TEST(FUNCNAME) \
  std_B = std::vector<std::vector<cpu_value_type> >(std_B.size(), std::vector<cpu_value_type>(std_B[0].size(), 1)); \
  for (std::size_t i=0; i<std_A.size(); ++i) \
    for (std::size_t j=0; j<std_A[i].size(); ++j) {\
      std_A[i][j] = 3 * std_B[i][j]; \
      std_C[i][j] = 2 * std_A[i][j]; \
    } \
  cuarma::copy(std_A, arma_A); \
  cuarma::copy(std_B, arma_B); \
  cuarma::copy(std_C, arma_C); \
  cuarma::copy(std_B, arma_B); \
  \
  for (std::size_t i=0; i<std_C.size(); ++i) \
    for (std::size_t j=0; j<std_C[i].size(); ++j) \
      std_C[i][j] = std::FUNCNAME(std_A[i][j]); \
  arma_C = cuarma::blas::element_##FUNCNAME(arma_A); \
 \
  if (!check_for_equality(std_C, arma_C)) \
  { \
    std::cout << "Failure at C = " << #FUNCNAME << "(A)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<std_C.size(); ++i) \
    for (std::size_t j=0; j<std_C[i].size(); ++j) \
      std_C[i][j] = std::FUNCNAME(std_A[i][j] + std_B[i][j]); \
  arma_C = cuarma::blas::element_##FUNCNAME(arma_A + arma_B); \
 \
  if (!check_for_equality(std_C, arma_C)) \
  { \
    std::cout << "Failure at C = " << #FUNCNAME << "(A + B)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<std_C.size(); ++i) \
    for (std::size_t j=0; j<std_C[i].size(); ++j) \
      std_C[i][j] += std::FUNCNAME(std_A[i][j]); \
  arma_C += cuarma::blas::element_##FUNCNAME(arma_A); \
 \
  if (!check_for_equality(std_C, arma_C)) \
  { \
    std::cout << "Failure at C += " << #FUNCNAME << "(A)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<std_C.size(); ++i) \
    for (std::size_t j=0; j<std_C[i].size(); ++j) \
      std_C[i][j] += std::FUNCNAME(std_A[i][j] + std_B[i][j]); \
  arma_C += cuarma::blas::element_##FUNCNAME(arma_A + arma_B); \
 \
  if (!check_for_equality(std_C, arma_C)) \
  { \
    std::cout << "Failure at C += " << #FUNCNAME << "(A + B)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<std_C.size(); ++i) \
    for (std::size_t j=0; j<std_C[i].size(); ++j) \
      std_C[i][j] -= std::FUNCNAME(std_A[i][j]); \
  arma_C -= cuarma::blas::element_##FUNCNAME(arma_A); \
 \
  if (!check_for_equality(std_C, arma_C)) \
  { \
    std::cout << "Failure at C -= " << #FUNCNAME << "(A)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \
  for (std::size_t i=0; i<std_C.size(); ++i) \
    for (std::size_t j=0; j<std_C[i].size(); ++j) \
      std_C[i][j] -= std::FUNCNAME(std_A[i][j] + std_B[i][j]); \
  arma_C -= cuarma::blas::element_##FUNCNAME(arma_A + arma_B); \
 \
  if (!check_for_equality(std_C, arma_C)) \
  { \
    std::cout << "Failure at C -= " << #FUNCNAME << "(A + B)" << std::endl; \
    return EXIT_FAILURE; \
  } \
 \

  GENERATE_UNARY_OP_TEST(abs);

  std::cout << "Complicated expressions: ";
  //std::cout << "std_A: " << std_A << std::endl;
  //std::cout << "std_B: " << std_B << std::endl;
  //std::cout << "std_C: " << std_C << std::endl;
  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_B[i][j] += alpha * (- std_A[i][j] - beta * std_C[i][j] + std_A[i][j]);
  arma_B += gpu_alpha * (- arma_A - arma_C * beta + arma_A);

  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_B[i][j] += (- std_A[i][j] - beta * std_C[i][j] + std_A[i][j] * beta) / gpu_alpha;
  arma_B   += (-   arma_A - arma_C * beta + gpu_beta * arma_A) / gpu_alpha;

  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;


  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_B[i][j] -= alpha * (- std_A[i][j] - beta * std_C[i][j] - std_A[i][j]);
  arma_B   -= gpu_alpha * (-   arma_A - arma_C * beta - arma_A);

  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_A.size(); ++i)
    for (std::size_t j=0; j<std_A[i].size(); ++j)
      std_B[i][j] -= (- std_A[i][j] - beta * std_C[i][j] - std_A[i][j] * beta) / alpha;
  arma_B   -= (-   arma_A - arma_C * beta - gpu_beta * arma_A) / gpu_alpha;

  if (!check_for_equality(std_B, arma_B))
    return EXIT_FAILURE;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;


  return EXIT_SUCCESS;
}




template<typename T, typename ScalarType>
int run_test()
{
    typedef cuarma::matrix<ScalarType, T>    VCLMatrixType;

    std::size_t dim_rows = 131;
    std::size_t dim_cols = 33;
    //std::size_t dim_rows = 5;
    //std::size_t dim_cols = 3;

    //setup std objects:
    std::vector<std::vector<ScalarType> > std_A(dim_rows, std::vector<ScalarType>(dim_cols));
    std::vector<std::vector<ScalarType> > std_B(dim_rows, std::vector<ScalarType>(dim_cols));
    std::vector<std::vector<ScalarType> > std_C(dim_rows, std::vector<ScalarType>(dim_cols));

    for (std::size_t i=0; i<std_A.size(); ++i)
      for (std::size_t j=0; j<std_A[i].size(); ++j)
      {
        std_A[i][j] = ScalarType((i+2) + (j+1)*(i+2));
        std_B[i][j] = ScalarType((j+2) + (j+1)*(j+2));
        std_C[i][j] = ScalarType((i+1) + (i+1)*(i+2));
      }

    std::vector<std::vector<ScalarType> > std_A_large(4 * dim_rows, std::vector<ScalarType>(4 * dim_cols));
    for (std::size_t i=0; i<std_A_large.size(); ++i)
      for (std::size_t j=0; j<std_A_large[i].size(); ++j)
        std_A_large[i][j] = ScalarType(i * std_A_large[i].size() + j);

    //Setup cuarma objects
    VCLMatrixType arma_A_full(4 * dim_rows, 4 * dim_cols);
    VCLMatrixType arma_B_full(4 * dim_rows, 4 * dim_cols);
    VCLMatrixType arma_C_full(4 * dim_rows, 4 * dim_cols);

    cuarma::copy(std_A_large, arma_A_full);
    cuarma::copy(std_A_large, arma_B_full);
    cuarma::copy(std_A_large, arma_C_full);

    //
    // Create A
    //
    VCLMatrixType arma_A(dim_rows, dim_cols);

    cuarma::range arma_A_r1(2 * dim_rows, 3 * dim_rows);
    cuarma::range arma_A_r2(dim_cols, 2 * dim_cols);
    cuarma::matrix_range<VCLMatrixType>   arma_range_A(arma_A_full, arma_A_r1, arma_A_r2);

    cuarma::slice arma_A_s1(2, 3, dim_rows);
    cuarma::slice arma_A_s2(2 * dim_cols, 2, dim_cols);
    cuarma::matrix_slice<VCLMatrixType>   arma_slice_A(arma_A_full, arma_A_s1, arma_A_s2);


    //
    // Create B
    //
    VCLMatrixType arma_B(dim_rows, dim_cols);

    cuarma::range arma_B_r1(dim_rows, 2 * dim_rows);
    cuarma::range arma_B_r2(2 * dim_cols, 3 * dim_cols);
    cuarma::matrix_range<VCLMatrixType>   arma_range_B(arma_B_full, arma_B_r1, arma_B_r2);

    cuarma::slice arma_B_s1(2 * dim_rows, 2, dim_rows);
    cuarma::slice arma_B_s2(dim_cols, 3, dim_cols);
    cuarma::matrix_slice<VCLMatrixType>   arma_slice_B(arma_B_full, arma_B_s1, arma_B_s2);


    //
    // Create C
    //
    VCLMatrixType arma_C(dim_rows, dim_cols);

    cuarma::range arma_C_r1(2 * dim_rows, 3 * dim_rows);
    cuarma::range arma_C_r2(3 * dim_cols, 4 * dim_cols);
    cuarma::matrix_range<VCLMatrixType>   arma_range_C(arma_C_full, arma_C_r1, arma_C_r2);

    cuarma::slice arma_C_s1(dim_rows, 2, dim_rows);
    cuarma::slice arma_C_s2(0, 3, dim_cols);
    cuarma::matrix_slice<VCLMatrixType>   arma_slice_C(arma_C_full, arma_C_s1, arma_C_s2);

    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_A, arma_slice_A);

    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_B, arma_slice_B);

    cuarma::copy(std_C, arma_C);
    cuarma::copy(std_C, arma_range_C);
    cuarma::copy(std_C, arma_slice_C);


    std::cout << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Copy CTOR //////////" << std::endl;
    std::cout << "//" << std::endl;

    {
      std::cout << "Testing matrix created from range... ";
      VCLMatrixType arma_temp = arma_range_A;
      if (check_for_equality(std_A, arma_temp))
        std::cout << "PASSED!" << std::endl;
      else
      {
        std::cout << "arma_temp: " << arma_temp << std::endl;
        std::cout << "arma_range_A: " << arma_range_A << std::endl;
        std::cout << "arma_A: " << arma_A << std::endl;
        std::cout << std::endl << "TEST failed!" << std::endl;
        return EXIT_FAILURE;
      }

      std::cout << "Testing matrix created from slice... ";
      VCLMatrixType arma_temp2 = arma_range_B;
      if (check_for_equality(std_B, arma_temp2))
        std::cout << "PASSED!" << std::endl;
      else
      {
        std::cout << std::endl << "TEST failed!" << std::endl;
        return EXIT_FAILURE;
      }
    }

    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Initializer for matrix type //////////" << std::endl;
    std::cout << "//" << std::endl;

    {
      std::vector<std::vector<ScalarType> > std_dummy1(std_A.size(), std::vector<ScalarType>(std_A.size()));
      for (std::size_t i=0; i<std_A.size(); ++i) std_dummy1[i][i] = ScalarType(1);
      std::vector<std::vector<ScalarType> > std_dummy2(std_A.size(), std::vector<ScalarType>(std_A.size(), 3));
      std::vector<std::vector<ScalarType> > std_dummy3(std_A.size(), std::vector<ScalarType>(std_A.size()));

      cuarma::matrix<ScalarType> arma_dummy1 = cuarma::identity_matrix<ScalarType>(std_A.size());
      cuarma::matrix<ScalarType> arma_dummy2 = cuarma::scalar_matrix<ScalarType>(std_A.size(), std_A.size(), 3);
      cuarma::matrix<ScalarType> arma_dummy3 = cuarma::zero_matrix<ScalarType>(std_A.size(), std_A.size());

      std::cout << "Testing initializer CTOR... ";
      if (   check_for_equality(std_dummy1, arma_dummy1)
          && check_for_equality(std_dummy2, arma_dummy2)
          && check_for_equality(std_dummy3, arma_dummy3)
         )
        std::cout << "PASSED!" << std::endl;
      else
      {
        std::cout << std::endl << "TEST failed!" << std::endl;
        return EXIT_FAILURE;
      }

      std_dummy1 = std::vector<std::vector<ScalarType> >(std_A.size(), std::vector<ScalarType>(std_A.size()));
      std_dummy2 = std::vector<std::vector<ScalarType> >(std_A.size(), std::vector<ScalarType>(std_A.size()));
      for (std::size_t i=0; i<std_A.size(); ++i) std_dummy2[i][i] = ScalarType(1);
      std_dummy3 = std::vector<std::vector<ScalarType> >(std_A.size(), std::vector<ScalarType>(std_A.size(), 3));

      arma_dummy1 = cuarma::zero_matrix<ScalarType>(std_A.size(), std_A.size());
      arma_dummy2 = cuarma::identity_matrix<ScalarType>(std_A.size());
      arma_dummy3 = cuarma::scalar_matrix<ScalarType>(std_A.size(), std_A.size(), 3);

      std::cout << "Testing initializer assignment... ";
      if (   check_for_equality(std_dummy1, arma_dummy1)
          && check_for_equality(std_dummy2, arma_dummy2)
          && check_for_equality(std_dummy3, arma_dummy3)
         )
        std::cout << "PASSED!" << std::endl;
      else
      {
        std::cout << std::endl << "TEST failed!" << std::endl;
        return EXIT_FAILURE;
      }
    }


    //
    // run operation tests:
    //

    /////// A=matrix:
    std::cout << "Testing A=matrix, B=matrix, C=matrix ..." << std::endl;
    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_C, arma_C);
    if (run_test(std_A, std_B, std_C,
                 arma_A, arma_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=matrix, C=range ..." << std::endl;
    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_C, arma_range_C);
    if (run_test(std_A, std_B, std_C,
                 arma_A, arma_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=matrix, C=slice ..." << std::endl;
    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_C, arma_slice_C);
    if (run_test(std_A, std_B, std_C,
                 arma_A, arma_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=range, C=matrix ..." << std::endl;
    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_C, arma_C);
    if (run_test(std_A, std_B, std_C,
                 arma_A, arma_range_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=range, C=range ..." << std::endl;
    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_C, arma_range_C);
    if (run_test(std_A, std_B, std_C,
                 arma_A, arma_range_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=range, C=slice ..." << std::endl;
    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_C, arma_slice_C);
    if (run_test(std_A, std_B, std_C,
                 arma_A, arma_range_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }


    std::cout << "Testing A=matrix, B=slice, C=matrix ..." << std::endl;
    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_B, arma_slice_B);
    cuarma::copy(std_C, arma_C);
    if (run_test(std_A, std_B, std_C,
                 arma_A, arma_slice_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=slice, C=range ..." << std::endl;
    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_B, arma_slice_B);
    cuarma::copy(std_C, arma_range_C);
    if (run_test(std_A, std_B, std_C,
                 arma_A, arma_slice_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=slice, C=slice ..." << std::endl;
    cuarma::copy(std_A, arma_A);
    cuarma::copy(std_B, arma_slice_B);
    cuarma::copy(std_C, arma_slice_C);
    if (run_test(std_A, std_B, std_C,
                 arma_A, arma_slice_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    /////// A=range:
    std::cout << "Testing A=range, B=matrix, C=matrix ..." << std::endl;
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_C, arma_C);
    if (run_test(std_A, std_B, std_C,
                 arma_range_A, arma_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=matrix, C=range ..." << std::endl;
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_C, arma_range_C);
    if (run_test(std_A, std_B, std_C,
                 arma_range_A, arma_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=matrix, C=slice ..." << std::endl;
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_C, arma_slice_C);
    if (run_test(std_A, std_B, std_C,
                 arma_range_A, arma_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    std::cout << "Testing A=range, B=range, C=matrix ..." << std::endl;
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_C, arma_C);
    if (run_test(std_A, std_B, std_C,
                 arma_range_A, arma_range_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=range, C=range ..." << std::endl;
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_C, arma_range_C);
    if (run_test(std_A, std_B, std_C,
                 arma_range_A, arma_range_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=range, C=slice ..." << std::endl;
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_C, arma_slice_C);
    if (run_test(std_A, std_B, std_C,
                 arma_range_A, arma_range_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    std::cout << "Testing A=range, B=slice, C=matrix ..." << std::endl;
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_B, arma_slice_B);
    cuarma::copy(std_C, arma_C);
    if (run_test(std_A, std_B, std_C,
                 arma_range_A, arma_slice_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=slice, C=range ..." << std::endl;
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_B, arma_slice_B);
    cuarma::copy(std_C, arma_range_C);
    if (run_test(std_A, std_B, std_C,
                 arma_range_A, arma_slice_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=slice, C=slice ..." << std::endl;
    cuarma::copy(std_A, arma_range_A);
    cuarma::copy(std_B, arma_slice_B);
    cuarma::copy(std_C, arma_slice_C);
    if (run_test(std_A, std_B, std_C,
                 arma_range_A, arma_slice_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }


    /////// A=slice:
    std::cout << "Testing A=slice, B=matrix, C=matrix ..." << std::endl;
    cuarma::copy(std_A, arma_slice_A);
    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_C, arma_C);
    if (run_test(std_A, std_B, std_C,
                 arma_slice_A, arma_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=matrix, C=range ..." << std::endl;
    cuarma::copy(std_A, arma_slice_A);
    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_C, arma_range_C);
    if (run_test(std_A, std_B, std_C,
                 arma_slice_A, arma_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=matrix, C=slice ..." << std::endl;
    cuarma::copy(std_A, arma_slice_A);
    cuarma::copy(std_B, arma_B);
    cuarma::copy(std_C, arma_slice_C);
    if (run_test(std_A, std_B, std_C,
                 arma_slice_A, arma_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    std::cout << "Testing A=slice, B=range, C=matrix ..." << std::endl;
    cuarma::copy(std_A, arma_slice_A);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_C, arma_C);
    if (run_test(std_A, std_B, std_C,
                 arma_slice_A, arma_range_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=range, C=range ..." << std::endl;
    cuarma::copy(std_A, arma_slice_A);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_C, arma_range_C);
    if (run_test(std_A, std_B, std_C,
                 arma_slice_A, arma_range_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=range, C=slice ..." << std::endl;
    cuarma::copy(std_A, arma_slice_A);
    cuarma::copy(std_B, arma_range_B);
    cuarma::copy(std_C, arma_slice_C);
    if (run_test(std_A, std_B, std_C,
                 arma_slice_A, arma_range_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    std::cout << "Testing A=slice, B=slice, C=matrix ..." << std::endl;
    cuarma::copy(std_A, arma_slice_A);
    cuarma::copy(std_B, arma_slice_B);
    cuarma::copy(std_C, arma_C);
    if (run_test(std_A, std_B, std_C,
                 arma_slice_A, arma_slice_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=slice, C=range ..." << std::endl;
    cuarma::copy(std_A, arma_slice_A);
    cuarma::copy(std_B, arma_slice_B);
    cuarma::copy(std_C, arma_range_C);
    if (run_test(std_A, std_B, std_C,
                 arma_slice_A, arma_slice_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=slice, C=slice ..." << std::endl;
    cuarma::copy(std_A, arma_slice_A);
    cuarma::copy(std_B, arma_slice_B);
    cuarma::copy(std_C, arma_slice_C);
    if (run_test(std_A, std_B, std_C,
                 arma_slice_A, arma_slice_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;
}


