/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   vector_int.cu
 *   @coding UTF-8
 *   @brief  Tests vector operations (BLAS level 1) for unsigned integer arithmetic.
 *   @brief  测试：向量-整数标量运算
 */

#include <iostream>
#include <iomanip>
#include <vector>

#include "head_define.h"

#include "cuarma/vector.hpp"
#include "cuarma/vector_proxy.hpp"
#include "cuarma/blas/inner_prod.hpp"
#include "cuarma/blas/norm_1.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/norm_inf.hpp"
#include "cuarma/blas/maxmin.hpp"
#include "cuarma/blas/sum.hpp"


//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
  cuarma::backend::finish();
  return s1 - s2;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, cuarma::scalar<ScalarType> const & s2)
{
  cuarma::backend::finish();
  return s1 - s2;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, cuarma::entry_proxy<ScalarType> const & s2)
{
  cuarma::backend::finish();
  return s1 - s2;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType, typename ARMAVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, ARMAVectorType const & v2)
{
   std::vector<ScalarType> v2_cpu(v2.size());
   cuarma::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   cuarma::copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if (v2_cpu[i] != v1[i])
        return 1;
   }

   return 0;
}


template<typename T1, typename T2>
int check(T1 const & t1, T2 const & t2)
{
  int retval = EXIT_SUCCESS;

  if (diff(t1, t2) != 0)
  {
    std::cout << "# Error! Difference: " << diff(t1, t2) << std::endl;
    retval = EXIT_FAILURE;
  }
  return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename STLVectorType, typename cuarmaVectorType1, typename cuarmaVectorType2 >
int test(STLVectorType       & std_v1, STLVectorType       & std_v2,
         cuarmaVectorType1 & arma_v1, cuarmaVectorType2 & arma_v2)
{
  int retval = EXIT_SUCCESS;

  NumericT                    cpu_result = 42;
  cuarma::scalar<NumericT>  gpu_result = 43;

  //
  // Initializer:
  //
  std::cout << "Checking for zero_vector initializer..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = 0;
  arma_v1 = cuarma::zero_vector<NumericT>(arma_v1.size());
  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for scalar_vector initializer..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = cpu_result;
  arma_v1 = cuarma::scalar_vector<NumericT>(arma_v1.size(), cpu_result);
  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = cpu_result + 1;
  arma_v1 = cuarma::scalar_vector<NumericT>(arma_v1.size(), gpu_result);
  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for unit_vector initializer..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (i == 5) ? 1 : 0;
  arma_v1 = cuarma::unit_vector<NumericT>(arma_v1.size(), 5);
  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(i);
    std_v2[i] = NumericT(i+42);
  }

  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());  //resync
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v2, arma_v2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  //
  // Part 1: Norms and inner product
  //

  // --------------------------------------------------------------------------
  std::cout << "Testing inner_prod..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * std_v2[i];
  NumericT cpu_result2 = cuarma::blas::inner_prod(arma_v1, arma_v2);
  gpu_result = cuarma::blas::inner_prod(arma_v1, arma_v2);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += (std_v1[i] + std_v2[i]) * (2*std_v2[i]);
  NumericT cpu_result3 = cuarma::blas::inner_prod(arma_v1 + arma_v2, 2*arma_v2);
  gpu_result = cuarma::blas::inner_prod(arma_v1 + arma_v2, 2*arma_v2);

  if (check(cpu_result, cpu_result3) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing norm_1..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)   //note: norm_1 broken for unsigned ints on MacOS
    cpu_result += std_v1[i];
  gpu_result = cuarma::blas::norm_1(arma_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result2 = 0; //reset
  for (std::size_t i=0; i<std_v1.size(); ++i)   //note: norm_1 broken for unsigned ints on MacOS
    cpu_result2 += std_v1[i];
  cpu_result = cuarma::blas::norm_1(arma_v1);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result2 = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)   //note: norm_1 broken for unsigned ints on MacOS
    cpu_result2 += std_v1[i] + std_v2[i];
  cpu_result = cuarma::blas::norm_1(arma_v1 + arma_v2);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing norm_inf..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] > cpu_result)
      cpu_result = std_v1[i];
  gpu_result = cuarma::blas::norm_inf(arma_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result2 = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] > cpu_result2)
      cpu_result2 = std_v1[i];
  cpu_result = cuarma::blas::norm_inf(arma_v1);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result2 = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] + std_v2[i] > cpu_result2)
      cpu_result2 = std_v1[i] + std_v2[i];
  cpu_result = cuarma::blas::norm_inf(arma_v1 + arma_v2);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing index_norm_inf..." << std::endl;

  std::size_t cpu_index = 0;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] > cpu_result)
    {
      cpu_result = std_v1[i];
      cpu_index = i;
    }
  std::size_t gpu_index = cuarma::blas::index_norm_inf(arma_v1);

  if (check(static_cast<NumericT>(cpu_index), static_cast<NumericT>(gpu_index)) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  // --------------------------------------------------------------------------
  gpu_result = arma_v1[cuarma::blas::index_norm_inf(arma_v1)];

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_index = 0;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    if (std_v1[i] + std_v2[i] > cpu_result)
    {
      cpu_result = std_v1[i];
      cpu_index = i;
    }
  gpu_result = arma_v1[cuarma::blas::index_norm_inf(arma_v1 + arma_v2)];

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing max..." << std::endl;
  cpu_result = std_v1[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max<NumericT>(cpu_result, std_v1[i]);
  gpu_result = cuarma::blas::max(arma_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std_v1[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max<NumericT>(cpu_result, std_v1[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = cuarma::blas::max(arma_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std_v1[0] + std_v2[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max<NumericT>(cpu_result, std_v1[i] + std_v2[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = cuarma::blas::max(arma_v1 + arma_v2);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  // --------------------------------------------------------------------------
  std::cout << "Testing min..." << std::endl;
  cpu_result = std_v1[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::min<NumericT>(cpu_result, std_v1[i]);
  gpu_result = cuarma::blas::min(arma_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std_v1[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::min<NumericT>(cpu_result, std_v1[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = cuarma::blas::min(arma_v1);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = std_v1[0] + std_v2[0];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::min<NumericT>(cpu_result, std_v1[i] + std_v2[i]);
  gpu_result = cpu_result;
  cpu_result *= 2; //reset
  cpu_result = cuarma::blas::min(arma_v1 + arma_v2);

  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  std::cout << "Testing sum..." << std::endl;
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i];
  cpu_result2 = cuarma::blas::sum(arma_v1);
  gpu_result = cuarma::blas::sum(arma_v1);

  if (check(cpu_result, cpu_result2) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] + std_v2[i];
  cpu_result3 = cuarma::blas::sum(arma_v1 + arma_v2);
  gpu_result = cuarma::blas::sum(arma_v1 + arma_v2);

  if (check(cpu_result, cpu_result3) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(cpu_result, gpu_result) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  // --------------------------------------------------------------------------

  std::cout << "Testing assignments..." << std::endl;
  NumericT val = static_cast<NumericT>(1);
  for (size_t i=0; i < std_v1.size(); ++i)
    std_v1[i] = val;

  for (size_t i=0; i < arma_v1.size(); ++i)
    arma_v1(i) = val;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // multiplication and division of vectors by scalars
  //
  std::cout << "Testing scaling with CPU scalar..." << std::endl;
  NumericT alpha = static_cast<NumericT>(3);
  cuarma::scalar<NumericT> gpu_alpha = alpha;

  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] *= alpha;
  arma_v1 *= alpha;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing scaling with GPU scalar..." << std::endl;
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] *= alpha;
  arma_v1 *= gpu_alpha;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  NumericT beta  = static_cast<NumericT>(2);
  cuarma::scalar<NumericT> gpu_beta = beta;

  std::cout << "Testing shrinking with CPU scalar..." << std::endl;
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] /= beta;
  arma_v1 /= beta;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing shrinking with GPU scalar..." << std::endl;
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] /= beta;
  arma_v1 /= gpu_beta;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // add and inplace_add of vectors
  //
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = NumericT(i);
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());  //resync
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  std::cout << "Testing add on vector..." << std::endl;

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v2, arma_v2) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i];
  arma_v1 = arma_v1 + arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace-add on vector..." << std::endl;
  for (size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v2[i];
  arma_v1 += arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // multiply-add
  //
  std::cout << "Testing multiply-add on vector with CPU scalar (right)..." << std::endl;
  for (size_t i=0; i < std_v1.size(); ++i)
    std_v1[i] = NumericT(i);
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + alpha * std_v2[i];
  arma_v1 = arma_v1 + alpha * arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with CPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = alpha * std_v1[i] + std_v2[i];
  arma_v1 = alpha * arma_v1 + arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with CPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = alpha * std_v1[i] + beta * std_v2[i];
  arma_v1 = alpha * arma_v1 + beta * arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += alpha * std_v2[i];
  arma_v1 += alpha * arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-add on vector with GPU scalar (right)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + alpha * std_v2[i];
  arma_v1   = arma_v1   + gpu_alpha *   arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with GPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + alpha * std_v2[i];
  arma_v1 = arma_v1 + gpu_alpha * arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing multiply-add on vector with GPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = alpha * std_v1[i] + beta * std_v2[i];
  arma_v1 = gpu_alpha * arma_v1 + gpu_beta * arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-add on vector with GPU scalar (both, adding)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += alpha * std_v1[i] + beta * std_v2[i];
  arma_v1 += gpu_alpha * arma_v1 + gpu_beta * arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace multiply-add on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += alpha * std_v2[i];
  arma_v1 += gpu_alpha * arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // division-add
  //
  std::cout << "Testing division-add on vector with CPU scalar (right)..." << std::endl;
  for (size_t i=0; i < std_v1.size(); ++i)
    std_v1[i] = NumericT(i);
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i] / alpha;
  arma_v1 = arma_v1 + arma_v2 / alpha;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing division-add on vector with CPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / alpha + std_v2[i];
  arma_v1 = arma_v1 / alpha + arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-add on vector with CPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / alpha + std_v2[i] / beta;
  arma_v1 = arma_v1 / alpha + arma_v2 / beta;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-multiply-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / alpha + std_v2[i] * beta;
  arma_v1 = arma_v1 / alpha + arma_v2 * beta;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing multiply-division-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * alpha + std_v2[i] / beta;
  arma_v1 = arma_v1 * alpha + arma_v2 / beta;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;



  std::cout << "Testing inplace division-add on vector with CPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v2[i] / alpha;
  arma_v1 += arma_v2 / alpha;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing division-add on vector with GPU scalar (right)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i] / alpha;
  arma_v1 = arma_v1 + arma_v2 / gpu_alpha;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-add on vector with GPU scalar (left)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i] / alpha;
  arma_v1   = arma_v1   +   arma_v2 / gpu_alpha;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing division-add on vector with GPU scalar (both)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / alpha + std_v2[i] / beta;
  arma_v1 = arma_v1 / gpu_alpha + arma_v2 / gpu_beta;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace division-add on vector with GPU scalar (both, adding)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] / alpha + std_v2[i] / beta;
  arma_v1 += arma_v1 / gpu_alpha + arma_v2 / gpu_beta;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing inplace division-multiply-add on vector with GPU scalar (adding)..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] / alpha + std_v2[i] * beta;
  arma_v1 += arma_v1 / gpu_alpha + arma_v2 * gpu_beta;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing inplace division-add on vector with GPU scalar..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v2[i] * alpha;
  arma_v1 += arma_v2 * gpu_alpha;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  //
  // More complicated expressions (for ensuring the operator overloads work correctly)
  //
  for (size_t i=0; i < std_v1.size(); ++i)
    std_v1[i] = NumericT(i);
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  std::cout << "Testing three vector additions..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v2[i] + std_v1[i] + std_v2[i];
  arma_v1 = arma_v2 + arma_v1 + arma_v2;

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v2[i] = 3 * std_v1[i];
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  std::cout << "Testing swap..." << std::endl;
  swap(std_v1, std_v2);
  swap(arma_v1, arma_v2);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Testing elementwise multiplication..." << std::endl;
  std::cout << " v1 = element_prod(v1, v2);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * std_v2[i];
  arma_v1 = cuarma::blas::element_prod(arma_v1, arma_v2);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1, v2);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] * std_v2[i];
  arma_v1 += cuarma::blas::element_prod(arma_v1, arma_v2);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  std::cout << " v1 = element_prod(v1 + v2, v2);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) * std_v2[i];
  arma_v1 = cuarma::blas::element_prod(arma_v1 + arma_v2, arma_v2);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1 + v2, v2);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += (std_v1[i] + std_v2[i]) * std_v2[i];
  arma_v1 += cuarma::blas::element_prod(arma_v1 + arma_v2, arma_v2);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  std::cout << " v1 = element_prod(v1, v2 + v1);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * (std_v2[i] + std_v1[i]);
  arma_v1 = cuarma::blas::element_prod(arma_v1, arma_v2 + arma_v1);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1, v2 + v1);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] * (std_v2[i] + std_v1[i]);
  arma_v1 += cuarma::blas::element_prod(arma_v1, arma_v2 + arma_v1);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  std::cout << " v1 = element_prod(v1 + v2, v2 + v1);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) * (std_v2[i] + std_v1[i]);
  arma_v1 = cuarma::blas::element_prod(arma_v1 + arma_v2, arma_v2 + arma_v1);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " v1 += element_prod(v1 + v2, v2 + v1);" << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += (std_v1[i] + std_v2[i]) * (std_v2[i] + std_v1[i]);
  arma_v1 += cuarma::blas::element_prod(arma_v1 + arma_v2, arma_v2 + arma_v1);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  std::cout << "Testing elementwise division..." << std::endl;
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(1 + i);
    std_v2[i] = NumericT(5 + i);
  }

  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / std_v2[i];
  arma_v1 = cuarma::blas::element_div(arma_v1, arma_v2);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] / std_v2[i];
  arma_v1 += cuarma::blas::element_div(arma_v1, arma_v2);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) / std_v2[i];
  arma_v1 = cuarma::blas::element_div(arma_v1 + arma_v2, arma_v2);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += (std_v1[i] + std_v2[i]) / std_v2[i];
  arma_v1 += cuarma::blas::element_div(arma_v1 + arma_v2, arma_v2);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / (std_v2[i] + std_v1[i]);
  arma_v1 = cuarma::blas::element_div(arma_v1, arma_v2 + arma_v1);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v1[i] / (std_v2[i] + std_v1[i]);
  arma_v1 += cuarma::blas::element_div(arma_v1, arma_v2 + arma_v1);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) / (std_v2[i] + std_v1[i]);
  arma_v1 = cuarma::blas::element_div(arma_v1 + arma_v2, arma_v2 + arma_v1);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += (std_v1[i] + std_v2[i]) / (std_v2[i] + std_v1[i]);
  arma_v1 += cuarma::blas::element_div(arma_v1 + arma_v2, arma_v2 + arma_v1);

  if (check(std_v1, arma_v1) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  // --------------------------------------------------------------------------
  return retval;
}


template< typename NumericT >
int test()
{
  int retval = EXIT_SUCCESS;
  std::size_t size = 12345;

  std::cout << "Running tests for vector of size " << size << std::endl;

  //
  // Set up STL objects
  //
  std::vector<NumericT> std_full_vec(size);
  std::vector<NumericT> std_full_vec2(std_full_vec.size());

  for (std::size_t i=0; i<std_full_vec.size(); ++i)
  {
    std_full_vec[i]  = NumericT(1.0) + NumericT(i);
    std_full_vec2[i] = NumericT(2.0) + NumericT(i) / NumericT(2);
  }

  std::vector<NumericT> std_range_vec (2 * std_full_vec.size() / 4 - std_full_vec.size() / 4);
  std::vector<NumericT> std_range_vec2(2 * std_full_vec.size() / 4 - std_full_vec.size() / 4);

  for (std::size_t i=0; i<std_range_vec.size(); ++i)
    std_range_vec[i] = std_full_vec[i + std_full_vec.size() / 4];
  for (std::size_t i=0; i<std_range_vec2.size(); ++i)
    std_range_vec2[i] = std_full_vec2[i + 2 * std_full_vec2.size() / 4];

  std::vector<NumericT> std_slice_vec (std_full_vec.size() / 4);
  std::vector<NumericT> std_slice_vec2(std_full_vec.size() / 4);

  for (std::size_t i=0; i<std_slice_vec.size(); ++i)
    std_slice_vec[i] = std_full_vec[3*i + std_full_vec.size() / 4];
  for (std::size_t i=0; i<std_slice_vec2.size(); ++i)
    std_slice_vec2[i] = std_full_vec2[2*i + 2 * std_full_vec2.size() / 4];

  //
  // Set up cuarma objects
  //
  cuarma::vector<NumericT> arma_full_vec(std_full_vec.size());
  cuarma::vector<NumericT> arma_full_vec2(std_full_vec2.size());

  cuarma::fast_copy(std_full_vec.begin(), std_full_vec.end(), arma_full_vec.begin());
  cuarma::copy(std_full_vec2.begin(), std_full_vec2.end(), arma_full_vec2.begin());

  cuarma::range arma_r1(    arma_full_vec.size() / 4, 2 * arma_full_vec.size() / 4);
  cuarma::range arma_r2(2 * arma_full_vec2.size() / 4, 3 * arma_full_vec2.size() / 4);
  cuarma::vector_range< cuarma::vector<NumericT> > arma_range_vec(arma_full_vec, arma_r1);
  cuarma::vector_range< cuarma::vector<NumericT> > arma_range_vec2(arma_full_vec2, arma_r2);

  {
    cuarma::vector<NumericT> arma_short_vec(arma_range_vec);
    cuarma::vector<NumericT> arma_short_vec2 = arma_range_vec2;

    std::vector<NumericT> std_short_vec(std_range_vec);
    std::vector<NumericT> std_short_vec2(std_range_vec2);

    std::cout << "Testing creation of vectors from range..." << std::endl;
    if (check(std_short_vec, arma_short_vec) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    if (check(std_short_vec2, arma_short_vec2) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

  cuarma::slice arma_s1(    arma_full_vec.size() / 4, 3, arma_full_vec.size() / 4);
  cuarma::slice arma_s2(2 * arma_full_vec2.size() / 4, 2, arma_full_vec2.size() / 4);
  cuarma::vector_slice< cuarma::vector<NumericT> > arma_slice_vec(arma_full_vec, arma_s1);
  cuarma::vector_slice< cuarma::vector<NumericT> > arma_slice_vec2(arma_full_vec2, arma_s2);

  cuarma::vector<NumericT> arma_short_vec(arma_slice_vec);
  cuarma::vector<NumericT> arma_short_vec2 = arma_slice_vec2;

  std::vector<NumericT> std_short_vec(std_slice_vec);
  std::vector<NumericT> std_short_vec2(std_slice_vec2);

  std::cout << "Testing creation of vectors from slice..." << std::endl;
  if (check(std_short_vec, arma_short_vec) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec2, arma_short_vec2) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** arma_v1 = vector, arma_v2 = vector **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          arma_short_vec, arma_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = vector, arma_v2 = range **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          arma_short_vec, arma_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = vector, arma_v2 = slice **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          arma_short_vec, arma_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** arma_v1 = range, arma_v2 = vector **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          arma_range_vec, arma_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = range, arma_v2 = range **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          arma_range_vec, arma_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = range, arma_v2 = slice **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          arma_range_vec, arma_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** arma_v1 = slice, arma_v2 = vector **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          arma_slice_vec, arma_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = slice, arma_v2 = range **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          arma_slice_vec, arma_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = slice, arma_v2 = slice **" << std::endl;
  retval = test<NumericT>(std_short_vec, std_short_vec2,
                          arma_slice_vec, arma_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}



//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Vector with Integer types" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  numeric: unsigned int" << std::endl;
    retval = test<unsigned int>();
    if ( retval == EXIT_SUCCESS )
      std::cout << "# Test passed" << std::endl;
    else
      return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  numeric: long" << std::endl;
    retval = test<unsigned long>();
    if ( retval == EXIT_SUCCESS )
      std::cout << "# Test passed" << std::endl;
    else
      return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return retval;
}
