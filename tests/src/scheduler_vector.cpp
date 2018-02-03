/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   scheduler_vector.cu
 *   @coding UTF-8
 *   @brief  Tests the scheduler for vector-operations.
 *   @brief  测试：向量操作调度器
 */

#include <iostream>
#include <iomanip>
#include <vector>

#include "head_define.h"

#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/vector_proxy.hpp"
#include "cuarma/blas/inner_prod.hpp"
#include "cuarma/blas/norm_1.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/norm_inf.hpp"

#include "cuarma/scheduler/execute.hpp"
#include "cuarma/scheduler/io.hpp"

#include "cuarma/tools/random.hpp"


template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
   cuarma::backend::finish();
   if (std::fabs(s1 - s2) > 0)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, cuarma::scalar<ScalarType> const & s2)
{
   cuarma::backend::finish();
   if (std::fabs(s1 - s2) > 0)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType const & s1, cuarma::entry_proxy<ScalarType> const & s2)
{
   cuarma::backend::finish();
   if (std::fabs(s1 - s2) > 0)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template<typename ScalarType, typename cuarmaVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, cuarmaVectorType const & arma_vec)
{
  std::vector<ScalarType> v2_cpu(arma_vec.size());
  cuarma::backend::finish();
  cuarma::copy(arma_vec, v2_cpu);

  ScalarType norm_inf_value = 0;
  for (std::size_t i=0;i<v1.size(); ++i)
  {
    ScalarType tmp = 0;
    if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
       tmp = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );

    norm_inf_value = (tmp > norm_inf_value) ? tmp : norm_inf_value;
  }

  return norm_inf_value;
}


template<typename T1, typename T2>
int check(T1 const & t1, T2 const & t2, double epsilon)
{
  int retval = EXIT_SUCCESS;

  double temp = std::fabs(diff(t1, t2));
  if (temp > epsilon)
  {
    std::cout << "# Error! Relative difference: " << temp << std::endl;
    retval = EXIT_FAILURE;
  }
  else
    std::cout << "PASSED!" << std::endl;
  return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon, typename STLVectorType, typename cuarmaVectorType1, typename cuarmaVectorType2 >
int test(Epsilon const& epsilon,
         STLVectorType       & std_v1, STLVectorType       & std_v2,
         cuarmaVectorType1 & arma_v1, cuarmaVectorType2 & arma_v2)
{
  int retval = EXIT_SUCCESS;

  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  NumericT                    cpu_result = 42.0;
  cuarma::scalar<NumericT>  gpu_result = 43.0;
  NumericT                    alpha      = NumericT(3.1415);
  NumericT                    beta       = NumericT(2.7172);

  //
  // Initializer:
  //
  std::cout << "Checking for zero_vector initializer..." << std::endl;
  std_v1 = std::vector<NumericT>(std_v1.size());
  arma_v1 = cuarma::zero_vector<NumericT>(arma_v1.size());
  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for scalar_vector initializer..." << std::endl;
  std_v1 = std::vector<NumericT>(std_v1.size(), cpu_result);
  arma_v1 = cuarma::scalar_vector<NumericT>(arma_v1.size(), cpu_result);
  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std_v1 = std::vector<NumericT>(std_v1.size(), gpu_result);
  arma_v1 = cuarma::scalar_vector<NumericT>(arma_v1.size(), gpu_result);
  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << "Checking for unit_vector initializer..." << std::endl;
  std_v1 = std::vector<NumericT>(std_v1.size()); std_v1[5] = NumericT(1);
  arma_v1 = cuarma::unit_vector<NumericT>(arma_v1.size(), 5);
  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(1.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }

  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());  //resync
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v2, arma_v2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  // --------------------------------------------------------------------------

  std::cout << "Testing simple assignments..." << std::endl;

  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), arma_v2); // same as arma_v1 = arma_v2;
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_inplace_add(), arma_v2); // same as arma_v1 += arma_v2;
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] -= std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_inplace_sub(), arma_v2); // same as arma_v1 -= arma_v2;
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "Testing composite assignments..." << std::endl;
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] + std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), arma_v1 + arma_v2); // same as arma_v1 = arma_v1 + arma_v2;
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] += alpha * std_v1[i] - beta * std_v2[i] + std_v1[i] / beta - std_v2[i] / alpha;
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_inplace_add(), alpha * arma_v1 - beta * arma_v2 + arma_v1 / beta - arma_v2 / alpha); // same as arma_v1 += alpha * arma_v1 - beta * arma_v2 + beta * arma_v1 - alpha * arma_v2;
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] - std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), arma_v1 - arma_v2); // same as arma_v1 = arma_v1 - arma_v2;
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "--- Testing reductions ---" << std::endl;
  std::cout << "inner_prod..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * std_v2[i];
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::inner_prod(arma_v1, arma_v2)); // same as gpu_result = inner_prod(arma_v1, arma_v2);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += (std_v1[i] + std_v2[i]) * std_v2[i];
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::inner_prod(arma_v1 + arma_v2, arma_v2)); // same as gpu_result = inner_prod(arma_v1 + arma_v2, arma_v2);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * (std_v2[i] - std_v1[i]);
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::inner_prod(arma_v1, arma_v2 - arma_v1)); // same as gpu_result = inner_prod(arma_v1, arma_v2 - arma_v1);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += (std_v1[i] - std_v2[i]) * (std_v2[i] + std_v1[i]);
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::inner_prod(arma_v1 - arma_v2, arma_v2 + arma_v1)); // same as gpu_result = inner_prod(arma_v1 - arma_v2, arma_v2 + arma_v1);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "norm_1..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std::fabs(std_v1[i]);
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::norm_1(arma_v1)); // same as gpu_result = norm_1(arma_v1);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std::fabs(std_v1[i] + std_v2[i]);
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::norm_1(arma_v1 + arma_v2)); // same as gpu_result = norm_1(arma_v1 + arma_v2);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "norm_2..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * std_v1[i];
  cpu_result = std::sqrt(cpu_result);
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::norm_2(arma_v1)); // same as gpu_result = norm_2(arma_v1);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += (std_v1[i] + std_v2[i]) * (std_v1[i] + std_v2[i]);
  cpu_result = std::sqrt(cpu_result);
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::norm_2(arma_v1 + arma_v2)); // same as gpu_result = norm_2(arma_v1 + arma_v2);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "norm_inf..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max(cpu_result, std::fabs(std_v1[i]));
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::norm_inf(arma_v1)); // same as gpu_result = norm_inf(arma_v1);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result = std::max(cpu_result, std::fabs(std_v1[i] - std_v2[i]));
  cuarma::scheduler::statement   my_statement(gpu_result, cuarma::op_assign(), cuarma::blas::norm_inf(arma_v1 - arma_v2)); // same as gpu_result = norm_inf(arma_v1 - arma_v2);
  cuarma::scheduler::execute(my_statement);

  if (check(cpu_result, gpu_result, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "--- Testing elementwise operations (binary) ---" << std::endl;
  std::cout << "x = element_prod(x, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_prod(arma_v1, arma_v2));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x + y, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) * std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_prod(arma_v1 + arma_v2, arma_v2));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x, x + y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] * (std_v1[i] + std_v2[i]);
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_prod(arma_v1, arma_v2 + arma_v1));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x - y, y + x)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] - std_v2[i]) * (std_v2[i] + std_v1[i]);
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_prod(arma_v1 - arma_v2, arma_v2 + arma_v1));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }



  std::cout << "x = element_div(x, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_div(arma_v1, arma_v2));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x + y, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] + std_v2[i]) / std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_div(arma_v1 + arma_v2, arma_v2));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x, x + y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v1[i] / (std_v1[i] + std_v2[i]);
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_div(arma_v1, arma_v2 + arma_v1));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x - y, y + x)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = (std_v1[i] - std_v2[i]) / (std_v2[i] + std_v1[i]);
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_div(arma_v1 - arma_v2, arma_v2 + arma_v1));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }


  std::cout << "x = element_pow(x, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(2.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }
  cuarma::copy(std_v1, arma_v1);
  cuarma::copy(std_v2, arma_v2);

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std::pow(std_v1[i], std_v2[i]);
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_pow(arma_v1, arma_v2));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_pow(x + y, y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(2.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }
  cuarma::copy(std_v1, arma_v1);
  cuarma::copy(std_v2, arma_v2);

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std::pow(std_v1[i]  + std_v2[i], std_v2[i]);
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_pow(arma_v1 + arma_v2, arma_v2));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_pow(x, x + y)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(2.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }
  cuarma::copy(std_v1, arma_v1);
  cuarma::copy(std_v2, arma_v2);

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std::pow(std_v1[i], std_v1[i] + std_v2[i]);
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_pow(arma_v1, arma_v2 + arma_v1));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_pow(x - y, y + x)... ";
  {
  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(2.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
  }
  cuarma::copy(std_v1, arma_v1);
  cuarma::copy(std_v2, arma_v2);

  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std::pow(std_v1[i] - std_v2[i], std_v2[i] + std_v1[i]);
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_pow(arma_v1 - arma_v2, arma_v2 + arma_v1));
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "--- Testing elementwise operations (unary) ---" << std::endl;
#define GENERATE_UNARY_OP_TEST(OPNAME) \
  std_v1 = std::vector<NumericT>(std_v1.size(), NumericT(0.21)); \
  for (std::size_t i=0; i<std_v1.size(); ++i) \
    std_v2[i] = NumericT(3.1415) * std_v1[i]; \
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin()); \
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin()); \
  { \
  for (std::size_t i=0; i<std_v1.size(); ++i) \
    std_v1[i] = std::OPNAME(std_v2[i]); \
  cuarma::scheduler::statement my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_##OPNAME(arma_v2)); \
  cuarma::scheduler::execute(my_statement); \
  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS) \
    return EXIT_FAILURE; \
  } \
  { \
  for (std::size_t i=0; i<std_v1.size(); ++i) \
  std_v1[i] = std::OPNAME(std_v2[i] / NumericT(2)); \
  cuarma::scheduler::statement my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::element_##OPNAME(arma_v2 / NumericT(2))); \
  cuarma::scheduler::execute(my_statement); \
  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS) \
    return EXIT_FAILURE; \
  }

  GENERATE_UNARY_OP_TEST(cos);
  GENERATE_UNARY_OP_TEST(cosh);
  GENERATE_UNARY_OP_TEST(exp);
  GENERATE_UNARY_OP_TEST(floor);
  GENERATE_UNARY_OP_TEST(fabs);
  GENERATE_UNARY_OP_TEST(log);
  GENERATE_UNARY_OP_TEST(log10);
  GENERATE_UNARY_OP_TEST(sin);
  GENERATE_UNARY_OP_TEST(sinh);
  GENERATE_UNARY_OP_TEST(fabs);
  GENERATE_UNARY_OP_TEST(sqrt);
  GENERATE_UNARY_OP_TEST(tan);
  GENERATE_UNARY_OP_TEST(tanh);

#undef GENERATE_UNARY_OP_TEST

  std::cout << "--- Testing complicated composite operations ---" << std::endl;
  std::cout << "x = inner_prod(x, y) * y..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std_v1[i] * std_v2[i];
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = cpu_result * std_v2[i];
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::inner_prod(arma_v1, arma_v2) * arma_v2);
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = y / norm_1(x)..." << std::endl;
  {
  cpu_result = 0;
  for (std::size_t i=0; i<std_v1.size(); ++i)
    cpu_result += std::fabs(std_v1[i]);
  for (std::size_t i=0; i<std_v1.size(); ++i)
    std_v1[i] = std_v2[i] / cpu_result;
  cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), arma_v2 / cuarma::blas::norm_1(arma_v1) );
  cuarma::scheduler::execute(my_statement);

  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }


  // --------------------------------------------------------------------------
  return retval;
}


template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int retval = EXIT_SUCCESS;
  std::size_t size = 24656;

  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  std::cout << "Running tests for vector of size " << size << std::endl;

  //
  // Set up STL objects
  //
  std::vector<NumericT> std_full_vec(size);
  std::vector<NumericT> std_full_vec2(std_full_vec.size());

  for (std::size_t i=0; i<std_full_vec.size(); ++i)
  {
    std_full_vec[i]  = NumericT(1.0) + randomNumber();
    std_full_vec2[i] = NumericT(1.0) + randomNumber();
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
    if (check(std_short_vec, arma_short_vec, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    if (check(std_short_vec2, arma_short_vec2, epsilon) != EXIT_SUCCESS)
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
  if (check(std_short_vec, arma_short_vec, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec2, arma_short_vec2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** arma_v1 = vector, arma_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          arma_short_vec, arma_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = vector, arma_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          arma_short_vec, arma_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = vector, arma_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          arma_short_vec, arma_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** arma_v1 = range, arma_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          arma_range_vec, arma_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = range, arma_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          arma_range_vec, arma_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = range, arma_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          arma_range_vec, arma_slice_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** arma_v1 = slice, arma_v2 = vector **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          arma_slice_vec, arma_short_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = slice, arma_v2 = range **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
                          arma_slice_vec, arma_range_vec2);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_v1 = slice, arma_v2 = slice **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec, std_short_vec2,
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
   std::cout << "## Test :: Vector" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = static_cast<NumericT>(1.0E-4);
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if ( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   {
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-12;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
         if ( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
   }

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;


   return retval;
}
