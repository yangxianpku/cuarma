/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   scheduler_matrix.cu
 *   @coding UTF-8
 *   @brief  Tests the scheduler for matrix-operations (not matrix-matrix).
 *   @brief  ²âÊÔ£º¾ØÕó²Ù×÷µ÷¶ÈÆ÷
 */

#define CUARMA_WITH_UBLAS

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <ctime>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/prod.hpp"

/*#include "cuarma/compressed_matrix.hpp"
#include "cuarma/blas/cg.hpp"
#include "cuarma/blas/inner_prod.hpp"
#include "cuarma/blas/ilu.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/io/matrix_market.hpp"*/

#include "cuarma/matrix_proxy.hpp"
#include "cuarma/vector_proxy.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"

#include "cuarma/scheduler/execute.hpp"

using namespace boost::numeric;

template<typename MatrixType, typename VCLMatrixType>
bool check_for_equality(MatrixType const & ublas_A, VCLMatrixType const & arma_A, double epsilon)
{
  typedef typename MatrixType::value_type   value_type;

  boost::numeric::ublas::matrix<value_type> arma_A_cpu(arma_A.size1(), arma_A.size2());
  cuarma::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
  cuarma::copy(arma_A, arma_A_cpu);

  for (std::size_t i=0; i<ublas_A.size1(); ++i)
  {
    for (std::size_t j=0; j<ublas_A.size2(); ++j)
    {
      if (std::fabs(ublas_A(i,j) - arma_A_cpu(i,j)) > 0)
      {
        if ( (std::fabs(ublas_A(i,j) - arma_A_cpu(i,j)) / std::max(std::fabs(ublas_A(i,j)), std::fabs(arma_A_cpu(i,j))) > epsilon) || std::fabs(arma_A_cpu(i,j) - arma_A_cpu(i,j)) > 0 )
        {
          std::cout << "Error at index (" << i << ", " << j << "): " << ublas_A(i,j) << " vs " << arma_A_cpu(i,j) << std::endl;
          std::cout << std::endl << "TEST failed!" << std::endl;
          return false;
        }
      }
    }
  }

  std::cout << "PASSED!" << std::endl;
  return true;
}




template<typename UBLASMatrixType,
          typename cuarmaMatrixType1, typename cuarmaMatrixType2, typename cuarmaMatrixType3>
int run_test(double epsilon,
             UBLASMatrixType & ublas_A, UBLASMatrixType & ublas_B, UBLASMatrixType & ublas_C,
             cuarmaMatrixType1 & arma_A, cuarmaMatrixType2 & arma_B, cuarmaMatrixType3 arma_C)
{

  typedef typename cuarma::result_of::cpu_value_type<typename cuarmaMatrixType1::value_type>::type  cpu_value_type;

  cpu_value_type alpha = cpu_value_type(3.1415);
  cuarma::scalar<cpu_value_type>   gpu_alpha = alpha;

  cpu_value_type beta = cpu_value_type(2.7182);
  cuarma::scalar<cpu_value_type>   gpu_beta = beta;


  //
  // Initializer:
  //
  std::cout << "Checking for zero_matrix initializer..." << std::endl;
  ublas_A = ublas::zero_matrix<cpu_value_type>(ublas_A.size1(), ublas_A.size2());
  arma_A = cuarma::zero_matrix<cpu_value_type>(arma_A.size1(), arma_A.size2());
  if (!check_for_equality(ublas_A, arma_A, epsilon))
    return EXIT_FAILURE;

  std::cout << "Checking for scalar_matrix initializer..." << std::endl;
  ublas_A = ublas::scalar_matrix<cpu_value_type>(ublas_A.size1(), ublas_A.size2(), alpha);
  arma_A = cuarma::scalar_matrix<cpu_value_type>(arma_A.size1(), arma_A.size2(), alpha);
  if (!check_for_equality(ublas_A, arma_A, epsilon))
    return EXIT_FAILURE;

  ublas_A =    ublas::scalar_matrix<cpu_value_type>(ublas_A.size1(), ublas_A.size2(), gpu_beta);
  arma_A   = cuarma::scalar_matrix<cpu_value_type>(  arma_A.size1(),   arma_A.size2(), gpu_beta);
  if (!check_for_equality(ublas_A, arma_A, epsilon))
    return EXIT_FAILURE;

  /*std::cout << "Checking for identity initializer..." << std::endl;
  ublas_A = ublas::identity_matrix<cpu_value_type>(ublas_A.size1());
  arma_A = cuarma::identity_matrix<cpu_value_type>(arma_A.size1());
  if (!check_for_equality(ublas_A, arma_A, epsilon))
    return EXIT_FAILURE;*/


  std::cout << std::endl;
  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test: Assignments //////////" << std::endl;
  //std::cout << "//" << std::endl;

  if (!check_for_equality(ublas_B, arma_B, epsilon))
    return EXIT_FAILURE;

  std::cout << "Testing matrix assignment... ";
  //std::cout << ublas_B(0,0) << " vs. " << arma_B(0,0) << std::endl;
  ublas_A = ublas_B;
  arma_A = arma_B;
  if (!check_for_equality(ublas_A, arma_A, epsilon))
    return EXIT_FAILURE;



  //std::cout << std::endl;
  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test 1: Copy to GPU //////////" << std::endl;
  //std::cout << "//" << std::endl;

  ublas_A = ublas_B;
  cuarma::copy(ublas_B, arma_A);
  std::cout << "Testing upper left copy to GPU... ";
  if (!check_for_equality(ublas_A, arma_A, epsilon))
    return EXIT_FAILURE;


  ublas_C = ublas_B;
  cuarma::copy(ublas_B, arma_C);
  std::cout << "Testing lower right copy to GPU... ";
  if (!check_for_equality(ublas_C, arma_C, epsilon))
    return EXIT_FAILURE;


  //std::cout << std::endl;
  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test 2: Copy from GPU //////////" << std::endl;
  //std::cout << "//" << std::endl;

  std::cout << "Testing upper left copy to A... ";
  if (!check_for_equality(ublas_A, arma_A, epsilon))
    return EXIT_FAILURE;

  std::cout << "Testing lower right copy to C... ";
  if (!check_for_equality(ublas_C, arma_C, epsilon))
    return EXIT_FAILURE;



  //std::cout << "//" << std::endl;
  //std::cout << "////////// Test 3: Addition //////////" << std::endl;
  //std::cout << "//" << std::endl;
  cuarma::copy(ublas_C, arma_C);

  std::cout << "Assignment: ";
  {
  ublas_C = ublas_B;
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), arma_B); // same as arma_C = arma_B;
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon))
    return EXIT_FAILURE;
  }


  std::cout << "Inplace add: ";
  {
  ublas_C += ublas_C;
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_inplace_add(), arma_C); // same as arma_C += arma_C;
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon))
    return EXIT_FAILURE;
  }

  std::cout << "Inplace sub: ";
  {
  ublas_C -= ublas_C;
  cuarma::scheduler::statement my_statement(arma_C, cuarma::op_inplace_sub(), arma_C); // same as arma_C -= arma_C;
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon))
    return EXIT_FAILURE;
  }


  std::cout << "Add: ";
  {
  ublas_C = ublas_A + ublas_B;
  cuarma::scheduler::statement my_statement(arma_C, cuarma::op_assign(), arma_A + arma_B); // same as arma_C = arma_A + arma_B;
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon))
    return EXIT_FAILURE;
  }

  std::cout << "Sub: ";
  {
  ublas_C = ublas_A - ublas_B;
  cuarma::scheduler::statement my_statement(arma_C, cuarma::op_assign(), arma_A - arma_B); // same as arma_C = arma_A - arma_B;
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon))
    return EXIT_FAILURE;
  }

  std::cout << "Composite assignments: ";
  {
  ublas_C += alpha * ublas_A - beta * ublas_B + ublas_A / beta - ublas_B / alpha;
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_inplace_add(), alpha * arma_A - beta * arma_B + arma_A / beta - arma_B / alpha); // same as arma_C += alpha * arma_A - beta * arma_B + arma_A / beta - arma_B / alpha;
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon))
    return EXIT_FAILURE;
  }


  std::cout << "--- Testing elementwise operations (binary) ---" << std::endl;
  std::cout << "x = element_prod(x, y)... ";
  {
  ublas_C = element_prod(ublas_A, ublas_B);
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_prod(arma_A, arma_B));
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x + y, y)... ";
  {
  ublas_C = element_prod(ublas_A + ublas_B, ublas_B);
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_prod(arma_A + arma_B, arma_B));
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x, x + y)... ";
  {
  ublas_C = element_prod(ublas_A, ublas_A + ublas_B);
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_prod(arma_A, arma_B + arma_A));
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_prod(x - y, y + x)... ";
  {
  ublas_C = element_prod(ublas_A - ublas_B, ublas_B + ublas_A);
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_prod(arma_A - arma_B, arma_B + arma_A));
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }



  std::cout << "x = element_div(x, y)... ";
  {
  ublas_C = element_div(ublas_A, ublas_B);
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_div(arma_A, arma_B));
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x + y, y)... ";
  {
  ublas_C = element_div(ublas_A + ublas_B, ublas_B);
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_div(arma_A + arma_B, arma_B));
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x, x + y)... ";
  {
  ublas_C = element_div(ublas_A, ublas_A + ublas_B);
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_div(arma_A, arma_B + arma_A));
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }

  std::cout << "x = element_div(x - y, y + x)... ";
  {
  ublas_C = element_div(ublas_A - ublas_B, ublas_B + ublas_A);
  cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_div(arma_A - arma_B, arma_B + arma_A));
  cuarma::scheduler::execute(my_statement);

  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  }


  std::cout << "--- Testing elementwise operations (unary) ---" << std::endl;
#define GENERATE_UNARY_OP_TEST(OPNAME) \
  ublas_A = ublas::scalar_matrix<cpu_value_type>(ublas_A.size1(), ublas_A.size2(), cpu_value_type(0.21)); \
  ublas_B = cpu_value_type(3.1415) * ublas_A; \
  cuarma::copy(ublas_A, arma_A); \
  cuarma::copy(ublas_B, arma_B); \
  { \
  for (std::size_t i=0; i<ublas_C.size1(); ++i) \
    for (std::size_t j=0; j<ublas_C.size2(); ++j) \
      ublas_C(i,j) = static_cast<cpu_value_type>(OPNAME(ublas_A(i,j))); \
  cuarma::scheduler::statement my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_##OPNAME(arma_A)); \
  cuarma::scheduler::execute(my_statement); \
  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS) \
    return EXIT_FAILURE; \
  } \
  { \
  for (std::size_t i=0; i<ublas_C.size1(); ++i) \
    for (std::size_t j=0; j<ublas_C.size2(); ++j) \
      ublas_C(i,j) = static_cast<cpu_value_type>(OPNAME(ublas_A(i,j) / cpu_value_type(2))); \
  cuarma::scheduler::statement my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_##OPNAME(arma_A / cpu_value_type(2))); \
  cuarma::scheduler::execute(my_statement); \
  if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS) \
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

  if (ublas_C.size1() == ublas_C.size2()) // transposition tests
  {
    std::cout << "z = element_fabs(x - trans(y))... ";
    {
    for (std::size_t i=0; i<ublas_C.size1(); ++i)
      for (std::size_t j=0; j<ublas_C.size2(); ++j)
        ublas_C(i,j) = std::fabs(ublas_A(i,j) - ublas_B(j,i));
    cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_fabs((arma_A) - trans(arma_B)));
    cuarma::scheduler::execute(my_statement);

    if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    }

    std::cout << "z = element_fabs(trans(x) - (y))... ";
    {
    for (std::size_t i=0; i<ublas_C.size1(); ++i)
      for (std::size_t j=0; j<ublas_C.size2(); ++j)
        ublas_C(i,j) = std::fabs(ublas_A(j,i) - ublas_B(i,j));
    cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_fabs(trans(arma_A) - (arma_B)));
    cuarma::scheduler::execute(my_statement);

    if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    }

    std::cout << "z = element_fabs(trans(x) - trans(y))... ";
    {
    for (std::size_t i=0; i<ublas_C.size1(); ++i)
      for (std::size_t j=0; j<ublas_C.size2(); ++j)
        ublas_C(i,j) = std::fabs(ublas_A(j,i) - ublas_B(j,i));
    cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_assign(), cuarma::blas::element_fabs(trans(arma_A) - trans(arma_B)));
    cuarma::scheduler::execute(my_statement);

    if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    }

    std::cout << "z += trans(x)... ";
    {
    for (std::size_t i=0; i<ublas_C.size1(); ++i)
      for (std::size_t j=0; j<ublas_C.size2(); ++j)
        ublas_C(i,j) += ublas_A(j,i);
    cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_inplace_add(), trans(arma_A));
    cuarma::scheduler::execute(my_statement);

    if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    }

    std::cout << "z -= trans(x)... ";
    {
    for (std::size_t i=0; i<ublas_C.size1(); ++i)
      for (std::size_t j=0; j<ublas_C.size2(); ++j)
        ublas_C(i,j) -= ublas_A(j,i);
    cuarma::scheduler::statement   my_statement(arma_C, cuarma::op_inplace_sub(), trans(arma_A));
    cuarma::scheduler::execute(my_statement);

    if (!check_for_equality(ublas_C, arma_C, epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    }

  }

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;


  return EXIT_SUCCESS;
}




template<typename T, typename ScalarType>
int run_test(double epsilon)
{
    //typedef float               ScalarType;
    typedef boost::numeric::ublas::matrix<ScalarType>       MatrixType;

    typedef cuarma::matrix<ScalarType, T>    VCLMatrixType;

    std::size_t dim_rows = 131;
    std::size_t dim_cols = 33;
    //std::size_t dim_rows = 4;
    //std::size_t dim_cols = 4;

    //setup ublas objects:
    MatrixType ublas_A(dim_rows, dim_cols);
    MatrixType ublas_B(dim_rows, dim_cols);
    MatrixType ublas_C(dim_rows, dim_cols);

    for (std::size_t i=0; i<ublas_A.size1(); ++i)
      for (std::size_t j=0; j<ublas_A.size2(); ++j)
      {
        ublas_A(i,j) = ScalarType((i+2) + (j+1)*(i+2));
        ublas_B(i,j) = ScalarType((j+2) + (j+1)*(j+2));
        ublas_C(i,j) = ScalarType((i+1) + (i+1)*(i+2));
      }

    MatrixType ublas_A_large(4 * dim_rows, 4 * dim_cols);
    for (std::size_t i=0; i<ublas_A_large.size1(); ++i)
      for (std::size_t j=0; j<ublas_A_large.size2(); ++j)
        ublas_A_large(i,j) = ScalarType(i * ublas_A_large.size2() + j);

    //Setup cuarma objects
    VCLMatrixType arma_A_full(4 * dim_rows, 4 * dim_cols);
    VCLMatrixType arma_B_full(4 * dim_rows, 4 * dim_cols);
    VCLMatrixType arma_C_full(4 * dim_rows, 4 * dim_cols);

    cuarma::copy(ublas_A_large, arma_A_full);
    cuarma::copy(ublas_A_large, arma_B_full);
    cuarma::copy(ublas_A_large, arma_C_full);

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

    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_A, arma_slice_A);

    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_B, arma_slice_B);

    cuarma::copy(ublas_C, arma_C);
    cuarma::copy(ublas_C, arma_range_C);
    cuarma::copy(ublas_C, arma_slice_C);


    std::cout << std::endl;
    std::cout << "//" << std::endl;
    std::cout << "////////// Test: Copy CTOR //////////" << std::endl;
    std::cout << "//" << std::endl;

    {
      std::cout << "Testing matrix created from range... ";
      VCLMatrixType arma_temp = arma_range_A;
      if (check_for_equality(ublas_A, arma_temp, epsilon))
        std::cout << "PASSED!" << std::endl;
      else
      {
        std::cout << "ublas_A: " << ublas_A << std::endl;
        std::cout << "arma_temp: " << arma_temp << std::endl;
        std::cout << "arma_range_A: " << arma_range_A << std::endl;
        std::cout << "arma_A: " << arma_A << std::endl;
        std::cout << std::endl << "TEST failed!" << std::endl;
        return EXIT_FAILURE;
      }

      std::cout << "Testing matrix created from slice... ";
      VCLMatrixType arma_temp2 = arma_range_B;
      if (check_for_equality(ublas_B, arma_temp2, epsilon))
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
      ublas::matrix<ScalarType> ublas_dummy1 = ublas::identity_matrix<ScalarType>(ublas_A.size1());
      ublas::matrix<ScalarType> ublas_dummy2 = ublas::scalar_matrix<ScalarType>(ublas_A.size1(), ublas_A.size1(), 3.0);
      ublas::matrix<ScalarType> ublas_dummy3 = ublas::zero_matrix<ScalarType>(ublas_A.size1(), ublas_A.size1());

      cuarma::matrix<ScalarType> arma_dummy1 = cuarma::identity_matrix<ScalarType>(ublas_A.size1());
      cuarma::matrix<ScalarType> arma_dummy2 = cuarma::scalar_matrix<ScalarType>(ublas_A.size1(), ublas_A.size1(), 3.0);
      cuarma::matrix<ScalarType> arma_dummy3 = cuarma::zero_matrix<ScalarType>(ublas_A.size1(), ublas_A.size1());

      std::cout << "Testing initializer CTOR... ";
      if (   check_for_equality(ublas_dummy1, arma_dummy1, epsilon)
          && check_for_equality(ublas_dummy2, arma_dummy2, epsilon)
          && check_for_equality(ublas_dummy3, arma_dummy3, epsilon)
         )
        std::cout << "PASSED!" << std::endl;
      else
      {
        std::cout << std::endl << "TEST failed!" << std::endl;
        return EXIT_FAILURE;
      }

      ublas_dummy1 = ublas::zero_matrix<ScalarType>(ublas_A.size1(), ublas_A.size1());
      ublas_dummy2 = ublas::identity_matrix<ScalarType>(ublas_A.size1());
      ublas_dummy3 = ublas::scalar_matrix<ScalarType>(ublas_A.size1(), ublas_A.size1(), 3.0);

      arma_dummy1 = cuarma::zero_matrix<ScalarType>(ublas_A.size1(), ublas_A.size1());
      arma_dummy2 = cuarma::identity_matrix<ScalarType>(ublas_A.size1());
      arma_dummy3 = cuarma::scalar_matrix<ScalarType>(ublas_A.size1(), ublas_A.size1(), 3.0);

      std::cout << "Testing initializer assignment... ";
      if (   check_for_equality(ublas_dummy1, arma_dummy1, epsilon)
          && check_for_equality(ublas_dummy2, arma_dummy2, epsilon)
          && check_for_equality(ublas_dummy3, arma_dummy3, epsilon)
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
    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_C, arma_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_A, arma_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=matrix, C=range ..." << std::endl;
    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_C, arma_range_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_A, arma_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=matrix, C=slice ..." << std::endl;
    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_C, arma_slice_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_A, arma_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=range, C=matrix ..." << std::endl;
    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_C, arma_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_A, arma_range_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=range, C=range ..." << std::endl;
    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_C, arma_range_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_A, arma_range_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=range, C=slice ..." << std::endl;
    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_C, arma_slice_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_A, arma_range_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }


    std::cout << "Testing A=matrix, B=slice, C=matrix ..." << std::endl;
    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_B, arma_slice_B);
    cuarma::copy(ublas_C, arma_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_A, arma_slice_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=slice, C=range ..." << std::endl;
    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_B, arma_slice_B);
    cuarma::copy(ublas_C, arma_range_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_A, arma_slice_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=matrix, B=slice, C=slice ..." << std::endl;
    cuarma::copy(ublas_A, arma_A);
    cuarma::copy(ublas_B, arma_slice_B);
    cuarma::copy(ublas_C, arma_slice_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_A, arma_slice_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    /////// A=range:
    std::cout << "Testing A=range, B=matrix, C=matrix ..." << std::endl;
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_C, arma_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_range_A, arma_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=matrix, C=range ..." << std::endl;
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_C, arma_range_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_range_A, arma_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=matrix, C=slice ..." << std::endl;
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_C, arma_slice_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_range_A, arma_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    std::cout << "Testing A=range, B=range, C=matrix ..." << std::endl;
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_C, arma_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_range_A, arma_range_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=range, C=range ..." << std::endl;
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_C, arma_range_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_range_A, arma_range_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=range, C=slice ..." << std::endl;
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_C, arma_slice_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_range_A, arma_range_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    std::cout << "Testing A=range, B=slice, C=matrix ..." << std::endl;
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_B, arma_slice_B);
    cuarma::copy(ublas_C, arma_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_range_A, arma_slice_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=slice, C=range ..." << std::endl;
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_B, arma_slice_B);
    cuarma::copy(ublas_C, arma_range_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_range_A, arma_slice_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=range, B=slice, C=slice ..." << std::endl;
    cuarma::copy(ublas_A, arma_range_A);
    cuarma::copy(ublas_B, arma_slice_B);
    cuarma::copy(ublas_C, arma_slice_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_range_A, arma_slice_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }


    /////// A=slice:
    std::cout << "Testing A=slice, B=matrix, C=matrix ..." << std::endl;
    cuarma::copy(ublas_A, arma_slice_A);
    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_C, arma_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_slice_A, arma_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=matrix, C=range ..." << std::endl;
    cuarma::copy(ublas_A, arma_slice_A);
    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_C, arma_range_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_slice_A, arma_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=matrix, C=slice ..." << std::endl;
    cuarma::copy(ublas_A, arma_slice_A);
    cuarma::copy(ublas_B, arma_B);
    cuarma::copy(ublas_C, arma_slice_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_slice_A, arma_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    std::cout << "Testing A=slice, B=range, C=matrix ..." << std::endl;
    cuarma::copy(ublas_A, arma_slice_A);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_C, arma_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_slice_A, arma_range_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=range, C=range ..." << std::endl;
    cuarma::copy(ublas_A, arma_slice_A);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_C, arma_range_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_slice_A, arma_range_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=range, C=slice ..." << std::endl;
    cuarma::copy(ublas_A, arma_slice_A);
    cuarma::copy(ublas_B, arma_range_B);
    cuarma::copy(ublas_C, arma_slice_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_slice_A, arma_range_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }



    std::cout << "Testing A=slice, B=slice, C=matrix ..." << std::endl;
    cuarma::copy(ublas_A, arma_slice_A);
    cuarma::copy(ublas_B, arma_slice_B);
    cuarma::copy(ublas_C, arma_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_slice_A, arma_slice_B, arma_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=slice, C=range ..." << std::endl;
    cuarma::copy(ublas_A, arma_slice_A);
    cuarma::copy(ublas_B, arma_slice_B);
    cuarma::copy(ublas_C, arma_range_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_slice_A, arma_slice_B, arma_range_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }

    std::cout << "Testing A=slice, B=slice, C=slice ..." << std::endl;
    cuarma::copy(ublas_A, arma_slice_A);
    cuarma::copy(ublas_B, arma_slice_B);
    cuarma::copy(ublas_C, arma_slice_C);
    if (run_test(epsilon,
                 ublas_A, ublas_B, ublas_C,
                 arma_slice_A, arma_slice_B, arma_slice_C) != EXIT_SUCCESS)
    {
      return EXIT_FAILURE;
    }


    return EXIT_SUCCESS;
}

int main (int, const char **)
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Matrix Range" << std::endl;
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
  std::cout << " --- column-major ---" << std::endl;
  if (run_test<cuarma::column_major, float>(epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  {
    epsilon = 1e-12;
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: double" << std::endl;

    if (run_test<cuarma::row_major, double>(epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
    if (run_test<cuarma::column_major, double>(epsilon) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

   std::cout << std::endl;
   std::cout << "------- Test completed --------" << std::endl;
   std::cout << std::endl;


  return EXIT_SUCCESS;
}

