/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   scheduler_matrix_vector.cu
 *   @coding UTF-8
 *   @brief  Tests the scheduler for matrix-vector-operations.
 *   @brief  ²âÊÔ£ºBLAS2²Ù×÷µ÷¶ÈÆ÷
 */

#include <iostream>
#include <vector>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/blas/lu.hpp"
#include "cuarma/tools/random.hpp"

#include "cuarma/scheduler/execute.hpp"
#include "cuarma/scheduler/io.hpp"


//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType & s1, cuarma::scalar<ScalarType> & s2)
{
   cuarma::backend::finish();
   if (s1 != s2)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}

template<typename ScalarType, typename VCLVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, VCLVectorType const & v2)
{
   std::vector<ScalarType> v2_cpu(v2.size());
   cuarma::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   cuarma::copy(v2.begin(), v2.end(), v2_cpu.begin());

   ScalarType norm_inf = 0;
   for (unsigned int i=0;i<v1.size(); ++i)
   {
     if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
     {
       ScalarType tmp = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
       if (tmp > norm_inf)
         norm_inf = tmp;
     }
   }

   return norm_inf;
}

template<typename ScalarType, typename VCLMatrixType>
ScalarType diff(std::vector<std::vector<ScalarType> > const & mat1, VCLMatrixType const & mat2)
{
   std::vector<std::vector<ScalarType> > mat2_cpu(mat2.size1(), std::vector<ScalarType>(mat2.size2()));
   cuarma::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   cuarma::copy(mat2, mat2_cpu);
   ScalarType ret = 0;
   ScalarType act = 0;

    for (std::size_t i = 0; i < mat2_cpu.size(); ++i)
    {
      for (std::size_t j = 0; j < mat2_cpu[i].size(); ++j)
      {
         act = std::fabs(mat2_cpu[i][j] - mat1[i][j]) / std::max( std::fabs(mat2_cpu[i][j]), std::fabs(mat1[i][j]) );
         if (act > ret)
           ret = act;
      }
    }
   //std::cout << ret << std::endl;
   return ret;
}
//
// -------------------------------------------------------------
//

template<typename NumericT, typename Epsilon,
          typename STLMatrixType, typename STLVectorType,
          typename VCLMatrixType, typename VCLVectorType1, typename VCLVectorType2>
int test_prod_rank1(Epsilon const & epsilon,
                    STLMatrixType & std_m1, STLVectorType & std_v1, STLVectorType & std_v2,
                    VCLMatrixType & arma_m1, VCLVectorType1 & arma_v1, VCLVectorType2 & arma_v2)
{
   int retval = EXIT_SUCCESS;

   // sync data:
   cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
   cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());
   cuarma::copy(std_m1, arma_m1);

   /* TODO: Add rank-1 operations here */

   //reset arma_matrix:
   cuarma::copy(std_m1, arma_m1);

   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product" << std::endl;
   for (std::size_t i=0; i<std_m1.size(); ++i)
   {
     std_v1[i] = 0;
     for (std::size_t j=0; j<std_m1[i].size(); ++j)
       std_v1[i] += std_m1[i][j] * std_v2[j];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::prod(arma_m1, arma_v2));
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v1, arma_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v1, arma_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Matrix-Vector product with inplace-add" << std::endl;
   for (std::size_t i=0; i<std_m1.size(); ++i)
   {
     for (std::size_t j=0; j<std_m1[i].size(); ++j)
       std_v1[i] += std_m1[i][j] * std_v2[j];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_inplace_add(), cuarma::blas::prod(arma_m1, arma_v2));
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v1, arma_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v1, arma_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Matrix-Vector product with inplace-sub" << std::endl;
   for (std::size_t i=0; i<std_m1.size(); ++i)
   {
     for (std::size_t j=0; j<std_m1[i].size(); ++j)
       std_v1[i] -= std_m1[i][j] * std_v2[j];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_inplace_sub(), cuarma::blas::prod(arma_m1, arma_v2));
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v1, arma_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v1, arma_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // --------------------------------------------------------------------------
   /*
   std::cout << "Matrix-Vector product with scaled matrix" << std::endl;
   std_v1 = cuarma::blas::prod(NumericT(2.0) * std_m1, std_v2);
   {
   cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::prod(NumericT(2.0) * arma_m1, arma_v2));
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v1, arma_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v1, arma_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }*/

   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with scaled vector" << std::endl;
   /*
   std_v1 = cuarma::blas::prod(std_m1, NumericT(2.0) * std_v2);
   {
   cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::prod(arma_m1, NumericT(2.0) * arma_v2));
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v1, arma_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v1, arma_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }*/

   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with scaled matrix and scaled vector" << std::endl;
   /*
   std_v1 = cuarma::blas::prod(NumericT(2.0) * std_m1, NumericT(2.0) * std_v2);
   {
   cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), cuarma::blas::prod(NumericT(2.0) * arma_m1, NumericT(2.0) * arma_v2));
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v1, arma_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v1, arma_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }*/


   // --------------------------------------------------------------------------
   std::cout << "Matrix-Vector product with scaled add" << std::endl;
   NumericT alpha = static_cast<NumericT>(2.786);
   NumericT beta = static_cast<NumericT>(3.1415);
   cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
   cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

   for (std::size_t i=0; i<std_m1.size(); ++i)
   {
     std_v1[i] = 0;
     for (std::size_t j=0; j<std_m1[i].size(); ++j)
       std_v1[i] += std_m1[i][j] * std_v2[j];
     std_v1[i] = alpha * std_v1[i] - beta * std_v1[i];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_assign(), alpha * cuarma::blas::prod(arma_m1, arma_v2) - beta * arma_v1);
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v1, arma_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v1, arma_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Matrix-Vector product with scaled add, inplace-add" << std::endl;
   cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
   cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

   for (std::size_t i=0; i<std_m1.size(); ++i)
   {
     NumericT tmp = 0;
     for (std::size_t j=0; j<std_m1[i].size(); ++j)
       tmp += std_m1[i][j] * std_v2[j];
     std_v1[i] += alpha * tmp - beta * std_v1[i];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_inplace_add(), alpha * cuarma::blas::prod(arma_m1, arma_v2) - beta * arma_v1);
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v1, arma_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v1, arma_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Matrix-Vector product with scaled add, inplace-sub" << std::endl;
   cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
   cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

   for (std::size_t i=0; i<std_m1.size(); ++i)
   {
     NumericT tmp = 0;
     for (std::size_t j=0; j<std_m1[i].size(); ++j)
       tmp += std_m1[i][j] * std_v2[j];
     std_v1[i] -= alpha * tmp - beta * std_v1[i];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v1, cuarma::op_inplace_sub(), alpha * cuarma::blas::prod(arma_m1, arma_v2) - beta * arma_v1);
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v1, arma_v1)) > epsilon )
   {
      std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v1, arma_v1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // --------------------------------------------------------------------------

   cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
   cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

   std::cout << "Transposed Matrix-Vector product" << std::endl;
   for (std::size_t i=0; i<std_m1[0].size(); ++i)
   {
     std_v2[i] = 0;
     for (std::size_t j=0; j<std_m1.size(); ++j)
       std_v2[i] += std_m1[j][i] * std_v1[j];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v2, cuarma::op_assign(), cuarma::blas::prod(trans(arma_m1), arma_v1));
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v2, arma_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v2, arma_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product, inplace-add" << std::endl;
   for (std::size_t i=0; i<std_m1[0].size(); ++i)
   {
     for (std::size_t j=0; j<std_m1.size(); ++j)
       std_v2[i] += std_m1[j][i] * std_v1[j];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v2, cuarma::op_inplace_add(), cuarma::blas::prod(trans(arma_m1), arma_v1));
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v2, arma_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v2, arma_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product, inplace-sub" << std::endl;
   for (std::size_t i=0; i<std_m1[0].size(); ++i)
   {
     for (std::size_t j=0; j<std_m1.size(); ++j)
       std_v2[i] -= std_m1[j][i] * std_v1[j];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v2, cuarma::op_inplace_sub(), cuarma::blas::prod(trans(arma_m1), arma_v1));
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v2, arma_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v2, arma_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // --------------------------------------------------------------------------
   std::cout << "Transposed Matrix-Vector product with scaled add" << std::endl;
   for (std::size_t i=0; i<std_m1[0].size(); ++i)
   {
     NumericT tmp = 0;
     for (std::size_t j=0; j<std_m1.size(); ++j)
       tmp += std_m1[j][i] * std_v1[j];
     std_v2[i] = alpha * tmp + beta * std_v2[i];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v2, cuarma::op_assign(), alpha * cuarma::blas::prod(trans(arma_m1), arma_v1) + beta * arma_v2);
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v2, arma_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v2, arma_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product with scaled add, inplace-add" << std::endl;
   for (std::size_t i=0; i<std_m1[0].size(); ++i)
   {
     NumericT tmp = 0;
     for (std::size_t j=0; j<std_m1.size(); ++j)
       tmp += std_m1[j][i] * std_v1[j];
     std_v2[i] += alpha * tmp + beta * std_v2[i];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v2, cuarma::op_inplace_add(), alpha * cuarma::blas::prod(trans(arma_m1), arma_v1) + beta * arma_v2);
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v2, arma_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v2, arma_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   std::cout << "Transposed Matrix-Vector product with scaled add, inplace-sub" << std::endl;
   for (std::size_t i=0; i<std_m1[0].size(); ++i)
   {
     NumericT tmp = 0;
     for (std::size_t j=0; j<std_m1.size(); ++j)
       tmp += std_m1[j][i] * std_v1[j];
     std_v2[i] -= alpha * tmp + beta * std_v2[i];
   }
   {
   cuarma::scheduler::statement   my_statement(arma_v2, cuarma::op_inplace_sub(), alpha * cuarma::blas::prod(trans(arma_m1), arma_v1) + beta * arma_v2);
   cuarma::scheduler::execute(my_statement);
   }

   if ( std::fabs(diff(std_v2, arma_v2)) > epsilon )
   {
      std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(std_v2, arma_v2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // --------------------------------------------------------------------------

   return retval;
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename F, typename Epsilon >
int test(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;

   cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

   std::size_t num_rows = 141;
   std::size_t num_cols = 79;

   // --------------------------------------------------------------------------
   std::vector<NumericT> std_v1(num_rows);
   for (std::size_t i = 0; i < std_v1.size(); ++i)
     std_v1[i] = randomNumber();
   std::vector<NumericT> std_v2 = std::vector<NumericT>(num_cols, NumericT(3.1415));


   std::vector<std::vector<NumericT> > std_m1(std_v1.size(), std::vector<NumericT>(std_v2.size()));

   for (std::size_t i = 0; i < std_m1.size(); ++i)
      for (std::size_t j = 0; j < std_m1[i].size(); ++j)
        std_m1[i][j] = static_cast<NumericT>(0.1) * randomNumber();


   std::vector<std::vector<NumericT> > std_m2(std_v1.size(), std::vector<NumericT>(std_v1.size()));

   for (std::size_t i = 0; i < std_m2.size(); ++i)
   {
      for (std::size_t j = 0; j < std_m2[i].size(); ++j)
         std_m2[i][j] = static_cast<NumericT>(-0.1) * randomNumber();
      std_m2[i][i] = static_cast<NumericT>(2) + randomNumber();
   }


   cuarma::vector<NumericT> arma_v1_native(std_v1.size());
   cuarma::vector<NumericT> arma_v1_large(4 * std_v1.size());
   cuarma::vector_range< cuarma::vector<NumericT> > arma_v1_range(arma_v1_large, cuarma::range(3, std_v1.size() + 3));
   cuarma::vector_slice< cuarma::vector<NumericT> > arma_v1_slice(arma_v1_large, cuarma::slice(2, 3, std_v1.size()));

   cuarma::vector<NumericT> arma_v2_native(std_v2.size());
   cuarma::vector<NumericT> arma_v2_large(4 * std_v2.size());
   cuarma::vector_range< cuarma::vector<NumericT> > arma_v2_range(arma_v2_large, cuarma::range(8, std_v2.size() + 8));
   cuarma::vector_slice< cuarma::vector<NumericT> > arma_v2_slice(arma_v2_large, cuarma::slice(6, 2, std_v2.size()));

   cuarma::matrix<NumericT, F> arma_m1_native(std_m1.size(), std_m1[0].size());
   cuarma::matrix<NumericT, F> arma_m1_large(4 * std_m1.size(), 4 * std_m1[0].size());
   cuarma::matrix_range< cuarma::matrix<NumericT, F> > arma_m1_range(arma_m1_large,
                                                                        cuarma::range(8, std_m1.size() + 8),
                                                                        cuarma::range(std_m1[0].size(), 2 * std_m1[0].size()) );
   cuarma::matrix_slice< cuarma::matrix<NumericT, F> > arma_m1_slice(arma_m1_large,
                                                                        cuarma::slice(6, 2, std_m1.size()),
                                                                        cuarma::slice(std_m1[0].size(), 2, std_m1[0].size()) );

   cuarma::matrix<NumericT, F> arma_m2_native(std_m2.size(), std_m2[0].size());
   cuarma::matrix<NumericT, F> arma_m2_large(4 * std_m2.size(), 4 * std_m2[0].size());
   cuarma::matrix_range< cuarma::matrix<NumericT, F> > arma_m2_range(arma_m2_large,
                                                                        cuarma::range(8, std_m2.size() + 8),
                                                                        cuarma::range(std_m2[0].size(), 2 * std_m2[0].size()) );
   cuarma::matrix_slice< cuarma::matrix<NumericT, F> > arma_m2_slice(arma_m2_large,
                                                                        cuarma::slice(6, 2, std_m2.size()),
                                                                        cuarma::slice(std_m2[0].size(), 2, std_m2[0].size()) );


/*   std::cout << "Matrix resizing (to larger)" << std::endl;
   matrix.resize(2*num_rows, 2*num_cols, true);
   for (unsigned int i = 0; i < matrix.size1(); ++i)
   {
      for (unsigned int j = (i<result.size() ? rhs.size() : 0); j < matrix.size2(); ++j)
         matrix(i,j) = 0;
   }
   arma_matrix.resize(2*num_rows, 2*num_cols, true);
   cuarma::copy(arma_matrix, matrix);
   if ( std::fabs(diff(matrix, arma_matrix)) > epsilon )
   {
      std::cout << "# Error at operation: matrix resize (to larger)" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(matrix, arma_matrix)) << std::endl;
      return EXIT_FAILURE;
   }

   matrix(12, 14) = NumericT(1.9);
   matrix(19, 16) = NumericT(1.0);
   matrix (13, 15) =  NumericT(-9);
   arma_matrix(12, 14) = NumericT(1.9);
   arma_matrix(19, 16) = NumericT(1.0);
   arma_matrix (13, 15) =  NumericT(-9);

   std::cout << "Matrix resizing (to smaller)" << std::endl;
   matrix.resize(result.size(), rhs.size(), true);
   arma_matrix.resize(result.size(), rhs.size(), true);
   if ( std::fabs(diff(matrix, arma_matrix)) > epsilon )
   {
      std::cout << "# Error at operation: matrix resize (to smaller)" << std::endl;
      std::cout << "  diff: " << std::fabs(diff(matrix, arma_matrix)) << std::endl;
      return EXIT_FAILURE;
   }
   */

   //
   // Run a bunch of tests for rank-1-updates, matrix-vector products
   //
   std::cout << "------------ Testing rank-1-updates and matrix-vector products ------------------" << std::endl;

   std::cout << "* m = full, v1 = full, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_native, arma_v1_native, arma_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_native, arma_v1_native, arma_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_native, arma_v1_native, arma_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   // v1 = range


   std::cout << "* m = full, v1 = range, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_native, arma_v1_range, arma_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_native, arma_v1_range, arma_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_native, arma_v1_range, arma_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;



   // v1 = slice

   std::cout << "* m = full, v1 = slice, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_native, arma_v1_slice, arma_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_native, arma_v1_slice, arma_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = full, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_native, arma_v1_slice, arma_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   ///////////////////////////// matrix_range

   std::cout << "* m = range, v1 = full, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_range, arma_v1_native, arma_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_range, arma_v1_native, arma_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_range, arma_v1_native, arma_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   // v1 = range


   std::cout << "* m = range, v1 = range, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_range, arma_v1_range, arma_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_range, arma_v1_range, arma_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_range, arma_v1_range, arma_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;



   // v1 = slice

   std::cout << "* m = range, v1 = slice, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_range, arma_v1_slice, arma_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_range, arma_v1_slice, arma_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = range, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_range, arma_v1_slice, arma_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   ///////////////////////////// matrix_slice

   std::cout << "* m = slice, v1 = full, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_slice, arma_v1_native, arma_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = full, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_slice, arma_v1_native, arma_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = full, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_slice, arma_v1_native, arma_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   // v1 = range


   std::cout << "* m = slice, v1 = range, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_slice, arma_v1_range, arma_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = range, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_slice, arma_v1_range, arma_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = range, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_slice, arma_v1_range, arma_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;



   // v1 = slice

   std::cout << "* m = slice, v1 = slice, v2 = full" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_slice, arma_v1_slice, arma_v2_native);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = slice, v2 = range" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_slice, arma_v1_slice, arma_v2_range);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;


   std::cout << "* m = slice, v1 = slice, v2 = slice" << std::endl;
   retval = test_prod_rank1<NumericT>(epsilon,
                                      std_m1, std_v1, std_v2,
                                      arma_m1_slice, arma_v1_slice, arma_v2_slice);
   if (retval == EXIT_FAILURE)
   {
     std::cout << " --- FAILED! ---" << std::endl;
     return retval;
   }
   else
     std::cout << " --- PASSED ---" << std::endl;

   return retval;
}
//
// -------------------------------------------------------------
//
int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Matrix" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = NumericT(1.0E-3);
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      std::cout << "  layout: row-major" << std::endl;
      retval = test<NumericT, cuarma::row_major>(epsilon);
      if ( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = NumericT(1.0E-3);
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      std::cout << "  layout: column-major" << std::endl;
      retval = test<NumericT, cuarma::column_major>(epsilon);
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
         NumericT epsilon = 1.0E-11;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         std::cout << "  layout: row-major" << std::endl;
         retval = test<NumericT, cuarma::row_major>(epsilon);
            if ( retval == EXIT_SUCCESS )
               std::cout << "# Test passed" << std::endl;
            else
              return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-11;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         std::cout << "  layout: column-major" << std::endl;
         retval = test<NumericT, cuarma::column_major>(epsilon);
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
