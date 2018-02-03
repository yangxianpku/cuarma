/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   scheduler_matrix_matrix.cu
 *   @coding UTF-8
 *   @brief  Tests the scheduler for dense matrix-matrix-operations.
 *   @brief  ²âÊÔ£ºBLAS3²Ù×÷µ÷¶ÈÆ÷
 */

#include <iostream>
#include <vector>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/matrix_proxy.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/tools/random.hpp"

#include "cuarma/scheduler/execute.hpp"
#include "cuarma/scheduler/io.hpp"


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
// Part 1: Matrix-matrix multiplications
//


template< typename NumericT, typename Epsilon,
          typename ReferenceMatrixTypeA, typename ReferenceMatrixTypeB, typename ReferenceMatrixTypeC,
          typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
int test_prod(Epsilon const& epsilon,

              ReferenceMatrixTypeA const & A, ReferenceMatrixTypeA const & A_trans,
              ReferenceMatrixTypeB const & B, ReferenceMatrixTypeB const & B_trans,
              ReferenceMatrixTypeC & C,

              MatrixTypeA const & arma_A, MatrixTypeA const & arma_A_trans,
              MatrixTypeB const & arma_B, MatrixTypeB const & arma_B_trans,
              MatrixTypeC & arma_C
             )
{
   int retval = EXIT_SUCCESS;
   NumericT act_diff = 0;


   // Test: C +-= A * B --------------------------------------------------------------------------
   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A[i][k] * B[k][j];
       C[i][j] = tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_assign(), cuarma::blas::prod(arma_A, arma_B));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C = A * B passed!" << std::endl;


   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A[i][k] * B[k][j];
       C[i][j] += tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_inplace_add(), cuarma::blas::prod(arma_A, arma_B));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C += A * B passed!" << std::endl;

   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A[i][k] * B[k][j];
       C[i][j] -= tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_inplace_sub(), cuarma::blas::prod(arma_A, arma_B));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C -= A * B passed!" << std::endl;





   // Test: C +-= A * trans(B) --------------------------------------------------------------------------
   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A[i][k] * B_trans[j][k];
       C[i][j] = tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_assign(), cuarma::blas::prod(arma_A, trans(arma_B_trans)));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C = A * trans(B) passed!" << std::endl;


   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A[i][k] * B_trans[j][k];
       C[i][j] += tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_inplace_add(), cuarma::blas::prod(arma_A, trans(arma_B_trans)));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C += A * trans(B) passed!" << std::endl;


   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A[i][k] * B_trans[j][k];
       C[i][j] -= tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_inplace_sub(), cuarma::blas::prod(arma_A, trans(arma_B_trans)));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C -= A * trans(B) passed!" << std::endl;



   // Test: C +-= trans(A) * B --------------------------------------------------------------------------
   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A_trans[k][i] * B[k][j];
       C[i][j] = tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_assign(), cuarma::blas::prod(trans(arma_A_trans), arma_B));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C = trans(A) * B passed!" << std::endl;


   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A_trans[k][i] * B[k][j];
       C[i][j] += tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_inplace_add(), cuarma::blas::prod(trans(arma_A_trans), arma_B));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C += trans(A) * B passed!" << std::endl;


   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A_trans[k][i] * B[k][j];
       C[i][j] -= tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_inplace_sub(), cuarma::blas::prod(trans(arma_A_trans), arma_B));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C -= trans(A) * B passed!" << std::endl;





   // Test: C +-= trans(A) * trans(B) --------------------------------------------------------------------------
   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A_trans[k][i] * B_trans[j][k];
       C[i][j] = tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_assign(), cuarma::blas::prod(trans(arma_A_trans), trans(arma_B_trans)));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C = trans(A) * trans(B) passed!" << std::endl;

   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A_trans[k][i] * B_trans[j][k];
       C[i][j] += tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_inplace_add(), cuarma::blas::prod(trans(arma_A_trans), trans(arma_B_trans)));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C += trans(A) * trans(B) passed!" << std::endl;


   for (std::size_t i=0; i<C.size(); ++i)
     for (std::size_t j=0; j<C[i].size(); ++j)
     {
       NumericT tmp = 0;
       for (std::size_t k=0; k<A[i].size(); ++k)
         tmp += A_trans[k][i] * B_trans[j][k];
       C[i][j] -= tmp;
     }
   {
   cuarma::scheduler::statement my_statement(arma_C, cuarma::op_inplace_sub(), cuarma::blas::prod(trans(arma_A_trans), trans(arma_B_trans)));
   cuarma::scheduler::execute(my_statement);
   }
   act_diff = std::fabs(diff(C, arma_C));

   if ( act_diff > epsilon )
   {
     std::cout << "# Error at operation: matrix-matrix product" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << "Test C -= trans(A) * trans(B) passed!" << std::endl;




   return retval;
}



template< typename NumericT, typename F_A, typename F_B, typename F_C, typename Epsilon >
int test_prod(Epsilon const& epsilon)
{
  int ret;

  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  std::size_t matrix_size1 = 29;  //some odd number, not too large
  std::size_t matrix_size2 = 47;  //some odd number, not too large
  std::size_t matrix_size3 = 33;  //some odd number, not too large
  //std::size_t matrix_size1 = 128;  //some odd number, not too large
  //std::size_t matrix_size2 = 64;  //some odd number, not too large
  //std::size_t matrix_size3 = 128;  //some odd number, not too large
  //std::size_t matrix_size1 = 256;  // for testing AMD kernels
  //std::size_t matrix_size2 = 256;  // for testing AMD kernels
  //std::size_t matrix_size3 = 256;  // for testing AMD kernels

  // --------------------------------------------------------------------------

  // ublas reference:
  std::vector<std::vector<NumericT> > A(matrix_size1, std::vector<NumericT>(matrix_size2));
  std::vector<std::vector<NumericT> > big_A(4*matrix_size1, std::vector<NumericT>(4*matrix_size2, NumericT(3.1415)));

  std::vector<std::vector<NumericT> > B(matrix_size2, std::vector<NumericT>(matrix_size3));
  std::vector<std::vector<NumericT> > big_B(4*matrix_size2, std::vector<NumericT>(4*matrix_size3, NumericT(42.0)));

  std::vector<std::vector<NumericT> > C(matrix_size1, std::vector<NumericT>(matrix_size3));

  //fill A and B:
  for (std::size_t i = 0; i < A.size(); ++i)
    for (std::size_t j = 0; j < A[0].size(); ++j)
      A[i][j] = static_cast<NumericT>(0.1) * randomNumber();
  for (std::size_t i = 0; i < B.size(); ++i)
    for (std::size_t j = 0; j < B[0].size(); ++j)
      B[i][j] = static_cast<NumericT>(0.1) * randomNumber();

  std::vector<std::vector<NumericT> >     A_trans(A[0].size(), std::vector<NumericT>(A.size()));
  for (std::size_t i = 0; i < A.size(); ++i)
    for (std::size_t j = 0; j < A[0].size(); ++j)
      A_trans[j][i] = A[i][j];

  std::vector<std::vector<NumericT> > big_A_trans(big_A[0].size(), std::vector<NumericT>(big_A.size()));
  for (std::size_t i = 0; i < big_A.size(); ++i)
    for (std::size_t j = 0; j < big_A[0].size(); ++j)
      big_A_trans[j][i] = big_A[i][j];


  std::vector<std::vector<NumericT> >     B_trans(B[0].size(), std::vector<NumericT>(B.size()));
  for (std::size_t i = 0; i < B.size(); ++i)
    for (std::size_t j = 0; j < B[0].size(); ++j)
      B_trans[j][i] = B[i][j];

  std::vector<std::vector<NumericT> > big_B_trans(big_B[0].size(), std::vector<NumericT>(big_B.size()));
  for (std::size_t i = 0; i < big_B.size(); ++i)
    for (std::size_t j = 0; j < big_B[0].size(); ++j)
      big_B_trans[j][i] = big_B[i][j];

  //
  // cuarma objects
  //

  // A
  cuarma::range range1_A(matrix_size1, 2*matrix_size1);
  cuarma::range range2_A(matrix_size2, 2*matrix_size2);
  cuarma::slice slice1_A(matrix_size1, 2, matrix_size1);
  cuarma::slice slice2_A(matrix_size2, 3, matrix_size2);

  cuarma::matrix<NumericT, F_A>    arma_A(matrix_size1, matrix_size2);
  cuarma::copy(A, arma_A);

  cuarma::matrix<NumericT, F_A>    arma_big_range_A(4*matrix_size1, 4*matrix_size2);
  cuarma::matrix_range<cuarma::matrix<NumericT, F_A> > arma_range_A(arma_big_range_A, range1_A, range2_A);
  cuarma::copy(A, arma_range_A);

  cuarma::matrix<NumericT, F_A>    arma_big_slice_A(4*matrix_size1, 4*matrix_size2);
  cuarma::matrix_slice<cuarma::matrix<NumericT, F_A> > arma_slice_A(arma_big_slice_A, slice1_A, slice2_A);
  cuarma::copy(A, arma_slice_A);


  // A^T
  cuarma::matrix<NumericT, F_A>    arma_A_trans(matrix_size2, matrix_size1);
  cuarma::copy(A_trans, arma_A_trans);

  cuarma::matrix<NumericT, F_A>    arma_big_range_A_trans(4*matrix_size2, 4*matrix_size1);
  cuarma::matrix_range<cuarma::matrix<NumericT, F_A> > arma_range_A_trans(arma_big_range_A_trans, range2_A, range1_A);
  cuarma::copy(A_trans, arma_range_A_trans);

  cuarma::matrix<NumericT, F_A>    arma_big_slice_A_trans(4*matrix_size2, 4*matrix_size1);
  cuarma::matrix_slice<cuarma::matrix<NumericT, F_A> > arma_slice_A_trans(arma_big_slice_A_trans, slice2_A, slice1_A);
  cuarma::copy(A_trans, arma_slice_A_trans);



  // B
  cuarma::range range1_B(2*matrix_size2, 3*matrix_size2);
  cuarma::range range2_B(2*matrix_size3, 3*matrix_size3);
  cuarma::slice slice1_B(matrix_size2, 3, matrix_size2);
  cuarma::slice slice2_B(matrix_size3, 2, matrix_size3);

  cuarma::matrix<NumericT, F_B>    arma_B(matrix_size2, matrix_size3);
  cuarma::copy(B, arma_B);

  cuarma::matrix<NumericT, F_B>    arma_big_range_B(4*matrix_size2, 4*matrix_size3);
  cuarma::matrix_range<cuarma::matrix<NumericT, F_B> > arma_range_B(arma_big_range_B, range1_B, range2_B);
  cuarma::copy(B, arma_range_B);

  cuarma::matrix<NumericT, F_B>    arma_big_slice_B(4*matrix_size2, 4*matrix_size3);
  cuarma::matrix_slice<cuarma::matrix<NumericT, F_B> > arma_slice_B(arma_big_slice_B, slice1_B, slice2_B);
  cuarma::copy(B, arma_slice_B);


  // B^T

  cuarma::matrix<NumericT, F_B>    arma_B_trans(matrix_size3, matrix_size2);
  cuarma::copy(B_trans, arma_B_trans);

  cuarma::matrix<NumericT, F_B>    arma_big_range_B_trans(4*matrix_size3, 4*matrix_size2);
  cuarma::matrix_range<cuarma::matrix<NumericT, F_B> > arma_range_B_trans(arma_big_range_B_trans, range2_B, range1_B);
  cuarma::copy(B_trans, arma_range_B_trans);

  cuarma::matrix<NumericT, F_B>    arma_big_slice_B_trans(4*matrix_size3, 4*matrix_size2);
  cuarma::matrix_slice<cuarma::matrix<NumericT, F_B> > arma_slice_B_trans(arma_big_slice_B_trans, slice2_B, slice1_B);
  cuarma::copy(B_trans, arma_slice_B_trans);


  // C

  cuarma::range range1_C(matrix_size1-1, 2*matrix_size1-1);
  cuarma::range range2_C(matrix_size3-1, 2*matrix_size3-1);
  cuarma::slice slice1_C(matrix_size1-1, 3, matrix_size1);
  cuarma::slice slice2_C(matrix_size3-1, 3, matrix_size3);

  cuarma::matrix<NumericT, F_C>    arma_C(matrix_size1, matrix_size3);

  cuarma::matrix<NumericT, F_C>    arma_big_range_C(4*matrix_size1, 4*matrix_size3);
  cuarma::matrix_range<cuarma::matrix<NumericT, F_C> > arma_range_C(arma_big_range_C, range1_C, range2_C);

  cuarma::matrix<NumericT, F_C>    arma_big_slice_C(4*matrix_size1, 4*matrix_size3);
  cuarma::matrix_slice<cuarma::matrix<NumericT, F_C> > arma_slice_C(arma_big_slice_C, slice1_C, slice2_C);


  std::cout << "--- Part 1: Testing matrix-matrix products ---" << std::endl;

  //////
  //////  A: matrix
  //////

  //
  //
  std::cout << "Now using A=matrix, B=matrix, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_A, arma_A_trans,
                            arma_B, arma_B_trans,
                            arma_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=matrix, B=matrix, C=range" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_A, arma_A_trans,
                            arma_B, arma_B_trans,
                            arma_range_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=matrix, B=matrix, C=slice" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_A, arma_A_trans,
                            arma_B, arma_B_trans,
                            arma_slice_C);
  if (ret != EXIT_SUCCESS)
    return ret;



  //
  //
  std::cout << "Now using A=matrix, B=range, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_A, arma_A_trans,
                            arma_range_B, arma_range_B_trans,
                            arma_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=matrix, B=range, C=range" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_A, arma_A_trans,
                            arma_range_B, arma_range_B_trans,
                            arma_range_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=matrix, B=range, C=slice" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_A, arma_A_trans,
                            arma_range_B, arma_range_B_trans,
                            arma_slice_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=matrix, B=slice, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_A, arma_A_trans,
                            arma_slice_B, arma_slice_B_trans,
                            arma_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=matrix, B=slice, C=range" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_A, arma_A_trans,
                            arma_slice_B, arma_slice_B_trans,
                            arma_range_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=matrix, B=slice, C=slice" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_A, arma_A_trans,
                            arma_slice_B, arma_slice_B_trans,
                            arma_slice_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //////
  //////  A: range
  //////

  //
  //
  std::cout << "Now using A=range, B=matrix, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_range_A, arma_range_A_trans,
                            arma_B, arma_B_trans,
                            arma_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=range, B=matrix, C=range" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_range_A, arma_range_A_trans,
                            arma_B, arma_B_trans,
                            arma_range_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=range, B=matrix, C=slice" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_range_A, arma_range_A_trans,
                            arma_B, arma_B_trans,
                            arma_slice_C);
  if (ret != EXIT_SUCCESS)
    return ret;



  //
  //
  std::cout << "Now using A=range, B=range, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_range_A, arma_range_A_trans,
                            arma_range_B, arma_range_B_trans,
                            arma_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=range, B=range, C=range" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_range_A, arma_range_A_trans,
                            arma_range_B, arma_range_B_trans,
                            arma_range_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=range, B=range, C=slice" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_range_A, arma_range_A_trans,
                            arma_range_B, arma_range_B_trans,
                            arma_slice_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=range, B=slice, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_range_A, arma_range_A_trans,
                            arma_slice_B, arma_slice_B_trans,
                            arma_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=range, B=slice, C=range" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_range_A, arma_range_A_trans,
                            arma_slice_B, arma_slice_B_trans,
                            arma_range_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=range, B=slice, C=slice" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_range_A, arma_range_A_trans,
                            arma_slice_B, arma_slice_B_trans,
                            arma_slice_C);
  if (ret != EXIT_SUCCESS)
    return ret;



  //////
  //////  A: slice
  //////

  //
  //
  std::cout << "Now using A=slice, B=matrix, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_slice_A, arma_slice_A_trans,
                            arma_B, arma_B_trans,
                            arma_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=slice, B=matrix, C=range" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_slice_A, arma_slice_A_trans,
                            arma_B, arma_B_trans,
                            arma_range_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=slice, B=matrix, C=slice" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_slice_A, arma_slice_A_trans,
                            arma_B, arma_B_trans,
                            arma_slice_C);
  if (ret != EXIT_SUCCESS)
    return ret;



  //
  //
  std::cout << "Now using A=slice, B=range, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_slice_A, arma_slice_A_trans,
                            arma_range_B, arma_range_B_trans,
                            arma_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=slice, B=range, C=range" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_slice_A, arma_slice_A_trans,
                            arma_range_B, arma_range_B_trans,
                            arma_range_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=slice, B=range, C=slice" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_slice_A, arma_slice_A_trans,
                            arma_range_B, arma_range_B_trans,
                            arma_slice_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=slice, B=slice, C=matrix" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_slice_A, arma_slice_A_trans,
                            arma_slice_B, arma_slice_B_trans,
                            arma_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  //
  //
  std::cout << "Now using A=slice, B=slice, C=range" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_slice_A, arma_slice_A_trans,
                            arma_slice_B, arma_slice_B_trans,
                            arma_range_C);
  if (ret != EXIT_SUCCESS)
    return ret;

  //
  //
  std::cout << "Now using A=slice, B=slice, C=slice" << std::endl;
  ret = test_prod<NumericT>(epsilon,
                            A, A_trans, B, B_trans, C,
                            arma_slice_A, arma_slice_A_trans,
                            arma_slice_B, arma_slice_B_trans,
                            arma_slice_C);
  if (ret != EXIT_SUCCESS)
    return ret;


  return ret;

}


//
// Control functions
//



template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=row, C=row ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, cuarma::row_major, cuarma::row_major, cuarma::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=row, C=col ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, cuarma::row_major, cuarma::row_major, cuarma::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=col, C=row ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, cuarma::row_major, cuarma::column_major, cuarma::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=col, C=col ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, cuarma::row_major, cuarma::column_major, cuarma::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=row, C=row ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, cuarma::column_major, cuarma::row_major, cuarma::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=row, C=col ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, cuarma::column_major, cuarma::row_major, cuarma::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=col, C=row ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, cuarma::column_major, cuarma::column_major, cuarma::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "///////////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=col, C=col ///" << std::endl;
  std::cout << "///////////////////////////////////////" << std::endl;
  ret = test_prod<NumericT, cuarma::column_major, cuarma::column_major, cuarma::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;



  return ret;
}

//
// -------------------------------------------------------------
//
int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: BLAS 3 routines" << std::endl;
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
        NumericT epsilon = 1.0E-11;
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
