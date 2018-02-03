/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   blas3_solve.cu
 *   @coding UTF-8
 *   @brief  ests the BLAS level 3 triangular solvers.
 *   @brief  ≤‚ ‘£∫BLAS level 3 triangular solvers
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

template<typename ScalarType>
ScalarType diff(ScalarType & s1, cuarma::scalar<ScalarType> & s2)
{
   cuarma::backend::finish();
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), fabs(s2));
   return 0;
}

template<typename ScalarType>
ScalarType diff(std::vector<ScalarType> & v1, cuarma::vector<ScalarType> & v2)
{
   std::vector<ScalarType> v2_cpu(v2.size());
   cuarma::backend::finish();
   cuarma::copy(v2.begin(), v2.end(), v2_cpu.begin());
   cuarma::backend::finish();

   for (std::size_t i=0;i<v1.size(); ++i)
   {
      if ( std::max( fabs(v2_cpu[i]), fabs(v1[i]) ) > 0 )
         v2_cpu[i] = fabs(v2_cpu[i] - v1[i]) / std::max( fabs(v2_cpu[i]), fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   return norm_inf(v2_cpu);
}


template<typename ScalarType, typename VCLMatrixType>
ScalarType diff(std::vector<std::vector<ScalarType> > & mat1, VCLMatrixType & mat2)
{
   std::vector<std::vector<ScalarType> > mat2_cpu(mat2.size1(), std::vector<ScalarType>(mat2.size2()));
   cuarma::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
   cuarma::copy(mat2, mat2_cpu);
   ScalarType ret = 0;
   ScalarType act = 0;

    for (unsigned int i = 0; i < mat2_cpu.size(); ++i)
    {
      for (unsigned int j = 0; j < mat2_cpu[i].size(); ++j)
      {
        act = std::fabs(mat2_cpu[i][j] - mat1[i][j]) / std::max( std::fabs(mat2_cpu[i][j]), std::fabs(mat1[i][j]) );
         if (act > ret)
           ret = act;
      }
    }
   //std::cout << ret << std::endl;
   return ret;
}


// Triangular solvers
template<typename NumericT>
void inplace_solve_lower(std::vector<std::vector<NumericT> > const & A, std::vector<std::vector<NumericT> > & B, bool unit_diagonal)
{
  for (std::size_t i=0; i<A.size(); ++i)
  {
    for (std::size_t j=0; j < i; ++j)
    {
      NumericT val_A = A[i][j];
      for (std::size_t k=0; k<B[i].size(); ++k)
        B[i][k] -= val_A * B[j][k];
    }

    NumericT diag_A = unit_diagonal ? NumericT(1) : A[i][i];

    for (std::size_t k=0; k<B[i].size(); ++k)
      B[i][k] /= diag_A;
  }
}

template<typename NumericT>
void inplace_solve(std::vector<std::vector<NumericT> > const & A, std::vector<std::vector<NumericT> > & B, cuarma::blas::lower_tag)
{
  inplace_solve_lower(A, B, false);
}

template<typename NumericT>
void inplace_solve(std::vector<std::vector<NumericT> > const & A, std::vector<std::vector<NumericT> > & B, cuarma::blas::unit_lower_tag)
{
  inplace_solve_lower(A, B, true);
}

template<typename NumericT>
void inplace_solve_upper(std::vector<std::vector<NumericT> > const & A, std::vector<std::vector<NumericT> > & B, bool unit_diagonal)
{
  for (std::size_t i2=0; i2<A.size(); ++i2)
  {
    std::size_t i = A.size() - i2 - 1;
    for (std::size_t j=i+1; j < A[0].size(); ++j)
    {
      NumericT val_A = A[i][j];
      for (std::size_t k=0; k<B[i].size(); ++k)
        B[i][k] -= val_A * B[j][k];
    }

    NumericT diag_A = unit_diagonal ? NumericT(1) : A[i][i];

    for (std::size_t k=0; k<B[i].size(); ++k)
      B[i][k] /= diag_A;
  }
}

template<typename NumericT>
void inplace_solve(std::vector<std::vector<NumericT> > const & A, std::vector<std::vector<NumericT> > & B, cuarma::blas::upper_tag)
{
  inplace_solve_upper(A, B, false);
}

template<typename NumericT>
void inplace_solve(std::vector<std::vector<NumericT> > const & A, std::vector<std::vector<NumericT> > & B, cuarma::blas::unit_upper_tag)
{
  inplace_solve_upper(A, B, true);
}

template<typename NumericT, typename SolverTagT>
std::vector<std::vector<NumericT> > solve(std::vector<std::vector<NumericT> > const & A, std::vector<std::vector<NumericT> > const & B, SolverTagT)
{
  std::vector<std::vector<NumericT> > C(B);
  inplace_solve(A, C, SolverTagT());
  return C;
}


template<typename RHSTypeRef, typename RHSTypeCheck, typename Epsilon >
void run_solver_check(RHSTypeRef & B_ref, RHSTypeCheck & B_check, int & retval, Epsilon const & epsilon)
{
   double act_diff = fabs(diff(B_ref, B_check));
   if ( act_diff > epsilon )
   {
     std::cout << " FAILED!" << std::endl;
     std::cout << "# Error at operation: matrix-matrix solve" << std::endl;
     std::cout << "  diff: " << act_diff << std::endl;
     retval = EXIT_FAILURE;
   }
   else
     std::cout << " passed! " << act_diff << std::endl;

}

template<typename NumericT>
std::vector<std::vector<NumericT> > trans(std::vector<std::vector<NumericT> > const & A)
{
  std::vector<std::vector<NumericT> > A_trans(A[0].size(), std::vector<NumericT>(A.size()));

  for (std::size_t i=0; i<A.size(); ++i)
    for (std::size_t j=0; j<A[i].size(); ++j)
      A_trans[j][i] = A[i][j];

  return A_trans;
}


template< typename NumericT, typename Epsilon, typename ReferenceMatrixTypeA, typename ReferenceMatrixTypeB, 
	typename ReferenceMatrixTypeC,typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeResult>
int test_solve(Epsilon const& epsilon,ReferenceMatrixTypeA const & A,ReferenceMatrixTypeB const & B_start,
              ReferenceMatrixTypeC const & C_start, MatrixTypeA const & arma_A, MatrixTypeB & arma_B, MatrixTypeC & arma_C, MatrixTypeResult const & )
{
   int retval = EXIT_SUCCESS;

   // --------------------------------------------------------------------------

   ReferenceMatrixTypeA result;
   ReferenceMatrixTypeC C_trans;

   ReferenceMatrixTypeB B = B_start;
   ReferenceMatrixTypeC C = C_start;

   MatrixTypeResult arma_result;

   // Test: A \ B with various tags --------------------------------------------------------------------------
   std::cout << "Testing A \\ B: " << std::endl;
   std::cout << " * upper_tag:      ";
   result = solve(A, B, cuarma::blas::upper_tag());
   arma_result = cuarma::blas::solve(arma_A, arma_B, cuarma::blas::upper_tag());
   run_solver_check(result, arma_result, retval, epsilon);

   std::cout << " * unit_upper_tag: ";
   result = solve(A, B, cuarma::blas::unit_upper_tag());
   arma_result = cuarma::blas::solve(arma_A, arma_B, cuarma::blas::unit_upper_tag());
   run_solver_check(result, arma_result, retval, epsilon);

   std::cout << " * lower_tag:      ";
   result = solve(A, B, cuarma::blas::lower_tag());
   arma_result = cuarma::blas::solve(arma_A, arma_B, cuarma::blas::lower_tag());
   run_solver_check(result, arma_result, retval, epsilon);

   std::cout << " * unit_lower_tag: ";
   result = solve(A, B, cuarma::blas::unit_lower_tag());
   arma_result = cuarma::blas::solve(arma_A, arma_B, cuarma::blas::unit_lower_tag());
   run_solver_check(result, arma_result, retval, epsilon);

   if (retval == EXIT_SUCCESS)
     std::cout << "Test A \\ B passed!" << std::endl;

   B = B_start;
   C = C_start;

   // Test: A \ B^T --------------------------------------------------------------------------
   std::cout << "Testing A \\ B^T: " << std::endl;
   std::cout << " * upper_tag:      ";
   cuarma::copy(C, arma_C); C_trans = trans(C);

   //check solve():
   result = solve(A, C_trans, cuarma::blas::upper_tag());
   arma_result = cuarma::blas::solve(arma_A, trans(arma_C), cuarma::blas::upper_tag());
   run_solver_check(result, arma_result, retval, epsilon);

   //check compute kernels:
   std::cout << " * upper_tag:      ";
   inplace_solve(A, C_trans, cuarma::blas::upper_tag());
   cuarma::blas::inplace_solve(arma_A, trans(arma_C), cuarma::blas::upper_tag());
   C = trans(C_trans); run_solver_check(C, arma_C, retval, epsilon);

   std::cout << " * unit_upper_tag: ";
   cuarma::copy(C, arma_C); C_trans = trans(C);
   inplace_solve(A, C_trans, cuarma::blas::unit_upper_tag());
   cuarma::blas::inplace_solve(arma_A, trans(arma_C), cuarma::blas::unit_upper_tag());
   C = trans(C_trans); run_solver_check(C, arma_C, retval, epsilon);

   std::cout << " * lower_tag:      ";
   cuarma::copy(C, arma_C); C_trans = trans(C);
   inplace_solve(A, C_trans, cuarma::blas::lower_tag());
   cuarma::blas::inplace_solve(arma_A, trans(arma_C), cuarma::blas::lower_tag());
   C = trans(C_trans); run_solver_check(C, arma_C, retval, epsilon);

   std::cout << " * unit_lower_tag: ";
   cuarma::copy(C, arma_C); C_trans = trans(C);
   inplace_solve(A, C_trans, cuarma::blas::unit_lower_tag());
   cuarma::blas::inplace_solve(arma_A, trans(arma_C), cuarma::blas::unit_lower_tag());
   C = trans(C_trans); run_solver_check(C, arma_C, retval, epsilon);

   if (retval == EXIT_SUCCESS)
     std::cout << "Test A \\ B^T passed!" << std::endl;

   B = B_start;
   C = C_start;

   // Test: A \ B with various tags --------------------------------------------------------------------------
   std::cout << "Testing A^T \\ B: " << std::endl;
   std::cout << " * upper_tag:      ";
   cuarma::copy(B, arma_B);
   result = solve(trans(A), B, cuarma::blas::upper_tag());
   arma_result = cuarma::blas::solve(trans(arma_A), arma_B, cuarma::blas::upper_tag());
   run_solver_check(result, arma_result, retval, epsilon);

   std::cout << " * unit_upper_tag: ";
   cuarma::copy(B, arma_B);
   result = solve(trans(A), B, cuarma::blas::unit_upper_tag());
   arma_result = cuarma::blas::solve(trans(arma_A), arma_B, cuarma::blas::unit_upper_tag());
   run_solver_check(result, arma_result, retval, epsilon);

   std::cout << " * lower_tag:      ";
   cuarma::copy(B, arma_B);
   result = solve(trans(A), B, cuarma::blas::lower_tag());
   arma_result = cuarma::blas::solve(trans(arma_A), arma_B, cuarma::blas::lower_tag());
   run_solver_check(result, arma_result, retval, epsilon);

   std::cout << " * unit_lower_tag: ";
   cuarma::copy(B, arma_B);
   result = solve(trans(A), B, cuarma::blas::unit_lower_tag());
   arma_result = cuarma::blas::solve(trans(arma_A), arma_B, cuarma::blas::unit_lower_tag());
   run_solver_check(result, arma_result, retval, epsilon);

   if (retval == EXIT_SUCCESS)
     std::cout << "Test A^T \\ B passed!" << std::endl;

   B = B_start;
   C = C_start;

   // Test: A^T \ B^T --------------------------------------------------------------------------
   std::cout << "Testing A^T \\ B^T: " << std::endl;
   std::cout << " * upper_tag:      ";
   cuarma::copy(C, arma_C); C_trans = trans(C);
   //check solve():
   result = solve(trans(A), C_trans, cuarma::blas::upper_tag());
   arma_result = cuarma::blas::solve(trans(arma_A), trans(arma_C), cuarma::blas::upper_tag());
   run_solver_check(result, arma_result, retval, epsilon);
   //check kernels:
   std::cout << " * upper_tag:      ";
   inplace_solve(trans(A), C_trans, cuarma::blas::upper_tag());
   cuarma::blas::inplace_solve(trans(arma_A), trans(arma_C), cuarma::blas::upper_tag());
   C = trans(C_trans); run_solver_check(C, arma_C, retval, epsilon);

   std::cout << " * unit_upper_tag: ";
   cuarma::copy(C, arma_C); C_trans = trans(C);
   inplace_solve(trans(A), C_trans, cuarma::blas::unit_upper_tag());
   cuarma::blas::inplace_solve(trans(arma_A), trans(arma_C), cuarma::blas::unit_upper_tag());
   C = trans(C_trans); run_solver_check(C, arma_C, retval, epsilon);

   std::cout << " * lower_tag:      ";
   cuarma::copy(C, arma_C); C_trans = trans(C);
   inplace_solve(trans(A), C_trans, cuarma::blas::lower_tag());
   cuarma::blas::inplace_solve(trans(arma_A), trans(arma_C), cuarma::blas::lower_tag());
   C = trans(C_trans); run_solver_check(C, arma_C, retval, epsilon);

   std::cout << " * unit_lower_tag: ";
   cuarma::copy(C, arma_C); C_trans = trans(C);
   inplace_solve(trans(A), C_trans, cuarma::blas::unit_lower_tag());
   cuarma::blas::inplace_solve(trans(arma_A), trans(arma_C), cuarma::blas::unit_lower_tag());
   C = trans(C_trans); run_solver_check(C, arma_C, retval, epsilon);

   if (retval == EXIT_SUCCESS)
     std::cout << "Test A^T \\ B^T passed!" << std::endl;

   return retval;
}


template< typename NumericT, typename F_A, typename F_B, typename Epsilon >
int test_solve(Epsilon const& epsilon)
{
  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  int ret = EXIT_SUCCESS;
  std::size_t matrix_size = 135;  //some odd number, not too large
  std::size_t rhs_num = 67;

  std::cout << "--- Part 2: Testing matrix-matrix solver ---" << std::endl;


  std::vector<std::vector<NumericT> > A(matrix_size, std::vector<NumericT>(matrix_size));
  std::vector<std::vector<NumericT> > B_start(matrix_size,  std::vector<NumericT>(rhs_num));
  std::vector<std::vector<NumericT> > C_start(rhs_num,  std::vector<NumericT>(matrix_size));

  for (std::size_t i = 0; i < A.size(); ++i)
  {
    for (std::size_t j = 0; j < A[i].size(); ++j)
        A[i][j] = static_cast<NumericT>(-0.5) * randomNumber();
    A[i][i] = NumericT(1.0) + NumericT(2.0) * randomNumber(); //some extra weight on diagonal for stability
  }

  for (std::size_t i = 0; i < B_start.size(); ++i)
    for (std::size_t j = 0; j < B_start[i].size(); ++j)
      B_start[i][j] = randomNumber();

  for (std::size_t i = 0; i < C_start.size(); ++i)
    for (std::size_t j = 0; j < C_start[i].size(); ++j)
      C_start[i][j] = randomNumber();


  // A
  cuarma::range range1_A(matrix_size, 2*matrix_size);
  cuarma::range range2_A(2*matrix_size, 3*matrix_size);
  cuarma::slice slice1_A(matrix_size, 2, matrix_size);
  cuarma::slice slice2_A(0, 3, matrix_size);

  cuarma::matrix<NumericT, F_A>    arma_A(matrix_size, matrix_size);
  cuarma::copy(A, arma_A);

  cuarma::matrix<NumericT, F_A>    arma_big_range_A(4*matrix_size, 4*matrix_size);
  cuarma::matrix_range<cuarma::matrix<NumericT, F_A> > arma_range_A(arma_big_range_A, range1_A, range2_A);
  cuarma::copy(A, arma_range_A);

  cuarma::matrix<NumericT, F_A>    arma_big_slice_A(4*matrix_size, 4*matrix_size);
  cuarma::matrix_slice<cuarma::matrix<NumericT, F_A> > arma_slice_A(arma_big_slice_A, slice1_A, slice2_A);
  cuarma::copy(A, arma_slice_A);


  // B
  cuarma::range range1_B(matrix_size, 2*matrix_size);
  cuarma::range range2_B(2*rhs_num, 3*rhs_num);
  cuarma::slice slice1_B(matrix_size, 2, matrix_size);
  cuarma::slice slice2_B(0, 3, rhs_num);

  cuarma::matrix<NumericT, F_B>    arma_B(matrix_size, rhs_num);
  cuarma::copy(B_start, arma_B);

  cuarma::matrix<NumericT, F_B>    arma_big_range_B(4*matrix_size, 4*rhs_num);
  cuarma::matrix_range<cuarma::matrix<NumericT, F_B> > arma_range_B(arma_big_range_B, range1_B, range2_B);
  cuarma::copy(B_start, arma_range_B);

  cuarma::matrix<NumericT, F_B>    arma_big_slice_B(4*matrix_size, 4*rhs_num);
  cuarma::matrix_slice<cuarma::matrix<NumericT, F_B> > arma_slice_B(arma_big_slice_B, slice1_B, slice2_B);
  cuarma::copy(B_start, arma_slice_B);


  // C
  cuarma::range range1_C(rhs_num, 2*rhs_num);
  cuarma::range range2_C(2*matrix_size, 3*matrix_size);
  cuarma::slice slice1_C(rhs_num, 2, rhs_num);
  cuarma::slice slice2_C(0, 3, matrix_size);

  cuarma::matrix<NumericT, F_B>    arma_C(rhs_num, matrix_size);
  cuarma::copy(C_start, arma_C);

  cuarma::matrix<NumericT, F_B>    arma_big_range_C(4*rhs_num, 4*matrix_size);
  cuarma::matrix_range<cuarma::matrix<NumericT, F_B> > arma_range_C(arma_big_range_C, range1_C, range2_C);
  cuarma::copy(C_start, arma_range_C);

  cuarma::matrix<NumericT, F_B>    arma_big_slice_C(4*rhs_num, 4*matrix_size);
  cuarma::matrix_slice<cuarma::matrix<NumericT, F_B> > arma_slice_C(arma_big_slice_C, slice1_C, slice2_C);
  cuarma::copy(C_start, arma_slice_C);


  std::cout << "Now using A=matrix, B=matrix" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             arma_A, arma_B, arma_C, arma_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=matrix, B=range" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             arma_A, arma_range_B, arma_range_C, arma_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=matrix, B=slice" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             arma_A, arma_slice_B, arma_slice_C, arma_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;



  std::cout << "Now using A=range, B=matrix" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             arma_range_A, arma_B, arma_C, arma_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=range, B=range" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             arma_range_A, arma_range_B, arma_range_C, arma_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=range, B=slice" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             arma_range_A, arma_slice_B, arma_slice_C, arma_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;


  std::cout << "Now using A=slice, B=matrix" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             arma_slice_A, arma_B, arma_C, arma_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=slice, B=range" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             arma_slice_A, arma_range_B, arma_range_C, arma_B
                            );
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "Now using A=slice, B=slice" << std::endl;
  ret = test_solve<NumericT>(epsilon,
                             A, B_start, C_start,
                             arma_slice_A, arma_slice_B, arma_slice_C, arma_B
                            );
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

  std::cout << "////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=row ///" << std::endl;
  std::cout << "////////////////////////////////" << std::endl;
  ret = test_solve<NumericT, cuarma::row_major, cuarma::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;


  std::cout << "////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=row, B=col ///" << std::endl;
  std::cout << "////////////////////////////////" << std::endl;
  ret = test_solve<NumericT, cuarma::row_major, cuarma::column_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=row ///" << std::endl;
  std::cout << "////////////////////////////////" << std::endl;
  ret = test_solve<NumericT, cuarma::column_major, cuarma::row_major>(epsilon);
  if (ret != EXIT_SUCCESS)
    return ret;

  std::cout << "////////////////////////////////" << std::endl;
  std::cout << "/// Now testing A=col, B=col ///" << std::endl;
  std::cout << "////////////////////////////////" << std::endl;
  ret = test_solve<NumericT, cuarma::column_major, cuarma::column_major>(epsilon);
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
