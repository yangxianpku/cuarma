/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   spmdm.cu
 *   @coding UTF-8
 *   @brief  Tests sparse-matrix-dense-matrix products.
 *   @brief  ≤‚ ‘£∫œ° Ëæÿ’Û-≥Ì√‹æÿ’Û≥Àª˝
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <map>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/coordinate_matrix.hpp"
#include "cuarma/ell_matrix.hpp"
#include "cuarma/hyb_matrix.hpp"
#include "cuarma/blas/prod.hpp"       //generic matrix-vector product
#include "cuarma/blas/norm_2.hpp"     //generic l2-norm for vectors
#include "cuarma/io/matrix_market.hpp"
#include "cuarma/tools/random.hpp"


template<typename NumericT>
int check_matrices(std::vector<std::vector<NumericT> > const & ref_mat,
                   std::vector<std::vector<NumericT> > const & mat,
                   NumericT eps)
{
  if ( (ref_mat.size() != mat.size()) || (ref_mat[0].size() != mat[0].size()) )
    return EXIT_FAILURE;

  for (std::size_t i = 0; i < ref_mat.size(); i++)
    for (std::size_t j = 0; j < ref_mat[0].size(); j++)
    {
      NumericT rel_error = std::abs(ref_mat[i][j] - mat[i][j]) / std::max(std::abs(ref_mat[i][j]), std::abs(mat[i][j]));
      if (rel_error > eps)
      {
        std::cout << "ERROR: Verification failed at (" << i <<", "<< j << "): "
                  << " Expected: " << ref_mat[i][j] << ", got: " << mat[i][j] << " (relative error: " << rel_error << ")" << std::endl;
        return EXIT_FAILURE;
      }
    }

  std::cout << "Everything went well!" << std::endl;
  return EXIT_SUCCESS;
}


// Computes C = A * B for a sparse matrix A and dense matrices B and C.
// C is initialized with zeros
template<typename IndexT, typename NumericT>
void compute_reference_result(std::vector<std::map<IndexT, NumericT> > const & A,
                              std::vector<std::vector<NumericT> > const & B,
                              std::vector<std::vector<NumericT> >       & C)
{
  typedef typename std::map<IndexT, NumericT>::const_iterator RowIterator;

  for (std::size_t i=0; i<C.size(); ++i)
    for (RowIterator it = A[i].begin(); it != A[i].end(); ++it)
    {
      IndexT   col_A = it->first;
      NumericT val_A = it->second;

      for (std::size_t j=0; j<C[i].size(); ++j)
        C[i][j] += val_A * B[col_A][j];
    }
}


template<typename NumericT, typename ResultLayoutT, typename FactorLayoutT>
int test(NumericT epsilon)
{
  int retVal = EXIT_SUCCESS;

  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  std::vector<std::map<unsigned int, NumericT> > std_A;
  if (cuarma::io::read_matrix_market_file(std_A, "../examples/testdata/mat65k.mtx") == EXIT_FAILURE)
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  // add some extra weight to diagonal in order to avoid issues with round-off errors
  for (std::size_t i=0; i<std_A.size(); ++i)
    std_A[i][static_cast<unsigned int>(i)] *= NumericT(1.5);

  std::size_t cols_rhs = 5;

  cuarma::compressed_matrix<NumericT> compressed_A;
  cuarma::ell_matrix<NumericT>        ell_A;
  cuarma::coordinate_matrix<NumericT> coo_A;
  cuarma::hyb_matrix<NumericT>        hyb_A;

  std::vector<std::vector<NumericT> >       std_C(std_A.size(), std::vector<NumericT>(cols_rhs));
  cuarma::matrix<NumericT, ResultLayoutT>     C;

  cuarma::copy(std_A, compressed_A);
  cuarma::copy(std_A, ell_A);
  cuarma::copy(std_A, coo_A);
  cuarma::copy(std_A, hyb_A);

  std::vector<std::vector<NumericT> >        std_B(std_A.size(), std::vector<NumericT>(cols_rhs));
  cuarma::matrix<NumericT, FactorLayoutT>  B1(std_A.size(), cols_rhs);
  cuarma::matrix<NumericT, FactorLayoutT>  B2;

  std::vector<std::vector<NumericT> > temp(std_A.size(), std::vector<NumericT>(cols_rhs));

  for (unsigned int i = 0; i < std_B.size(); i++)
    for (unsigned int j = 0; j < std_B[i].size(); j++)
      std_B[i][j] = NumericT(0.5) + NumericT(0.1) * randomNumber();
  cuarma::copy(std_B, B1);


  /* gold result */
  compute_reference_result(std_A, std_B, std_C);

  /******************************************************************/
  std::cout << "Testing compressed(CSR) lhs * dense rhs" << std::endl;
  C = cuarma::blas::prod(compressed_A, B1);

  for (std::size_t i=0; i<temp.size(); ++i)
    for (std::size_t j=0; j<temp[i].size(); ++j)
      temp[i][j] = 0;
  cuarma::copy(C, temp);
  retVal = check_matrices(std_C, temp, epsilon);
  if (retVal != EXIT_SUCCESS)
  {
    std::cerr << "Test failed!" << std::endl;
    return retVal;
  }

  /******************************************************************/
  std::cout << "Testing compressed(ELL) lhs * dense rhs" << std::endl;
  C.clear();
  C = cuarma::blas::prod(ell_A, B1);

  for (std::size_t i=0; i<temp.size(); ++i)
    for (std::size_t j=0; j<temp[i].size(); ++j)
      temp[i][j] = 0;
  cuarma::copy(C, temp);
  retVal = check_matrices(std_C, temp, epsilon);
  if (retVal != EXIT_SUCCESS)
  {
    std::cerr << "Test failed!" << std::endl;
    return retVal;
  }

  /******************************************************************/

  std::cout << "Testing compressed(COO) lhs * dense rhs" << std::endl;
  C.clear();
  C = cuarma::blas::prod(coo_A, B1);

  for (std::size_t i=0; i<temp.size(); ++i)
    for (std::size_t j=0; j<temp[i].size(); ++j)
      temp[i][j] = 0;
  cuarma::copy(C, temp);
  retVal = check_matrices(std_C, temp, epsilon);
  if (retVal != EXIT_SUCCESS)
  {
    std::cerr << "Test failed!" << std::endl;
    return retVal;
  }

  /******************************************************************/

  std::cout << "Testing compressed(HYB) lhs * dense rhs" << std::endl;
  C.clear();
  C = cuarma::blas::prod(hyb_A, B1);

  for (std::size_t i=0; i<temp.size(); ++i)
    for (std::size_t j=0; j<temp[i].size(); ++j)
      temp[i][j] = 0;
  cuarma::copy(C, temp);
  retVal = check_matrices(std_C, temp, epsilon);
  if (retVal != EXIT_SUCCESS)
  {
    std::cerr << "Test failed!" << std::endl;
    return retVal;
  }

  /******************************************************************/


  ///////////// transposed right hand side

  B2 = cuarma::trans(B1);

  /******************************************************************/
  std::cout << std::endl << "Testing compressed(CSR) lhs * transposed dense rhs:" << std::endl;
  C.clear();
  C = cuarma::blas::prod(compressed_A, cuarma::trans(B2));

  for (std::size_t i=0; i<temp.size(); ++i)
    for (std::size_t j=0; j<temp[i].size(); ++j)
      temp[i][j] = 0;
  cuarma::copy(C, temp);
  retVal = check_matrices(std_C, temp, epsilon);
  if (retVal != EXIT_SUCCESS)
  {
    std::cerr << "Test failed!" << std::endl;
    return retVal;
  }

  /******************************************************************/
  std::cout << "Testing compressed(ELL) lhs * transposed dense rhs" << std::endl;
  C.clear();
  C = cuarma::blas::prod(ell_A, cuarma::trans(B2));

  for (std::size_t i=0; i<temp.size(); ++i)
    for (std::size_t j=0; j<temp[i].size(); ++j)
      temp[i][j] = 0;
  cuarma::copy(C, temp);
  retVal = check_matrices(std_C, temp, epsilon);
  if (retVal != EXIT_SUCCESS)
  {
    std::cerr << "Test failed!" << std::endl;
    return retVal;
  }

  /******************************************************************/
  std::cout << "Testing compressed(COO) lhs * transposed dense rhs" << std::endl;
  C.clear();
  C = cuarma::blas::prod(coo_A, cuarma::trans(B2));

  for (std::size_t i=0; i<temp.size(); ++i)
    for (std::size_t j=0; j<temp[i].size(); ++j)
      temp[i][j] = 0;
  cuarma::copy(C, temp);
  retVal = check_matrices(std_C, temp, epsilon);
  if (retVal != EXIT_SUCCESS)
  {
    std::cerr << "Test failed!" << std::endl;
    return retVal;
  }

  /******************************************************************/

  std::cout << "Testing compressed(HYB) lhs * transposed dense rhs" << std::endl;
  C.clear();
  C = cuarma::blas::prod(hyb_A, cuarma::trans(B2));

  for (std::size_t i=0; i<temp.size(); ++i)
    for (std::size_t j=0; j<temp[i].size(); ++j)
      temp[i][j] = 0;
  cuarma::copy(C, temp);
  retVal = check_matrices(std_C, temp, epsilon);
  if (retVal != EXIT_SUCCESS)
  {
    std::cerr << "Test failed!" << std::endl;
    return retVal;
  }

  /******************************************************************/
  if (retVal == EXIT_SUCCESS) {
    std::cout << "Tests passed successfully" << std::endl;
  }

  return retVal;
}

//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Sparse-Dense Matrix Multiplication" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    typedef float NumericT;
    NumericT epsilon = static_cast<NumericT>(1E-4);
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    std::cout << "  layout:  row-major, row-major" << std::endl;
    retval = test<NumericT, cuarma::row_major, cuarma::row_major>(epsilon);
    if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;

    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    std::cout << "  layout:  row-major, column-major" << std::endl;
    retval = test<NumericT, cuarma::row_major, cuarma::column_major>(epsilon);
    if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;

    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    std::cout << "  layout:  column-major, row-major" << std::endl;
    retval = test<NumericT, cuarma::column_major, cuarma::row_major>(epsilon);
    if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;

    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    std::cout << "  layout:  column-major, column-major" << std::endl;
    retval = test<NumericT, cuarma::column_major, cuarma::column_major>(epsilon);
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
      std::cout << "  layout:  row-major, row-major" << std::endl;
      retval = test<NumericT, cuarma::row_major, cuarma::row_major>(epsilon);
      if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;

      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: double" << std::endl;
      std::cout << "  layout:  row-major, column-major" << std::endl;
      retval = test<NumericT, cuarma::row_major, cuarma::column_major>(epsilon);
      if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;

      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: double" << std::endl;
      std::cout << "  layout:  column-major, row-major" << std::endl;
      retval = test<NumericT, cuarma::column_major, cuarma::row_major>(epsilon);
      if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;

      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: double" << std::endl;
      std::cout << "  layout:  column-major, column-major" << std::endl;
      retval = test<NumericT, cuarma::column_major, cuarma::column_major>(epsilon);
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

