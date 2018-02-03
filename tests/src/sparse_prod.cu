/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   sparse_prod.cu
 *   @coding UTF-8
 *   @brief  Tests sparse matrix operations.
 *   @brief  ≤‚ ‘£∫œ° Ëæÿ’Û≥Àª˝
 */

#include <iostream>
#include <vector>
#include <map>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/blas/prod.hpp"

#include "cuarma/tools/random.hpp"

//
// -------------------------------------------------------------
//

/* Routine for computing the relative difference of two matrices. 1 is returned if the sparsity patterns do not match. */
template<typename IndexT, typename NumericT, typename MatrixT>
NumericT diff(std::vector<std::map<IndexT, NumericT> > const & stl_A,
              MatrixT & arma_A)
{
  cuarma::switch_memory_context(arma_A, cuarma::context(cuarma::MAIN_MEMORY));

  NumericT error = NumericT(-1.0);

  NumericT     const * arma_A_elements   = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(arma_A.handle());
  unsigned int const * arma_A_row_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(arma_A.handle1());
  unsigned int const * arma_A_col_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(arma_A.handle2());


  /* Simultaneously compare the sparsity patterns of both matrices against each other. */

  unsigned int const * arma_A_current_col_ptr = arma_A_col_buffer;
  NumericT     const * arma_A_current_val_ptr = arma_A_elements;

  for (std::size_t row = 0; row < stl_A.size(); ++row)
  {
    if (arma_A_current_col_ptr != arma_A_col_buffer + arma_A_row_buffer[row])
    {
      std::cerr << "Sparsity pattern mismatch detected: Start of row out of sync!" << std::endl;
      std::cerr << " STL row: " << row << std::endl;
      std::cerr << " cuarma col ptr is: " << arma_A_current_col_ptr << std::endl;
      std::cerr << " cuarma col ptr should: " << arma_A_col_buffer + arma_A_row_buffer[row] << std::endl;
      std::cerr << " cuarma col ptr value: " << *arma_A_current_col_ptr << std::endl;
      return NumericT(1.0);
    }

    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (typename std::map<IndexT, NumericT>::const_iterator col_it = stl_A[row].begin();
          col_it != stl_A[row].end();
          ++col_it, ++arma_A_current_col_ptr, ++arma_A_current_val_ptr)
    {
      if (col_it->first != std::size_t(*arma_A_current_col_ptr))
      {
        std::cerr << "Sparsity pattern mismatch detected!" << std::endl;
        std::cerr << " STL row: " << row << std::endl;
        std::cerr << " STL col: " << col_it->first << std::endl;
        std::cerr << " cuarma row entries: " << arma_A_row_buffer[row] << ", " << arma_A_row_buffer[row + 1] << std::endl;
        std::cerr << " cuarma entry in row: " << arma_A_current_col_ptr - (arma_A_col_buffer + arma_A_row_buffer[row]) << std::endl;
        std::cerr << " cuarma col: " << *arma_A_current_col_ptr << std::endl;
        return NumericT(1.0);
      }

      // compute relative error (we know for sure that the uBLAS matrix only carries nonzero entries:
      NumericT current_error = std::fabs(col_it->second - *arma_A_current_val_ptr) / std::max(std::fabs(col_it->second), std::fabs(*arma_A_current_val_ptr));

      if (current_error > 0.1)
      {
        std::cerr << "Value mismatch detected!" << std::endl;
        std::cerr << " STL row: " << row << std::endl;
        std::cerr << " STL col: " << col_it->first << std::endl;
        std::cerr << " STL value: " << col_it->second << std::endl;
        std::cerr << " cuarma value: " << *arma_A_current_val_ptr << std::endl;
        return NumericT(1.0);
      }

      if (current_error > error)
        error = current_error;
    }
  }

  return error;
}

template<typename IndexT, typename NumericT>
void prod(std::vector<std::map<IndexT, NumericT> > const & stl_A,
          std::vector<std::map<IndexT, NumericT> > const & stl_B,
          std::vector<std::map<IndexT, NumericT> >       & stl_C)
{
  for (std::size_t i=0; i<stl_A.size(); ++i)
    for (typename std::map<IndexT, NumericT>::const_iterator it_A = stl_A[i].begin(); it_A != stl_A[i].end(); ++it_A)
    {
      IndexT row_B = it_A->first;
      for (typename std::map<IndexT, NumericT>::const_iterator it_B = stl_B[row_B].begin(); it_B != stl_B[row_B].end(); ++it_B)
        stl_C[i][it_B->first] += it_A->second * it_B->second;
    }
}


//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int retval = EXIT_SUCCESS;

  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  std::size_t N = 210;
  std::size_t K = 300;
  std::size_t M = 420;
  std::size_t nnz_row = 40;
  // --------------------------------------------------------------------------
  std::vector<std::map<unsigned int, NumericT> > stl_A(N);
  std::vector<std::map<unsigned int, NumericT> > stl_B(K);
  std::vector<std::map<unsigned int, NumericT> > stl_C(N);

  for (std::size_t i=0; i<stl_A.size(); ++i)
    for (std::size_t j=0; j<nnz_row; ++j)
      stl_A[i][static_cast<unsigned int>(randomNumber() * NumericT(K))] = NumericT(1.0) + NumericT();

  for (std::size_t i=0; i<stl_B.size(); ++i)
    for (std::size_t j=0; j<nnz_row; ++j)
      stl_B[i][static_cast<unsigned int>(randomNumber() * NumericT(M))] = NumericT(1.0) + NumericT();


  cuarma::compressed_matrix<NumericT>  arma_A(N, K);
  cuarma::compressed_matrix<NumericT>  arma_B(K, M);
  cuarma::compressed_matrix<NumericT>  arma_C;

  cuarma::tools::sparse_matrix_adapter<NumericT> adapted_stl_A(stl_A, N, K);
  cuarma::tools::sparse_matrix_adapter<NumericT> adapted_stl_B(stl_B, K, M);
  cuarma::copy(adapted_stl_A, arma_A);
  cuarma::copy(adapted_stl_B, arma_B);

  // --------------------------------------------------------------------------
  std::cout << "Testing products: STL" << std::endl;
  prod(stl_A, stl_B, stl_C);

  std::cout << "Testing products: compressed_matrix" << std::endl;
  arma_C = cuarma::blas::prod(arma_A, arma_B);

  if ( std::fabs(diff(stl_C, arma_C)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-matrix product with compressed_matrix (arma_C)" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(stl_C, arma_C)) << std::endl;
    retval = EXIT_FAILURE;
  }

  cuarma::compressed_matrix<NumericT> arma_D = cuarma::blas::prod(arma_A, arma_B);
  if ( std::fabs(diff(stl_C, arma_D)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-matrix product with compressed_matrix (arma_D)" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(stl_C, arma_C)) << std::endl;
    retval = EXIT_FAILURE;
  }

  cuarma::compressed_matrix<NumericT> arma_E(cuarma::blas::prod(arma_A, arma_B));
  if ( std::fabs(diff(stl_C, arma_E)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-matrix product with compressed_matrix (arma_E)" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(stl_C, arma_C)) << std::endl;
    retval = EXIT_FAILURE;
  }

  // --------------------------------------------------------------------------
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
  std::cout << "## Test :: Sparse Matrix Product" << std::endl;
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
