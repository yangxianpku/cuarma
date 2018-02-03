/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   tql.cu
 *   @coding UTF-8
 *   @brief  Tests the tql algorithm for eigenvalue computations for symmetric tridiagonal matrices.
 *   @brief  ≤‚ ‘£∫tqlÀ„∑®
 */

#include <iostream>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/blas/tql2.hpp"

#define EPS 10.0e-5


typedef float     ScalarType;

// Test the eigenvectors
// Perform the multiplication (T - lambda * I) * Q, with the original tridiagonal matrx T, the
// eigenvalues lambda and the eigenvectors in Q. Result has to be 0.

template <typename MatrixLayout>
bool test_eigen_val_vec(cuarma::matrix<ScalarType, MatrixLayout> & Q,
                      std::vector<ScalarType> & eigenvalues,
                      std::vector<ScalarType> & d,
                      std::vector<ScalarType> & e)
{
  std::size_t Q_size = Q.size2();
  ScalarType value = 0;

  for(std::size_t j = 0; j < Q_size; j++)
  {
    // calculate first row
    value = (d[0]- eigenvalues[j]) * Q(0, j) + e[1] * Q(1, j);
    if (value > EPS)
      return false;

    // calcuate inner rows
    for(std::size_t i = 1; i < Q_size - 1; i++)
    {
      value = e[i] * Q(i - 1, j) + (d[i]- eigenvalues[j]) * Q(i, j) + e[i + 1] * Q(i + 1, j);
      if (value > EPS)
        return false;
    }

    // calculate last row
    value = e[Q_size - 1] * Q(Q_size - 2, j) + (d[Q_size - 1] - eigenvalues[j]) * Q(Q_size - 1, j);
    if (value > EPS)
      return false;
  }
  return true;
}


/**
 * Test the tql2 algorithm for symmetric tridiagonal matrices.
 */

template <typename MatrixLayout>
void test_qr_method_sym()
{
  std::size_t sz = 220;

  cuarma::matrix<ScalarType, MatrixLayout> Q = cuarma::identity_matrix<ScalarType>(sz);
  std::vector<ScalarType> d(sz), e(sz), d_ref(sz), e_ref(sz);

  std::cout << "Testing matrix of size " << sz << "-by-" << sz << std::endl << std::endl;

  // Initialize diagonal and superdiagonal elements
  for(unsigned int i = 0; i < sz; ++i)
  {
    d[i] = ((float)(i % 9)) - 4.5f;
    e[i] = ((float)(i % 5)) - 4.5f;
  }
  e[0] = 0.0f;
  d_ref = d;
  e_ref = e;

//---Run the tql2 algorithm-----------------------------------
  cuarma::blas::tql2(Q, d, e);


// ---Test the computed eigenvalues and eigenvectors
  if(!test_eigen_val_vec<MatrixLayout>(Q, d, d_ref, e_ref))
     exit(EXIT_FAILURE);
/*
  for( unsigned int i = 0; i < sz; ++i)
    std::cout << "Eigenvalue " << i << "= " << d[i] << std::endl;
    */
}

int main()
{

  std::cout << std::endl << "COMPUTATION OF EIGENVALUES AND EIGENVECTORS" << std::endl;
  std::cout << std::endl << "Testing QL algorithm for symmetric tridiagonal row-major matrices..." << std::endl;
  test_qr_method_sym<cuarma::row_major>();

  std::cout << std::endl << "Testing QL algorithm for symmetric tridiagonal column-major matrices..." << std::endl;
  test_qr_method_sym<cuarma::column_major>();

  std::cout << std::endl <<"--------TEST SUCCESSFULLY COMPLETED----------" << std::endl;
}
