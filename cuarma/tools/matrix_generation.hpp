#pragma once

/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/** @file cuarma/tools/matrix_generation.hpp
*   @encoding:UTF-8 文档编码
    @brief Helper routines for generating sparse matrices
*/

#include <string>
#include <fstream>
#include <sstream>
#include "cuarma/forwards.h"
#include "cuarma/meta/result_of.hpp"
#include "cuarma/tools/adapter.hpp"

#include <vector>
#include <map>

namespace cuarma
{
namespace tools
{

/** @brief Generates a sparse matrix obtained from a simple finite-difference discretization of the Laplace equation on the unit square (2d).
  *
  * @tparam MatrixType  An uBLAS-compatible matrix type supporting .clear(), .resize(), and operator()-access
  * @param A            A sparse matrix object from cuarma, total number of unknowns will be points_x*points_y
  * @param points_x     Number of points in x-direction
  * @param points_y     Number of points in y-direction
  */
template<typename MatrixType>
void generate_fdm_laplace(MatrixType & A, arma_size_t points_x, arma_size_t points_y)
{
  arma_size_t total_unknowns = points_x * points_y;

  A.clear();
  A.resize(total_unknowns, total_unknowns, false);

  for (arma_size_t i=0; i<points_x; ++i)
  {
    for (arma_size_t j=0; j<points_y; ++j)
    {
      arma_size_t row = i + j * points_x;

      A(row, row) = 4.0;

      if (i > 0)
      {
        arma_size_t col = (i-1) + j * points_x;
        A(row, col) = -1.0;
      }

      if (j > 0)
      {
        arma_size_t col = i + (j-1) * points_x;
        A(row, col) = -1.0;
      }

      if (i < points_x-1)
      {
        arma_size_t col = (i+1) + j * points_x;
        A(row, col) = -1.0;
      }

      if (j < points_y-1)
      {
        arma_size_t col = i + (j+1) * points_x;
        A(row, col) = -1.0;
      }
    }
  }

}

template<typename NumericT>
void generate_fdm_laplace(cuarma::compressed_matrix<NumericT> & A, arma_size_t points_x, arma_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  cuarma::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  cuarma::copy(temp_A, A);
}

template<typename NumericT>
void generate_fdm_laplace(cuarma::coordinate_matrix<NumericT> & A, arma_size_t points_x, arma_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  cuarma::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  cuarma::copy(temp_A, A);
}

template<typename NumericT>
void generate_fdm_laplace(cuarma::ell_matrix<NumericT> & A, arma_size_t points_x, arma_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  cuarma::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  cuarma::copy(temp_A, A);
}

template<typename NumericT>
void generate_fdm_laplace(cuarma::sliced_ell_matrix<NumericT> & A, arma_size_t points_x, arma_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  cuarma::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  cuarma::copy(temp_A, A);
}

template<typename NumericT>
void generate_fdm_laplace(cuarma::hyb_matrix<NumericT> & A, arma_size_t points_x, arma_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  cuarma::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  cuarma::copy(temp_A, A);
}


} //namespace tools
} //namespace cuarma