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

/** @file cuarma/blas/host_based/direct_solve.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of dense direct triangular solvers are found here.
*/

#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"

#include "cuarma/blas/host_based/common.hpp"

namespace cuarma
{
namespace blas
{
namespace host_based
{

namespace detail
{
  //
  // Upper solve:
  //
  template<typename MatrixT1, typename MatrixT2>
  void upper_inplace_solve_matrix(MatrixT1 & A, MatrixT2 & B, arma_size_t A_size, arma_size_t B_size, bool unit_diagonal)
  {
    typedef typename MatrixT2::value_type   value_type;

    for (arma_size_t i = 0; i < A_size; ++i)
    {
      arma_size_t current_row = A_size - i - 1;

      for (arma_size_t j = current_row + 1; j < A_size; ++j)
      {
        value_type A_element = A(current_row, j);
        for (arma_size_t k=0; k < B_size; ++k)
          B(current_row, k) -= A_element * B(j, k);
      }

      if (!unit_diagonal)
      {
        value_type A_diag = A(current_row, current_row);
        for (arma_size_t k=0; k < B_size; ++k)
          B(current_row, k) /= A_diag;
      }
    }
  }

  template<typename MatrixT1, typename MatrixT2>
  void inplace_solve_matrix(MatrixT1 & A, MatrixT2 & B, arma_size_t A_size, arma_size_t B_size, cuarma::blas::unit_upper_tag)
  {
    upper_inplace_solve_matrix(A, B, A_size, B_size, true);
  }

  template<typename MatrixT1, typename MatrixT2>
  void inplace_solve_matrix(MatrixT1 & A, MatrixT2 & B, arma_size_t A_size, arma_size_t B_size, cuarma::blas::upper_tag)
  {
    upper_inplace_solve_matrix(A, B, A_size, B_size, false);
  }

  //
  // Lower solve:
  //
  template<typename MatrixT1, typename MatrixT2>
  void lower_inplace_solve_matrix(MatrixT1 & A, MatrixT2 & B, arma_size_t A_size, arma_size_t B_size, bool unit_diagonal)
  {
    typedef typename MatrixT2::value_type   value_type;

    for (arma_size_t i = 0; i < A_size; ++i)
    {
      for (arma_size_t j = 0; j < i; ++j)
      {
        value_type A_element = A(i, j);
        for (arma_size_t k=0; k < B_size; ++k)
          B(i, k) -= A_element * B(j, k);
      }

      if (!unit_diagonal)
      {
        value_type A_diag = A(i, i);
        for (arma_size_t k=0; k < B_size; ++k)
          B(i, k) /= A_diag;
      }
    }
  }

  template<typename MatrixT1, typename MatrixT2>
  void inplace_solve_matrix(MatrixT1 & A, MatrixT2 & B, arma_size_t A_size, arma_size_t B_size, cuarma::blas::unit_lower_tag)
  {
    lower_inplace_solve_matrix(A, B, A_size, B_size, true);
  }

  template<typename MatrixT1, typename MatrixT2>
  void inplace_solve_matrix(MatrixT1 & A, MatrixT2 & B, arma_size_t A_size, arma_size_t B_size, cuarma::blas::lower_tag)
  {
    lower_inplace_solve_matrix(A, B, A_size, B_size, false);
  }

}

//
// Note: By convention, all size checks are performed in the calling frontend. No need to double-check here.
//

////////////////// upper triangular solver (upper_tag) //////////////////////////////////////
/** @brief Direct inplace solver for triangular systems with multiple right hand sides, i.e. A \ B   (MATLAB notation)
*
* @param A        The system matrix
* @param B        The matrix of row vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(matrix_base<NumericT> const & A,
                   matrix_base<NumericT> & B,
                   SolverTagT)
{
  typedef NumericT        value_type;

  value_type const * data_A = detail::extract_raw_pointer<value_type>(A);
  value_type       * data_B = detail::extract_raw_pointer<value_type>(B);

  arma_size_t A_start1 = cuarma::traits::start1(A);
  arma_size_t A_start2 = cuarma::traits::start2(A);
  arma_size_t A_inc1   = cuarma::traits::stride1(A);
  arma_size_t A_inc2   = cuarma::traits::stride2(A);
  //arma_size_t A_size1  = cuarma::traits::size1(A);
  arma_size_t A_size2  = cuarma::traits::size2(A);
  arma_size_t A_internal_size1  = cuarma::traits::internal_size1(A);
  arma_size_t A_internal_size2  = cuarma::traits::internal_size2(A);

  arma_size_t B_start1 = cuarma::traits::start1(B);
  arma_size_t B_start2 = cuarma::traits::start2(B);
  arma_size_t B_inc1   = cuarma::traits::stride1(B);
  arma_size_t B_inc2   = cuarma::traits::stride2(B);
  //arma_size_t B_size1  = cuarma::traits::size1(B);
  arma_size_t B_size2  = cuarma::traits::size2(B);
  arma_size_t B_internal_size1  = cuarma::traits::internal_size1(B);
  arma_size_t B_internal_size2  = cuarma::traits::internal_size2(B);


  if (A.row_major() && B.row_major())
  {
    detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

    detail::inplace_solve_matrix(wrapper_A, wrapper_B, A_size2, B_size2, SolverTagT());
  }
  else if (A.row_major() && !B.row_major())
  {
    detail::matrix_array_wrapper<value_type const, row_major,    false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

    detail::inplace_solve_matrix(wrapper_A, wrapper_B, A_size2, B_size2, SolverTagT());
  }
  else if (!A.row_major() && B.row_major())
  {
    detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type,       row_major,    false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

    detail::inplace_solve_matrix(wrapper_A, wrapper_B, A_size2, B_size2, SolverTagT());
  }
  else
  {
    detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

    detail::inplace_solve_matrix(wrapper_A, wrapper_B, A_size2, B_size2, SolverTagT());
  }
}


//
//  Solve on vector
//

namespace detail
{
  //
  // Upper solve:
  //
  template<typename MatrixT, typename VectorT>
  void upper_inplace_solve_vector(MatrixT & A, VectorT & b, arma_size_t A_size, bool unit_diagonal)
  {
    typedef typename VectorT::value_type   value_type;

    for (arma_size_t i = 0; i < A_size; ++i)
    {
      arma_size_t current_row = A_size - i - 1;

      for (arma_size_t j = current_row + 1; j < A_size; ++j)
      {
        value_type A_element = A(current_row, j);
        b(current_row) -= A_element * b(j);
      }

      if (!unit_diagonal)
        b(current_row) /= A(current_row, current_row);
    }
  }

  template<typename MatrixT, typename VectorT>
  void inplace_solve_vector(MatrixT & A, VectorT & b, arma_size_t A_size, cuarma::blas::unit_upper_tag)
  {
    upper_inplace_solve_vector(A, b, A_size, true);
  }

  template<typename MatrixT, typename VectorT>
  void inplace_solve_vector(MatrixT & A, VectorT & b, arma_size_t A_size, cuarma::blas::upper_tag)
  {
    upper_inplace_solve_vector(A, b, A_size, false);
  }

  //
  // Lower solve:
  //
  template<typename MatrixT, typename VectorT>
  void lower_inplace_solve_vector(MatrixT & A, VectorT & b, arma_size_t A_size, bool unit_diagonal)
  {
    typedef typename VectorT::value_type   value_type;

    for (arma_size_t i = 0; i < A_size; ++i)
    {
      for (arma_size_t j = 0; j < i; ++j)
      {
        value_type A_element = A(i, j);
        b(i) -= A_element * b(j);
      }

      if (!unit_diagonal)
        b(i) /= A(i, i);
    }
  }

  template<typename MatrixT, typename VectorT>
  void inplace_solve_vector(MatrixT & A, VectorT & b, arma_size_t A_size, cuarma::blas::unit_lower_tag)
  {
    lower_inplace_solve_vector(A, b, A_size, true);
  }

  template<typename MatrixT, typename VectorT>
  void inplace_solve_vector(MatrixT & A, VectorT & b, arma_size_t A_size, cuarma::blas::lower_tag)
  {
    lower_inplace_solve_vector(A, b, A_size, false);
  }

}

template<typename NumericT, typename SolverTagT>
void inplace_solve(matrix_base<NumericT> const & mat,
                   vector_base<NumericT> & vec,
                   SolverTagT)
{
  typedef NumericT        value_type;

  value_type const * data_A = detail::extract_raw_pointer<value_type>(mat);
  value_type       * data_v = detail::extract_raw_pointer<value_type>(vec);

  arma_size_t A_start1 = cuarma::traits::start1(mat);
  arma_size_t A_start2 = cuarma::traits::start2(mat);
  arma_size_t A_inc1   = cuarma::traits::stride1(mat);
  arma_size_t A_inc2   = cuarma::traits::stride2(mat);
  arma_size_t A_size2  = cuarma::traits::size2(mat);
  arma_size_t A_internal_size1  = cuarma::traits::internal_size1(mat);
  arma_size_t A_internal_size2  = cuarma::traits::internal_size2(mat);

  arma_size_t start1 = cuarma::traits::start(vec);
  arma_size_t inc1   = cuarma::traits::stride(vec);

  if (mat.row_major())
  {
    detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::vector_array_wrapper<value_type> wrapper_v(data_v, start1, inc1);

    detail::inplace_solve_vector(wrapper_A, wrapper_v, A_size2, SolverTagT());
  }
  else
  {
    detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::vector_array_wrapper<value_type> wrapper_v(data_v, start1, inc1);

    detail::inplace_solve_vector(wrapper_A, wrapper_v, A_size2, SolverTagT());
  }
}


} // namespace host_based
} // namespace blas
} // namespace cuarma