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

/** @file cuarma/blas/direct_solve.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of dense direct solvers are found here.
*/

#include "cuarma/forwards.h"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/vector_proxy.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/matrix_proxy.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/host_based/direct_solve.hpp"


#ifdef CUARMA_WITH_CUDA
  #include "cuarma/blas/cuda/direct_solve.hpp"
#endif

#define CUARMA_DIRECT_SOLVE_BLOCKSIZE 128

namespace cuarma
{
namespace blas
{

namespace detail
{

  //
  // A \ B:
  //

  /** @brief Direct inplace solver for dense triangular systems using a single kernel launch. Matlab notation: A \ B
  *
  * @param A    The system matrix
  * @param B    The matrix of row vectors, where the solution is directly written to
  */
  template<typename NumericT, typename SolverTagT>
  void inplace_solve_kernel(const matrix_base<NumericT>  & A, const matrix_base<NumericT> & B, SolverTagT)
  {
    assert( (cuarma::traits::size1(A) == cuarma::traits::size2(A)) && bool("Size check failed in inplace_solve(): size1(A) != size2(A)"));
    assert( (cuarma::traits::size1(A) == cuarma::traits::size1(B)) && bool("Size check failed in inplace_solve(): size1(A) != size1(B)"));
    switch (cuarma::traits::handle(A).get_active_handle_id())
    {
      case cuarma::MAIN_MEMORY:
        cuarma::blas::host_based::inplace_solve(A, const_cast<matrix_base<NumericT> &>(B), SolverTagT());
        break;
  
  #ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::inplace_solve(A, const_cast<matrix_base<NumericT> &>(B), SolverTagT());
        break;
  #endif
      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }


  //
  // A \ b
  //

  template<typename NumericT, typename SolverTagT>
  void inplace_solve_vec_kernel(const matrix_base<NumericT> & mat,
                                const vector_base<NumericT> & vec,
                                SolverTagT)
  {
    assert( (mat.size1() == vec.size()) && bool("Size check failed in inplace_solve(): size1(A) != size(b)"));
    assert( (mat.size2() == vec.size()) && bool("Size check failed in inplace_solve(): size2(A) != size(b)"));

    switch (cuarma::traits::handle(mat).get_active_handle_id())
    {
      case cuarma::MAIN_MEMORY:
        cuarma::blas::host_based::inplace_solve(mat, const_cast<vector_base<NumericT> &>(vec), SolverTagT());
        break;
  
  #ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
        cuarma::blas::cuda::inplace_solve(mat, const_cast<vector_base<NumericT> &>(vec), SolverTagT());
        break;
  #endif
      case cuarma::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }


  template<typename MatrixT1, typename MatrixT2, typename SolverTagT>
  void inplace_solve_lower_impl(MatrixT1 const & A, MatrixT2 & B, SolverTagT)
  {
    typedef typename cuarma::result_of::cpu_value_type<MatrixT1>::type  NumericType;

    arma_size_t blockSize = CUARMA_DIRECT_SOLVE_BLOCKSIZE;
    if (A.size1() <= blockSize)
      inplace_solve_kernel(A, B, SolverTagT());
    else
    {
      for (arma_size_t i = 0; i < A.size1(); i = i + blockSize)
      {
        arma_size_t Apos1 = i;
        arma_size_t Apos2 = std::min<arma_size_t>(A.size1(), i + blockSize);
        arma_size_t Bpos = B.size2();
        inplace_solve_kernel(cuarma::project(A, cuarma::range(Apos1, Apos2), cuarma::range(Apos1, Apos2)),
                             cuarma::project(B, cuarma::range(Apos1, Apos2), cuarma::range(0,     Bpos)),
                             SolverTagT());
        if (Apos2 < A.size1())
        {
          cuarma::matrix_range<MatrixT2> B_lower(B, cuarma::range(Apos2, B.size1()), cuarma::range(0, Bpos));
          cuarma::blas::prod_impl(cuarma::project(A, cuarma::range(Apos2, A.size1()), cuarma::range(Apos1, Apos2)),
                                      cuarma::project(B, cuarma::range(Apos1, Apos2),     cuarma::range(0,     Bpos)),
                                      B_lower,
                                      NumericType(-1.0), NumericType(1.0));
        }
      }
    }
  }

  template<typename MatrixT1, typename MatrixT2>
  void inplace_solve_impl(MatrixT1 const & A, MatrixT2 & B, cuarma::blas::lower_tag)
  {
    inplace_solve_lower_impl(A, B, cuarma::blas::lower_tag());
  }

  template<typename MatrixT1, typename MatrixT2>
  void inplace_solve_impl(MatrixT1 const & A, MatrixT2 & B, cuarma::blas::unit_lower_tag)
  {
    inplace_solve_lower_impl(A, B, cuarma::blas::unit_lower_tag());
  }

  template<typename MatrixT1, typename MatrixT2, typename SolverTagT>
  void inplace_solve_upper_impl(MatrixT1 const & A, MatrixT2 & B, SolverTagT)
  {
    typedef typename cuarma::result_of::cpu_value_type<MatrixT1>::type  NumericType;

    int blockSize = CUARMA_DIRECT_SOLVE_BLOCKSIZE;
    if (static_cast<int>(A.size1()) <= blockSize)
      inplace_solve_kernel(A, B, SolverTagT());
    else
    {
      for (int i = static_cast<int>(A.size1()); i > 0; i = i - blockSize)
      {
        arma_size_t Apos1 = arma_size_t(std::max<int>(0, i - blockSize));
        arma_size_t Apos2 = arma_size_t(i);
        arma_size_t Bpos = B.size2();
        inplace_solve_kernel(cuarma::project(A, cuarma::range(Apos1, Apos2), cuarma::range(Apos1, Apos2)),
                             cuarma::project(B, cuarma::range(Apos1, Apos2), cuarma::range(0, Bpos)),
                             SolverTagT());
        if (Apos1 > 0)
        {
          cuarma::matrix_range<MatrixT2> B_upper(B, cuarma::range(0, Apos1), cuarma::range(0, Bpos));

          cuarma::blas::prod_impl(cuarma::project(A, cuarma::range(0,     Apos1), cuarma::range(Apos1, Apos2)),
                                      cuarma::project(B, cuarma::range(Apos1, Apos2), cuarma::range(0,     Bpos)),
                                      B_upper,
                                      NumericType(-1.0), NumericType(1.0));
        }
      }
    }
  }

  template<typename MatrixT1, typename MatrixT2>
  void inplace_solve_impl(MatrixT1 const & A, MatrixT2 & B, cuarma::blas::upper_tag)
  {
    inplace_solve_upper_impl(A, B, cuarma::blas::upper_tag());
  }

  template<typename MatrixT1, typename MatrixT2>
  void inplace_solve_impl(MatrixT1 const & A, MatrixT2 & B, cuarma::blas::unit_upper_tag)
  {
    inplace_solve_upper_impl(A, B, cuarma::blas::unit_upper_tag());
  }

} // namespace detail

/** @brief Direct inplace solver for triangular systems with multiple right hand sides, i.e. A \ B   (MATLAB notation)
*
* @param A      The system matrix
* @param B      The matrix of row vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(const matrix_base<NumericT> & A,
                   matrix_base<NumericT> & B,
                   SolverTagT)
{
  detail::inplace_solve_impl(A,B,SolverTagT());
}

/** @brief Direct inplace solver for triangular systems with multiple transposed right hand sides, i.e. A \ B^T   (MATLAB notation)
*
* @param A       The system matrix
* @param proxy_B The proxy for the transposed matrix of row vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(const matrix_base<NumericT> & A,
                   matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> proxy_B,
                   SolverTagT)
{
  typedef typename matrix_base<NumericT>::handle_type    handle_type;

  matrix_base<NumericT> B(const_cast<handle_type &>(proxy_B.lhs().handle()),
                          proxy_B.lhs().size2(), proxy_B.lhs().start2(), proxy_B.lhs().stride2(), proxy_B.lhs().internal_size2(),
                          proxy_B.lhs().size1(), proxy_B.lhs().start1(), proxy_B.lhs().stride1(), proxy_B.lhs().internal_size1(),
                          !proxy_B.lhs().row_major());

  detail::inplace_solve_impl(A,B,SolverTagT());
}

//upper triangular solver for transposed lower triangular matrices
/** @brief Direct inplace solver for transposed triangular systems with multiple right hand sides, i.e. A^T \ B   (MATLAB notation)
*
* @param proxy_A  The transposed system matrix proxy
* @param B        The matrix holding the load vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>  & proxy_A,
                   matrix_base<NumericT> & B,
                   SolverTagT)
{
  typedef typename matrix_base<NumericT>::handle_type    handle_type;

  matrix_base<NumericT> A(const_cast<handle_type &>(proxy_A.lhs().handle()),
                          proxy_A.lhs().size2(), proxy_A.lhs().start2(), proxy_A.lhs().stride2(), proxy_A.lhs().internal_size2(),
                          proxy_A.lhs().size1(), proxy_A.lhs().start1(), proxy_A.lhs().stride1(), proxy_A.lhs().internal_size1(),
                          !proxy_A.lhs().row_major());

  detail::inplace_solve_impl(A,B,SolverTagT());
}

/** @brief Direct inplace solver for transposed triangular systems with multiple transposed right hand sides, i.e. A^T \ B^T   (MATLAB notation)
*
* @param proxy_A    The transposed system matrix proxy
* @param proxy_B    The transposed matrix holding the load vectors, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> const & proxy_A,
                   matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>         proxy_B,
                   SolverTagT)
{
  typedef typename matrix_base<NumericT>::handle_type    handle_type;

  matrix_base<NumericT> A(const_cast<handle_type &>(proxy_A.lhs().handle()),
                          proxy_A.lhs().size2(), proxy_A.lhs().start2(), proxy_A.lhs().stride2(), proxy_A.lhs().internal_size2(),
                          proxy_A.lhs().size1(), proxy_A.lhs().start1(), proxy_A.lhs().stride1(), proxy_A.lhs().internal_size1(),
                          !proxy_A.lhs().row_major());

  matrix_base<NumericT> B(const_cast<handle_type &>(proxy_B.lhs().handle()),
                          proxy_B.lhs().size2(), proxy_B.lhs().start2(), proxy_B.lhs().stride2(), proxy_B.lhs().internal_size2(),
                          proxy_B.lhs().size1(), proxy_B.lhs().start1(), proxy_B.lhs().stride1(), proxy_B.lhs().internal_size1(),
                          !proxy_B.lhs().row_major());

  detail::inplace_solve_impl(A,B,SolverTagT());
}


/////////////////// general wrappers for non-inplace solution //////////////////////


/** @brief Convenience functions for C = solve(A, B, some_tag()); Creates a temporary result matrix and forwards the request to inplace_solve()
*
* @param A    The system matrix
* @param B    The matrix of load vectors
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
matrix_base<NumericT> solve(const matrix_base<NumericT> & A,
                            const matrix_base<NumericT> & B,
                            SolverTagT tag)
{
  // do an inplace solve on the result vector:
  matrix_base<NumericT> result(B);
  inplace_solve(A, result, tag);
  return result;
}

/** @brief Convenience functions for C = solve(A, B^T, some_tag()); Creates a temporary result matrix and forwards the request to inplace_solve()
*
* @param A    The system matrix
* @param proxy  The transposed load vector
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
matrix_base<NumericT> solve(const matrix_base<NumericT> & A,
                            const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy,
                            SolverTagT tag)
{
  // do an inplace solve on the result vector:
  matrix_base<NumericT> result(proxy);
  inplace_solve(A, result, tag);
  return result;
}

/** @brief Convenience functions for result = solve(trans(mat), B, some_tag()); Creates a temporary result matrix and forwards the request to inplace_solve()
*
* @param proxy  The transposed system matrix proxy
* @param B      The matrix of load vectors
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
matrix_base<NumericT> solve(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy,
                            const matrix_base<NumericT> & B,
                            SolverTagT tag)
{
  // do an inplace solve on the result vector:
  matrix_base<NumericT> result(B);
  inplace_solve(proxy, result, tag);
  return result;
}

/** @brief Convenience functions for result = solve(trans(mat), vec, some_tag()); Creates a temporary result vector and forwards the request to inplace_solve()
*
* @param proxy_A  The transposed system matrix proxy
* @param proxy_B  The transposed matrix of load vectors, where the solution is directly written to
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
matrix_base<NumericT> solve(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy_A,
                            const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy_B,
                            SolverTagT tag)
{
  // run an inplace solve on the result vector:
  matrix_base<NumericT> result(proxy_B);
  inplace_solve(proxy_A, result, tag);
  return result;
}

//
/////////// solves with vector as right hand side ///////////////////
//

namespace detail
{
  template<typename MatrixT1, typename VectorT, typename SolverTagT>
  void inplace_solve_lower_vec_impl(MatrixT1 const & A, VectorT & b, SolverTagT)
  {
    arma_size_t blockSize = CUARMA_DIRECT_SOLVE_BLOCKSIZE;
    if (A.size1() <= blockSize)
      inplace_solve_vec_kernel(A, b, SolverTagT());
    else
    {
      VectorT temp(b);
      for (arma_size_t i = 0; i < A.size1(); i = i + blockSize)
      {
        arma_size_t Apos1 = i;
        arma_size_t Apos2 = std::min<arma_size_t>(A.size1(), i + blockSize);
        inplace_solve_vec_kernel(cuarma::project(A, cuarma::range(Apos1, Apos2), cuarma::range(Apos1, Apos2)),
                                 cuarma::project(b, cuarma::range(Apos1, Apos2)),
                                 SolverTagT());
        if (Apos2 < A.size1())
        {
          cuarma::project(temp, cuarma::range(Apos2, A.size1())) = cuarma::blas::prod(cuarma::project(A, cuarma::range(Apos2, A.size1()), cuarma::range(Apos1, Apos2)),
                                                                                              cuarma::project(b, cuarma::range(Apos1, Apos2)));
          cuarma::project(b, cuarma::range(Apos2, A.size1())) -= cuarma::project(temp, cuarma::range(Apos2, A.size1()));
        }
      }
    }
  }

  template<typename MatrixT1, typename VectorT>
  void inplace_solve_vec_impl(MatrixT1 const & A, VectorT & B, cuarma::blas::lower_tag)
  {
    inplace_solve_lower_vec_impl(A, B, cuarma::blas::lower_tag());
  }

  template<typename MatrixT1, typename VectorT>
  void inplace_solve_vec_impl(MatrixT1 const & A, VectorT & B, cuarma::blas::unit_lower_tag)
  {
    inplace_solve_lower_vec_impl(A, B, cuarma::blas::unit_lower_tag());
  }

  template<typename MatrixT1, typename VectorT, typename SolverTagT>
  void inplace_solve_upper_vec_impl(MatrixT1 const & A, VectorT & b, SolverTagT)
  {
    int blockSize = CUARMA_DIRECT_SOLVE_BLOCKSIZE;
    if (static_cast<int>(A.size1()) <= blockSize)
      inplace_solve_vec_kernel(A, b, SolverTagT());
    else
    {
      VectorT temp(b);
      for (int i = static_cast<int>(A.size1()); i > 0; i = i - blockSize)
      {
        arma_size_t Apos1 = arma_size_t(std::max<int>(0, i - blockSize));
        arma_size_t Apos2 = arma_size_t(i);
        inplace_solve_vec_kernel(cuarma::project(A, cuarma::range(Apos1, Apos2), cuarma::range(Apos1, Apos2)),
                                 cuarma::project(b, cuarma::range(Apos1, Apos2)),
                                 SolverTagT());
        if (Apos1 > 0)
        {
          cuarma::project(temp, cuarma::range(0, Apos1)) = cuarma::blas::prod(cuarma::project(A, cuarma::range(0,     Apos1), cuarma::range(Apos1, Apos2)),
                                                                                      cuarma::project(b, cuarma::range(Apos1, Apos2)));
          cuarma::project(b, cuarma::range(0, Apos1)) -= cuarma::project(temp, cuarma::range(0, Apos1));
        }
      }
    }
  }

  template<typename MatrixT1, typename VectorT>
  void inplace_solve_vec_impl(MatrixT1 const & A, VectorT & b, cuarma::blas::upper_tag)
  {
    inplace_solve_upper_vec_impl(A, b, cuarma::blas::upper_tag());
  }

  template<typename MatrixT1, typename VectorT>
  void inplace_solve_vec_impl(MatrixT1 const & A, VectorT & b, cuarma::blas::unit_upper_tag)
  {
    inplace_solve_upper_vec_impl(A, b, cuarma::blas::unit_upper_tag());
  }

} // namespace detail

/** @brief Inplace solution of a triangular system. Matlab notation A \ b.
*
* @param mat    The system matrix (a dense matrix for which only the respective triangular form is used)
* @param vec    The right hand side vector
* @param tag    The tag (either lower_tag, unit_lower_tag, upper_tag, or unit_upper_tag)
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(const matrix_base<NumericT> & mat,
                   vector_base<NumericT> & vec,
                   SolverTagT const & tag)
{

  detail::inplace_solve_vec_impl(mat, vec, tag);
}

/** @brief Inplace solution of a triangular system with transposed system matrix.. Matlab notation A' \ b.
*
* @param proxy  The transposed system matrix (a dense matrix for which only the respective triangular form is used)
* @param vec    The right hand side vector
* @param tag    The tag (either lower_tag, unit_lower_tag, upper_tag, or unit_upper_tag)
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> const & proxy,
                   vector_base<NumericT> & vec,
                   SolverTagT const & tag)
{
  typedef typename matrix_base<NumericT>::handle_type    handle_type;

  // wrap existing matrix in a new matrix_base object (no data copy)
  matrix_base<NumericT> mat(const_cast<handle_type &>(proxy.lhs().handle()),
                            proxy.lhs().size2(), proxy.lhs().start2(), proxy.lhs().stride2(), proxy.lhs().internal_size2(),
                            proxy.lhs().size1(), proxy.lhs().start1(), proxy.lhs().stride1(), proxy.lhs().internal_size1(),
                            !proxy.lhs().row_major());
  detail::inplace_solve_vec_impl(mat, vec, tag);
}


/** @brief Convenience function for result = solve(mat, vec, upper_tag()); for an upper triangular solve.
*
* Creates a temporary result vector and forwards the request to inplace_solve()
*
* @param mat    The system matrix
* @param vec    The load vector
* @param tag    Dispatch tag
*/
template<typename NumericT>
vector<NumericT> solve(const matrix_base<NumericT> & mat,
                       const vector_base<NumericT> & vec,
                       cuarma::blas::upper_tag const & tag)
{
  // run an inplace solve on the result vector:
  vector<NumericT> result(vec);
  inplace_solve(mat, result, tag);
  return result;
}

/** @brief Convenience function for result = solve(mat, vec, upper_tag()); for an upper triangular solve with unit diagonal.
*
* Creates a temporary result vector and forwards the request to inplace_solve()
*
* @param mat    The system matrix
* @param vec    The load vector
* @param tag    Dispatch tag
*/
template<typename NumericT>
vector<NumericT> solve(const matrix_base<NumericT> & mat,
                       const vector_base<NumericT> & vec,
                       cuarma::blas::unit_upper_tag const & tag)
{
  // run an inplace solve on the result vector:
  vector<NumericT> result(vec);
  inplace_solve(mat, result, tag);
  return result;
}

/** @brief Convenience function for result = solve(mat, vec, upper_tag()); for a lower triangular solve.
*
* Creates a temporary result vector and forwards the request to inplace_solve()
*
* @param mat    The system matrix
* @param vec    The load vector
* @param tag    Dispatch tag
*/
template<typename NumericT>
vector<NumericT> solve(const matrix_base<NumericT> & mat,
                       const vector_base<NumericT> & vec,
                       cuarma::blas::lower_tag const & tag)
{
  // run an inplace solve on the result vector:
  vector<NumericT> result(vec);
  inplace_solve(mat, result, tag);
  return result;
}

/** @brief Convenience function for result = solve(mat, vec, upper_tag()); for a lower triangular solve with unit diagonal.
*
* Creates a temporary result vector and forwards the request to inplace_solve()
*
* @param mat    The system matrix
* @param vec    The load vector
* @param tag    Dispatch tag
*/
template<typename NumericT>
vector<NumericT> solve(const matrix_base<NumericT> & mat,
                       const vector_base<NumericT> & vec,
                       cuarma::blas::unit_lower_tag const & tag)
{
  // run an inplace solve on the result vector:
  vector<NumericT> result(vec);
  inplace_solve(mat, result, tag);
  return result;
}

/** @brief Convenience functions for result = solve(trans(mat), vec, some_tag()); Creates a temporary result vector and forwards the request to inplace_solve()
*
* @param proxy  The transposed system matrix proxy
* @param vec    The load vector, where the solution is directly written to
* @param tag    Dispatch tag
*/
template<typename NumericT, typename SolverTagT>
vector<NumericT> solve(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & proxy,
                       const vector_base<NumericT> & vec,
                       SolverTagT const & tag)
{
  // run an inplace solve on the result vector:
  vector<NumericT> result(vec);
  inplace_solve(proxy, result, tag);
  return result;
}


}
}
