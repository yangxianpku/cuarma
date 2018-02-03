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

/** @file cuarma/blas/jacobi_precond.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementation of a simple Jacobi preconditioner
*/

#include <vector>
#include <cmath>
#include "cuarma/forwards.h"
#include "cuarma/vector.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/sparse_matrix_operations.hpp"
#include "cuarma/blas/row_scaling.hpp"

#include <map>

namespace cuarma
{
namespace blas
{

/** @brief A tag for a jacobi preconditioner
*/
class jacobi_tag {};


/** @brief Jacobi preconditioner class, can be supplied to solve()-routines. Generic version for non-cuarma matrices.
*/
template<typename MatrixT,
          bool is_cuarma = detail::row_scaling_for_cuarma<MatrixT>::value >
class jacobi_precond
{
  typedef typename MatrixT::value_type      NumericType;

  public:
    jacobi_precond(MatrixT const & mat, jacobi_tag const &) : diag_A_(cuarma::traits::size1(mat))
    {
      init(mat);
    }

    void init(MatrixT const & mat)
    {
      diag_A_.resize(cuarma::traits::size1(mat));  //resize without preserving values

      for (typename MatrixT::const_iterator1 row_it = mat.begin1();
            row_it != mat.end1();
            ++row_it)
      {
        bool diag_found = false;
        for (typename MatrixT::const_iterator2 col_it = row_it.begin();
              col_it != row_it.end();
              ++col_it)
        {
          if (col_it.index1() == col_it.index2())
          {
            diag_A_[col_it.index1()] = *col_it;
            diag_found = true;
          }
        }
        if (!diag_found)
          throw zero_on_diagonal_exception("cuarma: Zero in diagonal encountered while setting up Jacobi preconditioner!");
      }
    }


    /** @brief Apply to res = b - Ax, i.e. jacobi applied vec (right hand side),  */
    template<typename VectorT>
    void apply(VectorT & vec) const
    {
      assert(cuarma::traits::size(diag_A_) == cuarma::traits::size(vec) && bool("Size mismatch"));
      for (arma_size_t i=0; i<diag_A_.size(); ++i)
        vec[i] /= diag_A_[i];
    }

  private:
    std::vector<NumericType> diag_A_;
};


/** @brief Jacobi preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename MatrixT>
class jacobi_precond<MatrixT, true>
{
    typedef typename cuarma::result_of::cpu_value_type<typename MatrixT::value_type>::type  NumericType;

  public:
    jacobi_precond(MatrixT const & mat, jacobi_tag const &) : diag_A_(mat.size1(), cuarma::traits::context(mat))
    {
      init(mat);
    }


    void init(MatrixT const & mat)
    {
      detail::row_info(mat, diag_A_, detail::SPARSE_ROW_DIAGONAL);
    }


    template<unsigned int AlignmentV>
    void apply(cuarma::vector<NumericType, AlignmentV> & vec) const
    {
      assert(cuarma::traits::size(diag_A_) == cuarma::traits::size(vec) && bool("Size mismatch"));
      vec = element_div(vec, diag_A_);
    }

  private:
    cuarma::vector<NumericType> diag_A_;
};

}
}



