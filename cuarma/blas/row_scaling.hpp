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

/** @file cuarma/blas/row_scaling.hpp
 *  @encoding:UTF-8 文档编码
    @brief A row normalization preconditioner is implemented here
*/

#include <vector>
#include <cmath>
#include "cuarma/forwards.h"
#include "cuarma/vector.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/tools/tools.hpp"

#include <map>

namespace cuarma
{
  namespace blas
  {

    /** @brief A tag for a row scaling preconditioner which merely normalizes the equation system such that each row of the system matrix has unit norm. */
    class row_scaling_tag
    {
      public:
        /** @brief Constructor
        *
        * @param p   Integer selecting the desired row norm.
        */
        row_scaling_tag(unsigned int p = 2) : norm_(p) {}

        /** @brief Returns the index p of the l^p-norm (0 ... ||x||_sup, 1... sum(abs(x)), 2... sqrt(sum(x_i^2))). Currently only p=0, p=1, and p=2 supported.*/
        unsigned int norm() const { return norm_; }

      private:
        unsigned int norm_;
    };


    /** \cond */
    namespace detail
    {
      template<typename T>
      struct row_scaling_for_cuarma
      {
        enum { value = false };
      };

      template<typename ScalarType, unsigned int ALIGNMENT>
      struct row_scaling_for_cuarma< cuarma::compressed_matrix<ScalarType, ALIGNMENT> >
      {
        enum { value = true };
      };

      template<typename ScalarType, unsigned int ALIGNMENT>
      struct row_scaling_for_cuarma< cuarma::coordinate_matrix<ScalarType, ALIGNMENT> >
      {
        enum { value = true };
      };
    }
    /** \endcond */


    /** @brief Jacobi-type preconditioner class, can be supplied to solve()-routines. This is a diagonal preconditioner with the diagonal entries being (configurable) row norms of the matrix.
     *
     *  Default implementation for non-native cuarma matrices (e.g. uBLAS)
     */
    template<typename MatrixType,
              bool is_cuarma = detail::row_scaling_for_cuarma<MatrixType>::value >
    class row_scaling
    {
      typedef typename MatrixType::value_type      ScalarType;

      public:
        /** @brief Constructor for the preconditioner
        *
        * @param mat   The system matrix
        * @param tag   A row scaling tag holding the desired norm.
        */
        row_scaling(MatrixType const & mat, row_scaling_tag const & tag) : diag_M(cuarma::traits::size1(mat))
        {
          assert(mat.size1() == mat.size2() && bool("Size mismatch"));
          init(mat, tag);
        }

        void init(MatrixType const & mat, row_scaling_tag const & tag)
        {
          diag_M.resize(mat.size1());  //resize without preserving values

          for (typename MatrixType::const_iterator1 row_it = mat.begin1();
                row_it != mat.end1();
                ++row_it)
          {
            for (typename MatrixType::const_iterator2 col_it = row_it.begin();
                  col_it != row_it.end();
                  ++col_it)
            {
              if (tag.norm() == 0)
                diag_M[col_it.index1()] = std::max<ScalarType>(diag_M[col_it.index1()], std::fabs(*col_it));
              else if (tag.norm() == 1)
                diag_M[col_it.index1()] += std::fabs(*col_it);
              else if (tag.norm() == 2)
                diag_M[col_it.index1()] += (*col_it) * (*col_it);
            }
            if (!diag_M[row_it.index1()])
              throw zero_on_diagonal_exception("cuarma: Zero row encountered while setting up row scaling preconditioner!");

            if (tag.norm() == 2)
              diag_M[row_it.index1()] = std::sqrt(diag_M[row_it.index1()]);
          }
        }


        /** @brief Apply to res = b - Ax, i.e. row applied vec (right hand side),  */
        template<typename VectorType>
        void apply(VectorType & vec) const
        {
          assert(vec.size() == diag_M.size() && bool("Size mismatch"));
          for (arma_size_t i=0; i<vec.size(); ++i)
            vec[i] /= diag_M[i];
        }

      private:
        std::vector<ScalarType> diag_M;
    };


    /** @brief Jacobi preconditioner class, can be supplied to solve()-routines.
    *
    *  Specialization for compressed_matrix
    */
    template<typename MatrixType>
    class row_scaling< MatrixType, true>
    {
        typedef typename cuarma::result_of::cpu_value_type<typename MatrixType::value_type>::type  ScalarType;


      public:
        /** @brief Constructor for the preconditioner
        *
        * @param mat   The system matrix
        * @param tag   A row scaling tag holding the desired norm.
        */
        row_scaling(MatrixType const & mat, row_scaling_tag const & tag) : diag_M(mat.size1(), cuarma::traits::context(mat))
        {
          init(mat, tag);
        }

        void init(MatrixType const & mat, row_scaling_tag const & tag)
        {
          switch (tag.norm())
          {
            case 0:
              detail::row_info(mat, diag_M, detail::SPARSE_ROW_NORM_INF);
              break;
            case 1:
              detail::row_info(mat, diag_M, detail::SPARSE_ROW_NORM_1);
              break;
            case 2:
              detail::row_info(mat, diag_M, detail::SPARSE_ROW_NORM_2);
              break;
            default:
              throw unknown_norm_exception("Unknown norm when initializing row_scaling preconditioner!");
          }
        }

        template<unsigned int ALIGNMENT>
        void apply(cuarma::vector<ScalarType, ALIGNMENT> & vec) const
        {
          assert(cuarma::traits::size(diag_M) == cuarma::traits::size(vec) && bool("Size mismatch"));
          vec = element_div(vec, diag_M);
        }

      private:
        cuarma::vector<ScalarType> diag_M;
    };

  }
}