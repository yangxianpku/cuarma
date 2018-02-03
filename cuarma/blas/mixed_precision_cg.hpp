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

/** @file cuarma/blas/mixed_precision_cg.hpp
    @brief The conjugate gradient method using mixed precision is implemented here. Experimental.
*/

#include <vector>
#include <map>
#include <cmath>
#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/ilu.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/inner_prod.hpp"
#include "cuarma/traits/clear.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/meta/result_of.hpp"
#include "cuarma/backend/memory.hpp"

#include "cuarma/vector_proxy.hpp"

namespace cuarma
{
  namespace blas
  {

    /** @brief A tag for the conjugate gradient Used for supplying solver parameters and for dispatching the solve() function
    */
    class mixed_precision_cg_tag
    {
      public:
        /** @brief The constructor
        *
        * @param tol              Relative tolerance for the residual (solver quits if ||r|| < tol * ||r_initial||)
        * @param max_iterations   The maximum number of iterations
        * @param inner_tol        Inner tolerance for the low-precision iterations
        */
        mixed_precision_cg_tag(double tol = 1e-8, unsigned int max_iterations = 300, float inner_tol = 1e-2f) : tol_(tol), iterations_(max_iterations), inner_tol_(inner_tol) {}

        /** @brief Returns the relative tolerance */
        double tolerance() const { return tol_; }
        /** @brief Returns the relative tolerance */
        float inner_tolerance() const { return inner_tol_; }
        /** @brief Returns the maximum number of iterations */
        unsigned int max_iterations() const { return iterations_; }

        /** @brief Return the number of solver iterations: */
        unsigned int iters() const { return iters_taken_; }
        void iters(unsigned int i) const { iters_taken_ = i; }

        /** @brief Returns the estimated relative error at the end of the solver run */
        double error() const { return last_error_; }
        /** @brief Sets the estimated relative error at the end of the solver run */
        void error(double e) const { last_error_ = e; }


      private:
        double tol_;
        unsigned int iterations_;
        float inner_tol_;

        //return values from solver
        mutable unsigned int iters_taken_;
        mutable double last_error_;
    };


    /** @brief Implementation of the conjugate gradient solver without preconditioner
    *
    * Following the algorithm in the book by Y. Saad "Iterative Methods for sparse linear systems"
    *
    * @param matrix     The system matrix
    * @param rhs        The load vector
    * @param tag        Solver configuration tag
    * @return The result vector
    */
    template<typename MatrixType, typename VectorType>
    VectorType solve(const MatrixType & matrix, VectorType const & rhs, mixed_precision_cg_tag const & tag)
    {
      //typedef typename VectorType::value_type      ScalarType;
      typedef typename cuarma::result_of::cpu_value_type<VectorType>::type    CPU_ScalarType;

      //std::cout << "Starting CG" << std::endl;
      arma_size_t problem_size = cuarma::traits::size(rhs);
      VectorType result(rhs);
      cuarma::traits::clear(result);

      VectorType residual = rhs;

      CPU_ScalarType ip_rr = cuarma::blas::inner_prod(rhs, rhs);
      CPU_ScalarType new_ip_rr = 0;
      CPU_ScalarType norm_rhs_squared = ip_rr;

      if (norm_rhs_squared <= 0) //solution is zero if RHS norm is zero
        return result;

      cuarma::vector<float> residual_low_precision(problem_size, cuarma::traits::context(rhs));
      cuarma::vector<float> result_low_precision(problem_size, cuarma::traits::context(rhs));
      cuarma::vector<float> p_low_precision(problem_size, cuarma::traits::context(rhs));
      cuarma::vector<float> tmp_low_precision(problem_size, cuarma::traits::context(rhs));
      
      float inner_ip_rr = static_cast<float>(ip_rr);
      float new_inner_ip_rr = 0;
      float initial_inner_rhs_norm_squared = static_cast<float>(ip_rr);
      float alpha;
      float beta;

      // transfer rhs to single precision:
      p_low_precision = rhs;
      residual_low_precision = p_low_precision;

      // transfer matrix to single precision:
      cuarma::compressed_matrix<float> matrix_low_precision(matrix.size1(), matrix.size2(), matrix.nnz(), cuarma::traits::context(rhs));
      cuarma::backend::memory_copy(matrix.handle1(), const_cast<cuarma::backend::mem_handle &>(matrix_low_precision.handle1()), 0, 0, matrix_low_precision.handle1().raw_size() );
      cuarma::backend::memory_copy(matrix.handle2(), const_cast<cuarma::backend::mem_handle &>(matrix_low_precision.handle2()), 0, 0, matrix_low_precision.handle2().raw_size() );

      cuarma::vector_base<CPU_ScalarType> matrix_elements_high_precision(const_cast<cuarma::backend::mem_handle &>(matrix.handle()), matrix.nnz(), 0, 1);
      cuarma::vector_base<float>          matrix_elements_low_precision(matrix_low_precision.handle(), matrix.nnz(), 0, 1);
      matrix_elements_low_precision = matrix_elements_high_precision;
      matrix_low_precision.generate_row_block_information();

      for (unsigned int i = 0; i < tag.max_iterations(); ++i)
      {
        tag.iters(i+1);

        // lower precision 'inner iteration'
        tmp_low_precision = cuarma::blas::prod(matrix_low_precision, p_low_precision);

        alpha = inner_ip_rr / cuarma::blas::inner_prod(tmp_low_precision, p_low_precision);
        result_low_precision += alpha * p_low_precision;
        residual_low_precision -= alpha * tmp_low_precision;

        new_inner_ip_rr = cuarma::blas::inner_prod(residual_low_precision, residual_low_precision);

        beta = new_inner_ip_rr / inner_ip_rr;
        inner_ip_rr = new_inner_ip_rr;

        p_low_precision = residual_low_precision + beta * p_low_precision;

        //
        // If enough progress has been achieved, update current residual with high precision evaluation
        // This is effectively a restart of the CG method
        //
        if (new_inner_ip_rr < tag.inner_tolerance() * initial_inner_rhs_norm_squared || i == tag.max_iterations()-1)
        {
          residual = result_low_precision; // reusing residual vector as temporary buffer for conversion. Overwritten below anyway
          result += residual;

          // residual = b - Ax  (without introducing a temporary)
          residual = cuarma::blas::prod(matrix, result);
          residual = rhs - residual;

          new_ip_rr = cuarma::blas::inner_prod(residual, residual);
          if (new_ip_rr / norm_rhs_squared < tag.tolerance() *  tag.tolerance())//squared norms involved here
            break;

          p_low_precision = residual;

          result_low_precision.clear();
          residual_low_precision = p_low_precision;
          initial_inner_rhs_norm_squared = static_cast<float>(new_ip_rr);
          inner_ip_rr = static_cast<float>(new_ip_rr);
        }
      }

      //store last error estimate:
      tag.error(std::sqrt(new_ip_rr / norm_rhs_squared));

      return result;
    }

    template<typename MatrixType, typename VectorType>
    VectorType solve(const MatrixType & matrix, VectorType const & rhs, mixed_precision_cg_tag const & tag, cuarma::blas::no_precond)
    {
      return solve(matrix, rhs, tag);
    }


  }
}
