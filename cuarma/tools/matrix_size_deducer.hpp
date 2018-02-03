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

/** @file cuarma/tools/matrix_size_deducer.hpp
*   @encoding:UTF-8 文档编码
    @brief Helper implementations that deduce the dimensions of the supplied matrix-valued expressions.
*/

#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <map>

#include "cuarma/forwards.h"
#include "cuarma/tools/adapter.hpp"

namespace cuarma
{
namespace tools
{

/** @brief Deduces the size of the resulting vector represented by a vector_expression from the operands
*
* @tparam LHS   The left hand side operand
* @tparam RHS   The right hand side operand
* @tparam OP    The operation tag
*/
template<typename LHS, typename RHS, typename OP>
struct MATRIX_SIZE_DEDUCER
{
  //Standard case: size1 from lhs, size2 from rhs (fits most cases)
  static arma_size_t size1(LHS & lhs, RHS & /*rhs*/) { return lhs.size1(); }
  static arma_size_t size2(LHS & /*lhs*/, RHS & rhs) { return rhs.size2(); }
};

/** \cond */
//special case: outer vector product:
template<typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const cuarma::vector_base<ScalarType>,
    const cuarma::vector_base<ScalarType>,
    cuarma::op_prod>
{
  static arma_size_t size1(cuarma::vector_base<ScalarType> const & lhs,
                          cuarma::vector_base<ScalarType> const & /*rhs*/) { return lhs.size(); }

  static arma_size_t size2(cuarma::vector_base<ScalarType> const & /*lhs*/,
                          cuarma::vector_base<ScalarType> const & rhs) { return rhs.size(); }
};


//special case: multiplication with a scalar
template<typename LHS, typename RHS, typename OP, typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const cuarma::matrix_expression<const LHS, const RHS, OP>,
    const ScalarType,
    cuarma::op_mult>
{
  static arma_size_t size1(cuarma::matrix_expression<const LHS, const RHS, OP> const & lhs,
                          ScalarType const & /*rhs*/) { return MATRIX_SIZE_DEDUCER<const LHS, const RHS, OP>::size1(lhs.lhs(), lhs.rhs()); }

  static arma_size_t size2(cuarma::matrix_expression<const LHS, const RHS, OP> const & lhs,
                          ScalarType const & /*rhs*/) { return MATRIX_SIZE_DEDUCER<const LHS, const RHS, OP>::size2(lhs.lhs(), lhs.rhs()); }
};

//special case: multiplication with a scalar
template<typename T, typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const cuarma::matrix_base<T>,
    const ScalarType,
    cuarma::op_mult>
{
  static arma_size_t size1(cuarma::matrix_base<T> const & lhs,
                          ScalarType const & /*rhs*/) { return lhs.size1(); }

  static arma_size_t size2(cuarma::matrix_base<T> const & lhs,
                          ScalarType const & /*rhs*/) { return lhs.size2(); }
};


//special case: division with a scalar
template<typename LHS, typename RHS, typename OP, typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const cuarma::matrix_expression<const LHS, const RHS, OP>,
    const ScalarType,
    cuarma::op_div>
{
  static arma_size_t size1(cuarma::matrix_expression<const LHS, const RHS, OP> const & lhs,
                          ScalarType const & /*rhs*/) { return MATRIX_SIZE_DEDUCER<const LHS, const RHS, OP>::size1(lhs.lhs(), lhs.rhs()); }

  static arma_size_t size2(cuarma::matrix_expression<const LHS, const RHS, OP> const & lhs,
                          ScalarType const & /*rhs*/) { return MATRIX_SIZE_DEDUCER<const LHS, const RHS, OP>::size2(lhs.lhs(), lhs.rhs()); }
};

//special case: division with a scalar
template<typename T, typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const cuarma::matrix_base<T>,
    const ScalarType,
    cuarma::op_div>
{
  static arma_size_t size1(cuarma::matrix_base<T> const & lhs,
                          ScalarType const & /*rhs*/) { return lhs.size1(); }

  static arma_size_t size2(cuarma::matrix_base<T> const & lhs,
                          ScalarType const & /*rhs*/) { return lhs.size2(); }
};

//special case: diagonal from vector
template<typename T>
struct MATRIX_SIZE_DEDUCER<const cuarma::vector_base<T>,
    const int,
    cuarma::op_vector_diag>
{
  static arma_size_t size1(cuarma::vector_base<T> const & lhs,
                          const int k) { return lhs.size() + static_cast<arma_size_t>(std::fabs(double(k))); }

  static arma_size_t size2(cuarma::vector_base<T> const & lhs,
                          const int k) { return lhs.size() + static_cast<arma_size_t>(std::fabs(double(k))); }
};

//special case: transposed matrix-vector product: Return the number of rows of the matrix
template<typename MatrixType>
struct MATRIX_SIZE_DEDUCER<MatrixType,
    MatrixType,
    cuarma::op_trans>
{
  static arma_size_t size1(const MatrixType & lhs,
                          const MatrixType & /*rhs*/) { return lhs.size2(); }
  static arma_size_t size2(const MatrixType & lhs,
                          const MatrixType & /*rhs*/) { return lhs.size1(); }
};

// A^T * B
template<typename ScalarType, typename T1>
struct MATRIX_SIZE_DEDUCER<const cuarma::matrix_expression<T1,
    T1, op_trans>,
    const cuarma::matrix_base<ScalarType>,
    cuarma::op_mat_mat_prod>
{
  static arma_size_t size1(cuarma::matrix_expression<T1,
                          T1,
                          op_trans> const & lhs,
                          cuarma::matrix_base<ScalarType> const & /*rhs*/) { return lhs.lhs().size2(); }
  static arma_size_t size2(cuarma::matrix_expression<T1,
                          T1,
                          op_trans> const & /*lhs*/,
                          cuarma::matrix_base<ScalarType> const & rhs) { return rhs.size2(); }
};


// A * B^T
template<typename ScalarType, typename T2>
struct MATRIX_SIZE_DEDUCER<const cuarma::matrix_base<ScalarType>,
    const cuarma::matrix_expression<T2,
    T2, op_trans>,
    cuarma::op_mat_mat_prod>
{
  static arma_size_t size1(cuarma::matrix_base<ScalarType> const & lhs,
                          cuarma::matrix_expression<T2,
                          T2,
                          op_trans> const & /*rhs*/) { return lhs.size1(); }
  static arma_size_t size2(cuarma::matrix_base<ScalarType> const & /*lhs*/,
                          cuarma::matrix_expression<T2,
                          T2,
                          op_trans> const & rhs) { return rhs.lhs().size1(); }
};

// A^T * B^T
template<typename T1, typename T2>
struct MATRIX_SIZE_DEDUCER<const cuarma::matrix_expression<T1,
    T1, op_trans>,
    const cuarma::matrix_expression<T2,
    T2, op_trans>,
    cuarma::op_mat_mat_prod>
{
  typedef cuarma::matrix_expression<T1, T1, op_trans>   LHSType;
  typedef cuarma::matrix_expression<T2, T2, op_trans>   RHSType;

  static arma_size_t size1(LHSType const & lhs,
                          RHSType const & /*rhs*/) { return lhs.lhs().size2(); }
  static arma_size_t size2(LHSType const & /*lhs*/,
                          RHSType const & rhs) { return rhs.lhs().size1(); }
};
/** \endcond */

}
}