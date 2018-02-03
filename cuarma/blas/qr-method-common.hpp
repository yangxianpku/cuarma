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

#include <cmath>
#ifdef CUARMA_WITH_CUDA
  #include "cuarma/blas/cuda/matrix_operations.hpp"
#endif
#include "cuarma/meta/result_of.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"

/** @file cuarma/blas/qr-method-common.hpp
 *  @encoding:UTF-8 文档编码
    @brief Common routines used for the QR method and SVD. Experimental.
*/

namespace cuarma
{
namespace blas
{

const std::string SVD_HOUSEHOLDER_UPDATE_QR_KERNEL = "house_update_QR";
const std::string SVD_MATRIX_TRANSPOSE_KERNEL = "transpose_inplace";
const std::string SVD_INVERSE_SIGNS_KERNEL = "inverse_signs";
const std::string SVD_GIVENS_PREV_KERNEL = "givens_prev";
const std::string SVD_FINAL_ITER_UPDATE_KERNEL = "final_iter_update";
const std::string SVD_UPDATE_QR_COLUMN_KERNEL = "update_qr_column";
const std::string SVD_HOUSEHOLDER_UPDATE_A_LEFT_KERNEL = "house_update_A_left";
const std::string SVD_HOUSEHOLDER_UPDATE_A_RIGHT_KERNEL = "house_update_A_right";
const std::string SVD_HOUSEHOLDER_UPDATE_QL_KERNEL = "house_update_QL";

namespace detail
{
static const double EPS = 1e-10;
static const arma_size_t ITER_MAX = 50;

template <typename SCALARTYPE>
SCALARTYPE pythag(SCALARTYPE a, SCALARTYPE b)
{
  return std::sqrt(a*a + b*b);
}

template <typename SCALARTYPE>
SCALARTYPE sign(SCALARTYPE val)
{
    return (val >= 0) ? SCALARTYPE(1) : SCALARTYPE(-1);
}

// DEPRECATED: Replace with cuarma::blas::norm_2
template <typename VectorType>
typename VectorType::value_type norm_lcl(VectorType const & x, arma_size_t size)
{
  typename VectorType::value_type x_norm = 0.0;
  for(arma_size_t i = 0; i < size; i++)
    x_norm += std::pow(x[i], 2);
  return std::sqrt(x_norm);
}

template <typename VectorType>
void normalize(VectorType & x, arma_size_t size)
{
  typename VectorType::value_type x_norm = norm_lcl(x, size);
  for(arma_size_t i = 0; i < size; i++)
      x[i] /= x_norm;
}



template <typename VectorType>
void householder_vector(VectorType & v, arma_size_t start)
{
  typedef typename VectorType::value_type    ScalarType;
  ScalarType x_norm = norm_lcl(v, v.size());
  ScalarType alpha = -sign(v[start]) * x_norm;
  v[start] += alpha;
  normalize(v, v.size());
}

template <typename SCALARTYPE>
void transpose(matrix_base<SCALARTYPE> & A)
{
  (void)A;

}


template <typename T>
void cdiv(T xr, T xi, T yr, T yi, T& cdivr, T& cdivi)
{
    // Complex scalar division.
    T r;
    T d;
    if (std::fabs(yr) > std::fabs(yi))
    {
        r = yi / yr;
        d = yr + r * yi;
        cdivr = (xr + r * xi) / d;
        cdivi = (xi - r * xr) / d;
    }
    else
    {
        r = yr / yi;
        d = yi + r * yr;
        cdivr = (r * xr + xi) / d;
        cdivi = (r * xi - xr) / d;
    }
}


template<typename SCALARTYPE>
void prepare_householder_vector( matrix_base<SCALARTYPE>& A,vector_base<SCALARTYPE>& D, arma_size_t size,
                              arma_size_t row_start,arma_size_t col_start, arma_size_t start, bool is_column )
{
  std::vector<SCALARTYPE> tmp(size);
  copy_vec(A, D, row_start, col_start, is_column);
  fast_copy(D.begin(), D.begin() + arma_ptrdiff_t(size - start), tmp.begin() + arma_ptrdiff_t(start));

  detail::householder_vector(tmp, start);
  fast_copy(tmp, D);
}

} //detail
}
}
