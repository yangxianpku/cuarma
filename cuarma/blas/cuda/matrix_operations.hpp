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

/** @file  cuarma/blas/cuda/matrix_operations.hpp
 *  @encoding:UTF-8 文档编码
 *   @brief Implementations of dense matrix related operations, including matrix-vector products, using CUDA.
 */

#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/vector_proxy.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/meta/result_of.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/start.hpp"
#include "cuarma/traits/handle.hpp"
#include "cuarma/traits/stride.hpp"
#include "cuarma/blas/cuda/common.hpp"
#include "cuarma/blas/cuda/vector_operations.hpp"
#include "cuarma/blas/cuda/matrix_operations_row.hpp"
#include "cuarma/blas/cuda/matrix_operations_col.hpp"
#include "cuarma/blas/cuda/matrix_operations_prod.hpp"

namespace cuarma
{
namespace blas
{
namespace cuda
{
//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//

template<typename DestNumericT, typename SrcNumericT>
void convert(matrix_base<DestNumericT> & mat1, matrix_base<SrcNumericT> const & mat2)
{
  assert(mat1.row_major() == mat2.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  if (mat1.row_major())
  {
    convert_row_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                    cuarma::cuda_arg(mat2),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat2)),           static_cast<unsigned int>(cuarma::traits::start2(mat2)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat2)),          static_cast<unsigned int>(cuarma::traits::stride2(mat2)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat2)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat2))
                                  );
    CUARMA_CUDA_LAST_ERROR_CHECK("convert_row_kernel");
  }
  else
  {
    convert_col_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                    cuarma::cuda_arg(mat2),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat2)),           static_cast<unsigned int>(cuarma::traits::start2(mat2)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat2)),          static_cast<unsigned int>(cuarma::traits::stride2(mat2)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat2)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat2))
                                  );
    CUARMA_CUDA_LAST_ERROR_CHECK("convert_col_kernel");
  }
}

template<typename NumericT, typename SizeT, typename DistanceT>
void trans(matrix_expression<const matrix_base<NumericT, SizeT, DistanceT>,const matrix_base<NumericT, SizeT, DistanceT>, op_trans> const & proxy,
           matrix_base<NumericT> & temp_trans)
{
  trans_kernel<<<128,128>>>(cuarma::cuda_arg(proxy.lhs()),
                            static_cast<unsigned int>(proxy.lhs().start1()),          static_cast<unsigned int>(proxy.lhs().start2()),
                            static_cast<unsigned int>(proxy.lhs().internal_size1()),  static_cast<unsigned int>(proxy.lhs().internal_size2()),
                            static_cast<unsigned int>(proxy.lhs().size1()),           static_cast<unsigned int>(proxy.lhs().size2()),
                            static_cast<unsigned int>(proxy.lhs().stride1()),         static_cast<unsigned int>(proxy.lhs().stride2()),

                            cuarma::cuda_arg(temp_trans),
                            static_cast<unsigned int>(temp_trans.start1()),            static_cast<unsigned int>(temp_trans.start2()),
                            static_cast<unsigned int>(temp_trans.internal_size1()),    static_cast<unsigned int>(temp_trans.internal_size2()),
                            static_cast<unsigned int>(temp_trans.stride1()),           static_cast<unsigned int>(temp_trans.stride2()),
                            static_cast<bool>(proxy.lhs().row_major()));
  CUARMA_CUDA_LAST_ERROR_CHECK("trans_kernel");
}


template<typename NumericT, typename ScalarT>
void am(matrix_base<NumericT> & mat1,
        matrix_base<NumericT> const & mat2, ScalarT const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(mat1.row_major() == mat2.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  typedef NumericT        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  value_type temporary_alpha = 0;
  if (cuarma::is_cpu_scalar<ScalarT>::value)
    temporary_alpha = alpha;

  if (mat1.row_major())
  {
    am_row_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                options_alpha,
                                cuarma::cuda_arg(mat2),
                                static_cast<unsigned int>(cuarma::traits::start1(mat2)),           static_cast<unsigned int>(cuarma::traits::start2(mat2)),
                                static_cast<unsigned int>(cuarma::traits::stride1(mat2)),          static_cast<unsigned int>(cuarma::traits::stride2(mat2)),
                                static_cast<unsigned int>(cuarma::traits::internal_size1(mat2)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat2))
                              );
    CUARMA_CUDA_LAST_ERROR_CHECK("am_row_kernel");
  }
  else
  {
    am_col_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                options_alpha,
                                cuarma::cuda_arg(mat2),
                                static_cast<unsigned int>(cuarma::traits::start1(mat2)),           static_cast<unsigned int>(cuarma::traits::start2(mat2)),
                                static_cast<unsigned int>(cuarma::traits::stride1(mat2)),          static_cast<unsigned int>(cuarma::traits::stride2(mat2)),
                                static_cast<unsigned int>(cuarma::traits::internal_size1(mat2)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat2))
                              );
    CUARMA_CUDA_LAST_ERROR_CHECK("am_col_kernel");
  }
}


template<typename NumericT, typename ScalarT1, typename ScalarT2>
void ambm(matrix_base<NumericT> & mat1,
          matrix_base<NumericT> const & mat2, ScalarT1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          matrix_base<NumericT> const & mat3, ScalarT2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(mat1.row_major() == mat2.row_major() && mat1.row_major() == mat3.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  typedef NumericT        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  value_type temporary_alpha = 0;
  if (cuarma::is_cpu_scalar<ScalarT1>::value)
    temporary_alpha = alpha;


  unsigned int options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  value_type temporary_beta = 0;
  if (cuarma::is_cpu_scalar<ScalarT2>::value)
    temporary_beta = beta;


  if (mat1.row_major())
  {
    ambm_row_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                  static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                  static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                  static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                  static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                  cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                  options_alpha,
                                  cuarma::cuda_arg(mat2),
                                  static_cast<unsigned int>(cuarma::traits::start1(mat2)),           static_cast<unsigned int>(cuarma::traits::start2(mat2)),
                                  static_cast<unsigned int>(cuarma::traits::stride1(mat2)),          static_cast<unsigned int>(cuarma::traits::stride2(mat2)),
                                  static_cast<unsigned int>(cuarma::traits::internal_size1(mat2)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat2)),

                                  cuarma::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                                  options_beta,
                                  cuarma::cuda_arg(mat3),
                                  static_cast<unsigned int>(cuarma::traits::start1(mat3)),           static_cast<unsigned int>(cuarma::traits::start2(mat3)),
                                  static_cast<unsigned int>(cuarma::traits::stride1(mat3)),          static_cast<unsigned int>(cuarma::traits::stride2(mat3)),
                                  static_cast<unsigned int>(cuarma::traits::internal_size1(mat3)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat3))
                                );
    CUARMA_CUDA_LAST_ERROR_CHECK("ambm_row_kernel");
  }
  else
  {
    ambm_col_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                  static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                  static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                  static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                  static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                  cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                  options_alpha,
                                  cuarma::cuda_arg(mat2),
                                  static_cast<unsigned int>(cuarma::traits::start1(mat2)),           static_cast<unsigned int>(cuarma::traits::start2(mat2)),
                                  static_cast<unsigned int>(cuarma::traits::stride1(mat2)),          static_cast<unsigned int>(cuarma::traits::stride2(mat2)),
                                  static_cast<unsigned int>(cuarma::traits::internal_size1(mat2)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat2)),

                                  cuarma::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                                  options_beta,
                                  cuarma::cuda_arg(mat3),
                                  static_cast<unsigned int>(cuarma::traits::start1(mat3)),           static_cast<unsigned int>(cuarma::traits::start2(mat3)),
                                  static_cast<unsigned int>(cuarma::traits::stride1(mat3)),          static_cast<unsigned int>(cuarma::traits::stride2(mat3)),
                                  static_cast<unsigned int>(cuarma::traits::internal_size1(mat3)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat3))
                                );
    CUARMA_CUDA_LAST_ERROR_CHECK("ambm_col_kernel");
  }

}


template<typename NumericT, typename ScalarT1, typename ScalarT2>
void ambm_m(matrix_base<NumericT> & mat1,
            matrix_base<NumericT> const & mat2, ScalarT1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            matrix_base<NumericT> const & mat3, ScalarT2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(mat1.row_major() == mat2.row_major() && mat1.row_major() == mat3.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  typedef NumericT        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  value_type temporary_alpha = 0;
  if (cuarma::is_cpu_scalar<ScalarT1>::value)
    temporary_alpha = alpha;


  unsigned int options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  value_type temporary_beta = 0;
  if (cuarma::is_cpu_scalar<ScalarT2>::value)
    temporary_beta = beta;


  if (mat1.row_major())
  {
    ambm_m_row_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                    cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                    options_alpha,
                                    cuarma::cuda_arg(mat2),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat2)),           static_cast<unsigned int>(cuarma::traits::start2(mat2)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat2)),          static_cast<unsigned int>(cuarma::traits::stride2(mat2)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat2)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat2)),

                                    cuarma::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                                    options_beta,
                                    cuarma::cuda_arg(mat3),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat3)),           static_cast<unsigned int>(cuarma::traits::start2(mat3)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat3)),          static_cast<unsigned int>(cuarma::traits::stride2(mat3)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat3)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat3))
                                  );
    CUARMA_CUDA_LAST_ERROR_CHECK("ambm_m_row_kernel");
  }
  else
  {
    ambm_m_col_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                    cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                    options_alpha,
                                    cuarma::cuda_arg(mat2),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat2)),           static_cast<unsigned int>(cuarma::traits::start2(mat2)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat2)),          static_cast<unsigned int>(cuarma::traits::stride2(mat2)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat2)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat2)),

                                    cuarma::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                                    options_beta,
                                    cuarma::cuda_arg(mat3),
                                    static_cast<unsigned int>(cuarma::traits::start1(mat3)),           static_cast<unsigned int>(cuarma::traits::start2(mat3)),
                                    static_cast<unsigned int>(cuarma::traits::stride1(mat3)),          static_cast<unsigned int>(cuarma::traits::stride2(mat3)),
                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat3)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat3))
                                  );
    CUARMA_CUDA_LAST_ERROR_CHECK("ambm_m_col_kernel");
  }

}




template<typename NumericT>
void matrix_assign(matrix_base<NumericT> & mat, NumericT s, bool clear = false)
{
  typedef NumericT        value_type;
  value_type alpha = s;

  unsigned int s1  = clear ? cuarma::traits::internal_size1(mat) : cuarma::traits::size1(mat);
  unsigned int s2  = clear ? cuarma::traits::internal_size2(mat) : cuarma::traits::size2(mat);

  if (mat.row_major())
  {

    matrix_row_assign_kernel<<<128, 128>>>(cuarma::cuda_arg(mat),
                                           static_cast<unsigned int>(cuarma::traits::start1(mat)),           static_cast<unsigned int>(cuarma::traits::start2(mat)),
                                           static_cast<unsigned int>(cuarma::traits::stride1(mat)),          static_cast<unsigned int>(cuarma::traits::stride2(mat)),
                                           s1,                                                                 s2,
                                           static_cast<unsigned int>(cuarma::traits::internal_size1(mat)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat)),
                                           alpha);
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_assign_kernel");
  }
  else
  {
    matrix_col_assign_kernel<<<128, 128>>>(cuarma::cuda_arg(mat),
                                            static_cast<unsigned int>(cuarma::traits::start1(mat)),           static_cast<unsigned int>(cuarma::traits::start2(mat)),
                                            static_cast<unsigned int>(cuarma::traits::stride1(mat)),          static_cast<unsigned int>(cuarma::traits::stride2(mat)),
                                            s1,                                                                 s2,
                                            static_cast<unsigned int>(cuarma::traits::internal_size1(mat)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat)),
                                            alpha);
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_assign_kernel");
  }
}

template<typename NumericT>
void matrix_diagonal_assign(matrix_base<NumericT> & mat, NumericT s)
{
  typedef NumericT        value_type;
  value_type alpha = s;

  if (mat.row_major())
  {
    matrix_row_diagonal_assign_kernel<<<128, 128>>>(cuarma::cuda_arg(mat),
                                                    static_cast<unsigned int>(cuarma::traits::start1(mat)),           static_cast<unsigned int>(cuarma::traits::start2(mat)),
                                                    static_cast<unsigned int>(cuarma::traits::stride1(mat)),          static_cast<unsigned int>(cuarma::traits::stride2(mat)),
                                                    static_cast<unsigned int>(cuarma::traits::size1(mat)),            static_cast<unsigned int>(cuarma::traits::size2(mat)),
                                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat)),
                                                    alpha);
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_diagonal_assign_kernel");
  }
  else
  {
    matrix_col_diagonal_assign_kernel<<<128, 128>>>(cuarma::cuda_arg(mat),
                                                    static_cast<unsigned int>(cuarma::traits::start1(mat)),           static_cast<unsigned int>(cuarma::traits::start2(mat)),
                                                    static_cast<unsigned int>(cuarma::traits::stride1(mat)),          static_cast<unsigned int>(cuarma::traits::stride2(mat)),
                                                    static_cast<unsigned int>(cuarma::traits::size1(mat)),            static_cast<unsigned int>(cuarma::traits::size2(mat)),
                                                    static_cast<unsigned int>(cuarma::traits::internal_size1(mat)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat)),
                                                    alpha);
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_diagonal_assign_kernel");
  }
}


template<typename NumericT>
void matrix_diag_from_vector(const vector_base<NumericT> & vec, int k, matrix_base<NumericT> & mat)
{
  typedef NumericT        value_type;

  // Step 1: assign zero matrix:
  matrix_assign(mat, NumericT(0));

  // Step 2: Assign diagonal:
  unsigned int options_alpha = 0;

  arma_size_t mat_start = 0;
  arma_size_t mat_stride = 0;
  arma_size_t mat_size = cuarma::traits::size(vec);
  if (mat.row_major())
  {
    arma_size_t first_row_index = 0;
    arma_size_t first_col_index = 0;
    if (k < 0)
      first_row_index = arma_size_t(-k);
    else
      first_col_index = arma_size_t(k);
    mat_start  =  (cuarma::traits::start1(mat) + first_row_index * cuarma::traits::stride1(mat)) * cuarma::traits::internal_size2(mat)
                 + cuarma::traits::start2(mat) + first_col_index * cuarma::traits::stride2(mat);
    mat_stride = cuarma::traits::stride1(mat) * cuarma::traits::internal_size2(mat) + cuarma::traits::stride2(mat);
  }
  else
  {
    arma_size_t first_row_index = 0;
    arma_size_t first_col_index = 0;
    if (k < 0)
      first_row_index = arma_size_t(-k);
    else
      first_col_index = arma_size_t(k);
    mat_start  =    cuarma::traits::start1(mat) + first_row_index * cuarma::traits::stride1(mat)
                 + (cuarma::traits::start2(mat) + first_col_index * cuarma::traits::stride2(mat)) * cuarma::traits::internal_size1(mat);
    mat_stride = cuarma::traits::stride2(mat) * cuarma::traits::internal_size1(mat) + cuarma::traits::stride1(mat);
  }

  av_kernel<<<128, 128>>>(cuarma::cuda_arg(mat),
                          static_cast<unsigned int>(mat_start),
                          static_cast<unsigned int>(mat_stride),
                          static_cast<unsigned int>(mat_size),

                          cuarma::cuda_arg<value_type>(NumericT(1)),
                          options_alpha,
                          cuarma::cuda_arg(vec),
                          static_cast<unsigned int>(cuarma::traits::start(vec)),
                          static_cast<unsigned int>(cuarma::traits::stride(vec)) );
  CUARMA_CUDA_LAST_ERROR_CHECK("av_kernel");
}

template<typename NumericT>
void matrix_diag_to_vector(matrix_base<NumericT> const & mat, int k, vector_base<NumericT> & vec)
{
  typedef NumericT        value_type;

  unsigned int options_alpha = 0;

  arma_size_t mat_start = 0;
  arma_size_t mat_stride = 0;
  if (mat.row_major())
  {
    arma_size_t first_row_index = 0;
    arma_size_t first_col_index = 0;
    if (k < 0)
      first_row_index = arma_size_t(-k);
    else
      first_col_index = arma_size_t(k);
    mat_start  =  (cuarma::traits::start1(mat) + first_row_index * cuarma::traits::stride1(mat)) * cuarma::traits::internal_size2(mat)
                 + cuarma::traits::start2(mat) + first_col_index * cuarma::traits::stride2(mat);
    mat_stride = cuarma::traits::stride1(mat) * cuarma::traits::internal_size2(mat) + cuarma::traits::stride2(mat);
  }
  else
  {
    arma_size_t first_row_index = 0;
    arma_size_t first_col_index = 0;
    if (k < 0)
      first_row_index = arma_size_t(-k);
    else
      first_col_index = arma_size_t(k);
    mat_start  =    cuarma::traits::start1(mat) + first_row_index * cuarma::traits::stride1(mat)
                 + (cuarma::traits::start2(mat) + first_col_index * cuarma::traits::stride2(mat)) * cuarma::traits::internal_size1(mat);
    mat_stride = cuarma::traits::stride2(mat) * cuarma::traits::internal_size1(mat) + cuarma::traits::stride1(mat);
  }

  av_kernel<<<128, 128>>>(cuarma::cuda_arg(vec),
                          static_cast<unsigned int>(cuarma::traits::start(vec)),
                          static_cast<unsigned int>(cuarma::traits::stride(vec)),
                          static_cast<unsigned int>(cuarma::traits::size(vec)),

                          cuarma::cuda_arg<value_type>(NumericT(1)),
                          options_alpha,
                          cuarma::cuda_arg(mat),
                          static_cast<unsigned int>(mat_start),
                          static_cast<unsigned int>(mat_stride));
  CUARMA_CUDA_LAST_ERROR_CHECK("av_kernel");
}

template<typename NumericT>
void matrix_row(matrix_base<NumericT> const & mat, unsigned int i, vector_base<NumericT> & vec)
{
  typedef NumericT        value_type;

  unsigned int options_alpha = 0;

  arma_size_t mat_start = 0;
  arma_size_t mat_stride = 0;
  if (mat.row_major())
  {
    mat_start  = (cuarma::traits::start1(mat) + i * cuarma::traits::stride1(mat)) * cuarma::traits::internal_size2(mat) + cuarma::traits::start2(mat);
    mat_stride = cuarma::traits::stride2(mat);
  }
  else
  {
    mat_start  = cuarma::traits::start1(mat) + i * cuarma::traits::stride1(mat) + cuarma::traits::start2(mat) * cuarma::traits::internal_size1(mat);
    mat_stride = cuarma::traits::stride2(mat) * cuarma::traits::internal_size1(mat);
  }

  av_kernel<<<128, 128>>>(cuarma::cuda_arg(vec),
                          static_cast<unsigned int>(cuarma::traits::start(vec)),
                          static_cast<unsigned int>(cuarma::traits::stride(vec)),
                          static_cast<unsigned int>(cuarma::traits::size(vec)),

                          cuarma::cuda_arg<value_type>(NumericT(1)),
                          options_alpha,
                          cuarma::cuda_arg(mat),
                          static_cast<unsigned int>(mat_start),
                          static_cast<unsigned int>(mat_stride));
  CUARMA_CUDA_LAST_ERROR_CHECK("av_kernel");
}

template<typename NumericT>
void matrix_column(const matrix_base<NumericT> & mat, unsigned int j, vector_base<NumericT> & vec)
{
  typedef NumericT        value_type;

  unsigned int options_alpha = 0;

  arma_size_t mat_start = 0;
  arma_size_t mat_stride = 0;
  if (mat.row_major())
  {
    mat_start  = cuarma::traits::start1(mat) * cuarma::traits::internal_size2(mat) + cuarma::traits::start2(mat) + j * cuarma::traits::stride2(mat);
    mat_stride = cuarma::traits::stride2(mat) * cuarma::traits::internal_size2(mat);
  }
  else
  {
    mat_start  = cuarma::traits::start1(mat) + (cuarma::traits::start2(mat) + j * cuarma::traits::stride2(mat)) * cuarma::traits::internal_size1(mat);
    mat_stride = cuarma::traits::stride2(mat);
  }

  av_kernel<<<128, 128>>>(cuarma::cuda_arg(vec),
                          static_cast<unsigned int>(cuarma::traits::start(vec)),
                          static_cast<unsigned int>(cuarma::traits::stride(vec)),
                          static_cast<unsigned int>(cuarma::traits::size(vec)),

                          cuarma::cuda_arg<value_type>(NumericT(1)),
                          options_alpha,
                          cuarma::cuda_arg(mat),
                          static_cast<unsigned int>(mat_start),
                          static_cast<unsigned int>(mat_stride));
  CUARMA_CUDA_LAST_ERROR_CHECK("av_kernel");
}


//
/////////////////////////   binary element-wise operations    /////////////////////////////////
//


template<typename NumericT, typename SizeT, typename OpT>
void element_op(matrix_base<NumericT, SizeT> & A,
                matrix_expression<const matrix_base<NumericT, SizeT>, const matrix_base<NumericT, SizeT>, op_element_binary<OpT> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT      value_type;

  unsigned int op_type = 2; //0: product, 1: division, 2: power
  if (cuarma::is_division<OpT>::value)
    op_type = 1;
  else if (cuarma::is_product<OpT>::value)
    op_type = 0;

  if (A.row_major())
  {
    element_op_int_row_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
                                        static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
                                        static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

                                        cuarma::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs())),

                                        cuarma::cuda_arg(proxy.rhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.rhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.rhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.rhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.rhs())),

                                        op_type
                                      );
    CUARMA_CUDA_LAST_ERROR_CHECK("element_op_row_kernel");
  }
  else
  {
    element_op_int_col_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
                                        static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
                                        static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

                                        cuarma::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs())),

                                        cuarma::cuda_arg(proxy.rhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.rhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.rhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.rhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.rhs())),

                                        op_type
                                      );
    CUARMA_CUDA_LAST_ERROR_CHECK("element_op_col_kernel");
  }
}

template<typename SizeT, typename OpT>
void element_op(matrix_base<float, SizeT> & A,
                matrix_expression<const matrix_base<float, SizeT>, const matrix_base<float, SizeT>, op_element_binary<OpT> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef float        value_type;

  unsigned int op_type = 2; //0: product, 1: division, 2: power
  if (cuarma::is_division<OpT>::value)
    op_type = 1;
  else if (cuarma::is_product<OpT>::value)
    op_type = 0;

  if (A.row_major())
  {
    element_op_row_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
                                        static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
                                        static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

                                        cuarma::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs())),

                                        cuarma::cuda_arg(proxy.rhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.rhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.rhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.rhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.rhs())),

                                        op_type
                                      );
    CUARMA_CUDA_LAST_ERROR_CHECK("element_op_row_kernel");
  }
  else
  {
    element_op_col_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
                                        static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
                                        static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

                                        cuarma::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs())),

                                        cuarma::cuda_arg(proxy.rhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.rhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.rhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.rhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.rhs())),

                                        op_type
                                      );
    CUARMA_CUDA_LAST_ERROR_CHECK("element_op_col_kernel");
  }
}

template<typename SizeT, typename OpT>
void element_op(matrix_base<double, SizeT> & A,
                matrix_expression<const matrix_base<double, SizeT>, const matrix_base<double, SizeT>, op_element_binary<OpT> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef double        value_type;

  unsigned int op_type = 2; //0: product, 1: division, 2: power
  if (cuarma::is_division<OpT>::value)
    op_type = 1;
  else if (cuarma::is_product<OpT>::value)
    op_type = 0;

  if (A.row_major())
  {
    element_op_row_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
                                        static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
                                        static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

                                        cuarma::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs())),

                                        cuarma::cuda_arg(proxy.rhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.rhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.rhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.rhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.rhs())),

                                        op_type
                                      );
    CUARMA_CUDA_LAST_ERROR_CHECK("element_op_row_kernel");
  }
  else
  {
    element_op_col_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
                                        static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
                                        static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

                                        cuarma::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs())),

                                        cuarma::cuda_arg(proxy.rhs()),
                                        static_cast<unsigned int>(cuarma::traits::start1(proxy.rhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::stride1(proxy.rhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.rhs())),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.rhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.rhs())),

                                        op_type
                                      );
    CUARMA_CUDA_LAST_ERROR_CHECK("element_op_col_kernel");
  }
}

//
/////////////////////////   unary element-wise operations    /////////////////////////////////
//

// Note: Due to CUDA vs C-proprocessor interference (concatenation seems to be broken in at least CUDA 4.2),
//       we could not find a more 'automatic' way of generating the overloads below...

// abs
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_abs> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT value_type;

  if (A.row_major())
  {
    matrix_row_element_abs_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_abs_kernel");
  }
  else
  {
    matrix_col_element_abs_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_abs_kernel");
  }
}


// acos
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_acos> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT    value_type;

  if (A.row_major())
  {
    matrix_row_element_acos_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_acos_kernel");
  }
  else
  {
    matrix_col_element_acos_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_acos_kernel");
  }
}


// asin
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_asin> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT    value_type;

  if (A.row_major())
  {
    matrix_row_element_asin_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_asin_kernel");
  }
  else
  {
    matrix_col_element_asin_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_sin_kernel");
  }
}


// atan
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_atan> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT   value_type;

  if (A.row_major())
  {
    matrix_row_element_atan_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_atan_kernel");
  }
  else
  {
    matrix_col_element_atan_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_atan_kernel");
  }
}


// ceil
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_ceil> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT   value_type;

  if (A.row_major())
  {
    matrix_row_element_ceil_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_ceil_kernel");
  }
  else
  {
    matrix_col_element_ceil_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_ceil_kernel");
  }
}


// cos
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_cos> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT   value_type;

  if (A.row_major())
  {
    matrix_row_element_cos_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_cos_kernel");
  }
  else
  {
    matrix_col_element_cos_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_cos_kernel");
  }
}


// cosh
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_cosh> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT  value_type;

  if (A.row_major())
  {
    matrix_row_element_cosh_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_cosh_kernel");
  }
  else
  {
    matrix_col_element_cosh_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_cosh_kernel");
  }
}


// exp
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_exp> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT  value_type;

  if (A.row_major())
  {
    matrix_row_element_exp_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_exp_kernel");
  }
  else
  {
    matrix_col_element_exp_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_exp_kernel");
  }
}


// fabs
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_fabs> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT   value_type;

  if (A.row_major())
  {
    matrix_row_element_fabs_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_fabs_kernel");
  }
  else
  {
    matrix_col_element_fabs_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_fabs_kernel");
  }
}


// floor
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_floor> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT    value_type;

  if (A.row_major())
  {
    matrix_row_element_floor_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_floor_kernel");
  }
  else
  {
    matrix_col_element_floor_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_floor_kernel");
  }
}


// log
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_log> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT  value_type;

  if (A.row_major())
  {
    matrix_row_element_log_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_log_kernel");
  }
  else
  {
    matrix_col_element_log_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_log_kernel");
  }
}


// log10
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_log10> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT   value_type;

  if (A.row_major())
  {
    matrix_row_element_log10_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_log10_kernel");
  }
  else
  {
    matrix_col_element_log10_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_log10_kernel");
  }
}


// sin
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_sin> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT  value_type;

  if (A.row_major())
  {
    matrix_row_element_sin_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_sin_kernel");
  }
  else
  {
    matrix_col_element_sin_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_sin_kernel");
  }
}


// sinh
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_sinh> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT   value_type;

  if (A.row_major())
  {
    matrix_row_element_sinh_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_sinh_kernel");
  }
  else
  {
    matrix_col_element_sinh_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_sinh_kernel");
  }
}


// sqrt
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_sqrt> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT   value_type;

  if (A.row_major())
  {
    matrix_row_element_sqrt_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_sqrt_kernel");
  }
  else
  {
    matrix_col_element_sqrt_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_sqrt_kernel");
  }
}


// tan
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_tan> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT    value_type;

  if (A.row_major())
  {
    matrix_row_element_tan_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_tan_kernel");
  }
  else
  {
    matrix_col_element_tan_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
      static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
      static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
      static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
      static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

      cuarma::cuda_arg(proxy.lhs()),
      static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
      static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_tan_kernel");
  }
}


// tanh
template<typename NumericT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<op_tanh> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT   value_type;

  if (A.row_major())
  {
    matrix_row_element_tanh_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_row_element_tanh_kernel");
  }
  else
  {
    matrix_col_element_tanh_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
     static_cast<unsigned int>(cuarma::traits::start1(A)),           static_cast<unsigned int>(cuarma::traits::start2(A)),
     static_cast<unsigned int>(cuarma::traits::stride1(A)),          static_cast<unsigned int>(cuarma::traits::stride2(A)),
     static_cast<unsigned int>(cuarma::traits::size1(A)),            static_cast<unsigned int>(cuarma::traits::size2(A)),
     static_cast<unsigned int>(cuarma::traits::internal_size1(A)),   static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

     cuarma::cuda_arg(proxy.lhs()),
     static_cast<unsigned int>(cuarma::traits::start1(proxy.lhs())),           static_cast<unsigned int>(cuarma::traits::start2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::stride1(proxy.lhs())),          static_cast<unsigned int>(cuarma::traits::stride2(proxy.lhs())),
     static_cast<unsigned int>(cuarma::traits::internal_size1(proxy.lhs())),   static_cast<unsigned int>(cuarma::traits::internal_size2(proxy.lhs()))
    );
    CUARMA_CUDA_LAST_ERROR_CHECK("matrix_col_element_tanh_kernel");
  }
}


//
/////////////////////////   matrix-vector products /////////////////////////////////
//

// A * x

/** @brief Carries out matrix-vector multiplication
*
* Implementation of the convenience expressions result = prod(mat, vec); and result = prod(trans(mat), vec);
*
* @param mat    The matrix
* @param mat_transpose Whether the matrix is to be transposed.
* @param vec    The vector
* @param result The result vector
*/
template<typename NumericT>
void prod_impl(const matrix_base<NumericT> & mat, bool mat_transpose,
               const vector_base<NumericT> & vec,
                     vector_base<NumericT> & result)
{
  typedef NumericT        value_type;

  assert(cuarma::traits::handle(vec) != cuarma::traits::handle(result) && bool("No direct inplace matrix-vector product possible. Introduce a temporary!"));

  if (mat.row_major())
  {
    if (!mat_transpose)
    {
      vec_mul_row_kernel<<<128, 128>>>(cuarma::cuda_arg(mat),
                                       static_cast<unsigned int>(cuarma::traits::start1(mat)),         static_cast<unsigned int>(cuarma::traits::start2(mat)),
                                       static_cast<unsigned int>(cuarma::traits::stride1(mat)),        static_cast<unsigned int>(cuarma::traits::stride2(mat)),
                                       static_cast<unsigned int>(cuarma::traits::size1(mat)),          static_cast<unsigned int>(cuarma::traits::size2(mat)),
                                       static_cast<unsigned int>(cuarma::traits::internal_size1(mat)), static_cast<unsigned int>(cuarma::traits::internal_size2(mat)),

                                       cuarma::cuda_arg(vec),
                                       static_cast<unsigned int>(cuarma::traits::start(vec)),
                                       static_cast<unsigned int>(cuarma::traits::stride(vec)),
                                       static_cast<unsigned int>(cuarma::traits::size(vec)),

                                       cuarma::cuda_arg(result),
                                       static_cast<unsigned int>(cuarma::traits::start(result)),
                                       static_cast<unsigned int>(cuarma::traits::stride(result)),
                                       static_cast<unsigned int>(cuarma::traits::size(result))
                                      );
      CUARMA_CUDA_LAST_ERROR_CHECK("vec_mul_row_kernel");
    }
    else
    {
      trans_vec_mul_row_kernel<<<128, 128>>>(cuarma::cuda_arg(mat),
                                             static_cast<unsigned int>(cuarma::traits::start1(mat)),         static_cast<unsigned int>(cuarma::traits::start2(mat)),
                                             static_cast<unsigned int>(cuarma::traits::stride1(mat)),        static_cast<unsigned int>(cuarma::traits::stride2(mat)),
                                             static_cast<unsigned int>(cuarma::traits::size1(mat)),          static_cast<unsigned int>(cuarma::traits::size2(mat)),
                                             static_cast<unsigned int>(cuarma::traits::internal_size1(mat)), static_cast<unsigned int>(cuarma::traits::internal_size2(mat)),

                                             cuarma::cuda_arg(vec),
                                             static_cast<unsigned int>(cuarma::traits::start(vec)),
                                             static_cast<unsigned int>(cuarma::traits::stride(vec)),
                                             static_cast<unsigned int>(cuarma::traits::size(vec)),

                                             cuarma::cuda_arg(result),
                                             static_cast<unsigned int>(cuarma::traits::start(result)),
                                             static_cast<unsigned int>(cuarma::traits::stride(result)),
                                             static_cast<unsigned int>(cuarma::traits::size(result))
                                            );
      CUARMA_CUDA_LAST_ERROR_CHECK("trans_vec_mul_row_kernel");
    }
  }
  else
  {
    if (!mat_transpose)
    {
      vec_mul_col_kernel<<<128, 128>>>(cuarma::cuda_arg(mat),
                                       static_cast<unsigned int>(cuarma::traits::start1(mat)),         static_cast<unsigned int>(cuarma::traits::start2(mat)),
                                       static_cast<unsigned int>(cuarma::traits::stride1(mat)),        static_cast<unsigned int>(cuarma::traits::stride2(mat)),
                                       static_cast<unsigned int>(cuarma::traits::size1(mat)),          static_cast<unsigned int>(cuarma::traits::size2(mat)),
                                       static_cast<unsigned int>(cuarma::traits::internal_size1(mat)), static_cast<unsigned int>(cuarma::traits::internal_size2(mat)),

                                       cuarma::cuda_arg(vec),
                                       static_cast<unsigned int>(cuarma::traits::start(vec)),
                                       static_cast<unsigned int>(cuarma::traits::stride(vec)),
                                       static_cast<unsigned int>(cuarma::traits::size(vec)),

                                       cuarma::cuda_arg(result),
                                       static_cast<unsigned int>(cuarma::traits::start(result)),
                                       static_cast<unsigned int>(cuarma::traits::stride(result)),
                                       static_cast<unsigned int>(cuarma::traits::size(result))
                                      );
      CUARMA_CUDA_LAST_ERROR_CHECK("vec_mul_col_kernel");
    }
    else
    {
      trans_vec_mul_col_kernel<<<128, 128>>>(cuarma::cuda_arg(mat),
                                             static_cast<unsigned int>(cuarma::traits::start1(mat)),         static_cast<unsigned int>(cuarma::traits::start2(mat)),
                                             static_cast<unsigned int>(cuarma::traits::stride1(mat)),        static_cast<unsigned int>(cuarma::traits::stride2(mat)),
                                             static_cast<unsigned int>(cuarma::traits::size1(mat)),          static_cast<unsigned int>(cuarma::traits::size2(mat)),
                                             static_cast<unsigned int>(cuarma::traits::internal_size1(mat)), static_cast<unsigned int>(cuarma::traits::internal_size2(mat)),

                                             cuarma::cuda_arg(vec),
                                             static_cast<unsigned int>(cuarma::traits::start(vec)),
                                             static_cast<unsigned int>(cuarma::traits::stride(vec)),
                                             static_cast<unsigned int>(cuarma::traits::size(vec)),

                                             cuarma::cuda_arg(result),
                                             static_cast<unsigned int>(cuarma::traits::start(result)),
                                             static_cast<unsigned int>(cuarma::traits::stride(result)),
                                             static_cast<unsigned int>(cuarma::traits::size(result))
                                            );
      CUARMA_CUDA_LAST_ERROR_CHECK("trans_vec_mul_col_kernel");
    }
  }
}


//
/////////////////////////   matrix-matrix products /////////////////////////////////
//

namespace detail
{
  // C = A * B and possibly transposed variants
  template<typename MatrixT1, typename MatrixT2, typename MatrixT3, typename ScalarT>
  void prod_slow_kernel(const MatrixT1 & A, bool transposed_A,
                        const MatrixT2 & B, bool transposed_B,
                        MatrixT3 & C,
                        ScalarT alpha,
                        ScalarT beta)
  {
    typedef typename cuarma::result_of::cpu_value_type< typename MatrixT1::value_type >::type   cpu_value_type;

    cpu_value_type converted_alpha = static_cast<cpu_value_type>(alpha);
    cpu_value_type converted_beta  = static_cast<cpu_value_type>(beta);

    dim3 threads(16, 16);
    dim3 grid( (cuarma::traits::size1(C) - 1) / 16 + 1,
               (cuarma::traits::size2(C) - 1) / 16 + 1);

    bool row_major_A = A.row_major();
    bool row_major_B = B.row_major();
    bool row_major_C = C.row_major();


    if (!row_major_C && !row_major_A && !row_major_B && !transposed_A && !transposed_B)
    {
      matrix_matrix_col_col_col_prod_AA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && !row_major_A && !row_major_B && !transposed_A && transposed_B)
    {
      matrix_matrix_col_col_col_prod_AT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && !row_major_A && !row_major_B && transposed_A && !transposed_B)
    {
      matrix_matrix_col_col_col_prod_TA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && !row_major_A && !row_major_B && transposed_A && transposed_B)
    {
      matrix_matrix_col_col_col_prod_TT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    /////////////////////////////////

    else if (!row_major_C && !row_major_A && row_major_B && !transposed_A && !transposed_B)
    {
      matrix_matrix_col_col_row_prod_AA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && !row_major_A && row_major_B && !transposed_A && transposed_B)
    {
      matrix_matrix_col_col_row_prod_AT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && !row_major_A && row_major_B && transposed_A && !transposed_B)
    {
      matrix_matrix_col_col_row_prod_TA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && !row_major_A && row_major_B && transposed_A && transposed_B)
    {
      matrix_matrix_col_col_row_prod_TT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    /////////////////////////////////

    else if (!row_major_C && row_major_A && !row_major_B && !transposed_A && !transposed_B)
    {
      matrix_matrix_col_row_col_prod_AA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && row_major_A && !row_major_B && !transposed_A && transposed_B)
    {
      matrix_matrix_col_row_col_prod_AT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && row_major_A && !row_major_B && transposed_A && !transposed_B)
    {
      matrix_matrix_col_row_col_prod_TA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && row_major_A && !row_major_B && transposed_A && transposed_B)
    {
      matrix_matrix_col_row_col_prod_TT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    /////////////////////////////////

    else if (!row_major_C && row_major_A && row_major_B && !transposed_A && !transposed_B)
    {
      matrix_matrix_col_row_row_prod_AA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && row_major_A && row_major_B && !transposed_A && transposed_B)
    {
      matrix_matrix_col_row_row_prod_AT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && row_major_A && row_major_B && transposed_A && !transposed_B)
    {
      matrix_matrix_col_row_row_prod_TA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (!row_major_C && row_major_A && row_major_B && transposed_A && transposed_B)
    {
      matrix_matrix_col_row_row_prod_TT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    /////////////////////////////////

    else if (row_major_C && !row_major_A && !row_major_B && !transposed_A && !transposed_B)
    {
      matrix_matrix_row_col_col_prod_AA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && !row_major_A && !row_major_B && !transposed_A && transposed_B)
    {
      matrix_matrix_row_col_col_prod_AT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && !row_major_A && !row_major_B && transposed_A && !transposed_B)
    {
      matrix_matrix_row_col_col_prod_TA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && !row_major_A && !row_major_B && transposed_A && transposed_B)
    {
      matrix_matrix_row_col_col_prod_TT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    /////////////////////////////////

    else if (row_major_C && !row_major_A && row_major_B && !transposed_A && !transposed_B)
    {
      matrix_matrix_row_col_row_prod_AA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && !row_major_A && row_major_B && !transposed_A && transposed_B)
    {
      matrix_matrix_row_col_row_prod_AT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && !row_major_A && row_major_B && transposed_A && !transposed_B)
    {
      matrix_matrix_row_col_row_prod_TA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && !row_major_A && row_major_B && transposed_A && transposed_B)
    {
      matrix_matrix_row_col_row_prod_TT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    /////////////////////////////////

    else if (row_major_C && row_major_A && !row_major_B && !transposed_A && !transposed_B)
    {
      matrix_matrix_row_row_col_prod_AA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && row_major_A && !row_major_B && !transposed_A && transposed_B)
    {
      matrix_matrix_row_row_col_prod_AT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && row_major_A && !row_major_B && transposed_A && !transposed_B)
    {
      matrix_matrix_row_row_col_prod_TA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && row_major_A && !row_major_B && transposed_A && transposed_B)
    {
      matrix_matrix_row_row_col_prod_TT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }


    /////////////////////////////////

    else if (row_major_C && row_major_A && row_major_B && !transposed_A && !transposed_B)
    {
      matrix_matrix_row_row_row_prod_AA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && row_major_A && row_major_B && !transposed_A && transposed_B)
    {
      matrix_matrix_row_row_row_prod_AT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && row_major_A && row_major_B && transposed_A && !transposed_B)
    {
      matrix_matrix_row_row_row_prod_TA_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }
    else if (row_major_C && row_major_A && row_major_B && transposed_A && transposed_B)
    {
      matrix_matrix_row_row_row_prod_TT_kernel<<<grid, threads>>>
        (converted_alpha,
          cuarma::cuda_arg(A),
          static_cast<unsigned int>(cuarma::traits::start1(A)),         static_cast<unsigned int>(cuarma::traits::start2(A)),
          static_cast<unsigned int>(cuarma::traits::stride1(A)),        static_cast<unsigned int>(cuarma::traits::stride2(A)),
          static_cast<unsigned int>(cuarma::traits::size1(A)),          static_cast<unsigned int>(cuarma::traits::size2(A)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(A)), static_cast<unsigned int>(cuarma::traits::internal_size2(A)),

          cuarma::cuda_arg(B),
          static_cast<unsigned int>(cuarma::traits::start1(B)),         static_cast<unsigned int>(cuarma::traits::start2(B)),
          static_cast<unsigned int>(cuarma::traits::stride1(B)),        static_cast<unsigned int>(cuarma::traits::stride2(B)),
          static_cast<unsigned int>(cuarma::traits::size1(B)),          static_cast<unsigned int>(cuarma::traits::size2(B)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(B)), static_cast<unsigned int>(cuarma::traits::internal_size2(B)),

          converted_beta,
          cuarma::cuda_arg(C),
          static_cast<unsigned int>(cuarma::traits::start1(C)),         static_cast<unsigned int>(cuarma::traits::start2(C)),
          static_cast<unsigned int>(cuarma::traits::stride1(C)),        static_cast<unsigned int>(cuarma::traits::stride2(C)),
          static_cast<unsigned int>(cuarma::traits::size1(C)),          static_cast<unsigned int>(cuarma::traits::size2(C)),
          static_cast<unsigned int>(cuarma::traits::internal_size1(C)), static_cast<unsigned int>(cuarma::traits::internal_size2(C)) );
    }

  }


  template<typename MatrixT1, typename MatrixT2, typename MatrixT3, typename ScalarT>
  void prod(const MatrixT1 & A, bool transposed_A,
            const MatrixT2 & B, bool transposed_B,
            MatrixT3 & C,
            ScalarT alpha,
            ScalarT beta)
  {
    if (   (cuarma::traits::size1(A) < 64)
        || (cuarma::traits::size2(A) < 64)
        || (cuarma::traits::size1(B) < 64) )   //there is most likely not enough to compute, rendering kernel launch overhead considerable
    {
      prod_slow_kernel(A, transposed_A,
                       B, transposed_B,
                       C, alpha, beta);
    }
    /*else if (   (cuarma::traits::size1(A) % 64 == 0)
            && (cuarma::traits::size2(A) % 64 == 0)
            && (cuarma::traits::size1(B) % 64 == 0) )   // allows the use of the fast kernel only
    {
      prod_fast_kernel(A, B, C, alpha, beta);
      //prod_slow_kernel(A, B, C, slow_kernel_name);
    }*/
    else //TODO: use four kernels
    {
      prod_slow_kernel(A, transposed_A,
                       B, transposed_B,
                       C, alpha, beta);
    }

  }
} // namespace detail


/** @brief Carries out matrix-matrix multiplication
*
* Implementation of C = prod(A, B);
*
*/
template<typename NumericT, typename ScalarT>
void prod_impl(const matrix_base<NumericT> & A, bool trans_A,
               const matrix_base<NumericT> & B, bool trans_B,
                     matrix_base<NumericT> & C,
               ScalarT alpha,
               ScalarT beta)
{
  detail::prod(A, trans_A,
               B, trans_B,
               C, alpha, beta);
}




//
/////////////////////////   miscellaneous operations /////////////////////////////////
//


/** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
*
* Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
*
* @param mat1             The matrix to be updated
* @param alpha            The scaling factor (either a cuarma::scalar<>, float, or double)
* @param len_alpha        Length of the buffer for an eventual final reduction step (currently always '1')
* @param reciprocal_alpha Use 1/alpha instead of alpha
* @param flip_sign_alpha  Use -alpha instead of alpha
* @param vec1             The first vector
* @param vec2             The second vector
*/
template<typename NumericT, typename ScalarT>
void scaled_rank_1_update(matrix_base<NumericT> & mat1,
                          ScalarT const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                          const vector_base<NumericT> & vec1,
                          const vector_base<NumericT> & vec2)
{
  assert( (cuarma::traits::size1(mat1) == cuarma::traits::size(vec1)) && bool("Size mismatch in scaled_rank_1_update: size1(A) != size(v1)"));
  assert( (cuarma::traits::size2(mat1) == cuarma::traits::size(vec2)) && bool("Size mismatch in scaled_rank_1_update: size2(A) != size(v2)"));

  typedef NumericT        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  value_type temporary_alpha = 0;
  if (cuarma::is_cpu_scalar<ScalarT>::value)
    temporary_alpha = alpha;

  if (mat1.row_major())
  {
    scaled_rank1_update_row_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                                 static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                                 static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                                 static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                                 static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                                 cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                                 options_alpha,

                                                 cuarma::cuda_arg(vec1),
                                                 static_cast<unsigned int>(cuarma::traits::start(vec1)),
                                                 static_cast<unsigned int>(cuarma::traits::stride(vec1)),
                                                 static_cast<unsigned int>(cuarma::traits::size(vec1)),

                                                 cuarma::cuda_arg(vec2),
                                                 static_cast<unsigned int>(cuarma::traits::start(vec2)),
                                                 static_cast<unsigned int>(cuarma::traits::stride(vec2)),
                                                 static_cast<unsigned int>(cuarma::traits::size(vec2))
                                               );
    CUARMA_CUDA_LAST_ERROR_CHECK("scaled_rank1_update_row_kernel");
  }
  else
  {
    scaled_rank1_update_col_kernel<<<128, 128>>>(cuarma::cuda_arg(mat1),
                                                 static_cast<unsigned int>(cuarma::traits::start1(mat1)),           static_cast<unsigned int>(cuarma::traits::start2(mat1)),
                                                 static_cast<unsigned int>(cuarma::traits::stride1(mat1)),          static_cast<unsigned int>(cuarma::traits::stride2(mat1)),
                                                 static_cast<unsigned int>(cuarma::traits::size1(mat1)),            static_cast<unsigned int>(cuarma::traits::size2(mat1)),
                                                 static_cast<unsigned int>(cuarma::traits::internal_size1(mat1)),   static_cast<unsigned int>(cuarma::traits::internal_size2(mat1)),

                                                 cuarma::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                                 options_alpha,

                                                 cuarma::cuda_arg(vec1),
                                                 static_cast<unsigned int>(cuarma::traits::start(vec1)),
                                                 static_cast<unsigned int>(cuarma::traits::stride(vec1)),
                                                 static_cast<unsigned int>(cuarma::traits::size(vec1)),

                                                 cuarma::cuda_arg(vec2),
                                                 static_cast<unsigned int>(cuarma::traits::start(vec2)),
                                                 static_cast<unsigned int>(cuarma::traits::stride(vec2)),
                                                 static_cast<unsigned int>(cuarma::traits::size(vec2))
                                                );
    CUARMA_CUDA_LAST_ERROR_CHECK("scaled_rank1_update_col_kernel");
  }
}


/** @brief This function stores the diagonal and the superdiagonal of a matrix in two vectors.
*
*
* @param A     The matrix from which the vectors will be extracted of.
* @param dh    The vector in which the diagonal of the matrix will be stored in.
* @param sh    The vector in which the superdiagonal of the matrix will be stored in.
*/
template <typename NumericT, typename VectorType>
void bidiag_pack(matrix_base<NumericT> & A,
                 VectorType & dh,
                 VectorType & sh
                )
{
   if (A.row_major())
    {
      cuarma::blas::cuda::bidiag_pack_row_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                                               cuarma::cuda_arg(dh),
                                                               cuarma::cuda_arg(sh),
                                                               static_cast<unsigned int>(cuarma::traits::size1(A)),
                                                               static_cast<unsigned int>(cuarma::traits::size2(A)),
                                                               static_cast<unsigned int>(cuarma::traits::internal_size2(A)));
    }
  else
    {
      cuarma::blas::cuda::bidiag_pack_column_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                                               cuarma::cuda_arg(dh),
                                                               cuarma::cuda_arg(sh),
                                                               static_cast<unsigned int>(cuarma::traits::size1(A)),
                                                               static_cast<unsigned int>(cuarma::traits::size2(A)),
                                                               static_cast<unsigned int>(cuarma::traits::internal_size1(A)));
    }
}



/** @brief This function copies a row or a column from a matrix to a vector.
*
*
* @param A          The matrix where to copy from.
* @param V          The vector to fill with data.
* @param row_start  The number of the first row to copy.
* @param col_start  The number of the first column to copy.
* @param copy_col   Set to TRUE to copy a column, FALSE to copy a row.
*/
template <typename NumericT>
void copy_vec(matrix_base<NumericT>& A,
              vector_base<NumericT> & V,
              arma_size_t row_start,
              arma_size_t col_start,
              bool copy_col
)
{
  if(copy_col)
    {
      if (A.row_major())
        {
          copy_col_row_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        cuarma::cuda_arg(V),
                                        static_cast<unsigned int>(row_start),
                                        static_cast<unsigned int>(col_start),
                                        static_cast<unsigned int>(cuarma::traits::size1(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size2(A)));
        }
      else
        {
          copy_col_column_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        cuarma::cuda_arg(V),
                                        static_cast<unsigned int>(row_start),
                                        static_cast<unsigned int>(col_start),
                                        static_cast<unsigned int>(cuarma::traits::size1(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(A)));
        }


    }
  else
    {
      if (A.row_major())
        {
          copy_row_row_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        cuarma::cuda_arg(V),
                                        static_cast<unsigned int>(row_start),
                                        static_cast<unsigned int>(col_start),
                                        static_cast<unsigned int>(cuarma::traits::size2(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size2(A)));
        }
      else
        {
          copy_row_column_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                        cuarma::cuda_arg(V),
                                        static_cast<unsigned int>(row_start),
                                        static_cast<unsigned int>(col_start),
                                        static_cast<unsigned int>(cuarma::traits::size2(A)),
                                        static_cast<unsigned int>(cuarma::traits::internal_size1(A)));
        }
    }
}


/** @brief This function applies a householder transformation to a matrix. A <- P * A with a householder reflection P
*
* @param A       The matrix to be updated.
* @param D       The normalized householder vector.
* @param start   The repetition counter.
*/
template <typename NumericT>
void house_update_A_left(matrix_base<NumericT> & A,
                         vector_base<NumericT> & D,
                         arma_size_t start)
{
  if (A.row_major())
  {
      house_update_A_left_row_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                               cuarma::cuda_arg(D),
                                               static_cast<unsigned int>(start + 1),
                                               static_cast<unsigned int>(start),
                                               static_cast<unsigned int>(cuarma::traits::size1(A)),
                                               static_cast<unsigned int>(cuarma::traits::size2(A)),
                                               static_cast<unsigned int>(cuarma::traits::internal_size2(A)));


  }
  else
    {
      house_update_A_left_column_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                               cuarma::cuda_arg(D),
                                               static_cast<unsigned int>(start + 1),
                                               static_cast<unsigned int>(start),
                                               static_cast<unsigned int>(cuarma::traits::size1(A)),
                                               static_cast<unsigned int>(cuarma::traits::size2(A)),
                                               static_cast<unsigned int>(cuarma::traits::internal_size1(A)));


    }

}


/** @brief This function applies a householder transformation to a matrix: A <- A * P with a householder reflection P
*
*
* @param A        The matrix to be updated.
* @param D        The normalized householder vector.
*/
template <typename NumericT>
void house_update_A_right(matrix_base<NumericT> & A,
                         vector_base<NumericT> & D)
{
  if (A.row_major())
    {
      house_update_A_right_row_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                                cuarma::cuda_arg(D),
                                                static_cast<unsigned int>(0),
                                                static_cast<unsigned int>(0),
                                                static_cast<unsigned int>(cuarma::traits::size1(A)),
                                                static_cast<unsigned int>(cuarma::traits::size2(A)),
                                                static_cast<unsigned int>(cuarma::traits::internal_size2(A)));


    }
  else
    {
      house_update_A_right_column_major_kernel<<<128, 128>>>(cuarma::cuda_arg(A),
                                                cuarma::cuda_arg(D),
                                                static_cast<unsigned int>(0),
                                                static_cast<unsigned int>(0),
                                                static_cast<unsigned int>(cuarma::traits::size1(A)),
                                                static_cast<unsigned int>(cuarma::traits::size2(A)),
                                                static_cast<unsigned int>(cuarma::traits::internal_size1(A)));

    }

}


/** @brief This function updates the matrix Q, which is needed for the computation of the eigenvectors.
*
* @param Q        The matrix to be updated.
* @param D        The householder vector.
* @param A_size1  size1 of matrix A
*/
template <typename NumericT>
void house_update_QL(matrix_base<NumericT> & Q,
                     vector_base<NumericT> & D,
                     arma_size_t A_size1)

{
  if (Q.row_major())
  {
    house_update_QL_row_major_kernel<<<128, 128>>>(cuarma::cuda_arg(Q),
                                           cuarma::cuda_arg(D),
                                           static_cast<unsigned int>(A_size1),
                                           static_cast<unsigned int>(cuarma::traits::internal_size2(Q)));
  }
  else
  {
    house_update_QL_column_major_kernel<<<128, 128>>>(cuarma::cuda_arg(Q),
                                           cuarma::cuda_arg(D),
                                           static_cast<unsigned int>(A_size1),
                                           static_cast<unsigned int>(cuarma::traits::internal_size1(Q)));
  }
}

/** @brief This function updates the matrix Q. It is part of the tql2 algorithm.
*
*
* @param Q       The matrix to be updated.
* @param tmp1    Vector with data from the tql2 algorithm.
* @param tmp2    Vector with data from the tql2 algorithm.
* @param l       Data from the tql2 algorithm.
* @param m       Data from the tql2 algorithm.
*/
template<typename NumericT>
void givens_next(matrix_base<NumericT> & Q,
                 vector_base<NumericT>& tmp1,
                 vector_base<NumericT>& tmp2,
                 int l,
                 int m)
  {
  if (Q.row_major())
    givens_next_row_major_kernel<<<128, 128>>>(cuarma::cuda_arg(Q),
                                     cuarma::cuda_arg(tmp1),
                                     cuarma::cuda_arg(tmp2),
                                     static_cast<unsigned int>(cuarma::traits::size1(Q)),
                                     static_cast<unsigned int>(cuarma::traits::internal_size2(Q)),
                                     static_cast<unsigned int>(l),
                                     static_cast<unsigned int>(m - 1));

  else
    givens_next_column_major_kernel<<<128, 128>>>(cuarma::cuda_arg(Q),
                                     cuarma::cuda_arg(tmp1),
                                     cuarma::cuda_arg(tmp2),
                                     static_cast<unsigned int>(cuarma::traits::size1(Q)),
                                     static_cast<unsigned int>(cuarma::traits::internal_size1(Q)),
                                     static_cast<unsigned int>(l),
                                     static_cast<unsigned int>(m - 1));
  }


} // namespace cuda
} //namespace blas
} //namespace cuarma