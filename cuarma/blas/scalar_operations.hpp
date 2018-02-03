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

/** @file cuarma/blas/scalar_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of scalar operations.
*/

#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/meta/predicate.hpp"
#include "cuarma/meta/enable_if.hpp"
#include "cuarma/traits/size.hpp"
#include "cuarma/traits/start.hpp"
#include "cuarma/traits/handle.hpp"
#include "cuarma/traits/stride.hpp"
#include "cuarma/blas/host_based/scalar_operations.hpp"

#ifdef CUARMA_WITH_CUDA
  #include "cuarma/blas/cuda/scalar_operations.hpp"
#endif

namespace cuarma
{
  namespace blas
  {

    /** @brief Interface for the generic operation s1 = s2 @ alpha, where s1 and s2 are GPU scalars, @ denotes multiplication or division, and alpha is either a GPU or a CPU scalar
     *
     * @param s1                The first  (GPU) scalar
     * @param s2                The second (GPU) scalar
     * @param alpha             The scalar alpha in the operation
     * @param len_alpha         If alpha is obtained from summing over a small GPU vector (e.g. the final summation after a multi-group reduction), then supply the length of the array here
     * @param reciprocal_alpha  If true, then s2 / alpha instead of s2 * alpha is computed
     * @param flip_sign_alpha   If true, then (-alpha) is used instead of alpha
     */
    template<typename S1, typename S2, typename ScalarType1>
    typename cuarma::enable_if< cuarma::is_scalar<S1>::value && cuarma::is_scalar<S2>::value && cuarma::is_any_scalar<ScalarType1>::value >::type
    as(S1 & s1, S2 const & s2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
    {
      switch (cuarma::traits::handle(s1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::as(s1, s2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::as(s1, s2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Interface for the generic operation s1 = s2 @ alpha + s3 @ beta, where s1, s2 and s3 are GPU scalars, @ denotes multiplication or division, and alpha, beta are either a GPU or a CPU scalar
     *
     * @param s1                The first  (GPU) scalar
     * @param s2                The second (GPU) scalar
     * @param alpha             The scalar alpha in the operation
     * @param len_alpha         If alpha is a small GPU vector, which needs to be summed in order to obtain the final scalar, then supply the length of the array here
     * @param reciprocal_alpha  If true, then s2 / alpha instead of s2 * alpha is computed
     * @param flip_sign_alpha   If true, then (-alpha) is used instead of alpha
     * @param s3                The third (GPU) scalar
     * @param beta              The scalar beta in the operation
     * @param len_beta          If beta is obtained from summing over a small GPU vector (e.g. the final summation after a multi-group reduction), then supply the length of the array here
     * @param reciprocal_beta   If true, then s2 / beta instead of s2 * beta is computed
     * @param flip_sign_beta    If true, then (-beta) is used instead of beta
     */
    template<typename S1, typename S2, typename ScalarType1, typename S3, typename ScalarType2>
    typename cuarma::enable_if< cuarma::is_scalar<S1>::value
                                  && cuarma::is_scalar<S2>::value
                                  && cuarma::is_scalar<S3>::value
                                  && cuarma::is_any_scalar<ScalarType1>::value
                                  && cuarma::is_any_scalar<ScalarType2>::value
                                >::type
    asbs(S1 & s1,
         S2 const & s2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
         S3 const & s3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      switch (cuarma::traits::handle(s1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::asbs(s1,
                                             s2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                             s3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::asbs(s1,
                                       s2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                       s3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Interface for the generic operation s1 += s2 @ alpha + s3 @ beta, where s1, s2 and s3 are GPU scalars, @ denotes multiplication or division, and alpha, beta are either a GPU or a CPU scalar
     *
     * @param s1                The first  (GPU) scalar
     * @param s2                The second (GPU) scalar
     * @param alpha             The scalar alpha in the operation
     * @param len_alpha         If alpha is a small GPU vector, which needs to be summed in order to obtain the final scalar, then supply the length of the array here
     * @param reciprocal_alpha  If true, then s2 / alpha instead of s2 * alpha is computed
     * @param flip_sign_alpha   If true, then (-alpha) is used instead of alpha
     * @param s3                The third (GPU) scalar
     * @param beta              The scalar beta in the operation
     * @param len_beta          If beta is obtained from summing over a small GPU vector (e.g. the final summation after a multi-group reduction), then supply the length of the array here
     * @param reciprocal_beta   If true, then s2 / beta instead of s2 * beta is computed
     * @param flip_sign_beta    If true, then (-beta) is used instead of beta
     */
    template<typename S1,
              typename S2, typename ScalarType1,
              typename S3, typename ScalarType2>
    typename cuarma::enable_if< cuarma::is_scalar<S1>::value
                                  && cuarma::is_scalar<S2>::value
                                  && cuarma::is_scalar<S3>::value
                                  && cuarma::is_any_scalar<ScalarType1>::value
                                  && cuarma::is_any_scalar<ScalarType2>::value
                                >::type
    asbs_s(S1 & s1,
           S2 const & s2, ScalarType1 const & alpha, arma_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
           S3 const & s3, ScalarType2 const & beta,  arma_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      switch (cuarma::traits::handle(s1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::asbs_s(s1,
                                               s2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                               s3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::asbs_s(s1,
                                         s2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         s3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }



    /** @brief Swaps the contents of two scalars
    *
    * @param s1   The first scalar
    * @param s2   The second scalar
    */
    template<typename S1, typename S2>
    typename cuarma::enable_if<    cuarma::is_scalar<S1>::value
                                  && cuarma::is_scalar<S2>::value
                                >::type
    swap(S1 & s1, S2 & s2)
    {
      switch (cuarma::traits::handle(s1).get_active_handle_id())
      {
        case cuarma::MAIN_MEMORY:
          cuarma::blas::host_based::swap(s1, s2);
          break;

#ifdef CUARMA_WITH_CUDA
        case cuarma::CUDA_MEMORY:
          cuarma::blas::cuda::swap(s1, s2);
          break;
#endif
        case cuarma::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


  } //namespace blas
} //namespace cuarma
