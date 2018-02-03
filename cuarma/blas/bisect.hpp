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

/** @file cuarma/blas/bisect.hpp
 *  @encoding:UTF-8 文档编码
*   @brief Implementation of the algorithm for finding eigenvalues of a tridiagonal matrix.
*
*/

#include <vector>
#include <cmath>
#include <limits>
#include <cstddef>
#include "cuarma/meta/result_of.hpp"

namespace cuarma
{
namespace blas
{

namespace detail
{
  /**
  *    @brief overloaded function for copying vectors
  */
  template<typename NumericT, typename OtherVectorT>
  void copy_vec_to_vec(cuarma::vector<NumericT> const & src, OtherVectorT & dest)
  {
    cuarma::copy(src, dest);
  }

  template<typename OtherVectorT, typename NumericT>
  void copy_vec_to_vec(OtherVectorT const & src, cuarma::vector<NumericT> & dest)
  {
    cuarma::copy(src, dest);
  }

  template<typename VectorT1, typename VectorT2>
  void copy_vec_to_vec(VectorT1 const & src, VectorT2 & dest)
  {
    for (arma_size_t i=0; i<src.size(); ++i)
      dest[i] = src[i];
  }

} //namespace detail

/**
*   @brief Implementation of the bisect-algorithm for the calculation of the eigenvalues of a tridiagonal matrix. Experimental - interface might change.
*
*   Refer to "Calculation of the Eigenvalues of a Symmetric Tridiagonal Matrix by the Method of Bisection" in the Handbook Series Linear Algebra, contributed by Barth, Martin, and Wilkinson.
*   http://www.maths.ed.ac.uk/~aar/papers/bamawi.pdf
*
*   @param alphas       Elements of the main diagonal
*   @param betas        Elements of the secondary diagonal
*   @return             Returns the eigenvalues of the tridiagonal matrix defined by alpha and beta
*/
template<typename VectorT>
std::vector<
        typename cuarma::result_of::cpu_value_type<typename VectorT::value_type>::type
        >
bisect(VectorT const & alphas, VectorT const & betas)
{
  typedef typename cuarma::result_of::value_type<VectorT>::type           NumericType;
  typedef typename cuarma::result_of::cpu_value_type<NumericType>::type   CPU_NumericType;

  arma_size_t size = betas.size();
  std::vector<CPU_NumericType>  x_temp(size);


  std::vector<CPU_NumericType> beta_bisect;
  std::vector<CPU_NumericType> wu;

  double rel_error = std::numeric_limits<CPU_NumericType>::epsilon();
  beta_bisect.push_back(0);

  for (arma_size_t i = 1; i < size; i++)
    beta_bisect.push_back(betas[i] * betas[i]);

  double xmin = alphas[size - 1] - std::fabs(betas[size - 1]);
  double xmax = alphas[size - 1] + std::fabs(betas[size - 1]);

  for (arma_size_t i = 0; i < size - 1; i++)
  {
    double h = std::fabs(betas[i]) + std::fabs(betas[i + 1]);
    if (alphas[i] + h > xmax)
      xmax = alphas[i] + h;
    if (alphas[i] - h < xmin)
      xmin = alphas[i] - h;
  }


  double eps1 = 1e-6;
  /*double eps2 = (xmin + xmax > 0) ? (rel_error * xmax) : (-rel_error * xmin);
  if (eps1 <= 0)
    eps1 = eps2;
  else
    eps2 = 0.5 * eps1 + 7.0 * eps2; */

  double x0 = xmax;

  for (arma_size_t i = 0; i < size; i++)
  {
    x_temp[i] = xmax;
    wu.push_back(xmin);
  }

  for (long k = static_cast<long>(size) - 1; k >= 0; --k)
  {
    double xu = xmin;
    for (long i = k; i >= 0; --i)
    {
      if (xu < wu[arma_size_t(k-i)])
      {
        xu = wu[arma_size_t(i)];
        break;
      }
    }

    if (x0 > x_temp[arma_size_t(k)])
      x0 = x_temp[arma_size_t(k)];

    double x1 = (xu + x0) / 2.0;
    while (x0 - xu > 2.0 * rel_error * (std::fabs(xu) + std::fabs(x0)) + eps1)
    {
      arma_size_t a = 0;
      double q = 1;
      for (arma_size_t i = 0; i < size; i++)
      {
        if (q > 0 || q < 0)
          q = alphas[i] - x1 - beta_bisect[i] / q;
        else
          q = alphas[i] - x1 - std::fabs(betas[i] / rel_error);

        if (q < 0)
          a++;
      }

      if (a <= static_cast<arma_size_t>(k))
      {
        xu = x1;
        if (a < 1)
          wu[0] = x1;
        else
        {
          wu[a] = x1;
          if (x_temp[a - 1] > x1)
              x_temp[a - 1] = x1;
        }
      }
      else
        x0 = x1;

      x1 = (xu + x0) / 2.0;
    }
    x_temp[arma_size_t(k)] = x1;
  }
  return x_temp;
}

} // end namespace blas
} // end namespace cuarma
