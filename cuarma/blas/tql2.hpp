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

/** @file cuarma/blas/tql2.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementation of the tql2-algorithm for eigenvalue computations.
*/


#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include <iomanip>

#include "cuarma/blas/qr-method-common.hpp"
#include "cuarma/blas/prod.hpp"

namespace cuarma
{
namespace blas
{
  // Symmetric tridiagonal QL algorithm.
  // This is derived from the Algol procedures tql1, by Bowdler, Martin, Reinsch, and Wilkinson,
  // Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.
  template <typename SCALARTYPE, typename VectorType>
  void tql1(arma_size_t n,
            VectorType & d,
            VectorType & e)
  {
      for (arma_size_t i = 1; i < n; i++)
          e[i - 1] = e[i];


      e[n - 1] = 0;

      SCALARTYPE f = 0.;
      SCALARTYPE tst1 = 0.;
      SCALARTYPE eps = static_cast<SCALARTYPE>(1e-6);


      for (arma_size_t l = 0; l < n; l++)
      {
          // Find small subdiagonal element.
          tst1 = std::max<SCALARTYPE>(tst1, std::fabs(d[l]) + std::fabs(e[l]));
          arma_size_t m = l;
          while (m < n)
          {
              if (std::fabs(e[m]) <= eps * tst1)
                  break;
              m++;
          }

          // If m == l, d[l) is an eigenvalue, otherwise, iterate.
          if (m > l)
          {
              arma_size_t iter = 0;
              do
              {
                  iter = iter + 1;  // (Could check iteration count here.)

                  // Compute implicit shift
                  SCALARTYPE g = d[l];
                  SCALARTYPE p = (d[l + 1] - g) / (2 * e[l]);
                  SCALARTYPE r = cuarma::blas::detail::pythag<SCALARTYPE>(p, 1);
                  if (p < 0)
                  {
                      r = -r;
                  }

                  d[l] = e[l] / (p + r);
                  d[l + 1] = e[l] * (p + r);
                  SCALARTYPE h = g - d[l];
                  for (arma_size_t i = l + 2; i < n; i++)
                  {
                      d[i] -= h;
                  }

                  f = f + h;

                  // Implicit QL transformation.
                  p = d[m];
                  SCALARTYPE c = 1;
                  SCALARTYPE s = 0;
                  for (int i = int(m - 1); i >= int(l); i--)
                  {
                      g = c * e[arma_size_t(i)];
                      h = c * p;
                      r = cuarma::blas::detail::pythag<SCALARTYPE>(p, e[arma_size_t(i)]);
                      e[arma_size_t(i) + 1] = s * r;
                      s = e[arma_size_t(i)] / r;
                      c = p / r;
                      p = c * d[arma_size_t(i)] - s * g;
                      d[arma_size_t(i) + 1] = h + s * (c * g + s * d[arma_size_t(i)]);
                  }
                  e[l] = s * p;
                  d[l] = c * p;
              // Check for convergence.
              }
              while (std::fabs(e[l]) > eps * tst1);
          }
          d[l] = d[l] + f;
          e[l] = 0;
      }
  }


// Symmetric tridiagonal QL algorithm.
// This is derived from the Algol procedures tql2, by Bowdler, Martin, Reinsch, and Wilkinson,
// Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.
template <typename SCALARTYPE, typename VectorType, typename F>
void tql2(matrix_base<SCALARTYPE, F> & Q,
          VectorType & d,
          VectorType & e)
{
    arma_size_t n = static_cast<arma_size_t>(cuarma::traits::size1(Q));

    std::vector<SCALARTYPE> cs(n), ss(n);
    cuarma::vector<SCALARTYPE> tmp1(n), tmp2(n);

    for (arma_size_t i = 1; i < n; i++)
        e[i - 1] = e[i];

    e[n - 1] = 0;

    SCALARTYPE f = 0;
    SCALARTYPE tst1 = 0;
    SCALARTYPE eps = static_cast<SCALARTYPE>(cuarma::blas::detail::EPS);

    for (arma_size_t l = 0; l < n; l++)
    {
        // Find small subdiagonal element.
        tst1 = std::max<SCALARTYPE>(tst1, std::fabs(d[l]) + std::fabs(e[l]));
        arma_size_t m = l;
        while (m < n)
        {
            if (std::fabs(e[m]) <= eps * tst1)
                break;
            m++;
        }

        // If m == l, d[l) is an eigenvalue, otherwise, iterate.
        if (m > l)
        {
            arma_size_t iter = 0;
            do
            {
                iter = iter + 1;  // (Could check iteration count here.)

                // Compute implicit shift
                SCALARTYPE g = d[l];
                SCALARTYPE p = (d[l + 1] - g) / (2 * e[l]);
                SCALARTYPE r = cuarma::blas::detail::pythag<SCALARTYPE>(p, 1);
                if (p < 0)
                {
                    r = -r;
                }

                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                SCALARTYPE dl1 = d[l + 1];
                SCALARTYPE h = g - d[l];
                for (arma_size_t i = l + 2; i < n; i++)
                {
                    d[i] -= h;
                }

                f = f + h;

                // Implicit QL transformation.
                p = d[m];
                SCALARTYPE c = 1;
                SCALARTYPE c2 = c;
                SCALARTYPE c3 = c;
                SCALARTYPE el1 = e[l + 1];
                SCALARTYPE s = 0;
                SCALARTYPE s2 = 0;
                for (int i = int(m - 1); i >= int(l); i--)
                {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[arma_size_t(i)];
                    h = c * p;
                    r = cuarma::blas::detail::pythag(p, e[arma_size_t(i)]);
                    e[arma_size_t(i) + 1] = s * r;
                    s = e[arma_size_t(i)] / r;
                    c = p / r;
                    p = c * d[arma_size_t(i)] - s * g;
                    d[arma_size_t(i) + 1] = h + s * (c * g + s * d[arma_size_t(i)]);


                    cs[arma_size_t(i)] = c;
                    ss[arma_size_t(i)] = s;
                }


                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                cuarma::copy(cs, tmp1);
                cuarma::copy(ss, tmp2);

                cuarma::blas::givens_next(Q, tmp1, tmp2, int(l), int(m));

                // Check for convergence.
            }
            while (std::fabs(e[l]) > eps * tst1);
        }
        d[l] = d[l] + f;
        e[l] = 0;
    }
}
} // namespace blas
} // namespace cuarma
