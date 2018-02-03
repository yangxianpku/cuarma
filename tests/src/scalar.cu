/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   scalar.cu
 *   @coding UTF-8
 *   @brief  Tests operations for cuarma::scalar objects.
 *   @brief  ≤‚ ‘£∫cuarma::scalar ∂‘œÛ≤‚ ‘
 */

#include <iostream>
#include <algorithm>
#include <cmath>

#include "head_define.h"
#include "cuarma/scalar.hpp"

template<typename ScalarType>
ScalarType diff(ScalarType & s1, cuarma::scalar<ScalarType> & s2)
{
   cuarma::backend::finish();
   if (std::fabs(s1 - s2) > 0)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}
//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
   int retval = EXIT_SUCCESS;

   NumericT s1 = NumericT(3.1415926);
   NumericT s2 = NumericT(2.71763);
   NumericT s3 = NumericT(42);

   cuarma::scalar<NumericT> arma_s1;
   cuarma::scalar<NumericT> arma_s2;
   cuarma::scalar<NumericT> arma_s3 = 1.0;

   arma_s1 = s1;
   if ( std::fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: arma_s1 = s1;" << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   arma_s2 = s2;
   if ( std::fabs(diff(s2, arma_s2)) > epsilon )
   {
      std::cout << "# Error at operation: arma_s2 = s2;" << std::endl;
      std::cout << "  diff: " << fabs(diff(s2, arma_s2)) << std::endl;
      retval = EXIT_FAILURE;
   }

   arma_s3 = s3;
   if ( std::fabs(diff(s3, arma_s3)) > epsilon )
   {
      std::cout << "# Error at operation: arma_s3 = s3;" << std::endl;
      std::cout << "  diff: " << s3 - arma_s3 << std::endl;
      retval = EXIT_FAILURE;
   }

   NumericT tmp = s2;
   s2 = s1;
   s1 = tmp;
   cuarma::blas::swap(arma_s1, arma_s2);
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: swap " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2;
   arma_s1 += arma_s2;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: += " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 *= s3;
   arma_s1 *= arma_s3;

   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: *= " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2;
   arma_s1 -= arma_s2;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: -= " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 /= s3;
   arma_s1 /= arma_s3;

   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: /= " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = arma_s1;

   s1 = s2 + s3;
   arma_s1 = arma_s2 + arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 + s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 + s3;
   arma_s1 += arma_s2 + arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 + s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 + s3;
   arma_s1 -= arma_s2 + arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 + s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 - s3;
   arma_s1 = arma_s2 - arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 - s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 - s3;
   arma_s1 += arma_s2 - arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 - s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 - s3;
   arma_s1 -= arma_s2 - arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 - s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 * s3;
   arma_s1 = arma_s2 * arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 * s3;
   arma_s1 += arma_s2 * arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 * s3;
   arma_s1 -= arma_s2 * arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 / s3;
   arma_s1 = arma_s2 / arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 / s3;
   arma_s1 += arma_s2 / arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 / s3;
   arma_s1 -= arma_s2 / arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // addition with factors, =
   arma_s1 = s1;

   s1 = s2 * s2 + s3 * s3;
   arma_s1 = arma_s2 * s2 + arma_s3 * s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   arma_s1 = arma_s2 * arma_s2 + arma_s3 * arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s2 + s3 * s3, second test " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 * s2 + s3 / s3;
   arma_s1 = arma_s2 * s2 + arma_s3 / s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   arma_s1 = arma_s2 * arma_s2 + arma_s3 / arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 * s2 + s3 / s3, second test " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 / s2 + s3 * s3;
   arma_s1 = arma_s2 / s2 + arma_s3 * s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   arma_s1 = arma_s2 / arma_s2 + arma_s3 * arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s2 + s3 * s3, second test " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 = s2 / s2 + s3 / s3;
   arma_s1 = arma_s2 / s2 + arma_s3 / s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }
   arma_s1 = arma_s2 / arma_s2 + arma_s3 / arma_s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 = s2 / s2 + s3 / s3, second test " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // addition with factors, +=
   arma_s1 = s1;

   s1 += s2 * s2 + s3 * s3;
   arma_s1 += arma_s2 * s2 + arma_s3 * s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 * s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 * s2 + s3 / s3;
   arma_s1 += arma_s2 * s2 + arma_s3 / s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 * s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 / s2 + s3 * s3;
   arma_s1 += arma_s2 / s2 + arma_s3 * s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 / s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 += s2 / s2 + s3 / s3;
   arma_s1 += arma_s2 / s2 + arma_s3 / s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 += s2 / s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // addition with factors, -=
   arma_s1 = s1;

   s1 -= s2 * s2 + s3 * s3;
   arma_s1 -= arma_s2 * s2 + arma_s3 * s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 * s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 * s2 + s3 / s3;
   arma_s1 -= arma_s2 * s2 + arma_s3 / s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 * s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 / s2 + s3 * s3;
   arma_s1 -= arma_s2 / s2 + arma_s3 * s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 / s2 + s3 * s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   s1 -= s2 / s2 + s3 / s3;
   arma_s1 -= arma_s2 / s2 + arma_s3 / s3;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: s1 -= s2 / s2 + s3 / s3 " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   // lenghty expression:
   s1 = s2 + s3 * s2 - s3 / s1;
   arma_s1 = arma_s2 + arma_s3 * arma_s2 - arma_s3 / arma_s1;
   if ( fabs(diff(s1, arma_s1)) > epsilon )
   {
      std::cout << "# Error at operation: + * - / " << std::endl;
      std::cout << "  diff: " << fabs(diff(s1, arma_s1)) << std::endl;
      retval = EXIT_FAILURE;
   }

   return retval;
}

int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Scalar" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = NumericT(1.0E-5);
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>(epsilon);
      if ( retval == EXIT_SUCCESS )
         std::cout << "# Test passed" << std::endl;
      else
         return retval;
   }
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   {
      {
         typedef double NumericT;
         NumericT epsilon = 1.0E-10;
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  eps:     " << epsilon << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>(epsilon);
         if ( retval == EXIT_SUCCESS )
           std::cout << "# Test passed" << std::endl;
         else
           return retval;
      }
      std::cout << std::endl;
   }

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;
   return retval;
}