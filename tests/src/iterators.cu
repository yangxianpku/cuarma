/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   iterators.cu
 *   @coding UTF-8
 *   @brief  Tests the iterators in cuarma.
 *   @brief  µü´úÆ÷²âÊÔ
 */

#include <iostream>
#include <stdlib.h>

#include "head_define.h"

#include "cuarma/matrix.hpp"
#include "cuarma/vector.hpp"


template< typename NumericT >
int test()
{
   int retval = EXIT_SUCCESS;
   // --------------------------------------------------------------------------
   typedef cuarma::vector<NumericT>  armaVector;

   armaVector arma_cont(3);
   arma_cont[0] = 1;
   arma_cont[1] = 2;
   arma_cont[2] = 3;

   //typename armaVector::const_iterator const_iter_def_const;
   //typename armaVector::iterator       iter_def_const;

   for (typename armaVector::const_iterator iter = arma_cont.begin();iter != arma_cont.end(); iter++)
   {
      std::cout << *iter << std::endl;
   }

   for (typename armaVector::iterator iter = arma_cont.begin();iter != arma_cont.end(); iter++)
   {
      std::cout << *iter << std::endl;
   }

   // --------------------------------------------------------------------------
   return retval;
}

int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Iterators" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  numeric: float" << std::endl;
      retval = test<NumericT>();
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
         std::cout << "# Testing setup:" << std::endl;
         std::cout << "  numeric: double" << std::endl;
         retval = test<NumericT>();
            if ( retval == EXIT_SUCCESS )
              std::cout << "# Test passed" << std::endl;
            else
              return retval;
      }
      std::cout << std::endl;
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
   }

   std::cout << std::endl;
   std::cout << "------- Test completed --------" << std::endl;
   std::cout << std::endl;

   return retval;
}
