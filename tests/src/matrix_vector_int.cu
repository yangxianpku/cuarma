/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   matrix_vector_int.cu
 *   @coding UTF-8
 *   @brief  Tests routines for matrix-vector operaions (BLAS level 2) using integer arithmetic.
 *   @brief  测试：测试矩阵-向量运算，整型
 */


#include <iostream>
#include <vector>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/blas/lu.hpp"
#include "cuarma/blas/sum.hpp"

//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType & s1, cuarma::scalar<ScalarType> & s2)
{
  cuarma::backend::finish();
  if (s1 != s2)
    return 1;
  return 0;
}

template<typename ScalarType, typename ARMAVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, ARMAVectorType const & v2)
{
  std::vector<ScalarType> v2_cpu(v2.size());
  cuarma::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
  cuarma::copy(v2.begin(), v2.end(), v2_cpu.begin());

  for (unsigned int i=0;i<v1.size(); ++i)
  {
    if (v2_cpu[i] != v1[i])
      return 1;
  }

  return 0;
}

template<typename NumericT, typename ARMAMatrixType>
NumericT diff(std::vector<std::vector<NumericT> > const & mat1, ARMAMatrixType const & mat2)
{
  std::vector<std::vector<NumericT> > mat2_cpu(mat2.size1(), std::vector<NumericT>(mat2.size2()));
  cuarma::backend::finish();  //workaround for a bug in APP SDK 2.7 on Trinity APUs (with Catalyst 12.8)
  cuarma::copy(mat2, mat2_cpu);

  for (unsigned int i = 0; i < mat2_cpu.size(); ++i)
  {
    for (unsigned int j = 0; j < mat2_cpu[i].size(); ++j)
    {
      if (mat2_cpu[i][j] != mat1[i][j])
        return 1;
    }
  }
  //std::cout << ret << std::endl;
  return 0;
}

template<typename NumericT, typename STLMatrixType, typename STLVectorType,typename ARMAMatrixType, typename ARMAVectorType1, typename ARMAVectorType2>
int test_prod_rank1(STLMatrixType & std_m1, STLVectorType & std_v1, STLVectorType & std_v2,
                    ARMAMatrixType & arma_m1, ARMAVectorType1 & arma_v1, ARMAVectorType2 & arma_v2)
{
  int retval = EXIT_SUCCESS;

  // sync data:
  std_v1 = std::vector<NumericT>(std_v1.size(), NumericT(2));
  std_v2 = std::vector<NumericT>(std_v2.size(), NumericT(3));
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());
  cuarma::copy(std_m1, arma_m1);

  // --------------------------------------------------------------------------
  std::cout << "Rank 1 update" << std::endl;

  for (std::size_t i=0; i<std_m1.size(); ++i)
   for (std::size_t j=0; j<std_m1[i].size(); ++j)
     std_m1[i][j] += std_v1[i] * std_v2[j];
  arma_m1 += cuarma::blas::outer_prod(arma_v1, arma_v2);
  if ( diff(std_m1, arma_m1) != 0 )
  {
    std::cout << "# Error at operation: rank 1 update" << std::endl;
    std::cout << "  diff: " << diff(std_m1, arma_m1) << std::endl;
    return EXIT_FAILURE;
  }



  // --------------------------------------------------------------------------
  std::cout << "Scaled rank 1 update - CPU Scalar" << std::endl;
  for (std::size_t i=0; i<std_m1.size(); ++i)
   for (std::size_t j=0; j<std_m1[i].size(); ++j)
     std_m1[i][j] += NumericT(4) * std_v1[i] * std_v2[j];
  arma_m1 += NumericT(2) * cuarma::blas::outer_prod(arma_v1, arma_v2);
  arma_m1 += cuarma::blas::outer_prod(arma_v1, arma_v2) * NumericT(2);  //check proper compilation
  if ( diff(std_m1, arma_m1) != 0 )
  {
    std::cout << "# Error at operation: scaled rank 1 update - CPU Scalar" << std::endl;
    std::cout << "  diff: " << diff(std_m1, arma_m1) << std::endl;
    return EXIT_FAILURE;
  }

    // --------------------------------------------------------------------------
  std::cout << "Scaled rank 1 update - GPU Scalar" << std::endl;
  for (std::size_t i=0; i<std_m1.size(); ++i)
   for (std::size_t j=0; j<std_m1[i].size(); ++j)
     std_m1[i][j] += NumericT(4) * std_v1[i] * std_v2[j];
  arma_m1 += cuarma::scalar<NumericT>(2) * cuarma::blas::outer_prod(arma_v1, arma_v2);
  arma_m1 += cuarma::blas::outer_prod(arma_v1, arma_v2) * cuarma::scalar<NumericT>(2);  //check proper compilation
  if ( diff(std_m1, arma_m1) != 0 )
  {
    std::cout << "# Error at operation: scaled rank 1 update - GPU Scalar" << std::endl;
    std::cout << "  diff: " << diff(std_m1, arma_m1) << std::endl;
    return EXIT_FAILURE;
  }

  //reset arma_matrix:
  cuarma::copy(std_m1, arma_m1);

  // --------------------------------------------------------------------------
  std::cout << "Matrix-Vector product" << std::endl;
  for (std::size_t i=0; i<std_m1.size(); ++i)
  {
   NumericT temp = 0;
   for (std::size_t j=0; j<std_m1[i].size(); ++j)
     temp += std_m1[i][j] * std_v2[j];
   std_v1[i] = temp;
  }
  std_v1 = cuarma::blas::prod(std_m1, std_v2);
  arma_v1   = cuarma::blas::prod(arma_m1, arma_v2);

  if ( diff(std_v1, arma_v1) != 0 )
  {
    std::cout << "# Error at operation: matrix-vector product" << std::endl;
    std::cout << "  diff: " << diff(std_v1, arma_v1) << std::endl;
    retval = EXIT_FAILURE;
  }
  // --------------------------------------------------------------------------
  std::cout << "Matrix-Vector product with scaled add" << std::endl;
  NumericT alpha = static_cast<NumericT>(2);
  NumericT beta = static_cast<NumericT>(3);
  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  for (std::size_t i=0; i<std_m1.size(); ++i)
  {
   NumericT temp = 0;
   for (std::size_t j=0; j<std_m1[i].size(); ++j)
     temp += std_m1[i][j] * std_v2[j];
   std_v1[i] = alpha * temp + beta * std_v1[i];
  }
  arma_v1   = alpha * cuarma::blas::prod(arma_m1, arma_v2) + beta * arma_v1;

  if ( diff(std_v1, arma_v1) != 0 )
  {
    std::cout << "# Error at operation: matrix-vector product with scaled additions" << std::endl;
    std::cout << "  diff: " << diff(std_v1, arma_v1) << std::endl;
    retval = EXIT_FAILURE;
  }
  // --------------------------------------------------------------------------

  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());

  std::cout << "Transposed Matrix-Vector product" << std::endl;
  for (std::size_t i=0; i<std_m1[0].size(); ++i)
  {
   NumericT temp = 0;
   for (std::size_t j=0; j<std_m1.size(); ++j)
     temp += std_m1[j][i] * std_v1[j];
   std_v2[i] = alpha * temp;
  }
  arma_v2   = alpha * cuarma::blas::prod(trans(arma_m1), arma_v1);

  if ( diff(std_v2, arma_v2) != 0 )
  {
    std::cout << "# Error at operation: transposed matrix-vector product" << std::endl;
    std::cout << "  diff: " << diff(std_v2, arma_v2) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Transposed Matrix-Vector product with scaled add" << std::endl;
  for (std::size_t i=0; i<std_m1[0].size(); ++i)
  {
   NumericT temp = 0;
   for (std::size_t j=0; j<std_m1.size(); ++j)
     temp += std_m1[j][i] * std_v1[j];
   std_v2[i] = alpha * temp + beta * std_v2[i];
  }
  arma_v2   = alpha * cuarma::blas::prod(trans(arma_m1), arma_v1) + beta * arma_v2;

  if ( diff(std_v2, arma_v2) != 0 )
  {
    std::cout << "# Error at operation: transposed matrix-vector product with scaled additions" << std::endl;
    std::cout << "  diff: " << diff(std_v2, arma_v2) << std::endl;
    retval = EXIT_FAILURE;
  }
  // --------------------------------------------------------------------------


  std::cout << "Row sum with matrix" << std::endl;
  for (std::size_t i=0; i<std_m1.size(); ++i)
  {
   NumericT temp = 0;
   for (std::size_t j=0; j<std_m1[i].size(); ++j)
     temp += std_m1[i][j];
   std_v1[i] = temp;
  }
  arma_v1   = cuarma::blas::row_sum(arma_m1);

  if ( diff(std_v1, arma_v1) != 0 )
  {
    std::cout << "# Error at operation: row sum" << std::endl;
    std::cout << "  diff: " << diff(std_v1, arma_v1) << std::endl;
    retval = EXIT_FAILURE;
  }
  // --------------------------------------------------------------------------

  std::cout << "Row sum with matrix expression" << std::endl;
  for (std::size_t i=0; i<std_m1.size(); ++i)
  {
   NumericT temp = 0;
   for (std::size_t j=0; j<std_m1[i].size(); ++j)
     temp += std_m1[i][j] + std_m1[i][j];
   std_v1[i] = temp;
  }
  arma_v1   = cuarma::blas::row_sum(arma_m1 + arma_m1);

  if ( diff(std_v1, arma_v1) != 0 )
  {
    std::cout << "# Error at operation: row sum (with expression)" << std::endl;
    std::cout << "  diff: " << diff(std_v1, arma_v1) << std::endl;
    retval = EXIT_FAILURE;
  }
  // --------------------------------------------------------------------------

  std::cout << "Column sum with matrix" << std::endl;
  for (std::size_t i=0; i<std_m1[0].size(); ++i)
  {
   NumericT temp = 0;
   for (std::size_t j=0; j<std_m1.size(); ++j)
     temp += std_m1[j][i];
   std_v2[i] = temp;
  }
  arma_v2   = cuarma::blas::column_sum(arma_m1);

  if ( diff(std_v2, arma_v2) != 0 )
  {
    std::cout << "# Error at operation: column sum" << std::endl;
    std::cout << "  diff: " << diff(std_v2, arma_v2) << std::endl;
    retval = EXIT_FAILURE;
  }
  // --------------------------------------------------------------------------

  std::cout << "Column sum with matrix expression" << std::endl;
  for (std::size_t i=0; i<std_m1[0].size(); ++i)
  {
   NumericT temp = 0;
   for (std::size_t j=0; j<std_m1.size(); ++j)
     temp += std_m1[j][i] + std_m1[j][i];
   std_v2[i] = temp;
  }
  arma_v2   = cuarma::blas::column_sum(arma_m1 + arma_m1);

  if ( diff(std_v2, arma_v2) != 0 )
  {
    std::cout << "# Error at operation: column sum (with expression)" << std::endl;
    std::cout << "  diff: " << diff(std_v2, arma_v2) << std::endl;
    retval = EXIT_FAILURE;
  }
  // --------------------------------------------------------------------------

  return retval;
}


template< typename NumericT, typename F>
int test()
{
  int retval = EXIT_SUCCESS;

  std::size_t num_rows = 141;
  std::size_t num_cols = 103;

  // --------------------------------------------------------------------------
  std::vector<NumericT> std_v1(num_rows);
  for (std::size_t i = 0; i < std_v1.size(); ++i)
   std_v1[i] = NumericT(i);
  std::vector<NumericT> std_v2 = std::vector<NumericT>(num_cols, NumericT(3));


  std::vector<std::vector<NumericT> > std_m1(std_v1.size(), std::vector<NumericT>(std_v2.size()));
  std::vector<std::vector<NumericT> > std_m2(std_v1.size(), std::vector<NumericT>(std_v1.size()));


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);


  for (std::size_t i = 0; i < std_m2.size(); ++i)
  for (std::size_t j = 0; j < std_m2[i].size(); ++j)
     std_m2[i][j] = NumericT(j - i*j + i);


  cuarma::vector<NumericT> arma_v1_native(std_v1.size());
  cuarma::vector<NumericT> arma_v1_large(4 * std_v1.size());
  cuarma::vector_range< cuarma::vector<NumericT> > arma_v1_range(arma_v1_large, cuarma::range(3, std_v1.size() + 3));
  cuarma::vector_slice< cuarma::vector<NumericT> > arma_v1_slice(arma_v1_large, cuarma::slice(2, 3, std_v1.size()));

  cuarma::vector<NumericT> arma_v2_native(std_v2.size());
  cuarma::vector<NumericT> arma_v2_large(4 * std_v2.size());
  cuarma::vector_range< cuarma::vector<NumericT> > arma_v2_range(arma_v2_large, cuarma::range(8, std_v2.size() + 8));
  cuarma::vector_slice< cuarma::vector<NumericT> > arma_v2_slice(arma_v2_large, cuarma::slice(6, 2, std_v2.size()));

  cuarma::matrix<NumericT, F> arma_m1_native(std_m1.size(), std_m1[0].size());
  cuarma::matrix<NumericT, F> arma_m1_large(4 * std_m1.size(), 4 * std_m1[0].size());
  cuarma::matrix_range< cuarma::matrix<NumericT, F> > arma_m1_range(arma_m1_large,
                                                                       cuarma::range(8, std_m1.size() + 8),
                                                                       cuarma::range(std_m1[0].size(), 2 * std_m1[0].size()) );
  cuarma::matrix_slice< cuarma::matrix<NumericT, F> > arma_m1_slice(arma_m1_large,
                                                                       cuarma::slice(6, 2, std_m1.size()),
                                                                       cuarma::slice(std_m1[0].size(), 2, std_m1[0].size()) );


  //
  // Run a bunch of tests for rank-1-updates, matrix-vector products
  //
  std::cout << "------------ Testing rank-1-updates and matrix-vector products ------------------" << std::endl;

  std::cout << "* m = full, v1 = full, v2 = full" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_native, arma_v1_native, arma_v2_native);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;

  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = full, v1 = full, v2 = range" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_native, arma_v1_native, arma_v2_range);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = full, v1 = full, v2 = slice" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_native, arma_v1_native, arma_v2_slice);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  // v1 = range


  std::cout << "* m = full, v1 = range, v2 = full" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_native, arma_v1_range, arma_v2_native);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = full, v1 = range, v2 = range" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_native, arma_v1_range, arma_v2_range);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = full, v1 = range, v2 = slice" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_native, arma_v1_range, arma_v2_slice);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);


  // v1 = slice

  std::cout << "* m = full, v1 = slice, v2 = full" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_native, arma_v1_slice, arma_v2_native);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = full, v1 = slice, v2 = range" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_native, arma_v1_slice, arma_v2_range);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = full, v1 = slice, v2 = slice" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_native, arma_v1_slice, arma_v2_slice);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  ///////////////////////////// matrix_range

  std::cout << "* m = range, v1 = full, v2 = full" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_range, arma_v1_native, arma_v2_native);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = range, v1 = full, v2 = range" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_range, arma_v1_native, arma_v2_range);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = range, v1 = full, v2 = slice" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_range, arma_v1_native, arma_v2_slice);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  // v1 = range


  std::cout << "* m = range, v1 = range, v2 = full" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_range, arma_v1_range, arma_v2_native);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = range, v1 = range, v2 = range" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_range, arma_v1_range, arma_v2_range);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = range, v1 = range, v2 = slice" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_range, arma_v1_range, arma_v2_slice);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);


  // v1 = slice

  std::cout << "* m = range, v1 = slice, v2 = full" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_range, arma_v1_slice, arma_v2_native);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = range, v1 = slice, v2 = range" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_range, arma_v1_slice, arma_v2_range);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = range, v1 = slice, v2 = slice" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_range, arma_v1_slice, arma_v2_slice);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  ///////////////////////////// matrix_slice

  std::cout << "* m = slice, v1 = full, v2 = full" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_slice, arma_v1_native, arma_v2_native);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = slice, v1 = full, v2 = range" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_slice, arma_v1_native, arma_v2_range);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = slice, v1 = full, v2 = slice" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_slice, arma_v1_native, arma_v2_slice);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  // v1 = range


  std::cout << "* m = slice, v1 = range, v2 = full" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_slice, arma_v1_range, arma_v2_native);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = slice, v1 = range, v2 = range" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_slice, arma_v1_range, arma_v2_range);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = slice, v1 = range, v2 = slice" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_slice, arma_v1_range, arma_v2_slice);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;



  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  // v1 = slice

  std::cout << "* m = slice, v1 = slice, v2 = full" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_slice, arma_v1_slice, arma_v2_native);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;

  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);


  std::cout << "* m = slice, v1 = slice, v2 = range" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2,
                                    arma_m1_slice, arma_v1_slice, arma_v2_range);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  for (std::size_t i = 0; i < std_m1.size(); ++i)
  for (std::size_t j = 0; j < std_m1[i].size(); ++j)
    std_m1[i][j] = NumericT(i+j);

  std::cout << "* m = slice, v1 = slice, v2 = slice" << std::endl;
  retval = test_prod_rank1<NumericT>(std_m1, std_v1, std_v2, arma_m1_slice, arma_v1_slice, arma_v2_slice);
  if (retval == EXIT_FAILURE)
  {
   std::cout << " --- FAILED! ---" << std::endl;
   return retval;
  }
  else
   std::cout << " --- PASSED ---" << std::endl;


  return retval;
}
//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Matrix" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    typedef int NumericT;
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  numeric: int" << std::endl;
    std::cout << "  layout: row-major" << std::endl;
    retval = test<NumericT, cuarma::row_major>();
    if ( retval == EXIT_SUCCESS )
       std::cout << "# Test passed" << std::endl;
    else
       return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    typedef int NumericT;
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  numeric: int" << std::endl;
    std::cout << "  layout: column-major" << std::endl;
    retval = test<NumericT, cuarma::column_major>();
    if ( retval == EXIT_SUCCESS )
       std::cout << "# Test passed" << std::endl;
    else
       return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;


  {
    typedef long NumericT;
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  numeric: long" << std::endl;
    std::cout << "  layout: row-major" << std::endl;
    retval = test<NumericT, cuarma::row_major>();
    if ( retval == EXIT_SUCCESS )
       std::cout << "# Test passed" << std::endl;
    else
      return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    typedef long NumericT;
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  numeric: long" << std::endl;
    std::cout << "  layout: column-major" << std::endl;
    retval = test<NumericT, cuarma::column_major>();
    if ( retval == EXIT_SUCCESS )
       std::cout << "# Test passed" << std::endl;
    else
      return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;


  return retval;
}
