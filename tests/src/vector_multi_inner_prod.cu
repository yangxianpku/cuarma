/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   vector_multi_inner_prod.cu
 *   @coding UTF-8
 *   @brief  Tests the performance of multiple inner products with a common vector.
 *   @brief  测试：多向量内积运算性能测试
 */

#include <iostream>
#include <iomanip>
#include <iterator>

#include "head_define.h"

#include "cuarma/vector.hpp"
#include "cuarma/vector_proxy.hpp"
#include "cuarma/blas/inner_prod.hpp"
#include "cuarma/blas/norm_1.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/norm_inf.hpp"
#include "cuarma/tools/random.hpp"

template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
   cuarma::backend::finish();
   if (s1 != s2)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}

template<typename ScalarType>
ScalarType diff(ScalarType const & s1, cuarma::scalar<ScalarType> const & s2)
{
   cuarma::backend::finish();
   if (s1 != s2)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}

template<typename ScalarType>
ScalarType diff(ScalarType const & s1, cuarma::entry_proxy<ScalarType> const & s2)
{
   cuarma::backend::finish();
   if (s1 != s2)
      return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
   return 0;
}


template<typename ScalarType, typename cuarmaVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, cuarmaVectorType const & arma_vec)
{
   std::vector<ScalarType> v2_cpu(arma_vec.size());
   cuarma::backend::finish();
   cuarma::copy(arma_vec, v2_cpu);

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
         v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   ScalarType norm_inf = 0;
   for (std::size_t i=0; i<v2_cpu.size(); ++i)
     norm_inf = std::max<ScalarType>(norm_inf, std::fabs(v2_cpu[i]));

   return norm_inf;
}



template<typename T1, typename T2>
int check(T1 const & t1, T2 const & t2, double epsilon)
{
  int retval = EXIT_SUCCESS;

  double temp = std::fabs(diff(t1, t2));
  if (temp > epsilon)
  {
    std::cout << "# Error! Relative difference: " << temp << std::endl;
    retval = EXIT_FAILURE;
  }
  return retval;
}


template< typename NumericT, typename Epsilon,
          typename STLVectorType1,      typename STLVectorType2,      typename STLVectorType3,      typename STLVectorType4,
          typename cuarmaVectorType1, typename cuarmaVectorType2, typename cuarmaVectorType3, typename cuarmaVectorType4 >
int test(Epsilon const& epsilon,
         STLVectorType1      & std_v1, STLVectorType2      & std_v2, STLVectorType3      & std_v3, STLVectorType4      & std_v4,
         cuarmaVectorType1 & arma_v1, cuarmaVectorType2 & arma_v2, cuarmaVectorType3 & arma_v3, cuarmaVectorType4 & arma_v4)
{
  int retval = EXIT_SUCCESS;

  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  for (std::size_t i=0; i<std_v1.size(); ++i)
  {
    std_v1[i] = NumericT(1.0) + randomNumber();
    std_v2[i] = NumericT(1.0) + randomNumber();
    std_v3[i] = NumericT(1.0) + randomNumber();
    std_v4[i] = NumericT(1.0) + randomNumber();
  }

  cuarma::copy(std_v1.begin(), std_v1.end(), arma_v1.begin());  //resync
  cuarma::copy(std_v2.begin(), std_v2.end(), arma_v2.begin());
  cuarma::copy(std_v3.begin(), std_v3.end(), arma_v3.begin());
  cuarma::copy(std_v4.begin(), std_v4.end(), arma_v4.begin());

  std::cout << "Checking for successful copy..." << std::endl;
  if (check(std_v1, arma_v1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v2, arma_v2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v3, arma_v3, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_v4, arma_v4, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::vector<NumericT> ref_result(40, 0.0);
  cuarma::vector<NumericT> result = cuarma::scalar_vector<NumericT>(40, 0.0);

  std::cout << "Testing inner_prod with two vectors..." << std::endl;
  ref_result[2] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[2] += std_v1[i] * std_v1[i];
  ref_result[5] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5] += std_v1[i] * std_v2[i];
  cuarma::project(result, cuarma::slice(2, 3, 2)) = cuarma::blas::inner_prod(arma_v1, cuarma::tie(arma_v1, arma_v2));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[3] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3] += std_v1[i] * std_v3[i];
  ref_result[7] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[7] += std_v1[i] * std_v4[i];
  cuarma::project(result, cuarma::slice(3, 4, 2)) = cuarma::blas::inner_prod(arma_v1, cuarma::tie(arma_v3, arma_v4));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }


  std::cout << "Testing inner_prod with three vectors..." << std::endl;
  ref_result[1] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[1] += std_v1[i] * std_v1[i];
  ref_result[3] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3] += std_v1[i] * std_v2[i];
  ref_result[5] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5] += std_v1[i] * std_v3[i];
  cuarma::project(result, cuarma::slice(1, 2, 3)) = cuarma::blas::inner_prod(arma_v1, cuarma::tie(arma_v1, arma_v2, arma_v3));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[2]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[2]  += std_v1[i] * std_v3[i];
  ref_result[6]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[6]  += std_v1[i] * std_v2[i];
  ref_result[10] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[10] += std_v1[i] * std_v4[i];
  cuarma::project(result, cuarma::slice(2, 4, 3)) = cuarma::blas::inner_prod(arma_v1, cuarma::tie(arma_v3, arma_v2, arma_v4));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing inner_prod with four vectors..." << std::endl;
  ref_result[4] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[4] += std_v1[i] * std_v1[i];
  ref_result[5] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5] += std_v1[i] * std_v2[i];
  ref_result[6] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[6] += std_v1[i] * std_v3[i];
  ref_result[7] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[7] += std_v1[i] * std_v4[i];
  cuarma::project(result, cuarma::slice(4, 1, 4)) = cuarma::blas::inner_prod(arma_v1, cuarma::tie(arma_v1, arma_v2, arma_v3, arma_v4));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[3]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3]  += std_v1[i] * std_v3[i];
  ref_result[6]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[6]  += std_v1[i] * std_v2[i];
  ref_result[9]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[9]  += std_v1[i] * std_v4[i];
  ref_result[12] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[12] += std_v1[i] * std_v1[i];
  cuarma::project(result, cuarma::slice(3, 3, 4)) = cuarma::blas::inner_prod(arma_v1, cuarma::tie(arma_v3, arma_v2, arma_v4, arma_v1));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Testing inner_prod with five vectors..." << std::endl;
  ref_result[1] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[1] += std_v1[i] * std_v1[i];
  ref_result[3] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3] += std_v1[i] * std_v2[i];
  ref_result[5] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5] += std_v1[i] * std_v3[i];
  ref_result[7] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[7] += std_v1[i] * std_v4[i];
  ref_result[9] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[9] += std_v1[i] * std_v2[i];
  cuarma::project(result, cuarma::slice(1, 2, 5)) = cuarma::blas::inner_prod(arma_v1, cuarma::tie(arma_v1, arma_v2, arma_v3, arma_v4, arma_v2));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[2]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[2]  += std_v1[i] * std_v3[i];
  ref_result[4]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[4]  += std_v1[i] * std_v2[i];
  ref_result[6]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[6]  += std_v1[i] * std_v4[i];
  ref_result[8]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[8]  += std_v1[i] * std_v1[i];
  ref_result[10] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[10] += std_v1[i] * std_v2[i];
  cuarma::project(result, cuarma::slice(2, 2, 5)) = cuarma::blas::inner_prod(arma_v1, cuarma::tie(arma_v3, arma_v2, arma_v4, arma_v1, arma_v2));
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }


  std::cout << "Testing inner_prod with eight vectors..." << std::endl;
  ref_result[1]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[1]  += std_v1[i] * std_v1[i];
  ref_result[5]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5]  += std_v1[i] * std_v2[i];
  ref_result[9]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[9]  += std_v1[i] * std_v3[i];
  ref_result[13] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[13] += std_v1[i] * std_v4[i];
  ref_result[17] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[17] += std_v1[i] * std_v3[i];
  ref_result[21] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[21] += std_v1[i] * std_v2[i];
  ref_result[25] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[25] += std_v1[i] * std_v1[i];
  ref_result[29] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[29] += std_v1[i] * std_v2[i];
  std::vector<cuarma::vector_base<NumericT> const *> vecs1(8);
  vecs1[0] = &arma_v1;
  vecs1[1] = &arma_v2;
  vecs1[2] = &arma_v3;
  vecs1[3] = &arma_v4;
  vecs1[4] = &arma_v3;
  vecs1[5] = &arma_v2;
  vecs1[6] = &arma_v1;
  vecs1[7] = &arma_v2;
  cuarma::vector_tuple<NumericT> tuple1(vecs1);
  cuarma::project(result, cuarma::slice(1, 4, 8)) = cuarma::blas::inner_prod(arma_v1, tuple1);
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }

  ref_result[3]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[3]  += std_v1[i] * std_v2[i];
  ref_result[5]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[5]  += std_v1[i] * std_v4[i];
  ref_result[7]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[7]  += std_v1[i] * std_v1[i];
  ref_result[9]  = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[9]  += std_v1[i] * std_v2[i];
  ref_result[11] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[11] += std_v1[i] * std_v2[i];
  ref_result[13] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[13] += std_v1[i] * std_v1[i];
  ref_result[15] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[15] += std_v1[i] * std_v4[i];
  ref_result[17] = 0; for (std::size_t i=0; i<std_v1.size(); ++i) ref_result[17] += std_v1[i] * std_v2[i];
  std::vector<cuarma::vector_base<NumericT> const *> vecs2(8);
  vecs2[0] = &arma_v2;
  vecs2[1] = &arma_v4;
  vecs2[2] = &arma_v1;
  vecs2[3] = &arma_v2;
  vecs2[4] = &arma_v2;
  vecs2[5] = &arma_v1;
  vecs2[6] = &arma_v4;
  vecs2[7] = &arma_v2;
  cuarma::vector_tuple<NumericT> tuple2(vecs2);
  cuarma::project(result, cuarma::slice(3, 2, 8)) = cuarma::blas::inner_prod(arma_v1, tuple2);
  if (check(ref_result, result, epsilon) != EXIT_SUCCESS)
  {
    std::copy(ref_result.begin(), ref_result.end(), std::ostream_iterator<NumericT>(std::cout, " ")); std::cout << std::endl;
    std::cout << result << std::endl;
    return EXIT_FAILURE;
  }


  // --------------------------------------------------------------------------
  return retval;
}


template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  int retval = EXIT_SUCCESS;
  std::size_t size = 8 * 1337;

  std::cout << "Running tests for vector of size " << size << std::endl;

  //
  // Set up STL objects
  //
  std::vector<NumericT> std_full_vec1(size);
  std::vector<NumericT> std_full_vec2(std_full_vec1.size());

  for (std::size_t i=0; i<std_full_vec1.size(); ++i)
  {
    std_full_vec1[i] = NumericT(1.0) + randomNumber();
    std_full_vec2[i] = NumericT(1.0) + randomNumber();
  }

  std::vector<NumericT> std_slice_vec1(std_full_vec1.size() / 8); for (std::size_t i=0; i<std_slice_vec1.size(); ++i) std_slice_vec1[i] = std_full_vec1[    std_full_vec1.size() / 8 + i * 3];
  std::vector<NumericT> std_slice_vec2(std_full_vec2.size() / 8); for (std::size_t i=0; i<std_slice_vec2.size(); ++i) std_slice_vec2[i] = std_full_vec2[2 * std_full_vec2.size() / 8 + i * 1];
  std::vector<NumericT> std_slice_vec3(std_full_vec1.size() / 8); for (std::size_t i=0; i<std_slice_vec3.size(); ++i) std_slice_vec3[i] = std_full_vec1[4 * std_full_vec1.size() / 8 + i * 2];
  std::vector<NumericT> std_slice_vec4(std_full_vec2.size() / 8); for (std::size_t i=0; i<std_slice_vec4.size(); ++i) std_slice_vec4[i] = std_full_vec2[3 * std_full_vec2.size() / 8 + i * 4];

  //
  // Set up cuarma objects
  //
  cuarma::vector<NumericT> arma_full_vec1(std_full_vec1.size());
  cuarma::vector<NumericT> arma_full_vec2(std_full_vec2.size());

  cuarma::fast_copy(std_full_vec1.begin(), std_full_vec1.end(), arma_full_vec1.begin());
  cuarma::copy     (std_full_vec2.begin(), std_full_vec2.end(), arma_full_vec2.begin());

  cuarma::slice arma_s1(    arma_full_vec1.size() / 8, 3, arma_full_vec1.size() / 8);
  cuarma::slice arma_s2(2 * arma_full_vec2.size() / 8, 1, arma_full_vec2.size() / 8);
  cuarma::slice arma_s3(4 * arma_full_vec1.size() / 8, 2, arma_full_vec1.size() / 8);
  cuarma::slice arma_s4(3 * arma_full_vec2.size() / 8, 4, arma_full_vec2.size() / 8);
  cuarma::vector_slice< cuarma::vector<NumericT> > arma_slice_vec1(arma_full_vec1, arma_s1);
  cuarma::vector_slice< cuarma::vector<NumericT> > arma_slice_vec2(arma_full_vec2, arma_s2);
  cuarma::vector_slice< cuarma::vector<NumericT> > arma_slice_vec3(arma_full_vec1, arma_s3);
  cuarma::vector_slice< cuarma::vector<NumericT> > arma_slice_vec4(arma_full_vec2, arma_s4);

  cuarma::vector<NumericT> arma_short_vec1(arma_slice_vec1);
  cuarma::vector<NumericT> arma_short_vec2 = arma_slice_vec2;
  cuarma::vector<NumericT> arma_short_vec3 = arma_slice_vec2 + arma_slice_vec1;
  cuarma::vector<NumericT> arma_short_vec4 = arma_short_vec1 + arma_slice_vec2;

  std::vector<NumericT> std_short_vec1(std_slice_vec1);
  std::vector<NumericT> std_short_vec2(std_slice_vec2);
  std::vector<NumericT> std_short_vec3(std_slice_vec2.size()); for (std::size_t i=0; i<std_short_vec3.size(); ++i) std_short_vec3[i] = std_slice_vec2[i] + std_slice_vec1[i];
  std::vector<NumericT> std_short_vec4(std_slice_vec2.size()); for (std::size_t i=0; i<std_short_vec4.size(); ++i) std_short_vec4[i] = std_slice_vec1[i] + std_slice_vec2[i];

  std::cout << "Testing creation of vectors from slice..." << std::endl;
  if (check(std_short_vec1, arma_short_vec1, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec2, arma_short_vec2, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec3, arma_short_vec3, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (check(std_short_vec4, arma_short_vec4, epsilon) != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** [vector|vector|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_short_vec2, std_short_vec2, std_short_vec2,
                          arma_short_vec1, arma_short_vec2, arma_short_vec3, arma_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|vector|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_short_vec2, std_short_vec2, std_slice_vec2,
                          arma_short_vec1, arma_short_vec2, arma_short_vec3, arma_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|vector|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_short_vec2, std_slice_vec2, std_short_vec2,
                          arma_short_vec1, arma_short_vec2, arma_slice_vec3, arma_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|vector|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_short_vec2, std_slice_vec2, std_slice_vec2,
                          arma_short_vec1, arma_short_vec2, arma_slice_vec3, arma_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_slice_vec2, std_short_vec2, std_short_vec2,
                          arma_short_vec1, arma_slice_vec2, arma_short_vec3, arma_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_slice_vec2, std_short_vec2, std_slice_vec2,
                          arma_short_vec1, arma_slice_vec2, arma_short_vec3, arma_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_slice_vec2, std_slice_vec2, std_short_vec2,
                          arma_short_vec1, arma_slice_vec2, arma_slice_vec3, arma_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [vector|slice|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_short_vec1, std_slice_vec2, std_slice_vec2, std_slice_vec2,
                          arma_short_vec1, arma_slice_vec2, arma_slice_vec3, arma_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;


  //////////////////


  std::cout << " ** [slice|vector|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_short_vec2, std_short_vec2, std_short_vec2,
                          arma_slice_vec1, arma_short_vec2, arma_short_vec3, arma_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|vector|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_short_vec2, std_short_vec2, std_slice_vec2,
                          arma_slice_vec1, arma_short_vec2, arma_short_vec3, arma_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|vector|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_short_vec2, std_slice_vec2, std_short_vec2,
                          arma_slice_vec1, arma_short_vec2, arma_slice_vec3, arma_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|vector|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_short_vec2, std_slice_vec2, std_slice_vec2,
                          arma_slice_vec1, arma_short_vec2, arma_slice_vec3, arma_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|vector|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_slice_vec2, std_short_vec2, std_short_vec2,
                          arma_slice_vec1, arma_slice_vec2, arma_short_vec3, arma_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|vector|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_slice_vec2, std_short_vec2, std_slice_vec2,
                          arma_slice_vec1, arma_slice_vec2, arma_short_vec3, arma_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|slice|vector] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_slice_vec2, std_slice_vec2, std_short_vec2,
                          arma_slice_vec1, arma_slice_vec2, arma_slice_vec3, arma_short_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** [slice|slice|slice|slice] **" << std::endl;
  retval = test<NumericT>(epsilon,
                          std_slice_vec1, std_slice_vec2, std_slice_vec2, std_slice_vec2,
                          arma_slice_vec1, arma_slice_vec2, arma_slice_vec3, arma_slice_vec4);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}



//
// -------------------------------------------------------------
//
int main()
{
   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "## Test :: Vector multiple inner products" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;

   int retval = EXIT_SUCCESS;

   std::cout << std::endl;
   std::cout << "----------------------------------------------" << std::endl;
   std::cout << std::endl;
   {
      typedef float NumericT;
      NumericT epsilon = static_cast<NumericT>(1.0E-4);
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
         NumericT epsilon = 1.0E-12;
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
      std::cout << "----------------------------------------------" << std::endl;
      std::cout << std::endl;
   }

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;


   return retval;
}
