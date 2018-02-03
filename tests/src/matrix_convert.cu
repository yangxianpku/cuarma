/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   matrix_convert.cu
 *   @coding UTF-8
 *   @brief  Tests conversion between matrices with different numeric type
 *   @brief  测试：测试矩阵和不同数值类型之间的转换
 */


#include <iostream>
#include <iomanip>
#include <vector>

#include "cuarma/backend/memory.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/matrix_proxy.hpp"


template<typename NumericT, typename MatrixT>
int check(std::vector<NumericT> const & std_dest,
          std::size_t start1, std::size_t inc1, std::size_t size1,
          std::size_t start2, std::size_t inc2, std::size_t size2, std::size_t internal_size2,
          MatrixT const & arma_dest)
{
  cuarma::backend::typesafe_host_array<NumericT> tempmat(arma_dest.handle(), arma_dest.internal_size());
  cuarma::backend::memory_read(arma_dest.handle(), 0, tempmat.raw_size(), reinterpret_cast<NumericT*>(tempmat.get()));

  for (std::size_t i=0; i < size1; ++i)
  {
    for (std::size_t j=0; j < size2; ++j)
    {
      NumericT value_std  = std_dest[(i*inc1 + start1) * internal_size2 + (j*inc2 + start2)];
      NumericT value_dest = arma_dest.row_major() ? tempmat[(i * arma_dest.stride1() + arma_dest.start1()) * arma_dest.internal_size2() + (j * arma_dest.stride2() + arma_dest.start2())]
                                                 : tempmat[(i * arma_dest.stride1() + arma_dest.start1())                             + (j * arma_dest.stride2() + arma_dest.start2()) * arma_dest.internal_size1()];

      if (value_std < value_dest || value_std > value_dest)
      {
        std::cerr << "Failure at row " << i << ", col " << j << ": STL value " << value_std << ", cuarma value " << value_dest << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  return EXIT_SUCCESS;
}

// -------------------------------------------------------------
template<typename STLVectorT1, typename STLVectorT2, typename cuarmaVectorT1, typename cuarmaVectorT2 >
int test(STLVectorT1 & std_src,  std::size_t start1_src,  std::size_t inc1_src,  std::size_t size1_src,  std::size_t start2_src,  std::size_t inc2_src,  std::size_t size2_src,  std::size_t internal_size2_src,
         STLVectorT2 & std_dest, std::size_t start1_dest, std::size_t inc1_dest, std::size_t size1_dest, std::size_t start2_dest, std::size_t inc2_dest, std::size_t size2_dest, std::size_t internal_size2_dest,
         cuarmaVectorT1 const & arma_src, cuarmaVectorT2 & arma_dest)
{
  assert(size1_src       == size1_dest       && bool("Size1 mismatch for STL matrices"));
  assert(size2_src       == size2_dest       && bool("Size2 mismatch for STL matrices"));
  assert(arma_src.size1() == arma_dest.size1() && bool("Size1 mismatch for cuarma matrices"));
  assert(arma_src.size2() == arma_dest.size2() && bool("Size2 mismatch for cuarma matrices"));
  assert(size1_src       == arma_src.size1()  && bool("Size1 mismatch for STL and cuarma matrices"));
  assert(size2_src       == arma_src.size2()  && bool("Size2 mismatch for STL and cuarma matrices"));

  typedef typename STLVectorT2::value_type  DestNumericT;

  for (std::size_t i=0; i<size1_src; ++i)
    for (std::size_t j=0; j<size2_src; ++j)
      std_dest[(start1_dest + i * inc1_dest) * internal_size2_dest + (start2_dest + j * inc2_dest)] = static_cast<DestNumericT>(std_src[(start1_src + i * inc1_src) * internal_size2_src + (start2_src + j * inc2_src)]);

  arma_dest = arma_src; // here is the conversion taking place

  if (check(std_dest, start1_dest, inc1_dest, size1_dest, start2_dest, inc2_dest, size2_dest, internal_size2_dest, arma_dest) != EXIT_SUCCESS)
    return EXIT_FAILURE;

  if (arma_src.row_major())
  {
    cuarma::matrix<DestNumericT> A(arma_src);
    if (check(std_dest, start1_dest, inc1_dest, size1_dest, start2_dest, inc2_dest, size2_dest, internal_size2_dest, A) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }
  else
  {
    cuarma::matrix<DestNumericT, cuarma::column_major> A(arma_src);
    if (check(std_dest, start1_dest, inc1_dest, size1_dest, start2_dest, inc2_dest, size2_dest, internal_size2_dest, A) != EXIT_SUCCESS)
      return EXIT_FAILURE;
  }

  // --------------------------------------------------------------------------
  return EXIT_SUCCESS;
}

inline std::string type_string(unsigned int)    { return "unsigned int"; }
inline std::string type_string(int)             { return "int"; }
inline std::string type_string(unsigned long)   { return "unsigned long"; }
inline std::string type_string(long)            { return "long"; }
inline std::string type_string(float)           { return "float"; }
inline std::string type_string(double)          { return "double"; }

template<typename LayoutT, typename FromNumericT, typename ToNumericT>
int test()
{
  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "Conversion test from " << type_string(FromNumericT()) << " to " << type_string(ToNumericT()) << std::endl;
  std::cout << std::endl;

  std::size_t full_size1  = 578;
  std::size_t small_size1 = full_size1 / 4;

  std::size_t full_size2  = 687;
  std::size_t small_size2 = full_size2 / 4;

  //
  // Set up STL objects
  //
  std::vector<FromNumericT>               std_src(full_size1 * full_size2);
  std::vector<std::vector<FromNumericT> > std_src2(full_size1, std::vector<FromNumericT>(full_size2));
  std::vector<std::vector<FromNumericT> > std_src_small(small_size1, std::vector<FromNumericT>(small_size2));
  std::vector<ToNumericT> std_dest(std_src.size());

  for (std::size_t i=0; i<full_size1; ++i)
    for (std::size_t j=0; j<full_size2; ++j)
    {
      std_src[i * full_size2 + j]  = FromNumericT(1.0) + FromNumericT(i) + FromNumericT(j);
      std_src2[i][j]  = FromNumericT(1.0) + FromNumericT(i) + FromNumericT(j);
      if (i < small_size1 && j < small_size2)
        std_src_small[i][j]  = FromNumericT(1.0) + FromNumericT(i) + FromNumericT(j);
    }

  //
  // Set up cuarma objects
  //
  cuarma::matrix<FromNumericT, LayoutT> arma_src(full_size1, full_size2);
  cuarma::matrix<ToNumericT,   LayoutT> arma_dest(full_size1, full_size2);

  cuarma::copy(std_src2, arma_src);

  cuarma::matrix<FromNumericT, LayoutT> arma_src_small(small_size1, small_size2);
  cuarma::copy(std_src_small, arma_src_small);
  cuarma::matrix<ToNumericT, LayoutT> arma_dest_small(small_size1, small_size2);

  std::size_t r11_start = 1 + full_size1 / 4;
  std::size_t r11_stop  = r11_start + small_size1;
  cuarma::range arma_r11(r11_start, r11_stop);

  std::size_t r12_start = 2 * full_size1 / 4;
  std::size_t r12_stop  = r12_start + small_size1;
  cuarma::range arma_r12(r12_start, r12_stop);

  std::size_t r21_start = 2 * full_size2 / 4;
  std::size_t r21_stop  = r21_start + small_size2;
  cuarma::range arma_r21(r21_start, r21_stop);

  std::size_t r22_start = 1 + full_size2 / 4;
  std::size_t r22_stop  = r22_start + small_size2;
  cuarma::range arma_r22(r22_start, r22_stop);

  cuarma::matrix_range< cuarma::matrix<FromNumericT, LayoutT> > arma_range_src(arma_src, arma_r11, arma_r21);
  cuarma::matrix_range< cuarma::matrix<ToNumericT, LayoutT> >   arma_range_dest(arma_dest, arma_r12, arma_r22);



  std::size_t s11_start = 1 + full_size1 / 5;
  std::size_t s11_inc   = 3;
  std::size_t s11_size  = small_size1;
  cuarma::slice arma_s11(s11_start, s11_inc, s11_size);

  std::size_t s12_start = 2 * full_size1 / 5;
  std::size_t s12_inc   = 2;
  std::size_t s12_size  = small_size1;
  cuarma::slice arma_s12(s12_start, s12_inc, s12_size);

  std::size_t s21_start = 1 + full_size2 / 5;
  std::size_t s21_inc   = 3;
  std::size_t s21_size  = small_size2;
  cuarma::slice arma_s21(s21_start, s21_inc, s21_size);

  std::size_t s22_start = 2 * full_size2 / 5;
  std::size_t s22_inc   = 2;
  std::size_t s22_size  = small_size2;
  cuarma::slice arma_s22(s22_start, s22_inc, s22_size);

  cuarma::matrix_slice< cuarma::matrix<FromNumericT, LayoutT> > arma_slice_src(arma_src, arma_s11, arma_s21);
  cuarma::matrix_slice< cuarma::matrix<ToNumericT, LayoutT> >   arma_slice_dest(arma_dest, arma_s12, arma_s22);

  //
  // Now start running tests for vectors, ranges and slices:
  //

  std::cout << " ** arma_src = matrix, arma_dest = matrix **" << std::endl;
  retval = test(std_src,  0, 1, full_size1, 0, 1, full_size2, full_size2,
                std_dest, 0, 1, full_size1, 0, 1, full_size2, full_size2,
                arma_src, arma_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_src = matrix, arma_dest = range **" << std::endl;
  retval = test(std_src,          0, 1, small_size1,                  0, 1,          small_size2, full_size2,
                std_dest, r12_start, 1, r12_stop - r12_start, r22_start, 1, r22_stop - r22_start, full_size2,
                arma_src_small, arma_range_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_src = matrix, arma_dest = slice **" << std::endl;
  retval = test(std_src,          0,       1, small_size1,         0,       1, small_size2, full_size2,
                std_dest, s12_start, s12_inc,    s12_size, s22_start, s22_inc,    s22_size, full_size2,
                arma_src_small, arma_slice_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** arma_src = range, arma_dest = matrix **" << std::endl;
  retval = test(std_src,  r11_start, 1, r11_stop - r11_start, r21_start, 1, r21_stop - r21_start, full_size2,
                std_dest,         0, 1,          small_size1,         0, 1,          small_size2, full_size2,
                arma_range_src, arma_dest_small);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_src = range, arma_dest = range **" << std::endl;
  retval = test(std_src,  r11_start, 1, r11_stop - r11_start, r21_start, 1, r21_stop - r21_start, full_size2,
                std_dest, r12_start, 1, r12_stop - r12_start, r22_start, 1, r22_stop - r22_start, full_size2,
                arma_range_src, arma_range_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_src = range, arma_dest = slice **" << std::endl;
  retval = test(std_src,  r11_start,       1, r11_stop - r11_start, r21_start,       1, r21_stop - r21_start, full_size2,
                std_dest, s12_start, s12_inc,             s12_size, s22_start, s22_inc,             s22_size, full_size2,
                arma_range_src, arma_slice_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  ///////

  std::cout << " ** arma_src = slice, arma_dest = matrix **" << std::endl;
  retval = test(std_src,  s11_start, s11_inc,    s11_size, s21_start, s21_inc,    s21_size, full_size2,
                std_dest,         0,       1, small_size1,         0,       1, small_size2, full_size2,
                arma_slice_src, arma_dest_small);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_src = slice, arma_dest = range **" << std::endl;
  retval = test(std_src,  s11_start, s11_inc,             s11_size, s21_start, s21_inc,             s21_size, full_size2,
                std_dest, r12_start,       1, r12_stop - r12_start, r22_start,       1, r22_stop - r22_start, full_size2,
                arma_slice_src, arma_range_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  std::cout << " ** arma_src = slice, arma_dest = slice **" << std::endl;
  retval = test(std_src,  s11_start, s11_inc, s11_size, s21_start, s21_inc, s21_size, full_size2,
                std_dest, s12_start, s12_inc, s12_size, s22_start, s22_inc, s22_size, full_size2,
                arma_slice_src, arma_slice_dest);
  if (retval != EXIT_SUCCESS)
    return EXIT_FAILURE;

  return EXIT_SUCCESS;
}


template<typename FromNumericT, typename ToNumericT>
int test()
{
  int retval = test<cuarma::row_major, FromNumericT, ToNumericT>();
  if (retval == EXIT_SUCCESS)
  {
    retval = test<cuarma::column_major, FromNumericT, ToNumericT>();
    if (retval != EXIT_SUCCESS)
      std::cerr << "Test failed for column-major!" << std::endl;
  }
  else
    std::cerr << "Test failed for row-major!" << std::endl;

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
  std::cout << "## Test :: Type conversion test for matrices  " << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  int retval = EXIT_SUCCESS;

  //
  // from int
  //
  retval = test<int, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<int, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  {
    retval = test<int, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }

  // from unsigned int
  retval = test<unsigned int, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned int, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  {
    retval = test<unsigned int, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }


  //
  // from long
  //
  retval = test<long, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<long, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  {
    retval = test<long, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }

  // from unsigned long
  retval = test<unsigned long, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<unsigned long, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  {
    retval = test<unsigned long, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }

  // from float
  retval = test<float, int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, unsigned int>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, float>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  retval = test<float, unsigned long>();
  if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
  else return retval;

  {
    retval = test<float, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }

  // from double
  {
    retval = test<double, int>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, unsigned int>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, long>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, unsigned long>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, float>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, unsigned long>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;

    retval = test<double, double>();
    if ( retval == EXIT_SUCCESS ) std::cout << "# Test passed" << std::endl;
    else return retval;
  }


  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return retval;
}
