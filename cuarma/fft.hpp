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
/** @file cuarma/fft.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief 快速傅里叶变换常用函数工具，从一维到二维，原位和非原位
 *  @brief All routines related to the Fast Fourier Transform. Experimental.
 *  @link  https://en.wikipedia.org/wiki/Fast_Fourier_transform
 */

#include <cuarma/vector.hpp>
#include <cuarma/matrix.hpp>
#include "cuarma/blas/fft_operations.hpp"
#include "cuarma/traits/handle.hpp"
#include <cmath>
#include <stdexcept>
/// @cond
namespace cuarma
{
namespace detail
{
namespace fft
{
  inline bool is_radix2(arma_size_t data_size)
  {
    return !((data_size > 2) && (data_size & (data_size - 1)));
  }
} //namespace fft
} //namespace detail

/**
 * @brief Generic inplace version of 1-D Fourier transformation.
 *
 * @param input       Input vector, result will be stored here.
 * @param batch_num   Number of items in batch
 * @param sign        Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void inplace_fft(cuarma::vector<NumericT, AlignmentV>& input, arma_size_t batch_num = 1,
                 NumericT sign = -1.0)
{
  arma_size_t size = (input.size() >> 1) / batch_num;

  if (!cuarma::detail::fft::is_radix2(size))
  {
    cuarma::vector<NumericT, AlignmentV> output(input.size());
    cuarma::blas::direct(input, output, size, size, batch_num, sign);
    cuarma::copy(output, input);
  }
  else
    cuarma::blas::radix2(input, size, size, batch_num, sign);
}

/**
 * @brief Generic version of 1-D Fourier transformation.
 *
 * @param input      Input vector.
 * @param output     Output vector.
 * @param batch_num  Number of items in batch.
 * @param sign       Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void fft(cuarma::vector<NumericT, AlignmentV>& input,
         cuarma::vector<NumericT, AlignmentV>& output, arma_size_t batch_num = 1, NumericT sign = -1.0)
{
  arma_size_t size = (input.size() >> 1) / batch_num;
  if (cuarma::detail::fft::is_radix2(size))
  {
    cuarma::copy(input, output);
    cuarma::blas::radix2(output, size, size, batch_num, sign);
  }
  else
    cuarma::blas::direct(input, output, size, size, batch_num, sign);
}

/**
 * @brief Generic inplace version of 2-D Fourier transformation.
 *
 * @param input       Input matrix, result will be stored here.
 * @param sign        Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void inplace_fft(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>& input,
                 NumericT sign = -1.0)
{
  arma_size_t rows_num = input.size1();
  arma_size_t cols_num = input.size2() >> 1;

  arma_size_t cols_int = input.internal_size2() >> 1;

  // batch with rows
  if (cuarma::detail::fft::is_radix2(cols_num))
    cuarma::blas::radix2(input, cols_num, cols_int, rows_num, sign,
                             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR);
  else
  {
    cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> output(input.size1(),
                                                                       input.size2());

    cuarma::blas::direct(input, output, cols_num, cols_int, rows_num, sign,
                             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR);

    input = output;
  }

  // batch with cols
  if (cuarma::detail::fft::is_radix2(rows_num))
    cuarma::blas::radix2(input, rows_num, cols_int, cols_num, sign,
                             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR);
  else
  {
    cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> output(input.size1(),
                                                                       input.size2());

    cuarma::blas::direct(input, output, rows_num, cols_int, cols_num, sign,
                             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR);

    input = output;
  }

}

/**
 * @brief Generic version of 2-D Fourier transformation.
 *
 * @param input      Input vector.
 * @param output     Output vector.
 * @param sign       Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void fft(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>& input, //TODO
         cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>& output, NumericT sign = -1.0)
{

  arma_size_t rows_num = input.size1();
  arma_size_t cols_num = input.size2() >> 1;
  arma_size_t cols_int = input.internal_size2() >> 1;

  // batch with rows
  if (cuarma::detail::fft::is_radix2(cols_num))
  {
    output = input;
    cuarma::blas::radix2(output, cols_num, cols_int, rows_num, sign,
                             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR);
  }
  else
    cuarma::blas::direct(input, output, cols_num, cols_int, rows_num, sign,
                             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR);

  // batch with cols
  if (cuarma::detail::fft::is_radix2(rows_num))
  {
    //std::cout<<"output"<<output<<std::endl;

    cuarma::blas::radix2(output, rows_num, cols_int, cols_num, sign,
                             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR);
  }
  else
  {
    cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> tmp(output.size1(),
                                                                    output.size2());
    tmp = output;
    //std::cout<<"tmp"<<tmp<<std::endl;
    cuarma::blas::direct(tmp, output, rows_num, cols_int, cols_num, sign,
                             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR);
  }
}

/**
 * @brief Generic inplace version of inverse 1-D Fourier transformation.
 *
 * Shorthand function for fft(sign = 1.0)
 *
 * @param input      Input vector.
 * @param batch_num  Number of items in batch.
 * @param sign       Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void inplace_ifft(cuarma::vector<NumericT, AlignmentV>& input, arma_size_t batch_num = 1)
{
  cuarma::inplace_fft(input, batch_num, NumericT(1.0));
  cuarma::blas::normalize(input);
}

/**
 * @brief Generic version of inverse 1-D Fourier transformation.
 *
 * Shorthand function for fft(sign = 1.0)
 *
 * @param input      Input vector.
 * @param output     Output vector.
 * @param batch_num  Number of items in batch.
 * @param sign       Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void ifft(cuarma::vector<NumericT, AlignmentV>& input,
          cuarma::vector<NumericT, AlignmentV>& output, arma_size_t batch_num = 1)
{
  cuarma::fft(input, output, batch_num, NumericT(1.0));
  cuarma::blas::normalize(output);
}

namespace blas
{
  /**
   * @brief 1-D convolution of two vectors.
   *
   * This function does not make any changes to input vectors
   *
   * @param input1     Input vector #1.
   * @param input2     Input vector #2.
   * @param output     Output vector.
   */
  template<class NumericT, unsigned int AlignmentV>
  void convolve(cuarma::vector<NumericT, AlignmentV>& input1,
                cuarma::vector<NumericT, AlignmentV>& input2,
                cuarma::vector<NumericT, AlignmentV>& output)
  {
    assert(input1.size() == input2.size());
    assert(input1.size() == output.size());
    //temporal arrays
    cuarma::vector<NumericT, AlignmentV> tmp1(input1.size());
    cuarma::vector<NumericT, AlignmentV> tmp2(input2.size());
    cuarma::vector<NumericT, AlignmentV> tmp3(output.size());

    // align input arrays to equal size
    // FFT of input data
    cuarma::fft(input1, tmp1);
    cuarma::fft(input2, tmp2);

    // multiplication of input data
    cuarma::blas::multiply_complex(tmp1, tmp2, tmp3);
    // inverse FFT of input data
    cuarma::ifft(tmp3, output);
  }

  /**
   * @brief 1-D convolution of two vectors.
   *
   * This function can make changes to input vectors to avoid additional memory allocations.
   *
   * @param input1     Input vector #1.
   * @param input2     Input vector #2.
   * @param output     Output vector.
   */
  template<class NumericT, unsigned int AlignmentV>
  void convolve_i(cuarma::vector<NumericT, AlignmentV>& input1,
                  cuarma::vector<NumericT, AlignmentV>& input2,
                  cuarma::vector<NumericT, AlignmentV>& output)
  {
    assert(input1.size() == input2.size());
    assert(input1.size() == output.size());

    cuarma::inplace_fft(input1);
    cuarma::inplace_fft(input2);

    cuarma::blas::multiply_complex(input1, input2, output);

    cuarma::inplace_ifft(output);
  }
}      //namespace blas
}      //namespace cuarma

