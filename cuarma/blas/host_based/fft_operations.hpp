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

/** @file  cuarma/blas/host_based/fft_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of Fast Furier Transformation using a plain single-threaded execution on CPU
 */

//TODO openom Conditions
#include <cuarma/vector.hpp>
#include <cuarma/matrix.hpp>

#include "cuarma/blas/host_based/vector_operations.hpp"

#include <stdexcept>
#include <cmath>
#include <complex>

namespace cuarma
{
namespace blas
{
namespace host_based
{
namespace detail
{
  namespace fft
  {
    const arma_size_t MAX_LOCAL_POINTS_NUM = 512;

    namespace FFT_DATA_ORDER
    {
      enum DATA_ORDER
      {
        ROW_MAJOR, COL_MAJOR
      };
    }

    inline arma_size_t num_bits(arma_size_t size)
    {
      arma_size_t bits_datasize = 0;
      arma_size_t ds = 1;

      while (ds < size)
      {
        ds = ds << 1;
        bits_datasize++;
      }

      return bits_datasize;
    }

    inline arma_size_t next_power_2(arma_size_t n)
    {
      n = n - 1;

      arma_size_t power = 1;

      while (power < sizeof(arma_size_t) * 8)
      {
        n = n | (n >> power);
        power *= 2;
      }

      return n + 1;
    }

    inline arma_size_t get_reorder_num(arma_size_t v, arma_size_t bit_size)
    {
      v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
      v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
      v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
      v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
      v = (v >> 16) | (v << 16);
      v = v >> (32 - bit_size);
      return v;
    }

    template<typename NumericT, unsigned int AlignmentV>
    void copy_to_complex_array(std::complex<NumericT> * input_complex,
                               cuarma::vector<NumericT, AlignmentV> const & in, arma_size_t size)
    {

      for (long i2 = 0; i2 < long(size * 2); i2 += 2)
      { //change array to complex array
        arma_size_t i = arma_size_t(i2);
        input_complex[i / 2] = std::complex<NumericT>(in[i], in[i + 1]);
      }
    }

    template<typename NumericT>
    void copy_to_complex_array(std::complex<NumericT> * input_complex,
                               cuarma::vector_base<NumericT> const & in, arma_size_t size)
    {

      for (long i2 = 0; i2 < long(size * 2); i2 += 2)
      { //change array to complex array
        arma_size_t i = arma_size_t(i2);
        input_complex[i / 2] = std::complex<NumericT>(in[i], in[i + 1]);
      }
    }

    template<typename NumericT, unsigned int AlignmentV>
    void copy_to_vector(std::complex<NumericT> * input_complex,
                        cuarma::vector<NumericT, AlignmentV> & in, arma_size_t size)
    {

      for (long i2 = 0; i2 < long(size); i2++)
      {
        arma_size_t i = arma_size_t(i2);
        in(i * 2)     = static_cast<NumericT>(std::real(input_complex[i]));
        in(i * 2 + 1) = static_cast<NumericT>(std::imag(input_complex[i]));
      }
    }

    template<typename NumericT>
    void copy_to_complex_array(std::complex<NumericT> * input_complex,
                               NumericT const * in, arma_size_t size)
    {

      for (long i2 = 0; i2 < long(size * 2); i2 += 2)
      { //change array to complex array
        arma_size_t i = arma_size_t(i2);
        input_complex[i / 2] = std::complex<NumericT>(in[i], in[i + 1]);
      }
    }

    template<typename NumericT>
    void copy_to_vector(std::complex<NumericT> * input_complex, NumericT * in, arma_size_t size)
    {

      for (long i2 = 0; i2 < long(size); i2++)
      {
        arma_size_t i = arma_size_t(i2);
        in[i * 2]     = static_cast<NumericT>(std::real(input_complex[i]));
        in[i * 2 + 1] = static_cast<NumericT>(std::imag(input_complex[i]));
      }
    }

    template<typename NumericT>
    void copy_to_vector(std::complex<NumericT> * input_complex,
                        cuarma::vector_base<NumericT> & in, arma_size_t size)
    {
      std::vector<NumericT> temp(2 * size);

      for (long i2 = 0; i2 < long(size); i2++)
      {
        arma_size_t i = arma_size_t(i2);
        temp[i * 2]     = static_cast<NumericT>(std::real(input_complex[i]));
        temp[i * 2 + 1] = static_cast<NumericT>(std::imag(input_complex[i]));
      }
      cuarma::copy(temp, in);
    }

    template<typename NumericT>
    void zero2(NumericT *input1, NumericT *input2, arma_size_t size)
    {

      for (long i2 = 0; i2 < long(size); i2++)
      {
        arma_size_t i = arma_size_t(i2);
        input1[i] = 0;
        input2[i] = 0;
      }
    }

  } //namespace fft

} //namespace detail

/**
 * @brief Direct algoritm kenrnel
 */
template<typename NumericT>
void fft_direct(std::complex<NumericT> * input_complex, std::complex<NumericT> * output,
                arma_size_t size, arma_size_t stride, arma_size_t batch_num, NumericT sign,
                cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  NumericT const NUM_PI = NumericT(3.14159265358979323846);

  for (long batch_id2 = 0; batch_id2 < long(batch_num); batch_id2++)
  {
    arma_size_t batch_id = arma_size_t(batch_id2);
    for (arma_size_t k = 0; k < size; k += 1)
    {
      std::complex<NumericT> f = 0;
      for (arma_size_t n = 0; n < size; n++)
      {
        std::complex<NumericT> input;
        if (!data_order)
          input = input_complex[batch_id * stride + n]; //input index here
        else
          input = input_complex[n * stride + batch_id];
        NumericT arg = sign * 2 * NUM_PI * NumericT(k) / NumericT(size * n);
        NumericT sn  = std::sin(arg);
        NumericT cs  = std::cos(arg);

        std::complex<NumericT> ex(cs, sn);
        std::complex<NumericT> tmp(input.real() * ex.real() - input.imag() * ex.imag(),
                                   input.real() * ex.imag() + input.imag() * ex.real());
        f = f + tmp;
      }
      if (!data_order)
        output[batch_id * stride + k] = f;   // output index here
      else
        output[k * stride + batch_id] = f;
    }
  }

}

/**
 * @brief Direct 1D algorithm for computing Fourier transformation.
 *
 * Works on any sizes of data.
 * Serial implementation has o(n^2) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void direct(cuarma::vector<NumericT, AlignmentV> const & in,
            cuarma::vector<NumericT, AlignmentV>       & out,
            arma_size_t size, arma_size_t stride,
            arma_size_t batch_num, NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  std::vector<std::complex<NumericT> > input_complex(size * batch_num);
  std::vector<std::complex<NumericT> > output(size * batch_num);

  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input_complex[0], in, size * batch_num);

  fft_direct(&input_complex[0], &output[0], size, stride, batch_num, sign, data_order);

  cuarma::blas::host_based::detail::fft::copy_to_vector(&output[0], out, size * batch_num);
}

/**
 * @brief Direct 2D algorithm for computing Fourier transformation.
 *
 * Works on any sizes of data.
 * Serial implementation has o(n^2) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void direct(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> const & in,
            cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>       & out, arma_size_t size,
            arma_size_t stride, arma_size_t batch_num, NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  arma_size_t row_num = in.internal_size1();
  arma_size_t col_num = in.internal_size2() >> 1;

  arma_size_t size_mat = row_num * col_num;

  std::vector<std::complex<NumericT> > input_complex(size_mat);
  std::vector<std::complex<NumericT> > output(size_mat);

  NumericT const * data_A = detail::extract_raw_pointer<NumericT>(in);
  NumericT       * data_B = detail::extract_raw_pointer<NumericT>(out);

  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input_complex[0], data_A, size_mat);

  fft_direct(&input_complex[0], &output[0], size, stride, batch_num, sign, data_order);

  cuarma::blas::host_based::detail::fft::copy_to_vector(&output[0], data_B, size_mat);
}

/*
 * This function performs reorder of 1D input  data. Indexes are sorted in bit-reversal order.
 * Such reordering should be done before in-place FFT.
 */
template<typename NumericT, unsigned int AlignmentV>
void reorder(cuarma::vector<NumericT, AlignmentV>& in, arma_size_t size, arma_size_t stride,
             arma_size_t bits_datasize, arma_size_t batch_num,
             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  std::vector<std::complex<NumericT> > input(size * batch_num);
  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input[0], in, size * batch_num);

  for (long batch_id2 = 0; batch_id2 < long(batch_num); batch_id2++)
  {
    arma_size_t batch_id = arma_size_t(batch_id2);
    for (arma_size_t i = 0; i < size; i++)
    {
      arma_size_t v = cuarma::blas::host_based::detail::fft::get_reorder_num(i, bits_datasize);
      if (i < v)
      {
        if (!data_order)
        {
          std::complex<NumericT> tmp   = input[batch_id * stride + i]; // index
          input[batch_id * stride + i] = input[batch_id * stride + v]; //index
          input[batch_id * stride + v] = tmp;      //index
        }
        else
        {
          std::complex<NumericT> tmp   = input[i * stride + batch_id]; // index
          input[i * stride + batch_id] = input[v * stride + batch_id]; //index
          input[v * stride + batch_id] = tmp;      //index
        }
      }
    }
  }
  cuarma::blas::host_based::detail::fft::copy_to_vector(&input[0], in, size * batch_num);
}

/*
 * This function performs reorder of 2D input  data. Indexes are sorted in bit-reversal order.
 * Such reordering should be done before in-place FFT.
 */
template<typename NumericT, unsigned int AlignmentV>
void reorder(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>& in,
             arma_size_t size, arma_size_t stride, arma_size_t bits_datasize, arma_size_t batch_num,
             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{

  NumericT * data = detail::extract_raw_pointer<NumericT>(in);
  arma_size_t row_num = in.internal_size1();
  arma_size_t col_num = in.internal_size2() >> 1;
  arma_size_t size_mat = row_num * col_num;

  std::vector<std::complex<NumericT> > input(size_mat);

  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input[0], data, size_mat);


  for (long batch_id2 = 0; batch_id2 < long(batch_num); batch_id2++)
  {
    arma_size_t batch_id = arma_size_t(batch_id2);
    for (arma_size_t i = 0; i < size; i++)
    {
      arma_size_t v = cuarma::blas::host_based::detail::fft::get_reorder_num(i, bits_datasize);
      if (i < v)
      {
        if (!data_order)
        {
          std::complex<NumericT> tmp   = input[batch_id * stride + i]; // index
          input[batch_id * stride + i] = input[batch_id * stride + v]; //index
          input[batch_id * stride + v] = tmp;      //index
        } else
        {
          std::complex<NumericT> tmp   = input[i * stride + batch_id]; // index
          input[i * stride + batch_id] = input[v * stride + batch_id]; //index
          input[v * stride + batch_id] = tmp;      //index
        }
      }
    }
  }
  cuarma::blas::host_based::detail::fft::copy_to_vector(&input[0], data, size_mat);
}

/**
 * @brief Radix-2 algorithm for computing Fourier transformation.
 * Kernel for computing smaller amount of data
 */
template<typename NumericT>
void fft_radix2(std::complex<NumericT> * input_complex, arma_size_t batch_num,
                arma_size_t bit_size, arma_size_t size, arma_size_t stride, NumericT sign,
                cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  NumericT const NUM_PI = NumericT(3.14159265358979323846);

  for (arma_size_t step = 0; step < bit_size; step++)
  {
    arma_size_t ss = 1 << step;
    arma_size_t half_size = size >> 1;


    for (long batch_id2 = 0; batch_id2 < long(batch_num); batch_id2++)
    {
      arma_size_t batch_id = arma_size_t(batch_id2);
      for (arma_size_t tid = 0; tid < half_size; tid++)
      {
        arma_size_t group = (tid & (ss - 1));
        arma_size_t pos = ((tid >> step) << (step + 1)) + group;
        std::complex<NumericT> in1;
        std::complex<NumericT> in2;
        arma_size_t offset;
        if (!data_order)
        {
          offset = batch_id * stride + pos;
          in1 = input_complex[offset];
          in2 = input_complex[offset + ss];
        }
        else
        {
          offset = pos * stride + batch_id;
          in1 = input_complex[offset];
          in2 = input_complex[offset + ss * stride];
        }
        NumericT arg = NumericT(group) * sign * NUM_PI / NumericT(ss);
        NumericT sn = std::sin(arg);
        NumericT cs = std::cos(arg);
        std::complex<NumericT> ex(cs, sn);
        std::complex<NumericT> tmp(in2.real() * ex.real() - in2.imag() * ex.imag(),
                                   in2.real() * ex.imag() + in2.imag() * ex.real());
        if (!data_order)
          input_complex[offset + ss] = in1 - tmp;
        else
          input_complex[offset + ss * stride] = in1 - tmp;
        input_complex[offset] = in1 + tmp;
      }
    }
  }

}

/**
 * @brief Radix-2 algorithm for computing Fourier transformation.
 * Kernel for computing bigger amount of data
 */
template<typename NumericT>
void fft_radix2_local(std::complex<NumericT> * input_complex,
                      std::complex<NumericT> * lcl_input, arma_size_t batch_num, arma_size_t bit_size,
                      arma_size_t size, arma_size_t stride, NumericT sign,
                      cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  NumericT const NUM_PI = NumericT(3.14159265358979323846);

  for (arma_size_t batch_id = 0; batch_id < batch_num; batch_id++)
  {

    for (long p2 = 0; p2 < long(size); p2 += 1)
    {
      arma_size_t p = arma_size_t(p2);
      arma_size_t v = cuarma::blas::host_based::detail::fft::get_reorder_num(p, bit_size);

      if (!data_order)
        lcl_input[v] = input_complex[batch_id * stride + p]; //index
      else
        lcl_input[v] = input_complex[p * stride + batch_id];
    }

    for (arma_size_t s = 0; s < bit_size; s++)
    {
      arma_size_t ss = 1 << s;

      for (long tid2 = 0; tid2 < long(size)/2; tid2++)
      {
        arma_size_t tid = arma_size_t(tid2);
        arma_size_t group = (tid & (ss - 1));
        arma_size_t pos = ((tid >> s) << (s + 1)) + group;

        std::complex<NumericT> in1 = lcl_input[pos];
        std::complex<NumericT> in2 = lcl_input[pos + ss];

        NumericT arg = NumericT(group) * sign * NUM_PI / NumericT(ss);

        NumericT sn = std::sin(arg);
        NumericT cs = std::cos(arg);
        std::complex<NumericT> ex(cs, sn);

        std::complex<NumericT> tmp(in2.real() * ex.real() - in2.imag() * ex.imag(),
                                   in2.real() * ex.imag() + in2.imag() * ex.real());

        lcl_input[pos + ss] = in1 - tmp;
        lcl_input[pos] = in1 + tmp;
      }

    }

    //copy local array back to global memory
    for (long p2 = 0; p2 < long(size); p2 += 1)
    {
      arma_size_t p = arma_size_t(p2);
      if (!data_order)
        input_complex[batch_id * stride + p] = lcl_input[p];
      else
        input_complex[p * stride + batch_id] = lcl_input[p];

    }

  }

}

/**
 * @brief Radix-2 1D algorithm for computing Fourier transformation.
 *
 * Works only on power-of-two sizes of data.
 * Serial implementation has o(n * lg n) complexity.
 * This is a Cooley-Tukey algorithm
 */
template<typename NumericT, unsigned int AlignmentV>
void radix2(cuarma::vector<NumericT, AlignmentV>& in, arma_size_t size, arma_size_t stride,
            arma_size_t batch_num, NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{

  arma_size_t bit_size = cuarma::blas::host_based::detail::fft::num_bits(size);

  std::vector<std::complex<NumericT> > input_complex(size * batch_num);
  std::vector<std::complex<NumericT> > lcl_input(size * batch_num);
  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input_complex[0], in, size * batch_num);

  if (size <= cuarma::blas::host_based::detail::fft::MAX_LOCAL_POINTS_NUM)
  {
    cuarma::blas::host_based::fft_radix2_local(&input_complex[0], &lcl_input[0], batch_num, bit_size, size, stride, sign, data_order);
  }
  else
  {
    cuarma::blas::host_based::reorder<NumericT>(in, size, stride, bit_size, batch_num, data_order);
    cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input_complex[0], in, size * batch_num);
    cuarma::blas::host_based::fft_radix2(&input_complex[0], batch_num, bit_size, size, stride, sign, data_order);
  }

  cuarma::blas::host_based::detail::fft::copy_to_vector(&input_complex[0], in, size * batch_num);
}

/**
 * @brief Radix-2 2D algorithm for computing Fourier transformation.
 *
 * Works only on power-of-two sizes of data.
 * Serial implementation has o(n * lg n) complexity.
 * This is a Cooley-Tukey algorithm
 */
template<typename NumericT, unsigned int AlignmentV>
void radix2(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>& in, arma_size_t size,
            arma_size_t stride, arma_size_t batch_num, NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{

  arma_size_t bit_size = cuarma::blas::host_based::detail::fft::num_bits(size);

  NumericT * data = detail::extract_raw_pointer<NumericT>(in);

  arma_size_t row_num = in.internal_size1();
  arma_size_t col_num = in.internal_size2() >> 1;
  arma_size_t size_mat = row_num * col_num;

  std::vector<std::complex<NumericT> > input_complex(size_mat);

  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input_complex[0], data, size_mat);
  if (size <= cuarma::blas::host_based::detail::fft::MAX_LOCAL_POINTS_NUM)
  {
    //std::cout<<bit_size<<","<<size<<","<<stride<<","<<batch_num<<","<<size<<","<<sign<<","<<data_order<<std::endl;
    std::vector<std::complex<NumericT> > lcl_input(size_mat);
    cuarma::blas::host_based::fft_radix2_local(&input_complex[0], &lcl_input[0], batch_num, bit_size, size, stride, sign, data_order);
  }
  else
  {
    cuarma::blas::host_based::reorder<NumericT>(in, size, stride, bit_size, batch_num, data_order);
    cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input_complex[0], data, size_mat);
    cuarma::blas::host_based::fft_radix2(&input_complex[0], batch_num, bit_size, size, stride, sign, data_order);
  }

  cuarma::blas::host_based::detail::fft::copy_to_vector(&input_complex[0], data, size_mat);

}

/**
 * @brief Bluestein's algorithm for computing Fourier transformation.
 *
 * Currently,  Works only for sizes of input data which less than 2^16.
 * Uses a lot of additional memory, but should be fast for any size of data.
 * Serial implementation has something about o(n * lg n) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void bluestein(cuarma::vector<NumericT, AlignmentV>& in, cuarma::vector<NumericT, AlignmentV>& out, arma_size_t /*batch_num*/)
{

  arma_size_t size = in.size() >> 1;
  arma_size_t ext_size = cuarma::blas::host_based::detail::fft::next_power_2(2 * size - 1);

  cuarma::vector<NumericT, AlignmentV> A(ext_size << 1);
  cuarma::vector<NumericT, AlignmentV> B(ext_size << 1);
  cuarma::vector<NumericT, AlignmentV> Z(ext_size << 1);

  std::vector<std::complex<NumericT> > input_complex(size);
  std::vector<std::complex<NumericT> > output_complex(size);

  std::vector<std::complex<NumericT> > A_complex(ext_size);
  std::vector<std::complex<NumericT> > B_complex(ext_size);
  std::vector<std::complex<NumericT> > Z_complex(ext_size);

  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input_complex[0], in, size);

  for (long i2 = 0; i2 < long(ext_size); i2++)
  {
    arma_size_t i = arma_size_t(i2);
    A_complex[i] = 0;
    B_complex[i] = 0;
  }

  arma_size_t double_size = size << 1;

  NumericT const NUM_PI = NumericT(3.14159265358979323846);

  for (long i2 = 0; i2 < long(size); i2++)
  {
    arma_size_t i = arma_size_t(i2);
    arma_size_t rm = i * i % (double_size);
    NumericT angle = NumericT(rm) / NumericT(size) * NumericT(NUM_PI);

    NumericT sn_a = std::sin(-angle);
    NumericT cs_a = std::cos(-angle);

    std::complex<NumericT> a_i(cs_a, sn_a);
    std::complex<NumericT> b_i(cs_a, -sn_a);

    A_complex[i] = std::complex<NumericT>(input_complex[i].real() * a_i.real() - input_complex[i].imag() * a_i.imag(),
                                          input_complex[i].real() * a_i.imag() + input_complex[i].imag() * a_i.real());
    B_complex[i] = b_i;

    // very bad instruction, to be fixed
    if (i)
      B_complex[ext_size - i] = b_i;
  }

  cuarma::blas::host_based::detail::fft::copy_to_vector(&input_complex[0], in, size);
  cuarma::blas::host_based::detail::fft::copy_to_vector(&A_complex[0], A, ext_size);
  cuarma::blas::host_based::detail::fft::copy_to_vector(&B_complex[0], B, ext_size);

  cuarma::blas::convolve_i(A, B, Z);

  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&Z_complex[0], Z, ext_size);


  for (long i2 = 0; i2 < long(size); i2++)
  {
    arma_size_t i = arma_size_t(i2);
    arma_size_t rm = i * i % (double_size);
    NumericT angle = NumericT(rm) / NumericT(size) * NumericT(-NUM_PI);
    NumericT sn_a = std::sin(angle);
    NumericT cs_a = std::cos(angle);
    std::complex<NumericT> b_i(cs_a, sn_a);
    output_complex[i] = std::complex<NumericT>(Z_complex[i].real() * b_i.real() - Z_complex[i].imag() * b_i.imag(),
                                               Z_complex[i].real() * b_i.imag() + Z_complex[i].imag() * b_i.real());
  }
  cuarma::blas::host_based::detail::fft::copy_to_vector(&output_complex[0], out, size);

}

/**
 * @brief Normalize vector with his own size
 */
template<typename NumericT, unsigned int AlignmentV>
void normalize(cuarma::vector<NumericT, AlignmentV> & input)
{
  arma_size_t size = input.size() >> 1;
  NumericT norm_factor = static_cast<NumericT>(size);
  for (arma_size_t i = 0; i < size * 2; i++)
    input[i] /= norm_factor;

}

/**
 * @brief Complex multiplikation of two vectors
 */
template<typename NumericT, unsigned int AlignmentV>
void multiply_complex(cuarma::vector<NumericT, AlignmentV> const & input1,
                      cuarma::vector<NumericT, AlignmentV> const & input2,
                      cuarma::vector<NumericT, AlignmentV> & output)
{
  arma_size_t size = input1.size() >> 1;

  std::vector<std::complex<NumericT> > input1_complex(size);
  std::vector<std::complex<NumericT> > input2_complex(size);
  std::vector<std::complex<NumericT> > output_complex(size);
  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input1_complex[0], input1, size);
  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input2_complex[0], input2, size);


  for (long i2 = 0; i2 < long(size); i2++)
  {
    arma_size_t i = arma_size_t(i2);
    std::complex<NumericT> in1 = input1_complex[i];
    std::complex<NumericT> in2 = input2_complex[i];
    output_complex[i] = std::complex<NumericT>(in1.real() * in2.real() - in1.imag() * in2.imag(),
                                               in1.real() * in2.imag() + in1.imag() * in2.real());
  }
  cuarma::blas::host_based::detail::fft::copy_to_vector(&output_complex[0], output, size);

}
/**
 * @brief Inplace transpose of matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> & input)
{
  arma_size_t row_num = input.internal_size1() / 2;
  arma_size_t col_num = input.internal_size2() / 2;

  arma_size_t size = row_num * col_num;

  NumericT * data = detail::extract_raw_pointer<NumericT>(input);

  std::vector<std::complex<NumericT> > input_complex(size);

  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input_complex[0], data, size);

  for (long i2 = 0; i2 < long(size); i2++)
  {
    arma_size_t i = arma_size_t(i2);
    arma_size_t row = i / col_num;
    arma_size_t col = i - row * col_num;
    arma_size_t new_pos = col * row_num + row;

    if (i < new_pos)
    {
      std::complex<NumericT> val = input_complex[i];
      input_complex[i] = input_complex[new_pos];
      input_complex[new_pos] = val;
    }
  }
  cuarma::blas::host_based::detail::fft::copy_to_vector(&input_complex[0], data, size);

}

/**
 * @brief Transpose matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> const & input,
               cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>       & output)
{

  arma_size_t row_num = input.internal_size1() / 2;
  arma_size_t col_num = input.internal_size2() / 2;
  arma_size_t size = row_num * col_num;

  NumericT const * data_A = detail::extract_raw_pointer<NumericT>(input);
  NumericT       * data_B = detail::extract_raw_pointer<NumericT>(output);

  std::vector<std::complex<NumericT> > input_complex(size);
  cuarma::blas::host_based::detail::fft::copy_to_complex_array(&input_complex[0], data_A, size);

  std::vector<std::complex<NumericT> > output_complex(size);

  for (long i2 = 0; i2 < long(size); i2++)
  {
    arma_size_t i = arma_size_t(i2);
    arma_size_t row = i / col_num;
    arma_size_t col = i % col_num;
    arma_size_t new_pos = col * row_num + row;
    output_complex[new_pos] = input_complex[i];
  }
  cuarma::blas::host_based::detail::fft::copy_to_vector(&output_complex[0], data_B, size);
}

/**
 * @brief Create complex vector from real vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void real_to_complex(cuarma::vector_base<NumericT> const & in,
                     cuarma::vector_base<NumericT>       & out, arma_size_t size)
{
  NumericT const * data_in  = detail::extract_raw_pointer<NumericT>(in);
  NumericT       * data_out = detail::extract_raw_pointer<NumericT>(out);


  for (long i2 = 0; i2 < long(size); i2++)
  {
    arma_size_t i = static_cast<arma_size_t>(i2);
    data_out[2*i  ] = data_in[i];
    data_out[2*i+1] = NumericT(0);
  }
}

/**
 * @brief Create real vector from complex vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void complex_to_real(cuarma::vector_base<NumericT> const & in,
                     cuarma::vector_base<NumericT>       & out, arma_size_t size)
{
  NumericT const * data_in  = detail::extract_raw_pointer<NumericT>(in);
  NumericT       * data_out = detail::extract_raw_pointer<NumericT>(out);


  for (long i = 0; i < long(size); i++)
    data_out[i] = data_in[2*i];
}

/**
 * @brief Reverse vector to opposite order and save it in input vector
 */
template<typename NumericT>
void reverse(cuarma::vector_base<NumericT> & in)
{
  arma_size_t size = in.size();


  for (long i2 = 0; i2 < long(size); i2++)
  {
    arma_size_t i = arma_size_t(i2);
    NumericT val1 = in[i];
    NumericT val2 = in[size - i - 1];
    in[i] = val2;
    in[size - i - 1] = val1;
  }
}

}      //namespace host_based
}      //namespace blas
}      //namespace cuarma