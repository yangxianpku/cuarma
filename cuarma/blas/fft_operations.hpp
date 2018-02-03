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

/**  @file cuarma/blas/fft_operations.hpp
 *   @encoding:UTF-8 文档编码
 *   @brief Implementations of Fast Furier Transformation.
 */

#include <cuarma/vector.hpp>
#include <cuarma/matrix.hpp>
#include "cuarma/blas/host_based/fft_operations.hpp"

#ifdef CUARMA_WITH_CUDA
#include "cuarma/blas/cuda/fft_operations.hpp"
#endif

namespace cuarma
{
namespace blas
{

/**
 * @brief Direct 1D algorithm for computing Fourier transformation.
 *
 * Works on any sizes of data.
 * Serial implementation has o(n^2) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void direct(cuarma::vector<NumericT, AlignmentV> const & in,
            cuarma::vector<NumericT, AlignmentV>       & out, arma_size_t size, arma_size_t stride,
            arma_size_t batch_num, NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{

  switch (cuarma::traits::handle(in).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::direct(in, out, size, stride, batch_num, sign, data_order);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::direct(in, out, size, stride, batch_num,sign,data_order);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");

  }
}

/**
 * @brief Direct 2D algorithm for computing Fourier transformation.
 *
 * Works on any sizes of data.
 * Serial implementation has o(n^2) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void direct(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> const & in,
            cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>& out, arma_size_t size,
            arma_size_t stride, arma_size_t batch_num, NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{

  switch (cuarma::traits::handle(in).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::direct(in, out, size, stride, batch_num, sign, data_order);
    break;


#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::direct(in, out, size, stride, batch_num,sign,data_order);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");

  }
}

/*
 * This function performs reorder of input data. Indexes are sorted in bit-reversal order.
 * Such reordering should be done before in-place FFT.
 */
template<typename NumericT, unsigned int AlignmentV>
void reorder(cuarma::vector<NumericT, AlignmentV>& in, arma_size_t size, arma_size_t stride,
             arma_size_t bits_datasize, arma_size_t batch_num,
             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  switch (cuarma::traits::handle(in).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::reorder(in, size, stride, bits_datasize, batch_num, data_order);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::reorder(in, size, stride, bits_datasize, batch_num, data_order);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");

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
void radix2(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> & in, arma_size_t size,
            arma_size_t stride, arma_size_t batch_num, NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  switch (cuarma::traits::handle(in).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::radix2(in, size, stride, batch_num, sign, data_order);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::radix2(in, size, stride, batch_num, sign, data_order);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Radix-2 2D algorithm for computing Fourier transformation.
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

  switch (cuarma::traits::handle(in).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::radix2(in, size, stride, batch_num, sign, data_order);
    break;

#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::radix2(in, size, stride, batch_num, sign,data_order);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Bluestein's algorithm for computing Fourier transformation.
 *
 * Currently,  Works only for sizes of input data which less than 2^16.
 * Uses a lot of additional memory, but should be fast for any size of data.
 * Serial implementation has something about o(n * lg n) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void bluestein(cuarma::vector<NumericT, AlignmentV> & in,
               cuarma::vector<NumericT, AlignmentV> & out, arma_size_t /*batch_num*/)
{

  switch (cuarma::traits::handle(in).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::bluestein(in, out, 1);
    break;


#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::bluestein(in, out, 1);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Mutiply two complex vectors and store result in output
 */
template<typename NumericT, unsigned int AlignmentV>
void multiply_complex(cuarma::vector<NumericT, AlignmentV> const & input1,
                      cuarma::vector<NumericT, AlignmentV> const & input2,
                      cuarma::vector<NumericT, AlignmentV>       & output)
{
  switch (cuarma::traits::handle(input1).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::multiply_complex(input1, input2, output);
    break;


#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::multiply_complex(input1, input2, output);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Normalize vector on with his own size
 */
template<typename NumericT, unsigned int AlignmentV>
void normalize(cuarma::vector<NumericT, AlignmentV> & input)
{
  switch (cuarma::traits::handle(input).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::normalize(input);
    break;


#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::normalize(input);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Inplace_transpose matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> & input)
{
  switch (cuarma::traits::handle(input).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::transpose(input);
    break;


#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::transpose(input);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Transpose matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> const & input,
               cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>       & output)
{
  switch (cuarma::traits::handle(input).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::transpose(input, output);
    break;


#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::transpose(input, output);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Create complex vector from real vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void real_to_complex(cuarma::vector_base<NumericT> const & in, cuarma::vector_base<NumericT>       & out, arma_size_t size)
{
  switch (cuarma::traits::handle(in).get_active_handle_id())
  {
    case cuarma::MAIN_MEMORY:
      cuarma::blas::host_based::real_to_complex(in, out, size);
      break;


#ifdef CUARMA_WITH_CUDA
      case cuarma::CUDA_MEMORY:
      cuarma::blas::cuda::real_to_complex(in,out,size);
      break;
#endif

    case cuarma::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

/**
 * @brief Create real vector from complex vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void complex_to_real(cuarma::vector_base<NumericT> const & in,
                     cuarma::vector_base<NumericT>       & out, arma_size_t size)
{
  switch (cuarma::traits::handle(in).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::complex_to_real(in, out, size);
    break;


#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::complex_to_real(in, out, size);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Reverse vector to oposite order and save it in input vector
 */
template<typename NumericT>
void reverse(cuarma::vector_base<NumericT> & in)
{
  switch (cuarma::traits::handle(in).get_active_handle_id())
  {
  case cuarma::MAIN_MEMORY:
    cuarma::blas::host_based::reverse(in);
    break;


#ifdef CUARMA_WITH_CUDA
  case cuarma::CUDA_MEMORY:
    cuarma::blas::cuda::reverse(in);
    break;
#endif

  case cuarma::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

}
}
