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

/** @file cuarma/blas/cuda/fft_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of Fast Furier Transformation using cuda
*/

#include <cmath>
#include <cuarma/matrix.hpp>
#include <cuarma/vector.hpp>

#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/cuda/common.hpp"
#include "cuarma/blas/host_based/vector_operations.hpp"
#include "cuarma/blas/host_based/fft_operations.hpp"

namespace cuarma
{
namespace blas
{
namespace cuda
{
namespace detail
{
  namespace fft
  {
    const arma_size_t MAX_LOCAL_POINTS_NUM = 512;

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

  } //namespace fft
} //namespace detail

// addition
inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

// subtract
inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}
// division
template<typename SCALARTYPE>
inline __device__ float2 operator/(float2 a,SCALARTYPE b)
{
  return make_float2(a.x/b, a.y/b);
}

//multiplication
inline __device__ float2 operator*(float2 in1, float2 in2)
{
  return make_float2(in1.x * in2.x - in1.y * in2.y, in1.x * in2.y + in1.y * in2.x);
}

// addition
inline __host__ __device__ double2 operator+(double2 a, double2 b)
{
  return make_double2(a.x + b.x, a.y + b.y);
}

// subtraction
inline __host__ __device__ double2 operator-(double2 a, double2 b)
{
  return make_double2(a.x - b.x, a.y - b.y);
}

// division
template<typename SCALARTYPE>
inline __host__ __device__ double2 operator/(double2 a,SCALARTYPE b)
{
  return make_double2(a.x/b, a.y/b);
}

//multiplication
inline __host__ __device__ double2 operator*(double2 in1, double2 in2)
{
  return make_double2(in1.x * in2.x - in1.y * in2.y, in1.x * in2.y + in1.y * in2.x);
}

inline __device__ unsigned int get_reorder_num(unsigned int v, unsigned int bit_size)
{
  v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
  v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
  v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
  v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
  v = (v >> 16) | (v << 16);
  v = v >> (32 - bit_size);
  return v;
}

template<typename Numeric2T, typename NumericT>
__global__ void fft_direct(
    const Numeric2T * input,
    Numeric2T * output,
    unsigned int size,
    unsigned int stride,
    unsigned int batch_num,
    NumericT sign,
    bool is_row_major)
{

  const NumericT NUM_PI(3.14159265358979323846);

  for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
  {
    for (unsigned int k = blockIdx.x * blockDim.x + threadIdx.x; k < size; k += gridDim.x * blockDim.x)
    {
      Numeric2T f;
      f.x = 0;
      f.y = 0;

      for (unsigned int n = 0; n < size; n++)
      {
        Numeric2T in;
        if (!is_row_major)
          in = input[batch_id * stride + n];   //input index here
        else
          in = input[n * stride + batch_id];//input index here

        NumericT sn,cs;
        NumericT arg = sign * 2 * NUM_PI * k / size * n;
        sn = sin(arg);
        cs = cos(arg);

        Numeric2T ex;
        ex.x = cs;
        ex.y = sn;
        Numeric2T tmp;
        tmp.x = in.x * ex.x - in.y * ex.y;
        tmp.y = in.x * ex.y + in.y * ex.x;
        f = f + tmp;
      }

      if (!is_row_major)
        output[batch_id * stride + k] = f; // output index here
      else
        output[k * stride + batch_id] = f;// output index here
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
            arma_size_t size, arma_size_t stride, arma_size_t batch_num,
            NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  fft_direct<<<128,128>>>(reinterpret_cast<const numeric2_type *>(cuarma::cuda_arg(in)),
                          reinterpret_cast<      numeric2_type *>(cuarma::cuda_arg(out)),
                          static_cast<unsigned int>(size),
                          static_cast<unsigned int>(stride),
                          static_cast<unsigned int>(batch_num),
                          sign,
                          bool(data_order != cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR));
  CUARMA_CUDA_LAST_ERROR_CHECK("fft_direct");
}

/**
 * @brief Direct 2D algorithm for computing Fourier transformation.
 *
 * Works on any sizes of data.
 * Serial implementation has o(n^2) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void direct(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> const & in,
            cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>       & out,
            arma_size_t size, arma_size_t stride, arma_size_t batch_num,
            NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  fft_direct<<<128,128>>>(reinterpret_cast<const numeric2_type *>(cuarma::cuda_arg(in)),
                          reinterpret_cast<      numeric2_type *>(cuarma::cuda_arg(out)),
                          static_cast<unsigned int>(size),
                          static_cast<unsigned int>(stride),
                          static_cast<unsigned int>(batch_num),
                          sign,
                          bool(data_order != cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR));
  CUARMA_CUDA_LAST_ERROR_CHECK("fft_direct");
}

template<typename NumericT>
__global__ void fft_reorder(NumericT * input,
                            unsigned int bit_size,
                            unsigned int size,
                            unsigned int stride,
                            unsigned int batch_num,
                            bool is_row_major)
{

  unsigned int glb_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int glb_sz = gridDim.x * blockDim.x;

  for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
  {
    for (unsigned int i = glb_id; i < size; i += glb_sz)
    {
      unsigned int v = get_reorder_num(i, bit_size);

      if (i < v)
      {
        if (!is_row_major)
        {
          NumericT tmp = input[batch_id * stride + i];    // index
          input[batch_id * stride + i] = input[batch_id * stride + v];//index
          input[batch_id * stride + v] = tmp;//index
        }
        else
        {
          NumericT tmp = input[i * stride + batch_id];
          input[i * stride + batch_id] = input[v * stride + batch_id];
          input[v * stride + batch_id] = tmp;
        }
      }
    }
  }
}

/***
 * This function performs reorder of input data. Indexes are sorted in bit-reversal order.
 * Such reordering should be done before in-place FFT.
 */
template<typename NumericT, unsigned int AlignmentV>
void reorder(cuarma::vector<NumericT, AlignmentV> & in,
             arma_size_t size, arma_size_t stride, arma_size_t bits_datasize, arma_size_t batch_num,
             cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  fft_reorder<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(in)),
                           static_cast<unsigned int>(bits_datasize),
                           static_cast<unsigned int>(size),
                           static_cast<unsigned int>(stride),
                           static_cast<unsigned int>(batch_num),
                           static_cast<bool>(data_order));
  CUARMA_CUDA_LAST_ERROR_CHECK("fft_reorder");
}

template<typename Numeric2T, typename NumericT>
__global__ void fft_radix2_local(Numeric2T * input,
                                 unsigned int bit_size,
                                 unsigned int size,
                                 unsigned int stride,
                                 unsigned int batch_num,
                                 NumericT sign,
                                 bool is_row_major)
{
  __shared__ Numeric2T lcl_input[1024];
  unsigned int grp_id = blockIdx.x;
  unsigned int grp_num = gridDim.x;

  unsigned int lcl_sz = blockDim.x;
  unsigned int lcl_id = threadIdx.x;
  const NumericT NUM_PI(3.14159265358979323846);

  for (unsigned int batch_id = grp_id; batch_id < batch_num; batch_id += grp_num)
  {
    for (unsigned int p = lcl_id; p < size; p += lcl_sz)
    {
      unsigned int v = get_reorder_num(p, bit_size);
      if (!is_row_major)
        lcl_input[v] = input[batch_id * stride + p];
      else
        lcl_input[v] = input[p * stride + batch_id];
    }

    __syncthreads();

    //performs Cooley-Tukey FFT on local arrayfft
    for (unsigned int s = 0; s < bit_size; s++)
    {
      unsigned int ss = 1 << s;
      NumericT cs, sn;
      for (unsigned int tid = lcl_id; tid < size; tid += lcl_sz)
      {
        unsigned int group = (tid & (ss - 1));
        unsigned int pos = ((tid >> s) << (s + 1)) + group;

        Numeric2T in1 = lcl_input[pos];
        Numeric2T in2 = lcl_input[pos + ss];

        NumericT arg = group * sign * NUM_PI / ss;

        sn = sin(arg);
        cs = cos(arg);
        Numeric2T ex;
        ex.x = cs;
        ex.y = sn;

        Numeric2T tmp;
        tmp.x = in2.x * ex.x - in2.y * ex.y;
        tmp.y = in2.x * ex.y + in2.y * ex.x;

        lcl_input[pos + ss] = in1 - tmp;
        lcl_input[pos]      = in1 + tmp;
      }
      __syncthreads();
    }

    //copy local array back to global memory
    for (unsigned int p = lcl_id; p < size; p += lcl_sz)
    {
      if (!is_row_major)
        input[batch_id * stride + p] = lcl_input[p];   //index
      else
        input[p * stride + batch_id] = lcl_input[p];
    }

  }
}

template<typename Numeric2T, typename NumericT>
__global__ void fft_radix2(Numeric2T * input,
                           unsigned int s,
                           unsigned int bit_size,
                           unsigned int size,
                           unsigned int stride,
                           unsigned int batch_num,
                           NumericT sign,
                           bool is_row_major)
{

  unsigned int ss = 1 << s;
  unsigned int half_size = size >> 1;

  NumericT cs, sn;
  const NumericT NUM_PI(3.14159265358979323846);

  unsigned int glb_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int glb_sz = gridDim.x * blockDim.x;

  for (unsigned int batch_id = 0; batch_id < batch_num; batch_id++)
  {
    for (unsigned int tid = glb_id; tid < half_size; tid += glb_sz)
    {
      unsigned int group = (tid & (ss - 1));
      unsigned int pos = ((tid >> s) << (s + 1)) + group;
      Numeric2T in1;
      Numeric2T in2;
      unsigned int offset;
      if (!is_row_major)
      {
        offset = batch_id * stride + pos;
        in1 = input[offset];   //index
        in2 = input[offset + ss];//index
      }
      else
      {
        offset = pos * stride + batch_id;
        in1 = input[offset];   //index
        in2 = input[offset + ss * stride];//index
      }

      NumericT arg = group * sign * NUM_PI / ss;

      sn = sin(arg);
      cs = cos(arg);

      Numeric2T ex;
      ex.x = cs;
      ex.y = sn;

      Numeric2T tmp;
      tmp.x = in2.x * ex.x - in2.y * ex.y;
      tmp.y = in2.x * ex.y + in2.y * ex.x;

      if (!is_row_major)
        input[offset + ss] = in1 - tmp;  //index
      else
        input[offset + ss * stride] = in1 - tmp;  //index
      input[offset] = in1 + tmp;  //index
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
void radix2(cuarma::vector<NumericT, AlignmentV> & in,
            arma_size_t size, arma_size_t stride, arma_size_t batch_num, NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  unsigned int bit_size = cuarma::blas::cuda::detail::fft::num_bits(size);

  if (size <= cuarma::blas::cuda::detail::fft::MAX_LOCAL_POINTS_NUM)
  {
    fft_radix2_local<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(in)),
                                  static_cast<unsigned int>(bit_size),
                                  static_cast<unsigned int>(size),
                                  static_cast<unsigned int>(stride),
                                  static_cast<unsigned int>(batch_num),
                                  static_cast<NumericT>(sign),
                                  static_cast<bool>(data_order));
    CUARMA_CUDA_LAST_ERROR_CHECK("fft_radix2_local");
  }
  else
  {
    fft_reorder<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(in)),
                             static_cast<unsigned int>(bit_size),
                             static_cast<unsigned int>(size),
                             static_cast<unsigned int>(stride),
                             static_cast<unsigned int>(batch_num),
                             static_cast<bool>(data_order));
    CUARMA_CUDA_LAST_ERROR_CHECK("fft_reorder");

    for (arma_size_t step = 0; step < bit_size; step++)
    {
      fft_radix2<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(in)),
                              static_cast<unsigned int>(step),
                              static_cast<unsigned int>(bit_size),
                              static_cast<unsigned int>(size),
                              static_cast<unsigned int>(stride),
                              static_cast<unsigned int>(batch_num),
                              sign,
                              static_cast<bool>(data_order));
      CUARMA_CUDA_LAST_ERROR_CHECK("fft_radix2");
    }
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
void radix2(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV>& in,
            arma_size_t size, arma_size_t stride, arma_size_t batch_num, NumericT sign = NumericT(-1),
            cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = cuarma::blas::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  unsigned int bit_size = cuarma::blas::cuda::detail::fft::num_bits(size);

  if (size <= cuarma::blas::cuda::detail::fft::MAX_LOCAL_POINTS_NUM)
  {
    fft_radix2_local<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(in)),
                                  static_cast<unsigned int>(bit_size),
                                  static_cast<unsigned int>(size),
                                  static_cast<unsigned int>(stride),
                                  static_cast<unsigned int>(batch_num),
                                  sign,
                                  static_cast<bool>(data_order));
    CUARMA_CUDA_LAST_ERROR_CHECK("fft_radix2_local");
  }
  else
  {
    fft_reorder<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(in)),
                             static_cast<unsigned int>(bit_size),
                             static_cast<unsigned int>(size),
                             static_cast<unsigned int>(stride),
                             static_cast<unsigned int>(batch_num),
                             static_cast<bool>(data_order));
    CUARMA_CUDA_LAST_ERROR_CHECK("fft_reorder");
    for (arma_size_t step = 0; step < bit_size; step++)
    {
      fft_radix2<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(in)),
                              static_cast<unsigned int>(step),
                              static_cast<unsigned int>(bit_size),
                              static_cast<unsigned int>(size),
                              static_cast<unsigned int>(stride),
                              static_cast<unsigned int>(batch_num),
                              sign,
                              static_cast<bool>(data_order));
      CUARMA_CUDA_LAST_ERROR_CHECK("fft_radix2");
    }
  }
}

template<typename Numeric2T, typename NumericT>
__global__ void bluestein_post(Numeric2T * Z, Numeric2T * out, unsigned int size, NumericT sign)
{
  unsigned int glb_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int glb_sz =gridDim.x * blockDim.x;

  unsigned int double_size = size << 1;
  NumericT sn_a, cs_a;
  const NumericT NUM_PI(3.14159265358979323846);

  for (unsigned int i = glb_id; i < size; i += glb_sz)
  {
    unsigned int rm = i * i % (double_size);
    NumericT angle = (NumericT)rm / size * (-NUM_PI);

    sn_a = sin(angle);
    cs_a= cos(angle);

    Numeric2T b_i;
    b_i.x = cs_a;
    b_i.y = sn_a;
    out[i].x = Z[i].x * b_i.x - Z[i].y * b_i.y;
    out[i].y = Z[i].x * b_i.y + Z[i].y * b_i.x;
  }
}

template<typename Numeric2T, typename NumericT>
__global__ void bluestein_pre(Numeric2T * input, Numeric2T * A, Numeric2T * B,
                              unsigned int size, unsigned int ext_size, NumericT sign)
{
  unsigned int glb_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int glb_sz = gridDim.x * blockDim.x;

  unsigned int double_size = size << 1;

  NumericT sn_a, cs_a;
  const NumericT NUM_PI(3.14159265358979323846);

  for (unsigned int i = glb_id; i < size; i += glb_sz)
  {
    unsigned int rm = i * i % (double_size);
    NumericT angle = (NumericT)rm / size * NUM_PI;

    sn_a = sin(-angle);
    cs_a= cos(-angle);

    Numeric2T a_i;
    a_i.x =cs_a;
    a_i.y =sn_a;

    Numeric2T b_i;
    b_i.x =cs_a;
    b_i.y =-sn_a;

    A[i].x = input[i].x * a_i.x - input[i].y * a_i.y;
    A[i].y = input[i].x * a_i.y + input[i].y * a_i.x;
    B[i] = b_i;

    // very bad instruction, to be fixed
    if (i)
    B[ext_size - i] = b_i;
  }
}

template<typename NumericT>
__global__ void zero2(NumericT * input1, NumericT * input2, unsigned int size)
{
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
  {
    input1[i].x = 0;
    input1[i].y = 0;

    input2[i].x = 0;
    input2[i].y = 0;
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
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  arma_size_t size = in.size() >> 1;
  arma_size_t ext_size = cuarma::blas::cuda::detail::fft::next_power_2(2 * size - 1);

  cuarma::vector<NumericT, AlignmentV> A(ext_size << 1);
  cuarma::vector<NumericT, AlignmentV> B(ext_size << 1);
  cuarma::vector<NumericT, AlignmentV> Z(ext_size << 1);

  zero2<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(A)),
                     reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(B)),
                     static_cast<unsigned int>(ext_size));
  CUARMA_CUDA_LAST_ERROR_CHECK("zero2");

  bluestein_pre<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(in)),
                             reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(A)),
                             reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(B)),
                             static_cast<unsigned int>(size),
                             static_cast<unsigned int>(ext_size),
                             NumericT(1));
  CUARMA_CUDA_LAST_ERROR_CHECK("bluestein_pre");

  cuarma::blas::convolve_i(A, B, Z);

  bluestein_post<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(Z)),
                              reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(out)),
                              static_cast<unsigned int>(size),
                              NumericT(1));
  CUARMA_CUDA_LAST_ERROR_CHECK("bluestein_post");
}

template<typename NumericT>
__global__ void fft_mult_vec(const NumericT * input1,
                             const NumericT * input2,
                             NumericT * output,
                             unsigned int size)
{
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
  {
    NumericT in1 = input1[i];
    NumericT in2 = input2[i];
    output[i] = in1 * in2;
  }
}

/**
 * @brief Mutiply two complex vectors and store result in output
 */
template<typename NumericT, unsigned int AlignmentV>
void multiply_complex(cuarma::vector<NumericT, AlignmentV> const & input1,
                      cuarma::vector<NumericT, AlignmentV> const & input2,
                      cuarma::vector<NumericT, AlignmentV> & output)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  arma_size_t size = input1.size() / 2;

  fft_mult_vec<<<128,128>>>(reinterpret_cast<const numeric2_type *>(cuarma::cuda_arg(input1)),
                            reinterpret_cast<const numeric2_type *>(cuarma::cuda_arg(input2)),
                            reinterpret_cast<      numeric2_type *>(cuarma::cuda_arg(output)),
                            static_cast<unsigned int>(size));
  CUARMA_CUDA_LAST_ERROR_CHECK("fft_mult_vec");
}

template<typename Numeric2T, typename NumericT>
__global__ void fft_div_vec_scalar(Numeric2T * input1, unsigned int size, NumericT factor)
{
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x*blockDim.x)
    input1[i] = input1[i]/factor;
}

/**
 * @brief Normalize vector on with his own size
 */
template<typename NumericT, unsigned int AlignmentV>
void normalize(cuarma::vector<NumericT, AlignmentV> & input)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  arma_size_t size = input.size() >> 1;
  NumericT norm_factor = static_cast<NumericT>(size);
  fft_div_vec_scalar<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(input)),
                                  static_cast<unsigned int>(size),
                                  norm_factor);
  CUARMA_CUDA_LAST_ERROR_CHECK("fft_div_vec_scalar");
}

template<typename NumericT>
__global__ void transpose(const NumericT * input,
                          NumericT * output,
                          unsigned int row_num,
                          unsigned int col_num)
{
  unsigned int size = row_num * col_num;
  for (unsigned int i =blockIdx.x * blockDim.x + threadIdx.x; i < size; i+= gridDim.x * blockDim.x)
  {
    unsigned int row = i / col_num;
    unsigned int col = i - row*col_num;
    unsigned int new_pos = col * row_num + row;
    output[new_pos] = input[i];
  }
}

/**
 * @brief Transpose matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> const & input,
               cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> & output)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  transpose<<<128,128>>>(reinterpret_cast<const numeric2_type *>(cuarma::cuda_arg(input)),
                         reinterpret_cast<      numeric2_type *>(cuarma::cuda_arg(output)),
                         static_cast<unsigned int>(input.internal_size1()>>1),
                         static_cast<unsigned int>(input.internal_size2()>>1));
  CUARMA_CUDA_LAST_ERROR_CHECK("transpose");

}

template<typename NumericT>
__global__ void transpose_inplace( NumericT * input, unsigned int row_num, unsigned int col_num)
{
  unsigned int size = row_num * col_num;
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i+= gridDim.x * blockDim.x)
  {
    unsigned int row = i / col_num;
    unsigned int col = i - row*col_num;
    unsigned int new_pos = col * row_num + row;
    if (i < new_pos)
    {
      NumericT val = input[i];
      input[i] = input[new_pos];
      input[new_pos] = val;
    }
  }
}

/**
 * @brief Inplace_transpose matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(cuarma::matrix<NumericT, cuarma::row_major, AlignmentV> & input)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  transpose_inplace<<<128,128>>>(reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(input)),
                                 static_cast<unsigned int>(input.internal_size1()>>1),
                                 static_cast<unsigned int>(input.internal_size2() >> 1));
  CUARMA_CUDA_LAST_ERROR_CHECK("transpose_inplace");

}

template<typename RealT,typename ComplexT>
__global__ void real_to_complex(const RealT * in, ComplexT * out, unsigned int size)
{
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
  {
    ComplexT val;
    val.x = in[i];
    val.y = 0;
    out[i] = val;
  }
}

/**
 * @brief Create complex vector from real vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void real_to_complex(cuarma::vector_base<NumericT> const & in,
                     cuarma::vector_base<NumericT> & out, arma_size_t size)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  real_to_complex<<<128,128>>>(cuarma::cuda_arg(in),
                               reinterpret_cast<numeric2_type *>(cuarma::cuda_arg(out)),
                               static_cast<unsigned int>(size));
  CUARMA_CUDA_LAST_ERROR_CHECK("real_to_complex");
}

template<typename ComplexT,typename RealT>
__global__ void complex_to_real(const ComplexT * in, RealT * out, unsigned int size)
{
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
    out[i] = in[i].x;
}

/**
 * @brief Create real vector from complex vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void complex_to_real(cuarma::vector_base<NumericT> const & in,
                     cuarma::vector_base<NumericT>& out, arma_size_t size)
{
  typedef typename cuarma::blas::cuda::detail::type_to_type2<NumericT>::type  numeric2_type;

  complex_to_real<<<128,128>>>(reinterpret_cast<const numeric2_type *>(cuarma::cuda_arg(in)),
                               cuarma::cuda_arg(out),
                               static_cast<unsigned int>(size));
  CUARMA_CUDA_LAST_ERROR_CHECK("complex_to_real");

}

template<typename NumericT>
__global__ void reverse_inplace(NumericT * vec, unsigned int size)
{
  for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < (size >> 1); i+=gridDim.x * blockDim.x)
  {
    NumericT val1 = vec[i];
    NumericT val2 = vec[size - i - 1];
    vec[i] = val2;
    vec[size - i - 1] = val1;
  }
}

/**
 * @brief Reverse vector to oposite order and save it in input vector
 */
template<typename NumericT>
void reverse(cuarma::vector_base<NumericT>& in)
{
  arma_size_t size = in.size();
  reverse_inplace<<<128,128>>>(cuarma::cuda_arg(in), static_cast<unsigned int>(size));
  CUARMA_CUDA_LAST_ERROR_CHECK("reverse_inplace");
}

}  //namespace cuda
}  //namespace blas
}  //namespace cuarma
