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

/** @file cuarma/blas/cuda/sparse_matrix_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of operations using sparse matrices using CUDA
*/

#include <stdexcept>

#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/cuda/common.hpp"
#include "cuarma/tools/timer.hpp"
#include "cuarma/blas/cuda/sparse_matrix_operations_solve.hpp"
#include "thrust/device_ptr.h"
#include "thrust/scan.h"

namespace cuarma
{
namespace blas
{
namespace cuda
{

/** @brief Loads a value from the specified address. With CUDA arch 3.5 and above the value is also stored in global constant memory for later reuse */
template<typename NumericT>
static inline __device__ NumericT load_and_cache(const NumericT *address)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  return __ldg(address);
#else
  return *address;
#endif
}


//
// Stage 1: Obtain upper bound for number of elements per row in C:
//
template<typename IndexT>
__device__ IndexT round_to_next_power_of_2(IndexT val)
{
  if (val > 32)
    return 64; // just to indicate that we need to split/factor the matrix!
  else if (val > 16)
    return 32;
  else if (val > 8)
    return 16;
  else if (val > 4)
    return 8;
  else if (val > 2)
    return 4;
  else if (val > 1)
    return 2;
  else
    return 1;
}

template<typename IndexT>
__global__ void compressed_matrix_gemm_stage_1(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          const IndexT * B_row_indices,
          IndexT *subwarpsize_per_group,
          IndexT *max_nnz_row_A_per_group,
          IndexT *max_nnz_row_B_per_group)
{
  unsigned int subwarpsize_in_thread = 0;
  unsigned int max_nnz_row_A = 0;
  unsigned int max_nnz_row_B = 0;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + threadIdx.x; row < row_per_group_end; row += blockDim.x)
  {
    unsigned int A_row_start = A_row_indices[row];
    unsigned int A_row_end   = A_row_indices[row+1];
    unsigned int row_num = A_row_end - A_row_start;
    subwarpsize_in_thread = max(A_row_end - A_row_start, subwarpsize_in_thread);
    max_nnz_row_A = max(max_nnz_row_A, row_num);
    for (unsigned int j = A_row_start; j < A_row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      unsigned int row_len_B = B_row_indices[col + 1] - B_row_indices[col];
      max_nnz_row_B = max(row_len_B, max_nnz_row_B);
    }
  }

  // reduction to obtain maximum in thread block
  __shared__ unsigned int shared_subwarpsize[256];
  __shared__ unsigned int shared_max_nnz_row_A[256];
  __shared__ unsigned int shared_max_nnz_row_B[256];

    shared_subwarpsize[threadIdx.x] = subwarpsize_in_thread;
  shared_max_nnz_row_A[threadIdx.x] = max_nnz_row_A;
  shared_max_nnz_row_B[threadIdx.x] = max_nnz_row_B;
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
        shared_subwarpsize[threadIdx.x] = max(  shared_subwarpsize[threadIdx.x],   shared_subwarpsize[threadIdx.x + stride]);
      shared_max_nnz_row_A[threadIdx.x] = max(shared_max_nnz_row_A[threadIdx.x], shared_max_nnz_row_A[threadIdx.x + stride]);
      shared_max_nnz_row_B[threadIdx.x] = max(shared_max_nnz_row_B[threadIdx.x], shared_max_nnz_row_B[threadIdx.x + stride]);
    }
  }

  if (threadIdx.x == 0)
  {
      subwarpsize_per_group[blockIdx.x] = round_to_next_power_of_2(shared_subwarpsize[0]);
    max_nnz_row_A_per_group[blockIdx.x] = shared_max_nnz_row_A[0];
    max_nnz_row_B_per_group[blockIdx.x] = shared_max_nnz_row_B[0];
  }
}

//
// Stage 2: Determine sparsity pattern of C
//

// Using warp shuffle routines (CUDA arch 3.5)
template<unsigned int SubWarpSizeV, typename IndexT>
__device__ IndexT subwarp_minimum_shuffle(IndexT min_index)
{
  for (unsigned int i = SubWarpSizeV/2; i >= 1; i /= 2)
    min_index = min(min_index, __shfl_xor((int)min_index, (int)i));
  return min_index;
}

// Using shared memory
template<unsigned int SubWarpSizeV, typename IndexT>
__device__ IndexT subwarp_minimum_shared(IndexT min_index, IndexT id_in_warp, IndexT *shared_buffer)
{
  shared_buffer[threadIdx.x] = min_index;
  for (unsigned int i = SubWarpSizeV/2; i >= 1; i /= 2)
    shared_buffer[threadIdx.x] = min(shared_buffer[threadIdx.x], shared_buffer[(threadIdx.x + i) % 512]);
  return shared_buffer[threadIdx.x - id_in_warp];
}


template<unsigned int SubWarpSizeV, typename IndexT>
__global__ void compressed_matrix_gemm_stage_2(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          IndexT B_size2,
          IndexT * C_row_indices)
{
  __shared__ unsigned int shared_buffer[512];

  unsigned int num_warps  =  blockDim.x / SubWarpSizeV;
  unsigned int warp_id    = threadIdx.x / SubWarpSizeV;
  unsigned int id_in_warp = threadIdx.x % SubWarpSizeV;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    unsigned int my_row_B = row_A_start + id_in_warp;
    unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
    unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index) : 0;
    unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;

    unsigned int num_nnz = 0;
    if (row_A_end - row_A_start > 1) // zero or no row can be processed faster
    {
      unsigned int current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;

      while (1)
      {
        // determine current minimum (warp shuffle)
        unsigned int min_index = current_front_index;
        min_index = subwarp_minimum_shared<SubWarpSizeV>(min_index, id_in_warp, shared_buffer);

        if (min_index == B_size2)
          break;

        // update front:
        if (current_front_index == min_index)
        {
          ++row_B_start;
          current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;
        }

        ++num_nnz;
      }
    }
    else
    {
      num_nnz = row_B_end - row_B_start;
    }

    if (id_in_warp == 0)
      C_row_indices[row] = num_nnz;
  }

}


//
// Stage 3: Fill C with values
//

// Using warp shuffle routines (CUDA arch 3.5)
template<unsigned int SubWarpSizeV, typename NumericT>
__device__ NumericT subwarp_accumulate_shuffle(NumericT output_value)
{
  for (unsigned int i = SubWarpSizeV/2; i >= 1; i /= 2)
    output_value += __shfl_xor((int)output_value, (int)i);
  return output_value;
}

// Using shared memory
template<unsigned int SubWarpSizeV, typename NumericT>
__device__ NumericT subwarp_accumulate_shared(NumericT output_value, unsigned int id_in_warp, NumericT *shared_buffer)
{
  shared_buffer[threadIdx.x] = output_value;
  for (unsigned int i = SubWarpSizeV/2; i >= 1; i /= 2)
    shared_buffer[threadIdx.x] += shared_buffer[(threadIdx.x + i) % 512];
  return shared_buffer[threadIdx.x - id_in_warp];
}


template<unsigned int SubWarpSizeV, typename IndexT, typename NumericT>
__global__ void compressed_matrix_gemm_stage_3(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          const NumericT * A_elements,
          IndexT A_size1,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          const NumericT * B_elements,
          IndexT B_size2,
          IndexT const * C_row_indices,
          IndexT * C_col_indices,
          NumericT * C_elements)
{
  __shared__ unsigned int shared_indices[512];
  __shared__ NumericT     shared_values[512];

  unsigned int num_warps  =  blockDim.x / SubWarpSizeV;
  unsigned int warp_id    = threadIdx.x / SubWarpSizeV;
  unsigned int id_in_warp = threadIdx.x % SubWarpSizeV;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    unsigned int my_row_B = row_A_start + ((row_A_end - row_A_start > 1) ? id_in_warp : 0); // special case: single row
    unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
    unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index)     : 0;
    unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;
    NumericT val_A = (my_row_B < row_A_end) ? A_elements[my_row_B] : 0;

    unsigned int index_in_C = C_row_indices[row];

    if (row_A_end - row_A_start > 1)
    {
      unsigned int current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;
      NumericT     current_front_value = (row_B_start < row_B_end) ? load_and_cache(B_elements    + row_B_start) : 0;

      unsigned int index_buffer = 0;
      NumericT     value_buffer = 0;
      unsigned int buffer_size = 0;
      while (1)
      {
        // determine current minimum:
        unsigned int min_index = subwarp_minimum_shared<SubWarpSizeV>(current_front_index, id_in_warp, shared_indices);

        if (min_index == B_size2) // done
          break;

        // compute entry in C:
        NumericT output_value = (current_front_index == min_index) ? val_A * current_front_value : 0;
        output_value = subwarp_accumulate_shared<SubWarpSizeV>(output_value, id_in_warp, shared_values);

        // update front:
        if (current_front_index == min_index)
        {
          ++row_B_start;
          current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;
          current_front_value = (row_B_start < row_B_end) ? load_and_cache(B_elements    + row_B_start) : 0;
        }

        // write current front to register buffer:
        index_buffer = (id_in_warp == buffer_size) ? min_index    : index_buffer;
        value_buffer = (id_in_warp == buffer_size) ? output_value : value_buffer;
        ++buffer_size;

        // flush register buffer via a coalesced write once full:
        if (buffer_size == SubWarpSizeV)
        {
          C_col_indices[index_in_C + id_in_warp] = index_buffer;
          C_elements[index_in_C + id_in_warp]    = value_buffer;
        }

        index_in_C += (buffer_size == SubWarpSizeV) ? SubWarpSizeV : 0;
        buffer_size = (buffer_size == SubWarpSizeV) ?           0  : buffer_size;
      }

      // write remaining entries in register buffer to C:
      if (id_in_warp < buffer_size)
      {
        C_col_indices[index_in_C + id_in_warp] = index_buffer;
        C_elements[index_in_C + id_in_warp]  = value_buffer;
      }
    }
    else // write respective row using the full subwarp:
    {
      for (unsigned int i = row_B_start + id_in_warp; i < row_B_end; i += SubWarpSizeV)
      {
        C_col_indices[index_in_C + id_in_warp] = load_and_cache(B_col_indices + i);
        C_elements[index_in_C + id_in_warp]    = val_A * load_and_cache(B_elements    + i);
        index_in_C += SubWarpSizeV;
      }
    }

  }

}



//
// Decomposition kernels:
//
template<typename IndexT>
__global__ void compressed_matrix_gemm_decompose_1(
          const IndexT * A_row_indices,
          IndexT A_size1,
          IndexT max_per_row,
          IndexT *chunks_per_row)
{
  for (IndexT i = blockIdx.x * blockDim.x + threadIdx.x; i < A_size1; i += blockDim.x * gridDim.x)
  {
    IndexT num_entries = A_row_indices[i+1] - A_row_indices[i];
    chunks_per_row[i] = (num_entries < max_per_row) ? 1 : ((num_entries - 1)/ max_per_row + 1);
  }
}


template<typename IndexT, typename NumericT>
__global__ void compressed_matrix_gemm_A2(
          IndexT * A2_row_indices,
          IndexT * A2_col_indices,
          NumericT * A2_elements,
          IndexT A2_size1,
          IndexT *new_row_buffer)
{
  for (IndexT i = blockIdx.x * blockDim.x + threadIdx.x; i < A2_size1; i += blockDim.x * gridDim.x)
  {
    unsigned int index_start = new_row_buffer[i];
    unsigned int index_stop  = new_row_buffer[i+1];

    A2_row_indices[i] = index_start;

    for (IndexT j = index_start; j < index_stop; ++j)
    {
      A2_col_indices[j] = j;
      A2_elements[j] = NumericT(1);
    }
  }

  // write last entry in row_buffer with global thread 0:
  if (threadIdx.x == 0 && blockIdx.x == 0)
    A2_row_indices[A2_size1] = new_row_buffer[A2_size1];
}

template<typename IndexT, typename NumericT>
__global__ void compressed_matrix_gemm_G1(
          IndexT * G1_row_indices,
          IndexT * G1_col_indices,
          NumericT * G1_elements,
          IndexT G1_size1,
          IndexT const *A_row_indices,
          IndexT const *A_col_indices,
          NumericT const *A_elements,
          IndexT A_size1,
          IndexT A_nnz,
          IndexT max_per_row,
          IndexT *new_row_buffer)
{
  // Part 1: Copy column indices and entries:
  for (IndexT i = blockIdx.x * blockDim.x + threadIdx.x; i < A_nnz; i += blockDim.x * gridDim.x)
  {
    G1_col_indices[i] = A_col_indices[i];
    G1_elements[i]    = A_elements[i];
  }

  // Part 2: Derive new row indicies:
  for (IndexT i = blockIdx.x * blockDim.x + threadIdx.x; i < A_size1; i += blockDim.x * gridDim.x)
  {
    unsigned int old_start = A_row_indices[i];
    unsigned int new_start = new_row_buffer[i];
    unsigned int row_chunks = new_row_buffer[i+1] - new_start;

    for (IndexT j=0; j<row_chunks; ++j)
      G1_row_indices[new_start + j] = old_start + j * max_per_row;
  }

  // write last entry in row_buffer with global thread 0:
  if (threadIdx.x == 0 && blockIdx.x == 0)
    G1_row_indices[G1_size1] = A_row_indices[A_size1];
}


/** @brief Carries out sparse_matrix-sparse_matrix multiplication for CSR matrices
*
* Implementation of the convenience expression C = prod(A, B);
* Based on computing C(i, :) = A(i, :) * B via merging the respective rows of B
*
* @param A     Left factor
* @param B     Right factor
* @param C     Result matrix
*/
template<class NumericT, unsigned int AlignmentV>
void prod_impl(cuarma::compressed_matrix<NumericT, AlignmentV> const & A,
               cuarma::compressed_matrix<NumericT, AlignmentV> const & B,
               cuarma::compressed_matrix<NumericT, AlignmentV> & C)
{
  C.resize(A.size1(), B.size2(), false);

  unsigned int blocknum = 256;
  unsigned int threadnum = 128;

  cuarma::vector<unsigned int> subwarp_sizes(blocknum, cuarma::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
  cuarma::vector<unsigned int> max_nnz_row_A(blocknum, cuarma::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
  cuarma::vector<unsigned int> max_nnz_row_B(blocknum, cuarma::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

#ifdef CUARMA_WITH_SPGEMM_CUDA_TIMINGS
  cuarma::tools::timer timer;
#endif

  //
  // Stage 1: Determine upper bound for number of nonzeros
  //
#ifdef CUARMA_WITH_SPGEMM_CUDA_TIMINGS
  cudaDeviceSynchronize();
  timer.start();
#endif

  compressed_matrix_gemm_stage_1<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
                                                          cuarma::cuda_arg<unsigned int>(A.handle2()),
                                                          static_cast<unsigned int>(A.size1()),
                                                          cuarma::cuda_arg<unsigned int>(B.handle1()),
                                                          cuarma::cuda_arg(subwarp_sizes),
                                                          cuarma::cuda_arg(max_nnz_row_A),
                                                          cuarma::cuda_arg(max_nnz_row_B)
                                                         );
  CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_1");
#ifdef CUARMA_WITH_SPGEMM_CUDA_TIMINGS
  cudaDeviceSynchronize();
  std::cout << "Stage 1 device: " << timer.get() << std::endl;
  timer.start();
#endif

  subwarp_sizes.switch_memory_context(cuarma::context(MAIN_MEMORY));
  unsigned int * subwarp_sizes_ptr = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(subwarp_sizes.handle());

  max_nnz_row_A.switch_memory_context(cuarma::context(MAIN_MEMORY));
  unsigned int const * max_nnz_row_A_ptr = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(max_nnz_row_A.handle());

  max_nnz_row_B.switch_memory_context(cuarma::context(MAIN_MEMORY));
  unsigned int const * max_nnz_row_B_ptr = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(max_nnz_row_B.handle());

  //std::cout << "Subwarp sizes: " << subwarp_sizes << std::endl;

  cuarma::vector<unsigned int> scratchpad_offsets(blocknum, cuarma::context(MAIN_MEMORY)); // upper bound for the nonzeros per row encountered for each work group
  unsigned int * scratchpad_offsets_ptr = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(scratchpad_offsets.handle());

  unsigned int max_subwarp_size = 0;
  unsigned int A_max_nnz_per_row = 0;
  unsigned int scratchpad_offset = 0;
  //std::cout << "Scratchpad offsets: " << std::endl;
  for (std::size_t i=0; i<subwarp_sizes.size(); ++i)
  {
    max_subwarp_size = std::max(max_subwarp_size, subwarp_sizes_ptr[i]);
    A_max_nnz_per_row = std::max(A_max_nnz_per_row, max_nnz_row_A_ptr[i]);

    scratchpad_offsets_ptr[i] = scratchpad_offset;
    //std::cout << scratchpad_offset << " (with " << (max_nnz_row_A_ptr[i] / subwarp_sizes_ptr[i] + 1) << " warp reloads per group at " << max_nnz_row_A_ptr[i] << " max rows, "
    //                                            << upper_bound_nonzeros_per_row_C_ptr[i] << " row length, "
    //                                            << (256 / subwarp_sizes_ptr[i]) << " warps per group " << std::endl;
    unsigned int max_warp_reloads = max_nnz_row_A_ptr[i] / subwarp_sizes_ptr[i] + 1;
    unsigned int max_row_length_after_warp_merge = subwarp_sizes_ptr[i] * max_nnz_row_B_ptr[i];
    unsigned int warps_in_group = threadnum / subwarp_sizes_ptr[i];
    scratchpad_offset +=  max_warp_reloads
                        * max_row_length_after_warp_merge
                        * warps_in_group;
  }
  //std::cout << "Scratchpad memory for indices: " << scratchpad_offset << " entries (" << scratchpad_offset * sizeof(unsigned int) * 1e-6 << " MB)" << std::endl;

  if (max_subwarp_size > 32)
  {
    // determine augmented size:
    unsigned int max_entries_in_G = 1024;
    if (A_max_nnz_per_row <= 512*512)
      max_entries_in_G = 512;
    if (A_max_nnz_per_row <= 256*256)
      max_entries_in_G = 256;
    if (A_max_nnz_per_row <= 128*128)
      max_entries_in_G = 128;
    if (A_max_nnz_per_row <= 64*64)
      max_entries_in_G = 64;

    cuarma::vector<unsigned int> exclusive_scan_helper(A.size1() + 1, cuarma::traits::context(A));
    compressed_matrix_gemm_decompose_1<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
                                                                static_cast<unsigned int>(A.size1()),
                                                                static_cast<unsigned int>(max_entries_in_G),
                                                                cuarma::cuda_arg(exclusive_scan_helper)
                                                               );
    CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_decompose_1");

    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(cuarma::cuda_arg(exclusive_scan_helper)),
                           thrust::device_ptr<unsigned int>(cuarma::cuda_arg(exclusive_scan_helper) + exclusive_scan_helper.size()),
                           thrust::device_ptr<unsigned int>(cuarma::cuda_arg(exclusive_scan_helper)));

    unsigned int augmented_size = exclusive_scan_helper[A.size1()];

    // split A = A2 * G1
    cuarma::compressed_matrix<NumericT, AlignmentV> A2(A.size1(), augmented_size, augmented_size, cuarma::traits::context(A));
    cuarma::compressed_matrix<NumericT, AlignmentV> G1(augmented_size, A.size2(),        A.nnz(), cuarma::traits::context(A));

    // fill A2:
    compressed_matrix_gemm_A2<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A2.handle1()),
                                                       cuarma::cuda_arg<unsigned int>(A2.handle2()),
                                                       cuarma::cuda_arg<NumericT>(A2.handle()),
                                                       static_cast<unsigned int>(A2.size1()),
                                                       cuarma::cuda_arg(exclusive_scan_helper)
                                                      );
    CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_A2");

    // fill G1:
    compressed_matrix_gemm_G1<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(G1.handle1()),
                                                       cuarma::cuda_arg<unsigned int>(G1.handle2()),
                                                       cuarma::cuda_arg<NumericT>(G1.handle()),
                                                       static_cast<unsigned int>(G1.size1()),
                                                       cuarma::cuda_arg<unsigned int>(A.handle1()),
                                                       cuarma::cuda_arg<unsigned int>(A.handle2()),
                                                       cuarma::cuda_arg<NumericT>(A.handle()),
                                                       static_cast<unsigned int>(A.size1()),
                                                       static_cast<unsigned int>(A.nnz()),
                                                       static_cast<unsigned int>(max_entries_in_G),
                                                       cuarma::cuda_arg(exclusive_scan_helper)
                                                      );
    CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_G1");

    // compute tmp = G1 * B;
    // C = A2 * tmp;
    cuarma::compressed_matrix<NumericT, AlignmentV> tmp(G1.size1(), B.size2(), 0, cuarma::traits::context(A));
    prod_impl(G1, B, tmp); // this runs a standard RMerge without decomposition of G1
    prod_impl(A2, tmp, C); // this may split A2 again
    return;
  }

  subwarp_sizes.switch_memory_context(cuarma::traits::context(A));
  max_nnz_row_A.switch_memory_context(cuarma::traits::context(A));
  max_nnz_row_B.switch_memory_context(cuarma::traits::context(A));
  scratchpad_offsets.switch_memory_context(cuarma::traits::context(A));

  cuarma::vector<unsigned int> scratchpad_indices(scratchpad_offset, cuarma::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

#ifdef CUARMA_WITH_SPGEMM_CUDA_TIMINGS
  std::cout << "Intermediate host stage: " << timer.get() << std::endl;
  timer.start();
#endif

  //
  // Stage 2: Determine pattern of C
  //

  compressed_matrix_gemm_stage_2<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
                                                         cuarma::cuda_arg<unsigned int>(A.handle2()),
                                                         static_cast<unsigned int>(A.size1()),
                                                         cuarma::cuda_arg<unsigned int>(B.handle1()),
                                                         cuarma::cuda_arg<unsigned int>(B.handle2()),
                                                         static_cast<unsigned int>(B.size2()),
                                                         cuarma::cuda_arg<unsigned int>(C.handle1()),
                                                         cuarma::cuda_arg(subwarp_sizes),
                                                         cuarma::cuda_arg(max_nnz_row_A),
                                                         cuarma::cuda_arg(max_nnz_row_B),
                                                         cuarma::cuda_arg(scratchpad_offsets),
                                                         cuarma::cuda_arg(scratchpad_indices)
                                                        );
  CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_2");
#ifdef CUARMA_WITH_SPGEMM_CUDA_TIMINGS
  cudaDeviceSynchronize();
  std::cout << "Stage 2: " << timer.get() << std::endl;
  timer.start();
#endif


  // exclusive scan on C.handle1(), ultimately allowing to allocate remaining memory for C
  cuarma::backend::typesafe_host_array<unsigned int> row_buffer(C.handle1(), C.size1() + 1);
  cuarma::backend::memory_read(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
  unsigned int current_offset = 0;
  for (std::size_t i=0; i<C.size1(); ++i)
  {
    unsigned int tmp = row_buffer[i];
    row_buffer.set(i, current_offset);
    current_offset += tmp;
  }
  row_buffer.set(C.size1(), current_offset);
  cuarma::backend::memory_write(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());


  //
  // Stage 3: Compute entries in C
  //
  C.reserve(current_offset, false);

  cuarma::vector<NumericT> scratchpad_values(scratchpad_offset, cuarma::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

#ifdef CUARMA_WITH_SPGEMM_CUDA_TIMINGS
  std::cout << "Intermediate stage 2->3: " << timer.get() << std::endl;
  timer.start();
#endif

  compressed_matrix_gemm_stage_3<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
                                                          cuarma::cuda_arg<unsigned int>(A.handle2()),
                                                          cuarma::cuda_arg<NumericT>(A.handle()),
                                                          static_cast<unsigned int>(A.size1()),
                                                          cuarma::cuda_arg<unsigned int>(B.handle1()),
                                                          cuarma::cuda_arg<unsigned int>(B.handle2()),
                                                          cuarma::cuda_arg<NumericT>(B.handle()),
                                                          static_cast<unsigned int>(B.size2()),
                                                          cuarma::cuda_arg<unsigned int>(C.handle1()),
                                                          cuarma::cuda_arg<unsigned int>(C.handle2()),
                                                          cuarma::cuda_arg<NumericT>(C.handle()),
                                                          cuarma::cuda_arg(subwarp_sizes),
                                                          cuarma::cuda_arg(max_nnz_row_A),
                                                          cuarma::cuda_arg(max_nnz_row_B),
                                                          cuarma::cuda_arg(scratchpad_offsets),
                                                          cuarma::cuda_arg(scratchpad_indices),
                                                          cuarma::cuda_arg(scratchpad_values)
                                                         );
  CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_3");
#ifdef CUARMA_WITH_SPGEMM_CUDA_TIMINGS
  cudaDeviceSynchronize();
  std::cout << "Stage 3: " << timer.get() << std::endl;
  std::cout << "----------" << std::endl;
#endif

}

} // namespace cuda
} //namespace blas
} //namespace cuarma
