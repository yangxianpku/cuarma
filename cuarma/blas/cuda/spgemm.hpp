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

/** @file spgemm.hpp
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
#include "cuarma/blas/cuda/spgemm_rmerge.hpp"

namespace cuarma
{
namespace blas
{
namespace cuda
{


//
// Stage 2: Determine sparsity pattern of C
//
inline __device__ unsigned int merge_subwarp_symbolic(unsigned int row_B_start, unsigned int row_B_end, unsigned int const *B_col_indices, unsigned int B_size2, unsigned int subwarpsize)
{
  unsigned int current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;

  unsigned int num_nnz = 0;
  while (1)
  {
    // determine current minimum (warp shuffle)
    unsigned int min_index = current_front_index;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      min_index = min(min_index, __shfl_xor((int)min_index, (int)i));

    if (min_index == B_size2)
      break;

    // update front:
    current_front_index = (current_front_index == min_index) ? ((++row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2)
                                                             : current_front_index;
    ++num_nnz;
  }

  return num_nnz;
}

inline __device__ unsigned int merge_subwarp_symbolic_double(unsigned int row_B_start, unsigned int row_B_end, unsigned int const *B_col_indices, unsigned int B_size2,
                                                             unsigned int *output_array, unsigned int id_in_warp, unsigned int subwarpsize)
{
  unsigned int current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;

  unsigned int num_nnz = 0;
  unsigned int index_buffer = 0;
  unsigned int buffer_size = 0;
  while (1)
  {
    // determine current minimum (warp shuffle)
    unsigned int min_index = current_front_index;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      min_index = min(min_index, __shfl_xor((int)min_index, (int)i));

    if (min_index == B_size2)
      break;

    // update front:
    current_front_index = (current_front_index == min_index) ? ((++row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2)
                                                             : current_front_index;

    // write output
    index_buffer = (id_in_warp == buffer_size) ? min_index : index_buffer;
    ++buffer_size;

    if (buffer_size == subwarpsize) // register buffer full?
    {
      output_array[id_in_warp] = index_buffer;
      output_array += subwarpsize;
      buffer_size = 0;
    }

    ++num_nnz;
  }

  // write remaining entries from register buffer:
  if (id_in_warp < buffer_size)
    output_array[id_in_warp] = index_buffer;

  return num_nnz;
}

template<typename IndexT>
__global__ void compressed_matrix_gemm_stage_2(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          IndexT B_size2,
          IndexT * C_row_indices,
          unsigned int *subwarpsize_array,
          unsigned int *max_row_size_A,
          unsigned int *max_row_size_B,
          unsigned int *scratchpad_offsets,
          unsigned int *scratchpad_indices)
{
  unsigned int subwarpsize = subwarpsize_array[blockIdx.x];

  unsigned int num_warps  =  blockDim.x / subwarpsize;
  unsigned int warp_id    = threadIdx.x / subwarpsize;
  unsigned int id_in_warp = threadIdx.x % subwarpsize;

  unsigned int scratchpad_rowlength     = max_row_size_B[blockIdx.x] * subwarpsize;
  unsigned int scratchpad_rows_per_warp = max_row_size_A[blockIdx.x] / subwarpsize + 1;
  unsigned int *subwarp_scratchpad_start = scratchpad_indices + scratchpad_offsets[blockIdx.x] + warp_id * scratchpad_rows_per_warp * scratchpad_rowlength;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    if (row_A_end - row_A_start > subwarpsize)
    {
      unsigned int final_merge_start = 0;
      unsigned int final_merge_end   = 0;

      // merge to temporary scratchpad memory:
      unsigned int *subwarp_scratchpad = subwarp_scratchpad_start;
      unsigned int iter = 0;
      while (row_A_end > row_A_start)
      {
        unsigned int my_row_B = row_A_start + id_in_warp;
        unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
        unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index) : 0;
        unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;

        unsigned int nnz_in_merge = merge_subwarp_symbolic_double(row_B_start, row_B_end, B_col_indices, B_size2,
                                                                  subwarp_scratchpad, id_in_warp, subwarpsize);

        final_merge_start = (iter == id_in_warp) ? subwarp_scratchpad - scratchpad_indices : final_merge_start;
        final_merge_end   = (iter == id_in_warp) ? final_merge_start + nnz_in_merge        : final_merge_end;
        ++iter;

        row_A_start += subwarpsize;
        subwarp_scratchpad += scratchpad_rowlength; // write to next row in scratchpad
      }

      // final merge:
      unsigned int num_nnz = merge_subwarp_symbolic(final_merge_start, final_merge_end, scratchpad_indices, B_size2, subwarpsize);

      if (id_in_warp == 0)
        C_row_indices[row] = num_nnz;
    }
    else
    {
      // single merge
      unsigned int my_row_B = row_A_start + id_in_warp;
      unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
      unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index) : 0;
      unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;

      unsigned int num_nnz = merge_subwarp_symbolic(row_B_start, row_B_end, B_col_indices, B_size2, subwarpsize);

      if (id_in_warp == 0)
        C_row_indices[row] = num_nnz;
    }
  }

}


//
// Stage 3: Fill C with values
//
template<typename NumericT>
__device__ unsigned int merge_subwarp_numeric(NumericT scaling_factor,
                                              unsigned int input_start, unsigned int input_end, const unsigned int *input_indices, const NumericT *input_values, unsigned int invalid_token,
                                              unsigned int *output_indices, NumericT *output_values,
                                              unsigned int id_in_warp, unsigned int subwarpsize)
{
  unsigned int current_front_index = (input_start < input_end) ? load_and_cache(input_indices + input_start) : invalid_token;
  NumericT     current_front_value = (input_start < input_end) ? load_and_cache(input_values  + input_start) : 0;

  unsigned int index_buffer = 0;
  NumericT     value_buffer = 0;
  unsigned int buffer_size = 0;
  unsigned int nnz_written = 0;
  while (1)
  {
    // determine current minimum:
    unsigned int min_index = current_front_index;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      min_index = min(min_index, __shfl_xor((int)min_index, (int)i));

    if (min_index == invalid_token) // done
      break;

    // compute entry in C:
    NumericT output_value = (current_front_index == min_index) ? scaling_factor * current_front_value : 0;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      output_value += __shfl_xor((int)output_value, (int)i);

    // update front:
    if (current_front_index == min_index)
    {
      ++input_start;
      current_front_index = (input_start < input_end) ? load_and_cache(input_indices + input_start) : invalid_token;
      current_front_value = (input_start < input_end) ? load_and_cache(input_values  + input_start) : 0;
    }

    // write current front to register buffer:
    index_buffer = (id_in_warp == buffer_size) ? min_index    : index_buffer;
    value_buffer = (id_in_warp == buffer_size) ? output_value : value_buffer;
    ++buffer_size;

    // flush register buffer via a coalesced write once full:
    if (buffer_size == subwarpsize)
    {
      output_indices[id_in_warp] = index_buffer; output_indices += subwarpsize;
      output_values[id_in_warp]  = value_buffer; output_values  += subwarpsize;
      buffer_size = 0;
    }

    ++nnz_written;
  }

  // write remaining entries in register buffer to C:
  if (id_in_warp < buffer_size)
  {
    output_indices[id_in_warp] = index_buffer;
    output_values[id_in_warp]  = value_buffer;
  }

  return nnz_written;
}

template<typename IndexT, typename NumericT>
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
          NumericT * C_elements,
          unsigned int *subwarpsize_array,
          unsigned int *max_row_size_A,
          unsigned int *max_row_size_B,
          unsigned int *scratchpad_offsets,
          unsigned int *scratchpad_indices,
          NumericT *scratchpad_values)
{
  unsigned int subwarpsize = subwarpsize_array[blockIdx.x];

  unsigned int num_warps  =  blockDim.x / subwarpsize;
  unsigned int warp_id    = threadIdx.x / subwarpsize;
  unsigned int id_in_warp = threadIdx.x % subwarpsize;

  unsigned int scratchpad_rowlength     = max_row_size_B[blockIdx.x] * subwarpsize;
  unsigned int scratchpad_rows_per_warp = max_row_size_A[blockIdx.x] / subwarpsize + 1;
  unsigned int subwarp_scratchpad_shift = scratchpad_offsets[blockIdx.x] + warp_id * scratchpad_rows_per_warp * scratchpad_rowlength;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    if (row_A_end - row_A_start > subwarpsize)
    {
      // first merge stage:
      unsigned int final_merge_start = 0;
      unsigned int final_merge_end = 0;
      unsigned int iter = 0;
      unsigned int *scratchpad_indices_ptr = scratchpad_indices + subwarp_scratchpad_shift;
      NumericT     *scratchpad_values_ptr  = scratchpad_values  + subwarp_scratchpad_shift;

      while (row_A_start < row_A_end)
      {
        unsigned int my_row_B = row_A_start + id_in_warp;
        unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
        unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index)     : 0;
        unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;
        NumericT val_A = (my_row_B < row_A_end) ? A_elements[my_row_B] : 0;

        unsigned int nnz_written = merge_subwarp_numeric(val_A,
                                                         row_B_start, row_B_end, B_col_indices, B_elements, B_size2,
                                                         scratchpad_indices_ptr, scratchpad_values_ptr,
                                                         id_in_warp, subwarpsize);

        if (iter == id_in_warp)
        {
          final_merge_start = scratchpad_indices_ptr - scratchpad_indices;
          final_merge_end   = final_merge_start + nnz_written;
        }
        ++iter;

        row_A_start += subwarpsize;
        scratchpad_indices_ptr += scratchpad_rowlength;
        scratchpad_values_ptr  += scratchpad_rowlength;
      }

      // second merge stage:
      unsigned int index_in_C = C_row_indices[row];
      merge_subwarp_numeric(NumericT(1),
                            final_merge_start, final_merge_end, scratchpad_indices, scratchpad_values, B_size2,
                            C_col_indices + index_in_C, C_elements + index_in_C,
                            id_in_warp, subwarpsize);
    }
    else
    {
      unsigned int my_row_B = row_A_start + id_in_warp;
      unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
      unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index)     : 0;
      unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;
      NumericT val_A = (my_row_B < row_A_end) ? A_elements[my_row_B] : 0;

      unsigned int index_in_C = C_row_indices[row];

      merge_subwarp_numeric(val_A,
                            row_B_start, row_B_end, B_col_indices, B_elements, B_size2,
                            C_col_indices + index_in_C, C_elements + index_in_C,
                            id_in_warp, subwarpsize);
    }
  }

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
// template<class NumericT, unsigned int AlignmentV>
// void prod_impl(cuarma::compressed_matrix<NumericT, AlignmentV> const & A,
//                cuarma::compressed_matrix<NumericT, AlignmentV> const & B,
//                cuarma::compressed_matrix<NumericT, AlignmentV> & C)
// {
//   C.resize(A.size1(), B.size2(), false);

//   unsigned int blocknum = 256;
//   unsigned int threadnum = 128;

//   cuarma::vector<unsigned int> subwarp_sizes(blocknum, cuarma::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
//   cuarma::vector<unsigned int> max_nnz_row_A(blocknum, cuarma::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
//   cuarma::vector<unsigned int> max_nnz_row_B(blocknum, cuarma::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

//   //
//   // Stage 1: Determine upper bound for number of nonzeros
//   //
//   compressed_matrix_gemm_stage_1<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
//                                                           cuarma::cuda_arg<unsigned int>(A.handle2()),
//                                                           static_cast<unsigned int>(A.size1()),
//                                                           cuarma::cuda_arg<unsigned int>(B.handle1()),
//                                                           cuarma::cuda_arg(subwarp_sizes),
//                                                           cuarma::cuda_arg(max_nnz_row_A),
//                                                           cuarma::cuda_arg(max_nnz_row_B)
//                                                          );
//   CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_1");

//   subwarp_sizes.switch_memory_context(cuarma::context(MAIN_MEMORY));
//   unsigned int * subwarp_sizes_ptr = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(subwarp_sizes.handle());

//   max_nnz_row_A.switch_memory_context(cuarma::context(MAIN_MEMORY));
//   unsigned int const * max_nnz_row_A_ptr = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(max_nnz_row_A.handle());

//   max_nnz_row_B.switch_memory_context(cuarma::context(MAIN_MEMORY));
//   unsigned int const * max_nnz_row_B_ptr = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(max_nnz_row_B.handle());

//   unsigned int max_subwarp_size = 0;
//   //std::cout << "Scratchpad offsets: " << std::endl;
//   for (std::size_t i=0; i<subwarp_sizes.size(); ++i)
//     max_subwarp_size = std::max(max_subwarp_size, subwarp_sizes_ptr[i]);
//   unsigned int A_max_nnz_per_row = 0;
//   for (std::size_t i=0; i<max_nnz_row_A.size(); ++i)
//     A_max_nnz_per_row = std::max(A_max_nnz_per_row, max_nnz_row_A_ptr[i]);

//   if (max_subwarp_size > 32)
//   {
//     // determine augmented size:
//     unsigned int max_entries_in_G = 32;
//     if (A_max_nnz_per_row <= 256)
//       max_entries_in_G = 16;
//     if (A_max_nnz_per_row <= 64)
//       max_entries_in_G = 8;

//     cuarma::vector<unsigned int> exclusive_scan_helper(A.size1() + 1, cuarma::traits::context(A));
//     compressed_matrix_gemm_decompose_1<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
//                                                                 static_cast<unsigned int>(A.size1()),
//                                                                 static_cast<unsigned int>(max_entries_in_G),
//                                                                 cuarma::cuda_arg(exclusive_scan_helper)
//                                                                );
//     CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_decompose_1");

//     cuarma::blas::exclusive_scan(exclusive_scan_helper);
//     unsigned int augmented_size = exclusive_scan_helper[A.size1()];

//     // split A = A2 * G1
//     cuarma::compressed_matrix<NumericT, AlignmentV> A2(A.size1(), augmented_size, augmented_size, cuarma::traits::context(A));
//     cuarma::compressed_matrix<NumericT, AlignmentV> G1(augmented_size, A.size2(),        A.nnz(), cuarma::traits::context(A));

//     // fill A2:
//     compressed_matrix_gemm_A2<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A2.handle1()),
//                                                        cuarma::cuda_arg<unsigned int>(A2.handle2()),
//                                                        cuarma::cuda_arg<NumericT>(A2.handle()),
//                                                        static_cast<unsigned int>(A2.size1()),
//                                                        cuarma::cuda_arg(exclusive_scan_helper)
//                                                       );
//     CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_A2");

//     // fill G1:
//     compressed_matrix_gemm_G1<<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(G1.handle1()),
//                                                        cuarma::cuda_arg<unsigned int>(G1.handle2()),
//                                                        cuarma::cuda_arg<NumericT>(G1.handle()),
//                                                        static_cast<unsigned int>(G1.size1()),
//                                                        cuarma::cuda_arg<unsigned int>(A.handle1()),
//                                                        cuarma::cuda_arg<unsigned int>(A.handle2()),
//                                                        cuarma::cuda_arg<NumericT>(A.handle()),
//                                                        static_cast<unsigned int>(A.size1()),
//                                                        static_cast<unsigned int>(A.nnz()),
//                                                        static_cast<unsigned int>(max_entries_in_G),
//                                                        cuarma::cuda_arg(exclusive_scan_helper)
//                                                       );
//     CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_G1");

//     // compute tmp = G1 * B;
//     // C = A2 * tmp;
//     cuarma::compressed_matrix<NumericT, AlignmentV> tmp(G1.size1(), B.size2(), 0, cuarma::traits::context(A));
//     prod_impl(G1, B, tmp); // this runs a standard RMerge without decomposition of G1
//     prod_impl(A2, tmp, C); // this may split A2 again
//     return;
//   }

//   //std::cout << "Running RMerge with subwarp size " << max_subwarp_size << std::endl;

//   subwarp_sizes.switch_memory_context(cuarma::traits::context(A));
//   max_nnz_row_A.switch_memory_context(cuarma::traits::context(A));
//   max_nnz_row_B.switch_memory_context(cuarma::traits::context(A));

//   //
//   // Stage 2: Determine pattern of C
//   //

//   if (max_subwarp_size == 32)
//   {
//     compressed_matrix_gemm_stage_2<32><<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
//                                                            cuarma::cuda_arg<unsigned int>(A.handle2()),
//                                                            static_cast<unsigned int>(A.size1()),
//                                                            cuarma::cuda_arg<unsigned int>(B.handle1()),
//                                                            cuarma::cuda_arg<unsigned int>(B.handle2()),
//                                                            static_cast<unsigned int>(B.size2()),
//                                                            cuarma::cuda_arg<unsigned int>(C.handle1())
//                                                           );
//     CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_2");
//   }
//   else if (max_subwarp_size == 16)
//   {
//     compressed_matrix_gemm_stage_2<16><<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
//                                                            cuarma::cuda_arg<unsigned int>(A.handle2()),
//                                                            static_cast<unsigned int>(A.size1()),
//                                                            cuarma::cuda_arg<unsigned int>(B.handle1()),
//                                                            cuarma::cuda_arg<unsigned int>(B.handle2()),
//                                                            static_cast<unsigned int>(B.size2()),
//                                                            cuarma::cuda_arg<unsigned int>(C.handle1())
//                                                           );
//     CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_2");
//   }
//   else
//   {
//     compressed_matrix_gemm_stage_2<8><<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
//                                                            cuarma::cuda_arg<unsigned int>(A.handle2()),
//                                                            static_cast<unsigned int>(A.size1()),
//                                                            cuarma::cuda_arg<unsigned int>(B.handle1()),
//                                                            cuarma::cuda_arg<unsigned int>(B.handle2()),
//                                                            static_cast<unsigned int>(B.size2()),
//                                                            cuarma::cuda_arg<unsigned int>(C.handle1())
//                                                           );
//     CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_2");
//   }

//   // exclusive scan on C.handle1(), ultimately allowing to allocate remaining memory for C
//   cuarma::backend::typesafe_host_array<unsigned int> row_buffer(C.handle1(), C.size1() + 1);
//   cuarma::backend::memory_read(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
//   unsigned int current_offset = 0;
//   for (std::size_t i=0; i<C.size1(); ++i)
//   {
//     unsigned int tmp = row_buffer[i];
//     row_buffer.set(i, current_offset);
//     current_offset += tmp;
//   }
//   row_buffer.set(C.size1(), current_offset);
//   cuarma::backend::memory_write(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());


//   //
//   // Stage 3: Compute entries in C
//   //
//   C.reserve(current_offset, false);

//   if (max_subwarp_size == 32)
//   {
//     compressed_matrix_gemm_stage_3<32><<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
//                                                             cuarma::cuda_arg<unsigned int>(A.handle2()),
//                                                             cuarma::cuda_arg<NumericT>(A.handle()),
//                                                             static_cast<unsigned int>(A.size1()),
//                                                             cuarma::cuda_arg<unsigned int>(B.handle1()),
//                                                             cuarma::cuda_arg<unsigned int>(B.handle2()),
//                                                             cuarma::cuda_arg<NumericT>(B.handle()),
//                                                             static_cast<unsigned int>(B.size2()),
//                                                             cuarma::cuda_arg<unsigned int>(C.handle1()),
//                                                             cuarma::cuda_arg<unsigned int>(C.handle2()),
//                                                             cuarma::cuda_arg<NumericT>(C.handle())
//                                                            );
//     CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_3");
//   }
//   else if (max_subwarp_size == 16)
//   {
//     compressed_matrix_gemm_stage_3<16><<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
//                                                             cuarma::cuda_arg<unsigned int>(A.handle2()),
//                                                             cuarma::cuda_arg<NumericT>(A.handle()),
//                                                             static_cast<unsigned int>(A.size1()),
//                                                             cuarma::cuda_arg<unsigned int>(B.handle1()),
//                                                             cuarma::cuda_arg<unsigned int>(B.handle2()),
//                                                             cuarma::cuda_arg<NumericT>(B.handle()),
//                                                             static_cast<unsigned int>(B.size2()),
//                                                             cuarma::cuda_arg<unsigned int>(C.handle1()),
//                                                             cuarma::cuda_arg<unsigned int>(C.handle2()),
//                                                             cuarma::cuda_arg<NumericT>(C.handle())
//                                                            );
//     CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_3");
//   }
//   else
//   {
//     compressed_matrix_gemm_stage_3<8><<<blocknum, threadnum>>>(cuarma::cuda_arg<unsigned int>(A.handle1()),
//                                                             cuarma::cuda_arg<unsigned int>(A.handle2()),
//                                                             cuarma::cuda_arg<NumericT>(A.handle()),
//                                                             static_cast<unsigned int>(A.size1()),
//                                                             cuarma::cuda_arg<unsigned int>(B.handle1()),
//                                                             cuarma::cuda_arg<unsigned int>(B.handle2()),
//                                                             cuarma::cuda_arg<NumericT>(B.handle()),
//                                                             static_cast<unsigned int>(B.size2()),
//                                                             cuarma::cuda_arg<unsigned int>(C.handle1()),
//                                                             cuarma::cuda_arg<unsigned int>(C.handle2()),
//                                                             cuarma::cuda_arg<NumericT>(C.handle())
//                                                            );
//     CUARMA_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_3");
//   }

// }
} // namespace cuda
} //namespace blas
} //namespace cuarma