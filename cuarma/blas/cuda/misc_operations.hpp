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

/** @file cuarma/blas/cuda/misc_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of miscellaneous operations using CUDA
*/

#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/cuda/common.hpp"


namespace cuarma
{
namespace blas
{
namespace cuda
{
namespace detail
{

template<typename NumericT>
__global__ void level_scheduling_substitute_kernel(
          const unsigned int * row_index_array,
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
          NumericT * vec,
          unsigned int size)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < size;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int eq_row = row_index_array[row];
    NumericT vec_entry = vec[eq_row];
    unsigned int row_end = row_indices[row+1];

    for (unsigned int j = row_indices[row]; j < row_end; ++j)
      vec_entry -= vec[column_indices[j]] * elements[j];

    vec[eq_row] = vec_entry;
  }
}



template<typename NumericT>
void level_scheduling_substitute(vector<NumericT> & vec,
                             cuarma::backend::mem_handle const & row_index_array,
                             cuarma::backend::mem_handle const & row_buffer,
                             cuarma::backend::mem_handle const & col_buffer,
                             cuarma::backend::mem_handle const & element_buffer,
                             arma_size_t num_rows
                            )
{
  level_scheduling_substitute_kernel<<<128, 128>>>(cuarma::cuda_arg<unsigned int>(row_index_array),
                                                   cuarma::cuda_arg<unsigned int>(row_buffer),
                                                   cuarma::cuda_arg<unsigned int>(col_buffer),
                                                   cuarma::cuda_arg<NumericT>(element_buffer),
                                                   cuarma::cuda_arg(vec),
                                                   static_cast<unsigned int>(num_rows)
                                                  );
}

} //namespace detail
} //namespace cuda
} //namespace blas
} //namespace cuarma
