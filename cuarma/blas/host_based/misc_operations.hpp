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

/** @file cuarma/blas/host_based/misc_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of miscellaneous operations on the CPU using a single thread.
*/

#include <list>
#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/host_based/common.hpp"

namespace cuarma
{
namespace blas
{
namespace host_based
{
namespace detail
{
  template<typename NumericT>
  void level_scheduling_substitute(vector<NumericT> & vec,
                                   cuarma::backend::mem_handle const & row_index_array,
                                   cuarma::backend::mem_handle const & row_buffer,
                                   cuarma::backend::mem_handle const & col_buffer,
                                   cuarma::backend::mem_handle const & element_buffer,
                                   arma_size_t num_rows
                                  )
  {
    NumericT * vec_buf = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(vec.handle());

    unsigned int const * elim_row_index  = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(row_index_array);
    unsigned int const * elim_row_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(row_buffer);
    unsigned int const * elim_col_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(col_buffer);
    NumericT     const * elim_elements   = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(element_buffer);


    for (long row=0; row < static_cast<long>(num_rows); ++row)
    {
      unsigned int  eq_row = elim_row_index[row];
      unsigned int row_end = elim_row_buffer[row+1];
      NumericT   vec_entry = vec_buf[eq_row];

      for (arma_size_t j = elim_row_buffer[row]; j < row_end; ++j)
        vec_entry -= vec_buf[elim_col_buffer[j]] * elim_elements[j];

      vec_buf[eq_row] = vec_entry;
    }

  }
}

} // namespace host_based
} //namespace blas
} //namespace cuarma