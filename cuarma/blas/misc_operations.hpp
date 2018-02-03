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

/** @file cuarma/blas/misc_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of miscellaneous operations
*/

#include "cuarma/forwards.h"
#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/host_based/misc_operations.hpp"

#ifdef CUARMA_WITH_CUDA
  #include "cuarma/blas/cuda/misc_operations.hpp"
#endif

namespace cuarma
{
  namespace blas
  {
    namespace detail
    {
      template<typename ScalarType>
      void level_scheduling_substitute(vector<ScalarType> & vec,
                                  cuarma::backend::mem_handle const & row_index_array,
                                  cuarma::backend::mem_handle const & row_buffer,
                                  cuarma::backend::mem_handle const & col_buffer,
                                  cuarma::backend::mem_handle const & element_buffer,
                                  arma_size_t num_rows)
      {
        assert( cuarma::traits::handle(vec).get_active_handle_id() == row_index_array.get_active_handle_id() && bool("Incompatible memory domains"));
        assert( cuarma::traits::handle(vec).get_active_handle_id() ==      row_buffer.get_active_handle_id() && bool("Incompatible memory domains"));
        assert( cuarma::traits::handle(vec).get_active_handle_id() ==      col_buffer.get_active_handle_id() && bool("Incompatible memory domains"));
        assert( cuarma::traits::handle(vec).get_active_handle_id() ==  element_buffer.get_active_handle_id() && bool("Incompatible memory domains"));

        switch (cuarma::traits::handle(vec).get_active_handle_id())
        {
          case cuarma::MAIN_MEMORY:
            cuarma::blas::host_based::detail::level_scheduling_substitute(vec, row_index_array, row_buffer, col_buffer, element_buffer, num_rows);
            break;

#ifdef CUARMA_WITH_CUDA
          case cuarma::CUDA_MEMORY:
            cuarma::blas::cuda::detail::level_scheduling_substitute(vec, row_index_array, row_buffer, col_buffer, element_buffer, num_rows);
            break;
#endif
          case cuarma::MEMORY_NOT_INITIALIZED:
            throw memory_exception("not initialised!");
          default:
            throw memory_exception("not implemented");
        }
      }

    } //namespace detail

  } //namespace blas
} //namespace cuarma
