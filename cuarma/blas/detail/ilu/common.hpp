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

/** @file cuarma/blas/detail/ilu/common.hpp
 *  @encoding:UTF-8 文档编码
    @brief Common routines used within ILU-type preconditioners
*/

#include <vector>
#include <cmath>
#include <iostream>
#include <map>
#include <list>
#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/backend/memory.hpp"
#include "cuarma/blas/host_based/common.hpp"
#include "cuarma/blas/misc_operations.hpp"

namespace cuarma
{
namespace blas
{
namespace detail
{


//
// Level Scheduling Setup for ILU:
//

template<typename NumericT, unsigned int AlignmentV>
void level_scheduling_setup_impl(cuarma::compressed_matrix<NumericT, AlignmentV> const & LU,
                                 cuarma::vector<NumericT> const & diagonal_LU,
                                 std::list<cuarma::backend::mem_handle> & row_index_arrays,
                                 std::list<cuarma::backend::mem_handle> & row_buffers,
                                 std::list<cuarma::backend::mem_handle> & col_buffers,
                                 std::list<cuarma::backend::mem_handle> & element_buffers,
                                 std::list<arma_size_t> & row_elimination_num_list,
                                 bool setup_U)
{
  NumericT     const * diagonal_buf = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(diagonal_LU.handle());
  NumericT     const * elements     = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(LU.handle());
  unsigned int const * row_buffer   = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(LU.handle1());
  unsigned int const * col_buffer   = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(LU.handle2());

  //
  // Step 1: Determine row elimination order for each row and build up meta information about the number of entries taking part in each elimination step:
  //
  std::vector<arma_size_t> row_elimination(LU.size1());
  std::map<arma_size_t, std::map<arma_size_t, arma_size_t> > row_entries_per_elimination_step;

  arma_size_t max_elimination_runs = 0;
  for (arma_size_t row2 = 0; row2 < LU.size1(); ++row2)
  {
    arma_size_t row = setup_U ? (LU.size1() - row2) - 1 : row2;

    arma_size_t row_begin = row_buffer[row];
    arma_size_t row_end   = row_buffer[row+1];
    arma_size_t elimination_index = 0;  //Note: first run corresponds to elimination_index = 1 (otherwise, type issues with int <-> unsigned int would arise
    for (arma_size_t i = row_begin; i < row_end; ++i)
    {
      unsigned int col = col_buffer[i];
      if ( (!setup_U && col < row) || (setup_U && col > row) )
      {
        elimination_index = std::max<arma_size_t>(elimination_index, row_elimination[col]);
        row_entries_per_elimination_step[row_elimination[col]][row] += 1;
      }
    }
    row_elimination[row] = elimination_index + 1;
    max_elimination_runs = std::max<arma_size_t>(max_elimination_runs, elimination_index + 1);
  }

  //std::cout << "Number of elimination runs: " << max_elimination_runs << std::endl;

  //
  // Step 2: Build row-major elimination matrix for each elimination step
  //

  //std::cout << "Elimination order: " << std::endl;
  //for (arma_size_t i=0; i<row_elimination.size(); ++i)
  //  std::cout << row_elimination[i] << ", ";
  //std::cout << std::endl;

  //arma_size_t summed_rows = 0;
  for (arma_size_t elimination_run = 1; elimination_run <= max_elimination_runs; ++elimination_run)
  {
    std::map<arma_size_t, arma_size_t> const & current_elimination_info = row_entries_per_elimination_step[elimination_run];

    // count cols and entries handled in this elimination step
    arma_size_t num_tainted_cols = current_elimination_info.size();
    arma_size_t num_entries = 0;

    for (std::map<arma_size_t, arma_size_t>::const_iterator it  = current_elimination_info.begin();
                                                          it != current_elimination_info.end();
                                                        ++it)
      num_entries += it->second;

    //std::cout << "num_entries: " << num_entries << std::endl;
    //std::cout << "num_tainted_cols: " << num_tainted_cols << std::endl;

    if (num_tainted_cols > 0)
    {
      row_index_arrays.push_back(cuarma::backend::mem_handle());
      cuarma::backend::switch_memory_context<unsigned int>(row_index_arrays.back(), cuarma::traits::context(LU));
      cuarma::backend::typesafe_host_array<unsigned int> elim_row_index_array(row_index_arrays.back(), num_tainted_cols);

      row_buffers.push_back(cuarma::backend::mem_handle());
      cuarma::backend::switch_memory_context<unsigned int>(row_buffers.back(), cuarma::traits::context(LU));
      cuarma::backend::typesafe_host_array<unsigned int> elim_row_buffer(row_buffers.back(), num_tainted_cols + 1);

      col_buffers.push_back(cuarma::backend::mem_handle());
      cuarma::backend::switch_memory_context<unsigned int>(col_buffers.back(), cuarma::traits::context(LU));
      cuarma::backend::typesafe_host_array<unsigned int> elim_col_buffer(col_buffers.back(), num_entries);

      element_buffers.push_back(cuarma::backend::mem_handle());
      cuarma::backend::switch_memory_context<NumericT>(element_buffers.back(), cuarma::traits::context(LU));
      std::vector<NumericT> elim_elements_buffer(num_entries);

      row_elimination_num_list.push_back(num_tainted_cols);

      arma_size_t k=0;
      arma_size_t nnz_index = 0;
      elim_row_buffer.set(0, 0);

      for (std::map<arma_size_t, arma_size_t>::const_iterator it  = current_elimination_info.begin();
                                                              it != current_elimination_info.end();
                                                            ++it)
      {
        //arma_size_t col = setup_U ? (elimination_matrix.size() - it->first) - 1 : col2;
        arma_size_t row = it->first;
        elim_row_index_array.set(k, row);

        arma_size_t row_begin = row_buffer[row];
        arma_size_t row_end   = row_buffer[row+1];
        for (arma_size_t i = row_begin; i < row_end; ++i)
        {
          unsigned int col = col_buffer[i];
          if ( (!setup_U && col < row) || (setup_U && col > row) ) //entry of L/U
          {
            if (row_elimination[col] == elimination_run) // this entry is substituted in this run
            {
              elim_col_buffer.set(nnz_index, col);
              elim_elements_buffer[nnz_index] = setup_U ? elements[i] / diagonal_buf[it->first] : elements[i];
              ++nnz_index;
            }
          }
        }

        elim_row_buffer.set(++k, nnz_index);
      }

      //
      // Wrap in memory_handles:
      //
      cuarma::backend::memory_create(row_index_arrays.back(), elim_row_index_array.raw_size(),                cuarma::traits::context(row_index_arrays.back()), elim_row_index_array.get());
      cuarma::backend::memory_create(row_buffers.back(),      elim_row_buffer.raw_size(),                     cuarma::traits::context(row_buffers.back()),      elim_row_buffer.get());
      cuarma::backend::memory_create(col_buffers.back(),      elim_col_buffer.raw_size(),                     cuarma::traits::context(col_buffers.back()),      elim_col_buffer.get());
      cuarma::backend::memory_create(element_buffers.back(),  sizeof(NumericT) * elim_elements_buffer.size(), cuarma::traits::context(element_buffers.back()),  &(elim_elements_buffer[0]));
    }

    // Print some info:
    //std::cout << "Eliminated columns in run " << elimination_run << ": " << num_tainted_cols << " (tainted columns: " << num_tainted_cols << ")" << std::endl;
    //summed_rows += eliminated_rows_in_run;
    //if (eliminated_rows_in_run == 0)
    //  break;
  }
  //std::cout << "Eliminated rows: " << summed_rows << " out of " << row_elimination.size() << std::endl;
}


template<typename NumericT, unsigned int AlignmentV>
void level_scheduling_setup_L(cuarma::compressed_matrix<NumericT, AlignmentV> const & LU,
                              cuarma::vector<NumericT> const & diagonal_LU,
                              std::list<cuarma::backend::mem_handle> & row_index_arrays,
                              std::list<cuarma::backend::mem_handle> & row_buffers,
                              std::list<cuarma::backend::mem_handle> & col_buffers,
                              std::list<cuarma::backend::mem_handle> & element_buffers,
                              std::list<arma_size_t> & row_elimination_num_list)
{
  level_scheduling_setup_impl(LU, diagonal_LU, row_index_arrays, row_buffers, col_buffers, element_buffers, row_elimination_num_list, false);
}


//
// Multifrontal setup of U:
//

template<typename NumericT, unsigned int AlignmentV>
void level_scheduling_setup_U(cuarma::compressed_matrix<NumericT, AlignmentV> const & LU,
                              cuarma::vector<NumericT> const & diagonal_LU,
                              std::list<cuarma::backend::mem_handle> & row_index_arrays,
                              std::list<cuarma::backend::mem_handle> & row_buffers,
                              std::list<cuarma::backend::mem_handle> & col_buffers,
                              std::list<cuarma::backend::mem_handle> & element_buffers,
                              std::list<arma_size_t> & row_elimination_num_list)
{
  level_scheduling_setup_impl(LU, diagonal_LU, row_index_arrays, row_buffers, col_buffers, element_buffers, row_elimination_num_list, true);
}


//
// Multifrontal substitution (both L and U). Will partly be moved to single_threaded/cuda implementations
//
template<typename NumericT>
void level_scheduling_substitute(cuarma::vector<NumericT> & vec,
                                 std::list<cuarma::backend::mem_handle> const & row_index_arrays,
                                 std::list<cuarma::backend::mem_handle> const & row_buffers,
                                 std::list<cuarma::backend::mem_handle> const & col_buffers,
                                 std::list<cuarma::backend::mem_handle> const & element_buffers,
                                 std::list<arma_size_t> const & row_elimination_num_list)
{
  typedef typename std::list< cuarma::backend::mem_handle >::const_iterator  ListIterator;
  ListIterator row_index_array_it = row_index_arrays.begin();
  ListIterator row_buffers_it = row_buffers.begin();
  ListIterator col_buffers_it = col_buffers.begin();
  ListIterator element_buffers_it = element_buffers.begin();
  typename std::list< arma_size_t>::const_iterator row_elimination_num_it = row_elimination_num_list.begin();
  for (arma_size_t i=0; i<row_index_arrays.size(); ++i)
  {
    cuarma::blas::detail::level_scheduling_substitute(vec, *row_index_array_it, *row_buffers_it, *col_buffers_it, *element_buffers_it, *row_elimination_num_it);

    ++row_index_array_it;
    ++row_buffers_it;
    ++col_buffers_it;
    ++element_buffers_it;
    ++row_elimination_num_it;
  }
}

} // namespace detail
} // namespace blas
} // namespace cuarma