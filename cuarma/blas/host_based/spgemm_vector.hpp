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

/** @file cuarma/blas/host_based/sparse_matrix_operations.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of operations using sparse matrices on the CPU using a single thread.
*/

#include "cuarma/forwards.h"
#include "cuarma/blas/host_based/common.hpp"


namespace cuarma
{
namespace blas
{
namespace host_based
{


/** @brief Merges up to IndexNum rows from B into the result buffer.
*
* Because the input buffer also needs to be considered, this routine actually works on an index front of length (IndexNum+1)
**/
template<unsigned int IndexNum>
unsigned int row_C_scan_symbolic_vector_N(unsigned int const *row_indices_B,
                                          unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, unsigned int B_size2,
                                          unsigned int const *row_C_vector_input, unsigned int const *row_C_vector_input_end,
                                          unsigned int *row_C_vector_output)
{
  unsigned int index_front[IndexNum+1];
  unsigned int const *index_front_start[IndexNum+1];
  unsigned int const *index_front_end[IndexNum+1];

  // Set up pointers for loading the indices:
  for (unsigned int i=0; i<IndexNum; ++i, ++row_indices_B)
  {
    index_front_start[i] = B_col_buffer + B_row_buffer[*row_indices_B];
    index_front_end[i]   = B_col_buffer + B_row_buffer[*row_indices_B + 1];
  }
  index_front_start[IndexNum] = row_C_vector_input;
  index_front_end[IndexNum]   = row_C_vector_input_end;

  // load indices:
  for (unsigned int i=0; i<=IndexNum; ++i)
    index_front[i] = (index_front_start[i] < index_front_end[i]) ? *index_front_start[i] : B_size2;

  unsigned int *output_ptr = row_C_vector_output;

  while (1)
  {
    // get minimum index in current front:
    unsigned int min_index_in_front = B_size2;
    for (unsigned int i=0; i<=IndexNum; ++i)
      min_index_in_front = std::min(min_index_in_front, index_front[i]);

    if (min_index_in_front == B_size2) // we're done
      break;

    // advance index front where equal to minimum index:
    for (unsigned int i=0; i<=IndexNum; ++i)
    {
      if (index_front[i] == min_index_in_front)
      {
        index_front_start[i] += 1;
        index_front[i] = (index_front_start[i] < index_front_end[i]) ? *index_front_start[i] : B_size2;
      }
    }

    // write current entry:
    *output_ptr = min_index_in_front;
    ++output_ptr;
  }

  return static_cast<unsigned int>(output_ptr - row_C_vector_output);
}

struct spgemm_output_write_enabled  { static void apply(unsigned int *ptr, unsigned int value) { *ptr = value; } };
struct spgemm_output_write_disabled { static void apply(unsigned int *   , unsigned int      ) {               } };

template<typename OutputWriterT>
unsigned int row_C_scan_symbolic_vector_1(unsigned int const *input1_begin, unsigned int const *input1_end,
                                          unsigned int const *input2_begin, unsigned int const *input2_end,
                                          unsigned int termination_index,
                                          unsigned int *output_begin)
{
  unsigned int *output_ptr = output_begin;

  unsigned int val_1 = (input1_begin < input1_end) ? *input1_begin : termination_index;
  unsigned int val_2 = (input2_begin < input2_end) ? *input2_begin : termination_index;
  while (1)
  {
    unsigned int min_index = std::min(val_1, val_2);

    if (min_index == termination_index)
      break;

    if (min_index == val_1)
    {
      ++input1_begin;
      val_1 = (input1_begin < input1_end) ? *input1_begin : termination_index;
    }

    if (min_index == val_2)
    {
      ++input2_begin;
      val_2 = (input2_begin < input2_end) ? *input2_begin : termination_index;
    }

    // write current entry:
    OutputWriterT::apply(output_ptr, min_index); // *output_ptr = min_index;    if necessary
    ++output_ptr;
  }

  return static_cast<unsigned int>(output_ptr - output_begin);
}

inline
unsigned int row_C_scan_symbolic_vector(unsigned int row_start_A, unsigned int row_end_A, unsigned int const *A_col_buffer,
                                        unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, unsigned int B_size2,
                                        unsigned int *row_C_vector_1, unsigned int *row_C_vector_2, unsigned int *row_C_vector_3)
{
  // Trivial case: row length 0:
  if (row_start_A == row_end_A)
    return 0;

  // Trivial case: row length 1:
  if (row_end_A - row_start_A == 1)
  {
    unsigned int A_col = A_col_buffer[row_start_A];
    return B_row_buffer[A_col + 1] - B_row_buffer[A_col];
  }

  // Optimizations for row length 2:
  unsigned int row_C_len = 0;
  if (row_end_A - row_start_A == 2)
  {
    unsigned int A_col_1 = A_col_buffer[row_start_A];
    unsigned int A_col_2 = A_col_buffer[row_start_A + 1];
    return row_C_scan_symbolic_vector_1<spgemm_output_write_disabled>(B_col_buffer + B_row_buffer[A_col_1], B_col_buffer + B_row_buffer[A_col_1 + 1],
                                                                      B_col_buffer + B_row_buffer[A_col_2], B_col_buffer + B_row_buffer[A_col_2 + 1],
                                                                      B_size2,
                                                                      row_C_vector_1);
  }
  else // for more than two rows we can safely merge the first two:
  {
    unsigned int A_col_1 = A_col_buffer[row_start_A];
    unsigned int A_col_2 = A_col_buffer[row_start_A + 1];
    row_C_len =  row_C_scan_symbolic_vector_1<spgemm_output_write_enabled>(B_col_buffer + B_row_buffer[A_col_1], B_col_buffer + B_row_buffer[A_col_1 + 1],
                                                                           B_col_buffer + B_row_buffer[A_col_2], B_col_buffer + B_row_buffer[A_col_2 + 1],
                                                                           B_size2,
                                                                           row_C_vector_1);
    row_start_A += 2;
  }

  // all other row lengths:
  while (row_end_A > row_start_A)
  {

    if (row_start_A == row_end_A - 1) // last merge operation. No need to write output
    {
      // process last row
      unsigned int row_index_B = A_col_buffer[row_start_A];
      return row_C_scan_symbolic_vector_1<spgemm_output_write_disabled>(B_col_buffer + B_row_buffer[row_index_B], B_col_buffer + B_row_buffer[row_index_B + 1],
                                                                        row_C_vector_1, row_C_vector_1 + row_C_len,
                                                                        B_size2,
                                                                        row_C_vector_2);
    }
    else if (row_start_A + 1 < row_end_A)// at least two more rows left, so merge them
    {
      // process single row:
      unsigned int A_col_1 = A_col_buffer[row_start_A];
      unsigned int A_col_2 = A_col_buffer[row_start_A + 1];
      unsigned int merged_len =  row_C_scan_symbolic_vector_1<spgemm_output_write_enabled>(B_col_buffer + B_row_buffer[A_col_1], B_col_buffer + B_row_buffer[A_col_1 + 1],
                                                                                           B_col_buffer + B_row_buffer[A_col_2], B_col_buffer + B_row_buffer[A_col_2 + 1],
                                                                                           B_size2,
                                                                                           row_C_vector_3);
      if (row_start_A + 2 == row_end_A) // last merge does not need a write:
        return row_C_scan_symbolic_vector_1<spgemm_output_write_disabled>(row_C_vector_3, row_C_vector_3 + merged_len,
                                                                          row_C_vector_1, row_C_vector_1 + row_C_len,
                                                                          B_size2,
                                                                          row_C_vector_2);
      else
        row_C_len = row_C_scan_symbolic_vector_1<spgemm_output_write_enabled>(row_C_vector_3, row_C_vector_3 + merged_len,
                                                                              row_C_vector_1, row_C_vector_1 + row_C_len,
                                                                              B_size2,
                                                                              row_C_vector_2);
      row_start_A += 2;
    }
    else // at least two more rows left
    {
      // process single row:
      unsigned int row_index_B = A_col_buffer[row_start_A];
      row_C_len = row_C_scan_symbolic_vector_1<spgemm_output_write_enabled>(B_col_buffer + B_row_buffer[row_index_B], B_col_buffer + B_row_buffer[row_index_B + 1],
                                                                            row_C_vector_1, row_C_vector_1 + row_C_len,
                                                                            B_size2,
                                                                            row_C_vector_2);
      ++row_start_A;
    }

    std::swap(row_C_vector_1, row_C_vector_2);
  }

  return row_C_len;
}

//////////////////////////////

/** @brief Merges up to IndexNum rows from B into the result buffer.
*
* Because the input buffer also needs to be considered, this routine actually works on an index front of length (IndexNum+1)
**/
template<unsigned int IndexNum, typename NumericT>
unsigned int row_C_scan_numeric_vector_N(unsigned int const *row_indices_B, NumericT const *val_A,
                                          unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, NumericT const *B_elements, unsigned int B_size2,
                                          unsigned int const *row_C_vector_input, unsigned int const *row_C_vector_input_end, NumericT *row_C_vector_input_values,
                                          unsigned int *row_C_vector_output, NumericT *row_C_vector_output_values)
{
  unsigned int index_front[IndexNum+1];
  unsigned int const *index_front_start[IndexNum+1];
  unsigned int const *index_front_end[IndexNum+1];
  NumericT const * value_front_start[IndexNum+1];
  NumericT values_A[IndexNum+1];

  // Set up pointers for loading the indices:
  for (unsigned int i=0; i<IndexNum; ++i, ++row_indices_B)
  {
    unsigned int row_B = *row_indices_B;

    index_front_start[i] = B_col_buffer + B_row_buffer[row_B];
    index_front_end[i]   = B_col_buffer + B_row_buffer[row_B + 1];
    value_front_start[i] = B_elements   + B_row_buffer[row_B];
    values_A[i]          = val_A[i];
  }
  index_front_start[IndexNum] = row_C_vector_input;
  index_front_end[IndexNum]   = row_C_vector_input_end;
  value_front_start[IndexNum] = row_C_vector_input_values;
  values_A[IndexNum]          = NumericT(1);

  // load indices:
  for (unsigned int i=0; i<=IndexNum; ++i)
    index_front[i] = (index_front_start[i] < index_front_end[i]) ? *index_front_start[i] : B_size2;

  unsigned int *output_ptr = row_C_vector_output;

  while (1)
  {
    // get minimum index in current front:
    unsigned int min_index_in_front = B_size2;
    for (unsigned int i=0; i<=IndexNum; ++i)
      min_index_in_front = std::min(min_index_in_front, index_front[i]);

    if (min_index_in_front == B_size2) // we're done
      break;

    // advance index front where equal to minimum index:
    NumericT row_C_value = 0;
    for (unsigned int i=0; i<=IndexNum; ++i)
    {
      if (index_front[i] == min_index_in_front)
      {
        index_front_start[i] += 1;
        index_front[i] = (index_front_start[i] < index_front_end[i]) ? *index_front_start[i] : B_size2;

        row_C_value += values_A[i] * *value_front_start[i];
        value_front_start[i] += 1;
      }
    }

    // write current entry:
    *output_ptr = min_index_in_front;
    ++output_ptr;
    *row_C_vector_output_values = row_C_value;
    ++row_C_vector_output_values;
  }

  return static_cast<unsigned int>(output_ptr - row_C_vector_output);
}






template<typename NumericT>
unsigned int row_C_scan_numeric_vector_1(unsigned int const *input1_index_begin, unsigned int const *input1_index_end, NumericT const *input1_values_begin, NumericT factor1,
                                         unsigned int const *input2_index_begin, unsigned int const *input2_index_end, NumericT const *input2_values_begin, NumericT factor2,
                                         unsigned int termination_index,
                                         unsigned int *output_index_begin, NumericT *output_values_begin)
{
  unsigned int *output_ptr = output_index_begin;

  unsigned int index1 = (input1_index_begin < input1_index_end) ? *input1_index_begin : termination_index;
  unsigned int index2 = (input2_index_begin < input2_index_end) ? *input2_index_begin : termination_index;

  while (1)
  {
    unsigned int min_index = std::min(index1, index2);
    NumericT value = 0;

    if (min_index == termination_index)
      break;

    if (min_index == index1)
    {
      ++input1_index_begin;
      index1 = (input1_index_begin < input1_index_end) ? *input1_index_begin : termination_index;

      value += factor1 * *input1_values_begin;
      ++input1_values_begin;
    }

    if (min_index == index2)
    {
      ++input2_index_begin;
      index2 = (input2_index_begin < input2_index_end) ? *input2_index_begin : termination_index;

      value += factor2 * *input2_values_begin;
      ++input2_values_begin;
    }

    // write current entry:
    *output_ptr = min_index;
    ++output_ptr;
    *output_values_begin = value;
    ++output_values_begin;
  }

  return static_cast<unsigned int>(output_ptr - output_index_begin);
}

template<typename NumericT>
void row_C_scan_numeric_vector(unsigned int row_start_A, unsigned int row_end_A, unsigned int const *A_col_buffer, NumericT const *A_elements,
                               unsigned int const *B_row_buffer, unsigned int const *B_col_buffer, NumericT const *B_elements, unsigned int B_size2,
                               unsigned int row_start_C, unsigned int row_end_C, unsigned int *C_col_buffer, NumericT *C_elements,
                               unsigned int *row_C_vector_1, NumericT *row_C_vector_1_values,
                               unsigned int *row_C_vector_2, NumericT *row_C_vector_2_values,
                               unsigned int *row_C_vector_3, NumericT *row_C_vector_3_values)
{
  (void)row_end_C;

  // Trivial case: row length 0:
  if (row_start_A == row_end_A)
    return;

  // Trivial case: row length 1:
  if (row_end_A - row_start_A == 1)
  {
    unsigned int A_col = A_col_buffer[row_start_A];
    unsigned int B_end = B_row_buffer[A_col + 1];
    NumericT A_value   = A_elements[row_start_A];
    C_col_buffer += row_start_C;
    C_elements += row_start_C;
    for (unsigned int j = B_row_buffer[A_col]; j < B_end; ++j, ++C_col_buffer, ++C_elements)
    {
      *C_col_buffer = B_col_buffer[j];
      *C_elements = A_value * B_elements[j];
    }
    return;
  }

  unsigned int row_C_len = 0;
  if (row_end_A - row_start_A == 2) // directly merge to C:
  {
    unsigned int A_col_1 = A_col_buffer[row_start_A];
    unsigned int A_col_2 = A_col_buffer[row_start_A + 1];

    unsigned int B_offset_1 = B_row_buffer[A_col_1];
    unsigned int B_offset_2 = B_row_buffer[A_col_2];

    row_C_scan_numeric_vector_1(B_col_buffer + B_offset_1, B_col_buffer + B_row_buffer[A_col_1+1], B_elements + B_offset_1, A_elements[row_start_A],
                                B_col_buffer + B_offset_2, B_col_buffer + B_row_buffer[A_col_2+1], B_elements + B_offset_2, A_elements[row_start_A + 1],
                                B_size2,
                                C_col_buffer + row_start_C, C_elements + row_start_C);
    return;
  }

  else // safely merge two rows into temporary buffer:
  {
    unsigned int A_col_1 = A_col_buffer[row_start_A];
    unsigned int A_col_2 = A_col_buffer[row_start_A + 1];

    unsigned int B_offset_1 = B_row_buffer[A_col_1];
    unsigned int B_offset_2 = B_row_buffer[A_col_2];

    row_C_len = row_C_scan_numeric_vector_1(B_col_buffer + B_offset_1, B_col_buffer + B_row_buffer[A_col_1+1], B_elements + B_offset_1, A_elements[row_start_A],
                                            B_col_buffer + B_offset_2, B_col_buffer + B_row_buffer[A_col_2+1], B_elements + B_offset_2, A_elements[row_start_A + 1],
                                            B_size2,
                                            row_C_vector_1, row_C_vector_1_values);
    row_start_A += 2;
  }

  // process remaining rows:
  while (row_end_A > row_start_A)
  {

    if (row_start_A + 1 == row_end_A) // last row to merge, write directly to C:
    {
      unsigned int A_col    = A_col_buffer[row_start_A];
      unsigned int B_offset = B_row_buffer[A_col];

      row_C_len = row_C_scan_numeric_vector_1(B_col_buffer + B_offset, B_col_buffer + B_row_buffer[A_col+1], B_elements + B_offset, A_elements[row_start_A],
                                              row_C_vector_1, row_C_vector_1 + row_C_len, row_C_vector_1_values, NumericT(1.0),
                                              B_size2,
                                              C_col_buffer + row_start_C, C_elements + row_start_C);
      return;
    }
    else if (row_start_A + 2 < row_end_A)// at least three more rows left, so merge two
    {
      // process single row:
      unsigned int A_col_1 = A_col_buffer[row_start_A];
      unsigned int A_col_2 = A_col_buffer[row_start_A + 1];

      unsigned int B_offset_1 = B_row_buffer[A_col_1];
      unsigned int B_offset_2 = B_row_buffer[A_col_2];

      unsigned int merged_len = row_C_scan_numeric_vector_1(B_col_buffer + B_offset_1, B_col_buffer + B_row_buffer[A_col_1+1], B_elements + B_offset_1, A_elements[row_start_A],
                                                            B_col_buffer + B_offset_2, B_col_buffer + B_row_buffer[A_col_2+1], B_elements + B_offset_2, A_elements[row_start_A + 1],
                                                            B_size2,
                                                            row_C_vector_3, row_C_vector_3_values);
      row_C_len = row_C_scan_numeric_vector_1(row_C_vector_3, row_C_vector_3 + merged_len, row_C_vector_3_values, NumericT(1.0),
                                              row_C_vector_1, row_C_vector_1 + row_C_len,  row_C_vector_1_values, NumericT(1.0),
                                              B_size2,
                                              row_C_vector_2, row_C_vector_2_values);
      row_start_A += 2;
    }
    else
    {
      unsigned int A_col    = A_col_buffer[row_start_A];
      unsigned int B_offset = B_row_buffer[A_col];

      row_C_len = row_C_scan_numeric_vector_1(B_col_buffer + B_offset, B_col_buffer + B_row_buffer[A_col+1], B_elements + B_offset, A_elements[row_start_A],
                                              row_C_vector_1, row_C_vector_1 + row_C_len, row_C_vector_1_values, NumericT(1.0),
                                              B_size2,
                                              row_C_vector_2, row_C_vector_2_values);
      ++row_start_A;
    }

    std::swap(row_C_vector_1,        row_C_vector_2);
    std::swap(row_C_vector_1_values, row_C_vector_2_values);
  }
}


} // namespace host_based
} //namespace blas
} //namespace cuarma
