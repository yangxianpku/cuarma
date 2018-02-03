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

/** @file cuarma/blas/detail/ilu/ilu0.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of incomplete factorization preconditioners with static nonzero pattern.

  ILU0 (Incomplete LU with zero fill-in)
  - All preconditioner nonzeros exist at locations that were nonzero in the input matrix.
  - The number of nonzeros in the output preconditioner are exactly the same number as the input matrix
*/

#include <vector>
#include <cmath>
#include <iostream>
#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/detail/ilu/common.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/backend/memory.hpp"
#include "cuarma/blas/host_based/common.hpp"
#include <map>

namespace cuarma
{
namespace blas
{

/** @brief A tag for incomplete LU factorization with static pattern (ILU0)
*/
class ilu0_tag
{
public:
  ilu0_tag(bool with_level_scheduling = false) : use_level_scheduling_(with_level_scheduling) {}

  bool use_level_scheduling() const { return use_level_scheduling_; }
  void use_level_scheduling(bool b) { use_level_scheduling_ = b; }

private:
  bool use_level_scheduling_;
};


/** @brief Implementation of a ILU-preconditioner with static pattern. Optimized version for CSR matrices.
  * refer to the Algorithm in Saad's book (1996 edition)
  *  @param A       The sparse matrix matrix. The result is directly written to A.
  */
template<typename NumericT>
void precondition(cuarma::compressed_matrix<NumericT> & A, ilu0_tag const & /* tag */)
{
  assert( (A.handle1().get_active_handle_id() == cuarma::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
  assert( (A.handle2().get_active_handle_id() == cuarma::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
  assert( (A.handle().get_active_handle_id()  == cuarma::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );

  NumericT           * elements   = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * row_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * col_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  // Note: Line numbers in the following refer to the algorithm in Saad's book

  for (arma_size_t i=1; i<A.size1(); ++i)  // Line 1
  {
    unsigned int row_i_begin = row_buffer[i];
    unsigned int row_i_end   = row_buffer[i+1];
    for (unsigned int buf_index_k = row_i_begin; buf_index_k < row_i_end; ++buf_index_k) //Note: We do not assume that the column indices within a row are sorted
    {
      unsigned int k = col_buffer[buf_index_k];
      if (k >= i)
        continue; //Note: We do not assume that the column indices within a row are sorted

      unsigned int row_k_begin = row_buffer[k];
      unsigned int row_k_end   = row_buffer[k+1];

      // get a_kk:
      NumericT a_kk = 0;
      for (unsigned int buf_index_akk = row_k_begin; buf_index_akk < row_k_end; ++buf_index_akk)
      {
        if (col_buffer[buf_index_akk] == k)
        {
          a_kk = elements[buf_index_akk];
          break;
        }
      }

      NumericT & a_ik = elements[buf_index_k];
      a_ik /= a_kk;                                 //Line 3

      for (unsigned int buf_index_j = row_i_begin; buf_index_j < row_i_end; ++buf_index_j) //Note: We do not assume that the column indices within a row are sorted
      {
        unsigned int j = col_buffer[buf_index_j];
        if (j <= k)
          continue;

        // determine a_kj:
        NumericT a_kj = 0;
        for (unsigned int buf_index_akj = row_k_begin; buf_index_akj < row_k_end; ++buf_index_akj)
        {
          if (col_buffer[buf_index_akj] == j)
          {
            a_kj = elements[buf_index_akj];
            break;
          }
        }

        //a_ij -= a_ik * a_kj
        elements[buf_index_j] -= a_ik * a_kj;  //Line 5
      }
    }
  }

}


/** @brief ILU0 preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class ilu0_precond
{
  typedef typename MatrixT::value_type      NumericType;

public:
  ilu0_precond(MatrixT const & mat, ilu0_tag const & tag) : tag_(tag), LU_()
  {
    //initialize preconditioner:
    //std::cout << "Start CPU precond" << std::endl;
    init(mat);
    //std::cout << "End CPU precond" << std::endl;
  }

  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    unsigned int const * row_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(LU_.handle1());
    unsigned int const * col_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(LU_.handle2());
    NumericType  const * elements   = cuarma::blas::host_based::detail::extract_raw_pointer<NumericType>(LU_.handle());

    cuarma::blas::host_based::detail::csr_inplace_solve<NumericType>(row_buffer, col_buffer, elements, vec, LU_.size2(), unit_lower_tag());
    cuarma::blas::host_based::detail::csr_inplace_solve<NumericType>(row_buffer, col_buffer, elements, vec, LU_.size2(), upper_tag());
  }

private:
  void init(MatrixT const & mat)
  {
    cuarma::context host_context(cuarma::MAIN_MEMORY);
    cuarma::switch_memory_context(LU_, host_context);

    cuarma::copy(mat, LU_);
    cuarma::blas::precondition(LU_, tag_);
  }

  ilu0_tag                                   tag_;
  cuarma::compressed_matrix<NumericType>   LU_;
};


/** @brief ILU0 preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class ilu0_precond< cuarma::compressed_matrix<NumericT, AlignmentV> >
{
  typedef cuarma::compressed_matrix<NumericT, AlignmentV>   MatrixType;

public:
  ilu0_precond(MatrixType const & mat, ilu0_tag const & tag)
    : tag_(tag),
      LU_(mat.size1(), mat.size2(), cuarma::traits::context(mat))
  {
    //initialize preconditioner:
    //std::cout << "Start GPU precond" << std::endl;
    init(mat);
    //std::cout << "End GPU precond" << std::endl;
  }

  void apply(cuarma::vector<NumericT> & vec) const
  {
    cuarma::context host_context(cuarma::MAIN_MEMORY);
    if (vec.handle().get_active_handle_id() != cuarma::MAIN_MEMORY)
    {
      if (tag_.use_level_scheduling())
      {
        //std::cout << "Using multifrontal on GPU..." << std::endl;
        detail::level_scheduling_substitute(vec,
                                            multifrontal_L_row_index_arrays_,
                                            multifrontal_L_row_buffers_,
                                            multifrontal_L_col_buffers_,
                                            multifrontal_L_element_buffers_,
                                            multifrontal_L_row_elimination_num_list_);

        vec = cuarma::blas::element_div(vec, multifrontal_U_diagonal_);

        detail::level_scheduling_substitute(vec,
                                            multifrontal_U_row_index_arrays_,
                                            multifrontal_U_row_buffers_,
                                            multifrontal_U_col_buffers_,
                                            multifrontal_U_element_buffers_,
                                            multifrontal_U_row_elimination_num_list_);
      }
      else
      {
        cuarma::context old_context = cuarma::traits::context(vec);
        cuarma::switch_memory_context(vec, host_context);
        cuarma::blas::inplace_solve(LU_, vec, unit_lower_tag());
        cuarma::blas::inplace_solve(LU_, vec, upper_tag());
        cuarma::switch_memory_context(vec, old_context);
      }
    }
    else //apply ILU0 directly on CPU
    {
      if (tag_.use_level_scheduling())
      {
        //std::cout << "Using multifrontal..." << std::endl;
        detail::level_scheduling_substitute(vec,
                                            multifrontal_L_row_index_arrays_,
                                            multifrontal_L_row_buffers_,
                                            multifrontal_L_col_buffers_,
                                            multifrontal_L_element_buffers_,
                                            multifrontal_L_row_elimination_num_list_);

        vec = cuarma::blas::element_div(vec, multifrontal_U_diagonal_);

        detail::level_scheduling_substitute(vec,
                                            multifrontal_U_row_index_arrays_,
                                            multifrontal_U_row_buffers_,
                                            multifrontal_U_col_buffers_,
                                            multifrontal_U_element_buffers_,
                                            multifrontal_U_row_elimination_num_list_);
      }
      else
      {
        cuarma::blas::inplace_solve(LU_, vec, unit_lower_tag());
        cuarma::blas::inplace_solve(LU_, vec, upper_tag());
      }
    }
  }

  arma_size_t levels() const { return multifrontal_L_row_index_arrays_.size(); }

private:
  void init(MatrixType const & mat)
  {
    cuarma::context host_context(cuarma::MAIN_MEMORY);
    cuarma::switch_memory_context(LU_, host_context);
    LU_ = mat;
    cuarma::blas::precondition(LU_, tag_);

    if (!tag_.use_level_scheduling())
      return;

    // multifrontal part:
    cuarma::switch_memory_context(multifrontal_U_diagonal_, host_context);
    multifrontal_U_diagonal_.resize(LU_.size1(), false);
    host_based::detail::row_info(LU_, multifrontal_U_diagonal_, cuarma::blas::detail::SPARSE_ROW_DIAGONAL);

    detail::level_scheduling_setup_L(LU_,
                                     multifrontal_U_diagonal_, //dummy
                                     multifrontal_L_row_index_arrays_,
                                     multifrontal_L_row_buffers_,
                                     multifrontal_L_col_buffers_,
                                     multifrontal_L_element_buffers_,
                                     multifrontal_L_row_elimination_num_list_);


    detail::level_scheduling_setup_U(LU_,
                                     multifrontal_U_diagonal_,
                                     multifrontal_U_row_index_arrays_,
                                     multifrontal_U_row_buffers_,
                                     multifrontal_U_col_buffers_,
                                     multifrontal_U_element_buffers_,
                                     multifrontal_U_row_elimination_num_list_);

    //
    // Bring to device if necessary:
    //

    // L:
    for (typename std::list< cuarma::backend::mem_handle >::iterator it  = multifrontal_L_row_index_arrays_.begin();
                                                                       it != multifrontal_L_row_index_arrays_.end();
                                                                     ++it)
      cuarma::backend::switch_memory_context<unsigned int>(*it, cuarma::traits::context(mat));

    for (typename std::list< cuarma::backend::mem_handle >::iterator it  = multifrontal_L_row_buffers_.begin();
                                                                       it != multifrontal_L_row_buffers_.end();
                                                                     ++it)
      cuarma::backend::switch_memory_context<unsigned int>(*it, cuarma::traits::context(mat));

    for (typename std::list< cuarma::backend::mem_handle >::iterator it  = multifrontal_L_col_buffers_.begin();
                                                                       it != multifrontal_L_col_buffers_.end();
                                                                     ++it)
      cuarma::backend::switch_memory_context<unsigned int>(*it, cuarma::traits::context(mat));

    for (typename std::list< cuarma::backend::mem_handle >::iterator it  = multifrontal_L_element_buffers_.begin();
                                                                       it != multifrontal_L_element_buffers_.end();
                                                                     ++it)
      cuarma::backend::switch_memory_context<NumericT>(*it, cuarma::traits::context(mat));


    // U:

    cuarma::switch_memory_context(multifrontal_U_diagonal_, cuarma::traits::context(mat));

    for (typename std::list< cuarma::backend::mem_handle >::iterator it  = multifrontal_U_row_index_arrays_.begin();
                                                                       it != multifrontal_U_row_index_arrays_.end();
                                                                     ++it)
      cuarma::backend::switch_memory_context<unsigned int>(*it, cuarma::traits::context(mat));

    for (typename std::list< cuarma::backend::mem_handle >::iterator it  = multifrontal_U_row_buffers_.begin();
                                                                       it != multifrontal_U_row_buffers_.end();
                                                                     ++it)
      cuarma::backend::switch_memory_context<unsigned int>(*it, cuarma::traits::context(mat));

    for (typename std::list< cuarma::backend::mem_handle >::iterator it  = multifrontal_U_col_buffers_.begin();
                                                                       it != multifrontal_U_col_buffers_.end();
                                                                     ++it)
      cuarma::backend::switch_memory_context<unsigned int>(*it, cuarma::traits::context(mat));

    for (typename std::list< cuarma::backend::mem_handle >::iterator it  = multifrontal_U_element_buffers_.begin();
                                                                       it != multifrontal_U_element_buffers_.end();
                                                                     ++it)
      cuarma::backend::switch_memory_context<NumericT>(*it, cuarma::traits::context(mat));

  }

  ilu0_tag tag_;
  cuarma::compressed_matrix<NumericT> LU_;

  std::list<cuarma::backend::mem_handle> multifrontal_L_row_index_arrays_;
  std::list<cuarma::backend::mem_handle> multifrontal_L_row_buffers_;
  std::list<cuarma::backend::mem_handle> multifrontal_L_col_buffers_;
  std::list<cuarma::backend::mem_handle> multifrontal_L_element_buffers_;
  std::list<arma_size_t>                    multifrontal_L_row_elimination_num_list_;

  cuarma::vector<NumericT> multifrontal_U_diagonal_;
  std::list<cuarma::backend::mem_handle> multifrontal_U_row_index_arrays_;
  std::list<cuarma::backend::mem_handle> multifrontal_U_row_buffers_;
  std::list<cuarma::backend::mem_handle> multifrontal_U_col_buffers_;
  std::list<cuarma::backend::mem_handle> multifrontal_U_element_buffers_;
  std::list<arma_size_t>                    multifrontal_U_row_elimination_num_list_;

};

} // namespace blas
} // namespace cuarma