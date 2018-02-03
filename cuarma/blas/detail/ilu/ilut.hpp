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

/** @file cuarma/blas/detail/ilu/ilut.hpp
 *  @encoding:UTF-8 文档编码
    @brief Implementations of an incomplete factorization preconditioner with threshold (ILUT)
*/

#include <vector>
#include <cmath>
#include <iostream>
#include "cuarma/forwards.h"
#include "cuarma/tools/tools.hpp"
#include "cuarma/blas/detail/ilu/common.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/blas/host_based/common.hpp"
#include <map>

namespace cuarma
{
namespace blas
{

/** @brief A tag for incomplete LU factorization with threshold (ILUT)
*/
class ilut_tag
{
  public:
    /** @brief The constructor.
    *
    * @param entries_per_row        Number of nonzero entries per row in L and U. Note that L and U are stored in a single matrix, thus there are 2*entries_per_row in total.
    * @param drop_tolerance         The drop tolerance for ILUT
    * @param with_level_scheduling  Flag for enabling level scheduling on GPUs.
    */
    ilut_tag(unsigned int entries_per_row = 20,
             double       drop_tolerance = 1e-4,
             bool         with_level_scheduling = false)
      : entries_per_row_(entries_per_row),drop_tolerance_(drop_tolerance), use_level_scheduling_(with_level_scheduling) {}

    void set_drop_tolerance(double tol)
    {
      if (tol > 0)
        drop_tolerance_ = tol;
    }
    double get_drop_tolerance() const { return drop_tolerance_; }

    void set_entries_per_row(unsigned int e)
    {
      if (e > 0)
        entries_per_row_ = e;
    }

    unsigned int get_entries_per_row() const { return entries_per_row_; }

    bool use_level_scheduling() const { return use_level_scheduling_; }
    void use_level_scheduling(bool b) { use_level_scheduling_ = b; }

  private:
    unsigned int entries_per_row_;
    double       drop_tolerance_;
    bool         use_level_scheduling_;
};


namespace detail
{
  /** @brief Helper struct for holding a sparse vector in linear memory. For internal use only.
    *
    * Unfortunately, the 'naive' implementation using a std::map<> is almost always too slow.
    *
    */
  template<typename NumericT>
  struct ilut_sparse_vector
  {
    ilut_sparse_vector(arma_size_t alloc_size = 0) : size_(0), col_indices_(alloc_size), elements_(alloc_size) {}

    void resize_if_bigger(arma_size_t s)
    {
      if (s > elements_.size())
      {
        col_indices_.resize(s);
        elements_.resize(s);
      }
      size_ = s;
    }

    arma_size_t size_;
    std::vector<unsigned int> col_indices_;
    std::vector<NumericT>     elements_;
  };

  /** @brief Subtracts a scaled sparse vector u from a sparse vector w and writes the output to z: z = w - alpha * u
    *
    * Sparsity pattern of u and w are usually different.
    *
    * @return Length of new vector
    */
  template<typename IndexT, typename NumericT>
  IndexT merge_subtract_sparse_rows(IndexT const * w_coords, NumericT const * w_elements, IndexT w_size,
                                    IndexT const * u_coords, NumericT const * u_elements, IndexT u_size, NumericT alpha,
                                    IndexT       * z_coords, NumericT       * z_elements)
  {
    IndexT index_w = 0;
    IndexT index_u = 0;
    IndexT index_z = 0;

    while (1)
    {
      if (index_w < w_size && index_u < u_size)
      {
        if (w_coords[index_w] < u_coords[index_u])
        {
          z_coords[index_z]     = w_coords[index_w];
          z_elements[index_z++] = w_elements[index_w++];
        }
        else if (w_coords[index_w] == u_coords[index_u])
        {
          z_coords[index_z]     = w_coords[index_w];
          z_elements[index_z++] = w_elements[index_w++] - alpha * u_elements[index_u++];
        }
        else
        {
          z_coords[index_z]     = u_coords[index_u];
          z_elements[index_z++] = - alpha * u_elements[index_u++];
        }
      }
      else if (index_w == w_size && index_u < u_size)
      {
        z_coords[index_z]     = u_coords[index_u];
        z_elements[index_z++] = - alpha * u_elements[index_u++];
      }
      else if (index_w < w_size && index_u == u_size)
      {
        z_coords[index_z]     = w_coords[index_w];
        z_elements[index_z++] = w_elements[index_w++];
      }
      else
        return index_z;
    }
  }

  template<typename SizeT, typename NumericT>
  void insert_with_value_sort(std::vector<std::pair<SizeT, NumericT> > & map,
                              SizeT index, NumericT value)
  {
    NumericT abs_value = std::fabs(value);
    if (abs_value > 0)
    {
      // find first element with smaller absolute value:
      std::size_t first_smaller_index = 0;
      while (first_smaller_index < map.size() && std::fabs(map[first_smaller_index].second) > abs_value)
        ++first_smaller_index;

      std::pair<SizeT, NumericT> tmp(index, value);
      for (std::size_t j=first_smaller_index; j<map.size(); ++j)
        std::swap(map[j], tmp);
    }
  }

}

/** @brief Implementation of a ILU-preconditioner with threshold. Optimized implementation for compressed_matrix.
*
* refer to Algorithm 10.6 by Saad's book (1996 edition)
*
*  @param A       The input matrix. Either a compressed_matrix or of type std::vector< std::map<T, U> >
*  @param L       The output matrix for L.
*  @param U       The output matrix for U.
*  @param tag     An ilut_tag in order to dispatch among several other preconditioners.
*/
template<typename NumericT>
void precondition(cuarma::compressed_matrix<NumericT> const & A,
                  cuarma::compressed_matrix<NumericT>       & L,
                  cuarma::compressed_matrix<NumericT>       & U,
                  ilut_tag const & tag)
{
  assert(A.size1() == L.size1() && bool("Output matrix size mismatch") );
  assert(A.size1() == U.size1() && bool("Output matrix size mismatch") );

  L.reserve( tag.get_entries_per_row()      * A.size1());
  U.reserve((tag.get_entries_per_row() + 1) * A.size1());

  arma_size_t avg_nnz_per_row = static_cast<arma_size_t>(A.nnz() / A.size1());
  detail::ilut_sparse_vector<NumericT> w1(tag.get_entries_per_row() * (avg_nnz_per_row + 10));
  detail::ilut_sparse_vector<NumericT> w2(tag.get_entries_per_row() * (avg_nnz_per_row + 10));
  detail::ilut_sparse_vector<NumericT> * w_in  = &w1;
  detail::ilut_sparse_vector<NumericT> * w_out = &w2;
  std::vector<NumericT> diagonal_U(A.size1());

  NumericT     const * elements_A   = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * row_buffer_A = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * col_buffer_A = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  NumericT           * elements_L   = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(L.handle());
  unsigned int       * row_buffer_L = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(L.handle1()); row_buffer_L[0] = 0;
  unsigned int       * col_buffer_L = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(L.handle2());

  NumericT           * elements_U   = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(U.handle());
  unsigned int       * row_buffer_U = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(U.handle1()); row_buffer_U[0] = 0;
  unsigned int       * col_buffer_U = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(U.handle2());

  std::vector<std::pair<unsigned int, NumericT> > sorted_entries_L(tag.get_entries_per_row());
  std::vector<std::pair<unsigned int, NumericT> > sorted_entries_U(tag.get_entries_per_row());

  for (arma_size_t i=0; i<cuarma::traits::size1(A); ++i)  // Line 1
  {
    std::fill(sorted_entries_L.begin(), sorted_entries_L.end(), std::pair<unsigned int, NumericT>(0, NumericT(0)));
    std::fill(sorted_entries_U.begin(), sorted_entries_U.end(), std::pair<unsigned int, NumericT>(0, NumericT(0)));

    //line 2: set up w
    w_in->resize_if_bigger(row_buffer_A[i+1] - row_buffer_A[i]);
    NumericT row_norm = 0;
    unsigned int k = 0;
    for (unsigned int j = row_buffer_A[i]; j < row_buffer_A[i+1]; ++j, ++k)
    {
      w_in->col_indices_[k] = col_buffer_A[j];
      NumericT entry = elements_A[j];
      w_in->elements_[k] = entry;
      row_norm += entry * entry;
    }
    row_norm = std::sqrt(row_norm);
    NumericT tau_i = static_cast<NumericT>(tag.get_drop_tolerance()) * row_norm;

    //line 3: Iterate over lower diagonal parts of A:
    k = 0;
    unsigned int current_col = (row_buffer_A[i+1] > row_buffer_A[i]) ? w_in->col_indices_[k] : static_cast<unsigned int>(i); // mind empty rows here!
    while (current_col < i)
    {
      //line 4:
      NumericT a_kk = diagonal_U[current_col];

      NumericT w_k_entry = w_in->elements_[k] / a_kk;
      w_in->elements_[k] = w_k_entry;

      //lines 5,6: (dropping rule to w_k)
      if ( std::fabs(w_k_entry) > tau_i)
      {
        //line 7:
        unsigned int row_U_begin = row_buffer_U[current_col];
        unsigned int row_U_end   = row_buffer_U[current_col + 1];

        if (row_U_end > row_U_begin)
        {
          w_out->resize_if_bigger(w_in->size_ + (row_U_end - row_U_begin) - 1);
          w_out->size_ = detail::merge_subtract_sparse_rows(&(w_in->col_indices_[0]), &(w_in->elements_[0]), static_cast<unsigned int>(w_in->size_),
                                                            col_buffer_U + row_U_begin + 1, elements_U + row_U_begin + 1, (row_U_end - row_U_begin) - 1, w_k_entry,
                                                            &(w_out->col_indices_[0]), &(w_out->elements_[0])
                                                           );
          ++k;
        }
      }
      else // drop element
      {
        w_out->resize_if_bigger(w_in->size_ - 1);
        for (unsigned int r = 0; r < k; ++r)
        {
          w_out->col_indices_[r] = w_in->col_indices_[r];
          w_out->elements_[r]    = w_in->elements_[r];
        }
        for (unsigned int r = k+1; r < w_in->size_; ++r)
        {
          w_out->col_indices_[r-1] = w_in->col_indices_[r];
          w_out->elements_[r-1]    = w_in->elements_[r];
        }

        // Note: No increment to k here, element was dropped!
      }

      // swap pointers to w1 and w2
      std::swap(w_in, w_out);

      // process next entry:
      current_col = (k < w_in->size_) ? w_in->col_indices_[k] : static_cast<unsigned int>(i);
    } // while()

    // Line 10: Apply a dropping rule to w
    // To do so, we write values to a temporary array
    for (unsigned int r = 0; r < w_in->size_; ++r)
    {
      unsigned int col   = w_in->col_indices_[r];
      NumericT     value = w_in->elements_[r];

      if (col < i) // entry for L:
        detail::insert_with_value_sort(sorted_entries_L, col, value);
      else if (col == i) // do not drop diagonal element
      {
        diagonal_U[i] = value;
        if (value <= 0 && value >= 0)
        {
          std::cerr << "cuarma: FATAL ERROR in ILUT(): Diagonal entry computed to zero (" << value << ") in row " << i << "!" << std::endl;
          throw zero_on_diagonal_exception("ILUT zero diagonal!");
        }
      }
      else // entry for U:
        detail::insert_with_value_sort(sorted_entries_U, col, value);
    }

    //Lines 10-12: Apply a dropping rule to w, write the largest p values to L and U
    unsigned int offset_L = row_buffer_L[i];
    std::sort(sorted_entries_L.begin(), sorted_entries_L.end());
    for (unsigned int j=0; j<tag.get_entries_per_row(); ++j)
      if (std::fabs(sorted_entries_L[j].second) > 0)
      {
        col_buffer_L[offset_L] = sorted_entries_L[j].first;
        elements_L[offset_L]   = sorted_entries_L[j].second;
        ++offset_L;
      }
    row_buffer_L[i+1] = offset_L;

    unsigned int offset_U = row_buffer_U[i];
    col_buffer_U[offset_U] = static_cast<unsigned int>(i);
    elements_U[offset_U]   = diagonal_U[i];
    ++offset_U;
    std::sort(sorted_entries_U.begin(), sorted_entries_U.end());
    for (unsigned int j=0; j<tag.get_entries_per_row(); ++j)
      if (std::fabs(sorted_entries_U[j].second) > 0)
      {
        col_buffer_U[offset_U] = sorted_entries_U[j].first;
        elements_U[offset_U]   = sorted_entries_U[j].second;
        ++offset_U;
      }
    row_buffer_U[i+1] = offset_U;

  } //for i
}


/** @brief ILUT preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class ilut_precond
{
  typedef typename MatrixT::value_type      NumericType;

public:
  ilut_precond(MatrixT const & mat, ilut_tag const & tag) : tag_(tag), L_(mat.size1(), mat.size2()), U_(mat.size1(), mat.size2())
  {
    //initialize preconditioner:
    //std::cout << "Start CPU precond" << std::endl;
    init(mat);
    //std::cout << "End CPU precond" << std::endl;
  }

  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    //Note: Since vec can be a rather arbitrary vector type, we call the more generic version in the backend manually:
    {
      unsigned int const * row_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(L_.handle1());
      unsigned int const * col_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(L_.handle2());
      NumericType  const * elements   = cuarma::blas::host_based::detail::extract_raw_pointer<NumericType>(L_.handle());

      cuarma::blas::host_based::detail::csr_inplace_solve<NumericType>(row_buffer, col_buffer, elements, vec, L_.size2(), unit_lower_tag());
    }
    {
      unsigned int const * row_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(U_.handle1());
      unsigned int const * col_buffer = cuarma::blas::host_based::detail::extract_raw_pointer<unsigned int>(U_.handle2());
      NumericType  const * elements   = cuarma::blas::host_based::detail::extract_raw_pointer<NumericType>(U_.handle());

      cuarma::blas::host_based::detail::csr_inplace_solve<NumericType>(row_buffer, col_buffer, elements, vec, U_.size2(), upper_tag());
    }
  }

private:
  void init(MatrixT const & mat)
  {
    cuarma::context host_context(cuarma::MAIN_MEMORY);
    cuarma::compressed_matrix<NumericType> temp;
    cuarma::switch_memory_context(temp, host_context);
    cuarma::switch_memory_context(L_, host_context);
    cuarma::switch_memory_context(U_, host_context);

    cuarma::copy(mat, temp);

    cuarma::blas::precondition(temp, L_, U_, tag_);
  }

  ilut_tag tag_;
  cuarma::compressed_matrix<NumericType> L_;
  cuarma::compressed_matrix<NumericType> U_;
};


/** @brief ILUT preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class ilut_precond< cuarma::compressed_matrix<NumericT, AlignmentV> >
{
typedef cuarma::compressed_matrix<NumericT, AlignmentV>   MatrixType;

public:
  ilut_precond(MatrixType const & mat, ilut_tag const & tag)
    : tag_(tag),
      L_(mat.size1(), mat.size2(), cuarma::traits::context(mat)),
      U_(mat.size1(), mat.size2(), cuarma::traits::context(mat))
  {
    //initialize preconditioner:
    //std::cout << "Start GPU precond" << std::endl;
    init(mat);
    //std::cout << "End GPU precond" << std::endl;
  }

  void apply(cuarma::vector<NumericT> & vec) const
  {
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
        cuarma::context host_context(cuarma::MAIN_MEMORY);
        cuarma::context old_context = cuarma::traits::context(vec);
        cuarma::switch_memory_context(vec, host_context);
        cuarma::blas::inplace_solve(L_, vec, unit_lower_tag());
        cuarma::blas::inplace_solve(U_, vec, upper_tag());
        cuarma::switch_memory_context(vec, old_context);
      }
    }
    else //apply ILUT directly:
    {
      cuarma::blas::inplace_solve(L_, vec, unit_lower_tag());
      cuarma::blas::inplace_solve(U_, vec, upper_tag());
    }
  }

private:
  void init(MatrixType const & mat)
  {
    cuarma::context host_context(cuarma::MAIN_MEMORY);
    cuarma::switch_memory_context(L_, host_context);
    cuarma::switch_memory_context(U_, host_context);

    if (cuarma::traits::context(mat).memory_type() == cuarma::MAIN_MEMORY)
    {
      cuarma::blas::precondition(mat, L_, U_, tag_);
    }
    else //we need to copy to CPU
    {
      cuarma::compressed_matrix<NumericT> cpu_mat(mat.size1(), mat.size2(), cuarma::traits::context(mat));
      cuarma::switch_memory_context(cpu_mat, host_context);

      cpu_mat = mat;

      cuarma::blas::precondition(cpu_mat, L_, U_, tag_);
    }

    if (!tag_.use_level_scheduling())
      return;

    //
    // multifrontal part:
    //

    cuarma::switch_memory_context(multifrontal_U_diagonal_, host_context);
    multifrontal_U_diagonal_.resize(U_.size1(), false);
    host_based::detail::row_info(U_, multifrontal_U_diagonal_, cuarma::blas::detail::SPARSE_ROW_DIAGONAL);

    detail::level_scheduling_setup_L(L_,
                                     multifrontal_U_diagonal_, //dummy
                                     multifrontal_L_row_index_arrays_,
                                     multifrontal_L_row_buffers_,
                                     multifrontal_L_col_buffers_,
                                     multifrontal_L_element_buffers_,
                                     multifrontal_L_row_elimination_num_list_);


    detail::level_scheduling_setup_U(U_,
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

  ilut_tag tag_;
  cuarma::compressed_matrix<NumericT> L_;
  cuarma::compressed_matrix<NumericT> U_;

  std::list<cuarma::backend::mem_handle> multifrontal_L_row_index_arrays_;
  std::list<cuarma::backend::mem_handle> multifrontal_L_row_buffers_;
  std::list<cuarma::backend::mem_handle> multifrontal_L_col_buffers_;
  std::list<cuarma::backend::mem_handle> multifrontal_L_element_buffers_;
  std::list<arma_size_t > multifrontal_L_row_elimination_num_list_;

  cuarma::vector<NumericT> multifrontal_U_diagonal_;
  std::list<cuarma::backend::mem_handle> multifrontal_U_row_index_arrays_;
  std::list<cuarma::backend::mem_handle> multifrontal_U_row_buffers_;
  std::list<cuarma::backend::mem_handle> multifrontal_U_col_buffers_;
  std::list<cuarma::backend::mem_handle> multifrontal_U_element_buffers_;
  std::list<arma_size_t > multifrontal_U_row_elimination_num_list_;
};

} // namespace blas
} // namespace cuarma