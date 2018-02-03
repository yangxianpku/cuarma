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

/** @file cuarma/compressed_matrix.hpp
 *  @encoding:UTF-8 文档编码
 *  @brief Implementation of the compressed_matrix class
*/

#include <vector>
#include <list>
#include <map>
#include "cuarma/forwards.h"
#include "cuarma/vector.hpp"
#include "cuarma/blas/sparse_matrix_operations.hpp"
#include "cuarma/tools/tools.hpp"
#include "cuarma/tools/entry_proxy.hpp"

#ifdef CUARMA_WITH_UBLAS
#include <boost/numeric/ublas/matrix_sparse.hpp>
#endif

namespace cuarma
{
namespace detail
{

  /** @brief Implementation of the copy of a host-based sparse matrix to the device.
    *
    * See convenience copy() routines for type requirements of CPUMatrixT
    */
  template<typename CPUMatrixT, typename NumericT, unsigned int AlignmentV>
  void copy_impl(const CPUMatrixT & cpu_matrix,
                 compressed_matrix<NumericT, AlignmentV> & gpu_matrix,
                 arma_size_t nonzeros)
  {
    assert( (gpu_matrix.size1() == 0 || cuarma::traits::size1(cpu_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
    assert( (gpu_matrix.size2() == 0 || cuarma::traits::size2(cpu_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

    cuarma::backend::typesafe_host_array<unsigned int> row_buffer(gpu_matrix.handle1(), cpu_matrix.size1() + 1);
    cuarma::backend::typesafe_host_array<unsigned int> col_buffer(gpu_matrix.handle2(), nonzeros);
    std::vector<NumericT> elements(nonzeros);

    arma_size_t row_index  = 0;
    arma_size_t data_index = 0;

    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1();
         row_it != cpu_matrix.end1();
         ++row_it)
    {
      row_buffer.set(row_index, data_index);
      ++row_index;

      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin();
           col_it != row_it.end();
           ++col_it)
      {
        col_buffer.set(data_index, col_it.index2());
        elements[data_index] = *col_it;
        ++data_index;
      }
      data_index = cuarma::tools::align_to_multiple<arma_size_t>(data_index, AlignmentV); //take care of alignment
    }
    row_buffer.set(row_index, data_index);

    gpu_matrix.set(row_buffer.get(),
                   col_buffer.get(),
                   &elements[0],
        cpu_matrix.size1(),
        cpu_matrix.size2(),
        nonzeros);
  }
}

//
// host to device:
//

//provide copy-operation:
/** @brief Copies a sparse matrix from the host to the  device (either GPU or multi-core CPU)
  *
  * There are some type requirements on the CPUMatrixT type (fulfilled by e.g. boost::numeric::ublas):
  * - .size1() returns the number of rows
  * - .size2() returns the number of columns
  * - const_iterator1    is a type definition for an iterator along increasing row indices
  * - const_iterator2    is a type definition for an iterator along increasing columns indices
  * - The const_iterator1 type provides an iterator of type const_iterator2 via members .begin() and .end() that iterates along column indices in the current row.
  * - The types const_iterator1 and const_iterator2 provide members functions .index1() and .index2() that return the current row and column indices respectively.
  * - Dereferenciation of an object of type const_iterator2 returns the entry.
  *
  * @param cpu_matrix   A sparse matrix on the host.
  * @param gpu_matrix   A compressed_matrix from cuarma
  */
template<typename CPUMatrixT, typename NumericT, unsigned int AlignmentV>
void copy(const CPUMatrixT & cpu_matrix,
          compressed_matrix<NumericT, AlignmentV> & gpu_matrix )
{
  if ( cpu_matrix.size1() > 0 && cpu_matrix.size2() > 0 )
  {
    //determine nonzeros:
    arma_size_t num_entries = 0;
    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1();
         row_it != cpu_matrix.end1();
         ++row_it)
    {
      arma_size_t entries_per_row = 0;
      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin();
           col_it != row_it.end();
           ++col_it)
      {
        ++entries_per_row;
      }
      num_entries += cuarma::tools::align_to_multiple<arma_size_t>(entries_per_row, AlignmentV);
    }

    if (num_entries == 0) //we copy an empty matrix
      num_entries = 1;

    //set up matrix entries:
    cuarma::detail::copy_impl(cpu_matrix, gpu_matrix, num_entries);
  }
}


//adapted for std::vector< std::map < > > argument:
/** @brief Copies a sparse square matrix in the std::vector< std::map < > > format to an device. Use cuarma::tools::sparse_matrix_adapter for non-square matrices.
  *
  * @param cpu_matrix   A sparse square matrix on the host using STL types
  * @param gpu_matrix   A compressed_matrix from cuarma
  */
template<typename SizeT, typename NumericT, unsigned int AlignmentV>
void copy(const std::vector< std::map<SizeT, NumericT> > & cpu_matrix, compressed_matrix<NumericT, AlignmentV> & gpu_matrix )
{
  arma_size_t nonzeros = 0;
  arma_size_t max_col = 0;
  for (arma_size_t i=0; i<cpu_matrix.size(); ++i)
  {
    if (cpu_matrix[i].size() > 0)
      nonzeros += ((cpu_matrix[i].size() - 1) / AlignmentV + 1) * AlignmentV;
    if (cpu_matrix[i].size() > 0)
      max_col = std::max<arma_size_t>(max_col, (cpu_matrix[i].rbegin())->first);
  }

  cuarma::detail::copy_impl(tools::const_sparse_matrix_adapter<NumericT, SizeT>(cpu_matrix, cpu_matrix.size(), max_col + 1),
                              gpu_matrix,
                              nonzeros);
}

#ifdef CUARMA_WITH_UBLAS
/** @brief Convenience routine for copying a sparse uBLAS matrix to a cuarma matrix.
  *
  * Optimization which copies the data directly from the internal uBLAS buffers.
  */
template<typename ScalarType, typename F, arma_size_t IB, typename IA, typename TA>
void copy(const boost::numeric::ublas::compressed_matrix<ScalarType, F, IB, IA, TA> & ublas_matrix,
          cuarma::compressed_matrix<ScalarType, 1> & gpu_matrix)
{
  assert( (gpu_matrix.size1() == 0 || cuarma::traits::size1(ublas_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (gpu_matrix.size2() == 0 || cuarma::traits::size2(ublas_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

  //we just need to copy the CSR arrays:
  cuarma::backend::typesafe_host_array<unsigned int> row_buffer(gpu_matrix.handle1(), ublas_matrix.size1() + 1);
  for (arma_size_t i=0; i<=ublas_matrix.size1(); ++i)
    row_buffer.set(i, ublas_matrix.index1_data()[i]);

  cuarma::backend::typesafe_host_array<unsigned int> col_buffer(gpu_matrix.handle2(), ublas_matrix.nnz());
  for (arma_size_t i=0; i<ublas_matrix.nnz(); ++i)
    col_buffer.set(i, ublas_matrix.index2_data()[i]);

  gpu_matrix.set(row_buffer.get(),
                 col_buffer.get(),
                 &(ublas_matrix.value_data()[0]),
      ublas_matrix.size1(),
      ublas_matrix.size2(),
      ublas_matrix.nnz());

}
#endif


//
// device to host:
//
/** @brief Copies a sparse matrix from the  device (either GPU or multi-core CPU) to the host.
  *
  * There are two type requirements on the CPUMatrixT type (fulfilled by e.g. boost::numeric::ublas):
  * - resize(rows, cols)  A resize function to bring the matrix into the correct size
  * - operator(i,j)       Write new entries via the parenthesis operator
  *
  * @param gpu_matrix   A compressed_matrix from cuarma
  * @param cpu_matrix   A sparse matrix on the host.
  */
template<typename CPUMatrixT, typename NumericT, unsigned int AlignmentV>
void copy(const compressed_matrix<NumericT, AlignmentV> & gpu_matrix,
          CPUMatrixT & cpu_matrix )
{
  assert( (cuarma::traits::size1(cpu_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (cuarma::traits::size2(cpu_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

  if ( gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0 )
  {
    //get raw data from memory:
    cuarma::backend::typesafe_host_array<unsigned int> row_buffer(gpu_matrix.handle1(), cpu_matrix.size1() + 1);
    cuarma::backend::typesafe_host_array<unsigned int> col_buffer(gpu_matrix.handle2(), gpu_matrix.nnz());
    std::vector<NumericT> elements(gpu_matrix.nnz());

    //std::cout << "GPU->CPU, nonzeros: " << gpu_matrix.nnz() << std::endl;

    cuarma::backend::memory_read(gpu_matrix.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
    cuarma::backend::memory_read(gpu_matrix.handle2(), 0, col_buffer.raw_size(), col_buffer.get());
    cuarma::backend::memory_read(gpu_matrix.handle(),  0, sizeof(NumericT)* gpu_matrix.nnz(), &(elements[0]));

    //fill the cpu_matrix:
    arma_size_t data_index = 0;
    for (arma_size_t row = 1; row <= gpu_matrix.size1(); ++row)
    {
      while (data_index < row_buffer[row])
      {
        if (col_buffer[data_index] >= gpu_matrix.size2())
        {
          std::cerr << "cuarma encountered invalid data at colbuffer[" << data_index << "]: " << col_buffer[data_index] << std::endl;
          return;
        }

        if (std::fabs(elements[data_index]) > static_cast<NumericT>(0))
          cpu_matrix(row-1, static_cast<arma_size_t>(col_buffer[data_index])) = elements[data_index];
        ++data_index;
      }
    }
  }
}


/** @brief Copies a sparse matrix from an  device to the host. The host type is the std::vector< std::map < > > format .
  *
  * @param gpu_matrix   A compressed_matrix from cuarma
  * @param cpu_matrix   A sparse matrix on the host.
  */
template<typename NumericT, unsigned int AlignmentV>
void copy(const compressed_matrix<NumericT, AlignmentV> & gpu_matrix,
          std::vector< std::map<unsigned int, NumericT> > & cpu_matrix)
{
  assert( (cpu_matrix.size() == gpu_matrix.size1()) && bool("Size mismatch") );

  tools::sparse_matrix_adapter<NumericT> temp(cpu_matrix, gpu_matrix.size1(), gpu_matrix.size2());
  copy(gpu_matrix, temp);
}

#ifdef CUARMA_WITH_UBLAS
/** @brief Convenience routine for copying a cuarma sparse matrix back to a sparse uBLAS matrix
  *
  * Directly populates the internal buffer of the uBLAS matrix, thus avoiding a temporary STL matrix.
  */
template<typename ScalarType, unsigned int AlignmentV, typename F, arma_size_t IB, typename IA, typename TA>
void copy(cuarma::compressed_matrix<ScalarType, AlignmentV> const & gpu_matrix,
          boost::numeric::ublas::compressed_matrix<ScalarType> & ublas_matrix)
{
  assert( (cuarma::traits::size1(ublas_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (cuarma::traits::size2(ublas_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

  cuarma::backend::typesafe_host_array<unsigned int> row_buffer(gpu_matrix.handle1(), gpu_matrix.size1() + 1);
  cuarma::backend::typesafe_host_array<unsigned int> col_buffer(gpu_matrix.handle2(), gpu_matrix.nnz());

  cuarma::backend::memory_read(gpu_matrix.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
  cuarma::backend::memory_read(gpu_matrix.handle2(), 0, col_buffer.raw_size(), col_buffer.get());

  ublas_matrix.clear();
  ublas_matrix.reserve(gpu_matrix.nnz());

  ublas_matrix.set_filled(gpu_matrix.size1() + 1, gpu_matrix.nnz());

  for (arma_size_t i=0; i<ublas_matrix.size1() + 1; ++i)
    ublas_matrix.index1_data()[i] = row_buffer[i];

  for (arma_size_t i=0; i<ublas_matrix.nnz(); ++i)
    ublas_matrix.index2_data()[i] = col_buffer[i];

  cuarma::backend::memory_read(gpu_matrix.handle(),  0, sizeof(ScalarType) * gpu_matrix.nnz(), &(ublas_matrix.value_data()[0]));

}
#endif

//////////////////////// compressed_matrix //////////////////////////
/** @brief A sparse square matrix in compressed sparse rows format.
  *
  * @tparam NumericT    The floating point type (either float or double, checked at compile time)
  * @tparam AlignmentV     The internal memory size for the entries in each row is given by (size()/AlignmentV + 1) * AlignmentV. AlignmentV must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
  */
template<class NumericT, unsigned int AlignmentV /* see VCLForwards.h */>
class compressed_matrix
{
public:
  typedef cuarma::backend::mem_handle                                                              handle_type;
  typedef scalar<typename cuarma::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<NumericT>::ResultType>   value_type;
  typedef arma_size_t                                                                                 size_type;

  /** @brief Default construction of a compressed matrix. No memory is allocated */
  compressed_matrix() : rows_(0), cols_(0), nonzeros_(0), row_block_num_(0) {}

  /** @brief Construction of a compressed matrix with the supplied number of rows and columns. If the number of nonzeros is positive, memory is allocated
      *
      * @param rows     Number of rows
      * @param cols     Number of columns
      * @param nonzeros Optional number of nonzeros for memory preallocation
      * @param ctx      Optional context in which the matrix is created (one out of multiple  contexts, CUDA, host)
      */
  explicit compressed_matrix(arma_size_t rows, arma_size_t cols, arma_size_t nonzeros = 0, cuarma::context ctx = cuarma::context())
    : rows_(rows), cols_(cols), nonzeros_(nonzeros), row_block_num_(0)
  {
    row_buffer_.switch_active_handle_id(ctx.memory_type());
    col_buffer_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());
    row_blocks_.switch_active_handle_id(ctx.memory_type());


    if (rows > 0)
    {
      cuarma::backend::memory_create(row_buffer_, cuarma::backend::typesafe_host_array<unsigned int>().element_size() * (rows + 1), ctx);
      cuarma::vector_base<unsigned int> init_temporary(row_buffer_, size_type(rows+1), 0, 1);
      init_temporary = cuarma::zero_vector<unsigned int>(size_type(rows+1), ctx);
    }
    if (nonzeros > 0)
    {
      cuarma::backend::memory_create(col_buffer_, cuarma::backend::typesafe_host_array<unsigned int>().element_size() * nonzeros, ctx);
      cuarma::backend::memory_create(elements_, sizeof(NumericT) * nonzeros, ctx);
    }
  }

  /** @brief Construction of a compressed matrix with the supplied number of rows and columns. If the number of nonzeros is positive, memory is allocated
      *
      * @param rows     Number of rows
      * @param cols     Number of columns
      * @param ctx      Context in which to create the matrix
      */
  explicit compressed_matrix(arma_size_t rows, arma_size_t cols, cuarma::context ctx)
    : rows_(rows), cols_(cols), nonzeros_(0), row_block_num_(0)
  {
    row_buffer_.switch_active_handle_id(ctx.memory_type());
    col_buffer_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());
    row_blocks_.switch_active_handle_id(ctx.memory_type());


    if (rows > 0)
    {
      cuarma::backend::memory_create(row_buffer_, cuarma::backend::typesafe_host_array<unsigned int>().element_size() * (rows + 1), ctx);
      cuarma::vector_base<unsigned int> init_temporary(row_buffer_, size_type(rows+1), 0, 1);
      init_temporary = cuarma::zero_vector<unsigned int>(size_type(rows+1), ctx);
    }
  }

  /** @brief Creates an empty compressed_matrix, but sets the respective context information.
    *
    * This is useful if you want to want to populate e.g. a cuarma::compressed_matrix<> on the host with copy(), but the default backend is .
    */
  explicit compressed_matrix(cuarma::context ctx) : rows_(0), cols_(0), nonzeros_(0), row_block_num_(0)
  {
    row_buffer_.switch_active_handle_id(ctx.memory_type());
    col_buffer_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());
    row_blocks_.switch_active_handle_id(ctx.memory_type());
  }


  /** @brief Assignment a compressed matrix from the product of two compressed_matrix objects (C = A * B). */
  compressed_matrix(matrix_expression<const compressed_matrix, const compressed_matrix, op_prod> const & proxy)
    : rows_(0), cols_(0), nonzeros_(0), row_block_num_(0)
  {
    cuarma::context ctx = cuarma::traits::context(proxy.lhs());

    row_buffer_.switch_active_handle_id(ctx.memory_type());
    col_buffer_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());
    row_blocks_.switch_active_handle_id(ctx.memory_type());

    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), *this);
    generate_row_block_information();
  }

  /** @brief Assignment a compressed matrix from possibly another memory domain. */
  compressed_matrix & operator=(compressed_matrix const & other)
  {
    assert( (rows_ == 0 || rows_ == other.size1()) && bool("Size mismatch") );
    assert( (cols_ == 0 || cols_ == other.size2()) && bool("Size mismatch") );

    rows_ = other.size1();
    cols_ = other.size2();
    nonzeros_ = other.nnz();
    row_block_num_ = other.row_block_num_;

    cuarma::backend::typesafe_memory_copy<unsigned int>(other.row_buffer_, row_buffer_);
    cuarma::backend::typesafe_memory_copy<unsigned int>(other.col_buffer_, col_buffer_);
    cuarma::backend::typesafe_memory_copy<unsigned int>(other.row_blocks_, row_blocks_);
    cuarma::backend::typesafe_memory_copy<NumericT>(other.elements_, elements_);

    return *this;
  }

  /** @brief Assignment a compressed matrix from the product of two compressed_matrix objects (C = A * B). */
  compressed_matrix & operator=(matrix_expression<const compressed_matrix, const compressed_matrix, op_prod> const & proxy)
  {
    assert( (rows_ == 0 || rows_ == proxy.lhs().size1()) && bool("Size mismatch") );
    assert( (cols_ == 0 || cols_ == proxy.rhs().size2()) && bool("Size mismatch") );

    cuarma::blas::prod_impl(proxy.lhs(), proxy.rhs(), *this);
    generate_row_block_information();

    return *this;
  }


  /** @brief Sets the row, column and value arrays of the compressed matrix
    *
    * Type of row_jumper and col_buffer is 'unsigned int' for CUDA and OpenMP (host) backend, but *must* be cl_uint for .
    * The reason is that 'unsigned int' might have a different bit representation on the host than 'unsigned int' on the  device.
    * cl_uint is guaranteed to have the correct bit representation for  devices.
    *
    * @param row_jumper     Pointer to an array holding the indices of the first element of each row (starting with zero). E.g. row_jumper[10] returns the index of the first entry of the 11th row. The array length is 'cols + 1'
    * @param col_buffer     Pointer to an array holding the column index of each entry. The array length is 'nonzeros'
    * @param elements       Pointer to an array holding the entries of the sparse matrix. The array length is 'elements'
    * @param rows           Number of rows of the sparse matrix
    * @param cols           Number of columns of the sparse matrix
    * @param nonzeros       Number of nonzeros
    */
  void set(const void * row_jumper,
           const void * col_buffer,
           const NumericT * elements,
           arma_size_t rows,
           arma_size_t cols,
           arma_size_t nonzeros)
  {
    assert( (rows > 0)     && bool("Error in compressed_matrix::set(): Number of rows must be larger than zero!"));
    assert( (cols > 0)     && bool("Error in compressed_matrix::set(): Number of columns must be larger than zero!"));
    assert( (nonzeros > 0) && bool("Error in compressed_matrix::set(): Number of nonzeros must be larger than zero!"));
    //std::cout << "Setting memory: " << cols + 1 << ", " << nonzeros << std::endl;

    //row_buffer_.switch_active_handle_id(cuarma::backend::_MEMORY);
    cuarma::backend::memory_create(row_buffer_, cuarma::backend::typesafe_host_array<unsigned int>(row_buffer_).element_size() * (rows + 1), cuarma::traits::context(row_buffer_), row_jumper);

    //col_buffer_.switch_active_handle_id(cuarma::backend::_MEMORY);
    cuarma::backend::memory_create(col_buffer_, cuarma::backend::typesafe_host_array<unsigned int>(col_buffer_).element_size() * nonzeros, cuarma::traits::context(col_buffer_), col_buffer);

    //elements_.switch_active_handle_id(cuarma::backend::_MEMORY);
    cuarma::backend::memory_create(elements_, sizeof(NumericT) * nonzeros, cuarma::traits::context(elements_), elements);

    nonzeros_ = nonzeros;
    rows_ = rows;
    cols_ = cols;

    //generate block information for CSR-adaptive:
    generate_row_block_information();
  }

  /** @brief Allocate memory for the supplied number of nonzeros in the matrix. Old values are preserved. */
  void reserve(arma_size_t new_nonzeros, bool preserve = true)
  {
    if (new_nonzeros > nonzeros_)
    {
      if (preserve)
      {
        handle_type col_buffer_old;
        handle_type elements_old;
        cuarma::backend::memory_shallow_copy(col_buffer_, col_buffer_old);
        cuarma::backend::memory_shallow_copy(elements_,   elements_old);

        cuarma::backend::typesafe_host_array<unsigned int> size_deducer(col_buffer_);
        cuarma::backend::memory_create(col_buffer_, size_deducer.element_size() * new_nonzeros, cuarma::traits::context(col_buffer_));
        cuarma::backend::memory_create(elements_,   sizeof(NumericT) * new_nonzeros,          cuarma::traits::context(elements_));

        cuarma::backend::memory_copy(col_buffer_old, col_buffer_, 0, 0, size_deducer.element_size() * nonzeros_);
        cuarma::backend::memory_copy(elements_old,   elements_,   0, 0, sizeof(NumericT)* nonzeros_);
      }
      else
      {
        cuarma::backend::typesafe_host_array<unsigned int> size_deducer(col_buffer_);
        cuarma::backend::memory_create(col_buffer_, size_deducer.element_size() * new_nonzeros, cuarma::traits::context(col_buffer_));
        cuarma::backend::memory_create(elements_,   sizeof(NumericT)            * new_nonzeros, cuarma::traits::context(elements_));
      }

      nonzeros_ = new_nonzeros;
    }
  }

  /** @brief Resize the matrix.
      *
      * @param new_size1    New number of rows
      * @param new_size2    New number of columns
      * @param preserve     If true, the old values are preserved. At present, old values are always discarded.
      */
  void resize(arma_size_t new_size1, arma_size_t new_size2, bool preserve = true)
  {
    assert(new_size1 > 0 && new_size2 > 0 && bool("Cannot resize to zero size!"));

    if (new_size1 != rows_ || new_size2 != cols_)
    {
      if (!preserve)
      {
        cuarma::backend::typesafe_host_array<unsigned int> host_row_buffer(row_buffer_, new_size1 + 1);
        cuarma::backend::memory_create(row_buffer_, cuarma::backend::typesafe_host_array<unsigned int>().element_size() * (new_size1 + 1), cuarma::traits::context(row_buffer_), host_row_buffer.get());
        // faster version without initializing memory:
        //cuarma::backend::memory_create(row_buffer_, cuarma::backend::typesafe_host_array<unsigned int>().element_size() * (new_size1 + 1), cuarma::traits::context(row_buffer_));
        nonzeros_ = 0;
      }
      else
      {
        std::vector<std::map<unsigned int, NumericT> > stl_sparse_matrix;
        if (rows_ > 0)
        {
          stl_sparse_matrix.resize(rows_);
          cuarma::copy(*this, stl_sparse_matrix);
        } else {
          stl_sparse_matrix.resize(new_size1);
          stl_sparse_matrix[0][0] = 0;      //enforces nonzero array sizes if matrix was initially empty
        }

        stl_sparse_matrix.resize(new_size1);

        //discard entries with column index larger than new_size2
        if (new_size2 < cols_ && rows_ > 0)
        {
          for (arma_size_t i=0; i<stl_sparse_matrix.size(); ++i)
          {
            std::list<unsigned int> to_delete;
            for (typename std::map<unsigned int, NumericT>::iterator it = stl_sparse_matrix[i].begin();
                 it != stl_sparse_matrix[i].end();
                 ++it)
            {
              if (it->first >= new_size2)
                to_delete.push_back(it->first);
            }

            for (std::list<unsigned int>::iterator it = to_delete.begin(); it != to_delete.end(); ++it)
              stl_sparse_matrix[i].erase(*it);
          }
        }

        cuarma::tools::sparse_matrix_adapter<NumericT> adapted_matrix(stl_sparse_matrix, new_size1, new_size2);
        rows_ = new_size1;
        cols_ = new_size2;
        cuarma::copy(adapted_matrix, *this);
      }

      rows_ = new_size1;
      cols_ = new_size2;
    }
  }

  /** @brief Resets all entries in the matrix back to zero without changing the matrix size. Resets the sparsity pattern. */
  void clear()
  {
    cuarma::backend::typesafe_host_array<unsigned int> host_row_buffer(row_buffer_, rows_ + 1);
    cuarma::backend::typesafe_host_array<unsigned int> host_col_buffer(col_buffer_, 1);
    std::vector<NumericT> host_elements(1);

    cuarma::backend::memory_create(row_buffer_, host_row_buffer.element_size() * (rows_ + 1), cuarma::traits::context(row_buffer_), host_row_buffer.get());
    cuarma::backend::memory_create(col_buffer_, host_col_buffer.element_size() * 1,           cuarma::traits::context(col_buffer_), host_col_buffer.get());
    cuarma::backend::memory_create(elements_,   sizeof(NumericT) * 1,                         cuarma::traits::context(elements_), &(host_elements[0]));

    nonzeros_ = 0;
  }

  /** @brief Returns a reference to the (i,j)-th entry of the sparse matrix. If (i,j) does not exist (zero), it is inserted (slow!) */
  entry_proxy<NumericT> operator()(arma_size_t i, arma_size_t j)
  {
    assert( (i < rows_) && (j < cols_) && bool("compressed_matrix access out of bounds!"));

    arma_size_t index = element_index(i, j);

    // check for element in sparsity pattern
    if (index < nonzeros_)
      return entry_proxy<NumericT>(index, elements_);

    // Element not found. Copying required. Very slow, but direct entry manipulation is painful anyway...
    std::vector< std::map<unsigned int, NumericT> > cpu_backup(rows_);
    tools::sparse_matrix_adapter<NumericT> adapted_cpu_backup(cpu_backup, rows_, cols_);
    cuarma::copy(*this, adapted_cpu_backup);
    cpu_backup[i][static_cast<unsigned int>(j)] = 0.0;
    cuarma::copy(adapted_cpu_backup, *this);

    index = element_index(i, j);

    assert(index < nonzeros_);

    return entry_proxy<NumericT>(index, elements_);
  }

  /** @brief  Returns the number of rows */
  const arma_size_t & size1() const { return rows_; }
  /** @brief  Returns the number of columns */
  const arma_size_t & size2() const { return cols_; }
  /** @brief  Returns the number of nonzero entries */
  const arma_size_t & nnz() const { return nonzeros_; }
  /** @brief  Returns the internal number of row blocks for an adaptive SpMV */
  const arma_size_t & blocks1() const { return row_block_num_; }

  /** @brief  Returns the  handle to the row index array */
  const handle_type & handle1() const { return row_buffer_; }
  /** @brief  Returns the  handle to the column index array */
  const handle_type & handle2() const { return col_buffer_; }
  /** @brief  Returns the  handle to the row block array */
  const handle_type & handle3() const { return row_blocks_; }
  /** @brief  Returns the  handle to the matrix entry array */
  const handle_type & handle() const { return elements_; }

  /** @brief  Returns the  handle to the row index array */
  handle_type & handle1() { return row_buffer_; }
  /** @brief  Returns the  handle to the column index array */
  handle_type & handle2() { return col_buffer_; }
  /** @brief  Returns the  handle to the row block array */
  handle_type & handle3() { return row_blocks_; }
  /** @brief  Returns the  handle to the matrix entry array */
  handle_type & handle() { return elements_; }

  /** @brief Switches the memory context of the matrix.
    *
    * Allows for e.g. an migration of the full matrix from  memory to host memory for e.g. computing a preconditioner.
    */
  void switch_memory_context(cuarma::context new_ctx)
  {
    cuarma::backend::switch_memory_context<unsigned int>(row_buffer_, new_ctx);
    cuarma::backend::switch_memory_context<unsigned int>(col_buffer_, new_ctx);
    cuarma::backend::switch_memory_context<unsigned int>(row_blocks_, new_ctx);
    cuarma::backend::switch_memory_context<NumericT>(elements_, new_ctx);
  }

  /** @brief Returns the current memory context to determine whether the matrix is set up for OpenMP, , or CUDA. */
  cuarma::memory_types memory_context() const
  {
    return row_buffer_.get_active_handle_id();
  }

private:

  /** @brief Helper function for accessing the element (i,j) of the matrix. */
  arma_size_t element_index(arma_size_t i, arma_size_t j)
  {
    //read row indices
    cuarma::backend::typesafe_host_array<unsigned int> row_indices(row_buffer_, 2);
    cuarma::backend::memory_read(row_buffer_, row_indices.element_size()*i, row_indices.element_size()*2, row_indices.get());

    //get column indices for row i:
    cuarma::backend::typesafe_host_array<unsigned int> col_indices(col_buffer_, row_indices[1] - row_indices[0]);
    cuarma::backend::memory_read(col_buffer_, col_indices.element_size()*row_indices[0], row_indices.element_size()*col_indices.size(), col_indices.get());

    for (arma_size_t k=0; k<col_indices.size(); ++k)
    {
      if (col_indices[k] == j)
        return row_indices[0] + k;
    }

    // if not found, return index past the end of the matrix (cf. matrix.end() in the spirit of the STL)
    return nonzeros_;
  }

public:
  /** @brief Builds the row block information needed for fast sparse matrix-vector multiplications.
   *
   *  Required when manually populating the memory buffers with values. Not necessary when using cuarma::copy() or .set()
   */
  void generate_row_block_information()
  {
    cuarma::backend::typesafe_host_array<unsigned int> row_buffer(row_buffer_, rows_ + 1);
    cuarma::backend::memory_read(row_buffer_, 0, row_buffer.raw_size(), row_buffer.get());

    cuarma::backend::typesafe_host_array<unsigned int> row_blocks(row_buffer_, rows_ + 1);

    arma_size_t num_entries_in_current_batch = 0;

    const arma_size_t shared_mem_size = 1024; // number of column indices loaded to shared memory, number of floating point values loaded to shared memory

    row_block_num_ = 0;
    row_blocks.set(0, 0);
    for (arma_size_t i=0; i<rows_; ++i)
    {
      arma_size_t entries_in_row = arma_size_t(row_buffer[i+1]) - arma_size_t(row_buffer[i]);
      num_entries_in_current_batch += entries_in_row;

      if (num_entries_in_current_batch > shared_mem_size)
      {
        arma_size_t rows_in_batch = i - row_blocks[row_block_num_];
        if (rows_in_batch > 0) // at least one full row is in the batch. Use current row in next batch.
          row_blocks.set(++row_block_num_, i--);
        else // row is larger than buffer in shared memory
          row_blocks.set(++row_block_num_, i+1);
        num_entries_in_current_batch = 0;
      }
    }
    if (num_entries_in_current_batch > 0)
      row_blocks.set(++row_block_num_, rows_);

    if (row_block_num_ > 0) //matrix might be empty...
      cuarma::backend::memory_create(row_blocks_,
                                       row_blocks.element_size() * (row_block_num_ + 1),
                                       cuarma::traits::context(row_buffer_), row_blocks.get());

  }

private:
  // /** @brief Copy constructor is by now not available. */
  //compressed_matrix(compressed_matrix const &);

private:

  arma_size_t rows_;
  arma_size_t cols_;
  arma_size_t nonzeros_;
  arma_size_t row_block_num_;
  handle_type row_buffer_;
  handle_type row_blocks_;
  handle_type col_buffer_;
  handle_type elements_;
};

/** @brief Output stream support for compressed_matrix. Output format is same as MATLAB, Octave, or SciPy
  *
  * @param os   STL output stream
  * @param A    The compressed matrix to be printed.
*/
template<typename NumericT, unsigned int AlignmentV>
std::ostream & operator<<(std::ostream & os, compressed_matrix<NumericT, AlignmentV> const & A)
{
  std::vector<std::map<unsigned int, NumericT> > tmp(A.size1());
  cuarma::copy(A, tmp);
  os << "compressed_matrix of size (" << A.size1() << ", " << A.size2() << ") with " << A.nnz() << " nonzeros:" << std::endl;

  for (arma_size_t i=0; i<A.size1(); ++i)
  {
    for (typename std::map<unsigned int, NumericT>::const_iterator it = tmp[i].begin(); it != tmp[i].end(); ++it)
      os << "  (" << i << ", " << it->first << ")\t" << it->second << std::endl;
  }
  return os;
}

//
// Specify available operations:
//

/** \cond */

namespace blas
{
namespace detail
{
  // x = A * y
  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const compressed_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      // check for the special case x = A * x
      if (cuarma::traits::handle(lhs) == cuarma::traits::handle(rhs.rhs()))
      {
        cuarma::vector<T> temp(lhs);
        cuarma::blas::prod_impl(rhs.lhs(), rhs.rhs(), T(1), temp, T(0));
        lhs = temp;
      }
      else
        cuarma::blas::prod_impl(rhs.lhs(), rhs.rhs(), T(1), lhs, T(0));
    }
  };

  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const compressed_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      // check for the special case x += A * x
      if (cuarma::traits::handle(lhs) == cuarma::traits::handle(rhs.rhs()))
      {
        cuarma::vector<T> temp(lhs);
        cuarma::blas::prod_impl(rhs.lhs(), rhs.rhs(), T(1), temp, T(0));
        lhs += temp;
      }
      else
        cuarma::blas::prod_impl(rhs.lhs(), rhs.rhs(), T(1), lhs, T(1));
    }
  };

  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const compressed_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      // check for the special case x -= A * x
      if (cuarma::traits::handle(lhs) == cuarma::traits::handle(rhs.rhs()))
      {
        cuarma::vector<T> temp(lhs);
        cuarma::blas::prod_impl(rhs.lhs(), rhs.rhs(), T(1), temp, T(0));
        lhs -= temp;
      }
      else
        cuarma::blas::prod_impl(rhs.lhs(), rhs.rhs(), T(-1), lhs, T(1));
    }
  };


  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const compressed_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      cuarma::vector<T> temp(rhs.rhs(), cuarma::traits::context(rhs));
      cuarma::blas::prod_impl(rhs.lhs(), temp, lhs);
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const compressed_matrix<T, A>, vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_matrix<T, A>, vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      cuarma::vector<T> temp(rhs.rhs(), cuarma::traits::context(rhs));
      cuarma::vector<T> temp_result(lhs);
      cuarma::blas::prod_impl(rhs.lhs(), temp, temp_result);
      lhs += temp_result;
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const compressed_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      cuarma::vector<T> temp(rhs.rhs(), cuarma::traits::context(rhs));
      cuarma::vector<T> temp_result(lhs);
      cuarma::blas::prod_impl(rhs.lhs(), temp, temp_result);
      lhs -= temp_result;
    }
  };

} // namespace detail
} // namespace blas
  /** \endcond */
}
