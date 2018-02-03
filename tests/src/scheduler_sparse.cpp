/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   scheduler_sparse.cu
 *   @coding UTF-8
 *   @brief  Tests the scheduler for sparse matrix operations.
 *   @brief  ²âÊÔ£ºÏ¡Êè¾ØÕó²Ù×÷µ÷¶ÈÆ÷
 */

#include <iostream>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/coordinate_matrix.hpp"
#include "cuarma/ell_matrix.hpp"
#include "cuarma/hyb_matrix.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/vector_proxy.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/ilu.hpp"
#include "cuarma/blas/detail/ilu/common.hpp"
#include "cuarma/io/matrix_market.hpp"
#include "cuarma/tools/random.hpp"

#include "cuarma/scheduler/execute.hpp"
#include "cuarma/scheduler/io.hpp"


//
// -------------------------------------------------------------
//
template<typename ScalarType>
ScalarType diff(ScalarType & s1, cuarma::scalar<ScalarType> & s2)
{
   if (s1 != s2)
      return (s1 - s2) / std::max(fabs(s1), std::fabs(s2));
   return 0;
}

template<typename ScalarType>
ScalarType diff(std::vector<ScalarType> & v1, cuarma::vector<ScalarType> & v2)
{
   std::vector<ScalarType> v2_cpu(v2.size());
   cuarma::backend::finish();
   cuarma::copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
      {
        //if (std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) < 1e-10 )  //absolute tolerance (avoid round-off issues)
        //  v2_cpu[i] = 0;
        //else
          v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      }
      else
         v2_cpu[i] = 0.0;

      if (v2_cpu[i] > 0.0001)
      {
        //std::cout << "Neighbor: "      << i-1 << ": " << v1[i-1] << " vs. " << v2_cpu[i-1] << std::endl;
        std::cout << "Error at entry " << i   << ": Should: " << v1[i]   << " vs. Is: " << v2[i]   << std::endl;
        //std::cout << "Neighbor: "      << i+1 << ": " << v1[i+1] << " vs. " << v2_cpu[i+1] << std::endl;
        exit(EXIT_FAILURE);
      }
   }

   ScalarType norm_inf = 0;
   for (std::size_t i=0; i<v2_cpu.size(); ++i)
     norm_inf = std::max<ScalarType>(norm_inf, std::fabs(v2_cpu[i]));

   return norm_inf;
}


template<typename IndexT, typename NumericT, typename SparseMatrixT>
NumericT diff(std::vector<std::map<IndexT, NumericT> > & cpu_A, SparseMatrixT & arma_A)
{
  typedef typename std::map<IndexT, NumericT>::const_iterator  RowIterator;

  std::vector<std::map<IndexT, NumericT> > from_gpu(arma_A.size1());

  cuarma::backend::finish();
  cuarma::copy(arma_A, from_gpu);

  NumericT error = 0;

  //step 1: compare all entries from cpu_A with arma_A:
  for (std::size_t i=0; i<cpu_A.size(); ++i)
  {
    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (RowIterator it = cpu_A[i].begin(); it != cpu_A[i].end(); ++it)
    {
      //std::cout << "(" << col_it.index2() << ", " << *col_it << std::endl;
      NumericT current_error = 0;
      NumericT val_cpu_A = it->second;
      NumericT val_gpu_A = from_gpu[i][it->first];

      NumericT max_val = std::max(std::fabs(val_cpu_A), std::fabs(val_gpu_A));
      if (max_val > 0)
        current_error = std::fabs(val_cpu_A - val_gpu_A) / max_val;
      if (current_error > error)
        error = current_error;
    }
  }

  //step 2: compare all entries from gpu_matrix with cpu_matrix (sparsity pattern might differ):
  //std::cout << "cuarma matrix: " << std::endl;
  for (std::size_t i=0; i<from_gpu.size(); ++i)
  {
    //std::cout << "Row " << row_it.index1() << ": " << std::endl;
    for (RowIterator it = from_gpu[i].begin(); it != from_gpu[i].end(); ++it)
    {
      //std::cout << "(" << col_it.index2() << ", " << *col_it << std::endl;
      NumericT current_error = 0;
      NumericT val_gpu_A = it->second;
      NumericT val_cpu_A = cpu_A[i][it->first];

      NumericT max_val = std::max(std::fabs(val_cpu_A), std::fabs(val_gpu_A));
      if (max_val > 0)
        current_error = std::fabs(val_cpu_A - val_gpu_A) / max_val;
      if (current_error > error)
        error = current_error;
    }
  }

  return error;
}

// computes x = alpha * A * y + beta * x
template<typename IndexT, typename NumericT, typename VectorT>
void sparse_prod(std::vector<std::map<IndexT, NumericT> > const & A, VectorT const & y, VectorT & x, NumericT alpha, NumericT beta)
{
  typedef typename std::map<IndexT, NumericT>::const_iterator     RowIterator;

  for (std::size_t i=0; i<A.size(); ++i)
  {
    NumericT val = 0;
    for (RowIterator it = A[i].begin(); it != A[i].end(); ++it)
      val += it->second * y[it->first];
    x[i] = alpha * val + beta * x[i];
  }
}

//
// -------------------------------------------------------------
//
template< typename NumericT, typename Epsilon >
int test(Epsilon const& epsilon)
{
  int retval = EXIT_SUCCESS;

  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  // --------------------------------------------------------------------------
  NumericT alpha = static_cast<NumericT>(2.786);
  NumericT beta = static_cast<NumericT>(1.432);

  std::vector<NumericT> rhs;
  std::vector<NumericT> result;
  std::vector<std::map<unsigned int, NumericT> > std_matrix;

  if (cuarma::io::read_matrix_market_file(std_matrix, "../examples/testdata/mat65k.mtx") == EXIT_FAILURE)
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }
  //unsigned int cg_mat_size = cg_mat.size();
  std::cout << "done reading matrix" << std::endl;


  rhs.resize(std_matrix.size());
  for (std::size_t i=0; i<rhs.size(); ++i)
  {
    std_matrix[i][static_cast<unsigned int>(i)] = NumericT(0.5);   // Get rid of round-off errors by making row-sums unequal to zero:
    rhs[i] = NumericT(1) + randomNumber();
  }

  result = rhs;


  cuarma::vector<NumericT> arma_rhs(rhs.size());
  cuarma::vector<NumericT> arma_result(result.size());
  cuarma::vector<NumericT> arma_result2(result.size());
  cuarma::compressed_matrix<NumericT> arma_compressed_matrix(rhs.size(), rhs.size());
  cuarma::coordinate_matrix<NumericT> arma_coordinate_matrix(rhs.size(), rhs.size());
  cuarma::ell_matrix<NumericT> arma_ell_matrix;
  cuarma::hyb_matrix<NumericT> arma_hyb_matrix;

  cuarma::copy(rhs.begin(), rhs.end(), arma_rhs.begin());
  cuarma::copy(std_matrix, arma_compressed_matrix);
  cuarma::copy(std_matrix, arma_coordinate_matrix);

  // --------------------------------------------------------------------------
  std::cout << "Testing products: compressed_matrix" << std::endl;
  sparse_prod(std_matrix, rhs, result, NumericT(1), NumericT(0));
  {
  cuarma::scheduler::statement my_statement(arma_result, cuarma::op_assign(), cuarma::blas::prod(arma_compressed_matrix, arma_rhs));
  cuarma::scheduler::execute(my_statement);
  }
  arma_result = cuarma::blas::prod(arma_compressed_matrix, arma_rhs);

  if ( std::fabs(diff(result, arma_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with compressed_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, arma_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing products: coordinate_matrix" << std::endl;
  for (std::size_t i=0; i<rhs.size(); ++i)
    rhs[i] *= NumericT(1.1);
  arma_rhs *= NumericT(1.1);
  sparse_prod(std_matrix, rhs, result, NumericT(1), NumericT(0));
  {
  cuarma::scheduler::statement my_statement(arma_result, cuarma::op_assign(), cuarma::blas::prod(arma_coordinate_matrix, arma_rhs));
  cuarma::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, arma_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with coordinate_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, arma_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  //std::cout << "Copying ell_matrix" << std::endl;
  cuarma::copy(std_matrix, arma_ell_matrix);
  std_matrix.clear();
  cuarma::copy(arma_ell_matrix, std_matrix);// just to check that it's works


  std::cout << "Testing products: ell_matrix" << std::endl;
  for (std::size_t i=0; i<rhs.size(); ++i)
    rhs[i] *= NumericT(1.1);
  arma_rhs *= NumericT(1.1);
  sparse_prod(std_matrix, rhs, result, NumericT(1), NumericT(0));
  {
  //cuarma::scheduler::statement my_statement(arma_result, cuarma::op_assign(), cuarma::blas::prod(arma_ell_matrix, arma_rhs));
  //cuarma::scheduler::execute(my_statement);
  }
  arma_result = cuarma::blas::prod(arma_ell_matrix, arma_rhs);

  if ( std::fabs(diff(result, arma_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with ell_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, arma_result)) << std::endl;
    retval = EXIT_FAILURE;
  }

  //std::cout << "Copying hyb_matrix" << std::endl;
  cuarma::copy(std_matrix, arma_hyb_matrix);
  std_matrix.clear();
  cuarma::copy(arma_hyb_matrix, std_matrix);// just to check that it's works
  cuarma::copy(std_matrix, arma_hyb_matrix);

  std::cout << "Testing products: hyb_matrix" << std::endl;
  for (std::size_t i=0; i<rhs.size(); ++i)
    rhs[i] *= NumericT(1.1);
  arma_rhs *= NumericT(1.1);
  sparse_prod(std_matrix, rhs, result, NumericT(1), NumericT(0));
  {
  cuarma::scheduler::statement my_statement(arma_result, cuarma::op_assign(), cuarma::blas::prod(arma_hyb_matrix, arma_rhs));
  cuarma::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, arma_result)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product with hyb_matrix" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, arma_result)) << std::endl;
    retval = EXIT_FAILURE;
  }


  // --------------------------------------------------------------------------
  // --------------------------------------------------------------------------
  copy(rhs.begin(), rhs.end(), arma_rhs.begin());
  copy(result.begin(), result.end(), arma_result.begin());
  copy(result.begin(), result.end(), arma_result2.begin());
  copy(std_matrix, arma_compressed_matrix);
  copy(std_matrix, arma_coordinate_matrix);
  copy(std_matrix, arma_ell_matrix);
  copy(std_matrix, arma_hyb_matrix);

  std::cout << "Testing scaled additions of products and vectors: compressed_matrix" << std::endl;
  for (std::size_t i=0; i<rhs.size(); ++i)
    rhs[i] *= NumericT(1.1);
  arma_rhs *= NumericT(1.1);
  sparse_prod(std_matrix, rhs, result, NumericT(alpha), beta);
  {
  cuarma::scheduler::statement my_statement(arma_result2, cuarma::op_assign(), alpha * cuarma::blas::prod(arma_compressed_matrix, arma_rhs) + beta * arma_result);
  cuarma::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, arma_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (compressed_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, arma_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }


  std::cout << "Testing scaled additions of products and vectors: coordinate_matrix" << std::endl;
  copy(result.begin(), result.end(), arma_result.begin());
  for (std::size_t i=0; i<rhs.size(); ++i)
    rhs[i] *= NumericT(1.1);
  arma_rhs *= NumericT(1.1);
  sparse_prod(std_matrix, rhs, result, NumericT(alpha), beta);
  {
  cuarma::scheduler::statement my_statement(arma_result2, cuarma::op_assign(), alpha * cuarma::blas::prod(arma_coordinate_matrix, arma_rhs) + beta * arma_result);
  cuarma::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, arma_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (coordinate_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, arma_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing scaled additions of products and vectors: ell_matrix" << std::endl;
  copy(result.begin(), result.end(), arma_result.begin());
  for (std::size_t i=0; i<rhs.size(); ++i)
    rhs[i] *= NumericT(1.1);
  arma_rhs *= NumericT(1.1);
  sparse_prod(std_matrix, rhs, result, NumericT(alpha), beta);
  {
  cuarma::scheduler::statement my_statement(arma_result2, cuarma::op_assign(), alpha * cuarma::blas::prod(arma_ell_matrix, arma_rhs) + beta * arma_result);
  cuarma::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, arma_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (ell_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, arma_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }

  std::cout << "Testing scaled additions of products and vectors: hyb_matrix" << std::endl;
  copy(result.begin(), result.end(), arma_result.begin());
  for (std::size_t i=0; i<rhs.size(); ++i)
    rhs[i] *= NumericT(1.1);
  arma_rhs *= NumericT(1.1);
  sparse_prod(std_matrix, rhs, result, NumericT(alpha), beta);
  {
  cuarma::scheduler::statement my_statement(arma_result2, cuarma::op_assign(), alpha * cuarma::blas::prod(arma_hyb_matrix, arma_rhs) + beta * arma_result);
  cuarma::scheduler::execute(my_statement);
  }

  if ( std::fabs(diff(result, arma_result2)) > epsilon )
  {
    std::cout << "# Error at operation: matrix-vector product (hyb_matrix) with scaled additions" << std::endl;
    std::cout << "  diff: " << std::fabs(diff(result, arma_result2)) << std::endl;
    retval = EXIT_FAILURE;
  }
  // --------------------------------------------------------------------------
  return retval;
}
//
// -------------------------------------------------------------
//
int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Test :: Sparse Matrices" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  int retval = EXIT_SUCCESS;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  {
    typedef float NumericT;
    NumericT epsilon = static_cast<NumericT>(1E-4);
    std::cout << "# Testing setup:" << std::endl;
    std::cout << "  eps:     " << epsilon << std::endl;
    std::cout << "  numeric: float" << std::endl;
    retval = test<NumericT>(epsilon);
    if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
    else
        return retval;
  }
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;

  {
    {
      typedef double NumericT;
      NumericT epsilon = 1.0E-13;
      std::cout << "# Testing setup:" << std::endl;
      std::cout << "  eps:     " << epsilon << std::endl;
      std::cout << "  numeric: double" << std::endl;
      retval = test<NumericT>(epsilon);
      if ( retval == EXIT_SUCCESS )
        std::cout << "# Test passed" << std::endl;
      else
        return retval;
    }
    std::cout << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << std::endl;
  }

  std::cout << std::endl;
  std::cout << "------- Test completed --------" << std::endl;
  std::cout << std::endl;

  return retval;
}
