/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/* Sparse matrix operations, i.e. matrix-vector products. */

//#define VIENNACL_BUILD_INFO
#ifndef NDEBUG
 #define NDEBUG
#endif

#define VIENNACL_WITH_UBLAS 1

#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/lu.hpp>


#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/coordinate_matrix.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/ell_matrix.hpp"
#include "cuarma/hyb_matrix.hpp"
#include "cuarma/sliced_ell_matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/io/matrix_market.hpp"
#include "cuarma/blas/ilu.hpp"
#include "cuarma/tools/timer.hpp"


#include <iostream>
#include <vector>


#define BENCHMARK_RUNS          10


inline void printOps(double num_ops, double exec_time)
{
  std::cout << "GFLOPs: " << num_ops / (1000000 * exec_time * 1000) << std::endl;
}


template<typename ScalarType>
int run_benchmark()
{
  cuarma::tools::timer timer;
  double exec_time;

  ScalarType std_factor1 = ScalarType(3.1415);
  ScalarType std_factor2 = ScalarType(42.0);
  cuarma::scalar<ScalarType> cuarma_factor1(std_factor1);
  cuarma::scalar<ScalarType> cuarma_factor2(std_factor2);

  boost::numeric::ublas::vector<ScalarType> ublas_vec1;
  boost::numeric::ublas::vector<ScalarType> ublas_vec2;

  boost::numeric::ublas::compressed_matrix<ScalarType> ublas_matrix;
  if (!cuarma::io::read_matrix_market_file(ublas_matrix, "../examples/testdata/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return 0;
  }
  //unsigned int cg_mat_size = cg_mat.size();
  std::cout << "done reading matrix" << std::endl;

  ublas_vec1 = boost::numeric::ublas::scalar_vector<ScalarType>(ublas_matrix.size1(), ScalarType(1.0));
  ublas_vec2 = ublas_vec1;

  cuarma::compressed_matrix<ScalarType, 1> cuarma_compressed_matrix_1;
  cuarma::compressed_matrix<ScalarType, 4> cuarma_compressed_matrix_4;
  cuarma::compressed_matrix<ScalarType, 8> cuarma_compressed_matrix_8;

  cuarma::coordinate_matrix<ScalarType> cuarma_coordinate_matrix_128;

  cuarma::ell_matrix<ScalarType, 1> cuarma_ell_matrix_1;
  cuarma::hyb_matrix<ScalarType, 1> cuarma_hyb_matrix_1;
  cuarma::sliced_ell_matrix<ScalarType> cuarma_sliced_ell_matrix_1;

  cuarma::vector<ScalarType> cuarma_vec1(ublas_vec1.size());
  cuarma::vector<ScalarType> cuarma_vec2(ublas_vec1.size());

  //cpu to gpu:
  cuarma::copy(ublas_matrix, cuarma_compressed_matrix_1);
  #ifndef VIENNACL_EXPERIMENTAL_DOUBLE_PRECISION_WITH_STREAM_SDK_ON_GPU
  cuarma::copy(ublas_matrix, cuarma_compressed_matrix_4);
  cuarma::copy(ublas_matrix, cuarma_compressed_matrix_8);
  #endif
  cuarma::copy(ublas_matrix, cuarma_coordinate_matrix_128);
  cuarma::copy(ublas_matrix, cuarma_ell_matrix_1);
  cuarma::copy(ublas_matrix, cuarma_hyb_matrix_1);
  cuarma::copy(ublas_matrix, cuarma_sliced_ell_matrix_1);
  cuarma::copy(ublas_vec1, cuarma_vec1);
  cuarma::copy(ublas_vec2, cuarma_vec2);


  ///////////// Matrix operations /////////////////

  std::cout << "------- Matrix-Vector product on CPU ----------" << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    //ublas_vec1 = boost::numeric::ublas::prod(ublas_matrix, ublas_vec2);
    boost::numeric::ublas::axpy_prod(ublas_matrix, ublas_vec2, ublas_vec1, true);
  }
  exec_time = timer.get();
  std::cout << "CPU time: " << exec_time << std::endl;
  std::cout << "CPU "; printOps(2.0 * static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << ublas_vec1[0] << std::endl;


  std::cout << "------- Matrix-Vector product with compressed_matrix ----------" << std::endl;


  cuarma_vec1 = cuarma::blas::prod(cuarma_compressed_matrix_1, cuarma_vec2); //startup calculation
  cuarma_vec1 = cuarma::blas::prod(cuarma_compressed_matrix_4, cuarma_vec2); //startup calculation
  cuarma_vec1 = cuarma::blas::prod(cuarma_compressed_matrix_8, cuarma_vec2); //startup calculation
  //std_result = 0.0;

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    cuarma_vec1 = cuarma::blas::prod(cuarma_compressed_matrix_1, cuarma_vec2);
  }
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time align1: " << exec_time << std::endl;
  std::cout << "GPU align1 "; printOps(2.0 * static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << cuarma_vec1[0] << std::endl;

  std::cout << "Testing triangular solves: compressed_matrix" << std::endl;

  cuarma::copy(ublas_vec1, cuarma_vec1);
  cuarma::blas::inplace_solve(trans(cuarma_compressed_matrix_1), cuarma_vec1, cuarma::blas::unit_lower_tag());
  cuarma::copy(ublas_vec1, cuarma_vec1);
  std::cout << "ublas..." << std::endl;
  timer.start();
  boost::numeric::ublas::inplace_solve(trans(ublas_matrix), ublas_vec1, boost::numeric::ublas::unit_lower_tag());
  std::cout << "Time elapsed: " << timer.get() << std::endl;
  std::cout << "ViennaCL..." << std::endl;
  cuarma::backend::finish();
  timer.start();
  cuarma::blas::inplace_solve(trans(cuarma_compressed_matrix_1), cuarma_vec1, cuarma::blas::unit_lower_tag());
  cuarma::backend::finish();
  std::cout << "Time elapsed: " << timer.get() << std::endl;

  ublas_vec1 = boost::numeric::ublas::prod(ublas_matrix, ublas_vec2);

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    cuarma_vec1 = cuarma::blas::prod(cuarma_compressed_matrix_4, cuarma_vec2);
  }
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time align4: " << exec_time << std::endl;
  std::cout << "GPU align4 "; printOps(2.0 * static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << cuarma_vec1[0] << std::endl;

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    cuarma_vec1 = cuarma::blas::prod(cuarma_compressed_matrix_8, cuarma_vec2);
  }
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time align8: " << exec_time << std::endl;
  std::cout << "GPU align8 "; printOps(2.0 * static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << cuarma_vec1[0] << std::endl;


  std::cout << "------- Matrix-Vector product with coordinate_matrix ----------" << std::endl;
  cuarma_vec1 = cuarma::blas::prod(cuarma_coordinate_matrix_128, cuarma_vec2); //startup calculation
  cuarma::backend::finish();

  cuarma::copy(cuarma_vec1, ublas_vec2);
  long err_cnt = 0;
  for (std::size_t i=0; i<ublas_vec1.size(); ++i)
  {
    if ( fabs(ublas_vec1[i] - ublas_vec2[i]) / std::max(fabs(ublas_vec1[i]), fabs(ublas_vec2[i])) > 1e-2)
    {
      std::cout << "Error at index " << i << ": Should: " << ublas_vec1[i] << ", Is: " << ublas_vec2[i] << std::endl;
      ++err_cnt;
      if (err_cnt > 5)
        break;
    }
  }

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    cuarma_vec1 = cuarma::blas::prod(cuarma_coordinate_matrix_128, cuarma_vec2);
  }
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(2.0 * static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << cuarma_vec1[0] << std::endl;


  std::cout << "------- Matrix-Vector product with ell_matrix ----------" << std::endl;
  cuarma_vec1 = cuarma::blas::prod(cuarma_ell_matrix_1, cuarma_vec2); //startup calculation
  cuarma::backend::finish();

  cuarma::copy(cuarma_vec1, ublas_vec2);
  err_cnt = 0;
  for (std::size_t i=0; i<ublas_vec1.size(); ++i)
  {
    if ( fabs(ublas_vec1[i] - ublas_vec2[i]) / std::max(fabs(ublas_vec1[i]), fabs(ublas_vec2[i])) > 1e-2)
    {
      std::cout << "Error at index " << i << ": Should: " << ublas_vec1[i] << ", Is: " << ublas_vec2[i] << std::endl;
      ++err_cnt;
      if (err_cnt > 5)
        break;
    }
  }

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    cuarma_vec1 = cuarma::blas::prod(cuarma_ell_matrix_1, cuarma_vec2);
  }
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(2.0 * static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << cuarma_vec1[0] << std::endl;


  std::cout << "------- Matrix-Vector product with hyb_matrix ----------" << std::endl;
  cuarma_vec1 = cuarma::blas::prod(cuarma_hyb_matrix_1, cuarma_vec2); //startup calculation
  cuarma::backend::finish();

  cuarma::copy(cuarma_vec1, ublas_vec2);
  err_cnt = 0;
  for (std::size_t i=0; i<ublas_vec1.size(); ++i)
  {
    if ( fabs(ublas_vec1[i] - ublas_vec2[i]) / std::max(fabs(ublas_vec1[i]), fabs(ublas_vec2[i])) > 1e-2)
    {
      std::cout << "Error at index " << i << ": Should: " << ublas_vec1[i] << ", Is: " << ublas_vec2[i] << std::endl;
      ++err_cnt;
      if (err_cnt > 5)
        break;
    }
  }

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    cuarma_vec1 = cuarma::blas::prod(cuarma_hyb_matrix_1, cuarma_vec2);
  }
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(2.0 * static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << cuarma_vec1[0] << std::endl;


  std::cout << "------- Matrix-Vector product with sliced_ell_matrix ----------" << std::endl;
  cuarma_vec1 = cuarma::blas::prod(cuarma_sliced_ell_matrix_1, cuarma_vec2); //startup calculation
  cuarma::backend::finish();

  cuarma::copy(cuarma_vec1, ublas_vec2);
  err_cnt = 0;
  for (std::size_t i=0; i<ublas_vec1.size(); ++i)
  {
    if ( fabs(ublas_vec1[i] - ublas_vec2[i]) / std::max(fabs(ublas_vec1[i]), fabs(ublas_vec2[i])) > 1e-2)
    {
      std::cout << "Error at index " << i << ": Should: " << ublas_vec1[i] << ", Is: " << ublas_vec2[i] << std::endl;
      ++err_cnt;
      if (err_cnt > 5)
        break;
    }
  }

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    cuarma_vec1 = cuarma::blas::prod(cuarma_sliced_ell_matrix_1, cuarma_vec2);
  }
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "GPU time: " << exec_time << std::endl;
  std::cout << "GPU "; printOps(2.0 * static_cast<double>(ublas_matrix.nnz()), static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS));
  std::cout << cuarma_vec1[0] << std::endl;

  return EXIT_SUCCESS;
}


int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Benchmark :: Sparse" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  run_benchmark<float>();

  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    run_benchmark<double>();
  }
  return 0;
}

