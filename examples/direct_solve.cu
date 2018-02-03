/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/* Direct solve matrix-matrix and matrix-vecotor */

#include <iostream>

#include "cuarma/scalar.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/matrix_proxy.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/tools/random.hpp"
#include "cuarma/tools/timer.hpp"

#define BENCHMARK_RUNS 10


inline void printOps(double num_ops, double exec_time)
{
  std::cout << "GFLOPs: " << num_ops / (1000000 * exec_time * 1000) << std::endl;
}


template<typename NumericT>
void fill_matrix(cuarma::matrix<NumericT> & mat)
{
  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  for (std::size_t i = 0; i < mat.size1(); ++i)
  {
    for (std::size_t j = 0; j < mat.size2(); ++j)
      mat(i, j) = static_cast<NumericT>(-0.5) * randomNumber();
    mat(i, i) = NumericT(1.0) + NumericT(2.0) * randomNumber(); //some extra weight on diagonal for stability
  }
}

template<typename NumericT>
void fill_vector(cuarma::vector<NumericT> & vec)
{
  cuarma::tools::uniform_random_numbers<NumericT> randomNumber;

  for (std::size_t i = 0; i < vec.size(); ++i)
    vec(i) = NumericT(1.0) + NumericT(2.0) * randomNumber(); //some extra weight on diagonal for stability
}

template<typename NumericT,typename MatrixT1, typename MatrixT2,typename MatrixT3, typename SolverTag>
void run_solver_matrix(MatrixT1 const & matrix1, MatrixT2 const & matrix2,MatrixT3 & result, SolverTag)
{
  std::cout << "------- Solver tag: " <<SolverTag::name()<<" ----------" << std::endl;
  result = cuarma::blas::solve(matrix1, matrix2, SolverTag());

  cuarma::tools::timer timer;
  cuarma::backend::finish();

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    result = cuarma::blas::solve(matrix1, matrix2, SolverTag());

  double exec_time = timer.get();
  cuarma::backend::finish();
  std::cout << "GPU: ";printOps(double(matrix1.size1() * matrix1.size1() * matrix2.size2()),(static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS)));
  std::cout << "GPU: " << double(matrix1.size1() * matrix1.size1() * matrix2.size2() * sizeof(NumericT)) / (static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS)) / 1e9 << " GB/sec" << std::endl;
  std::cout << "Execution time: " << exec_time/BENCHMARK_RUNS << std::endl;
  std::cout << "------- Finnished: " << SolverTag::name() << " ----------" << std::endl;
}

template<typename NumericT,typename VectorT, typename VectorT2,typename MatrixT, typename SolverTag>
void run_solver_vector(MatrixT const & matrix, VectorT2 const & vector2,VectorT & result, SolverTag)
{
  std::cout << "------- Solver tag: " <<SolverTag::name()<<" ----------" << std::endl;
  result = cuarma::blas::solve(matrix, vector2, SolverTag());

  cuarma::tools::timer timer;
  cuarma::backend::finish();

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    result = cuarma::blas::solve(matrix, vector2, SolverTag());
  }
  double exec_time = timer.get();
  cuarma::backend::finish();
  std::cout << "GPU: ";printOps(double(matrix.size1() * matrix.size1()),(static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS)));
  std::cout << "GPU: "<< double(matrix.size1() * matrix.size1() * sizeof(NumericT)) / (static_cast<double>(exec_time) / static_cast<double>(BENCHMARK_RUNS)) / 1e9 << " GB/sec" << std::endl;
  std::cout << "Execution time: " << exec_time/BENCHMARK_RUNS << std::endl;
  std::cout << "------- Finished: " << SolverTag::name() << " ----------" << std::endl;
}

template<typename NumericT,typename F_A, typename F_B>
void run_benchmark()
{
  std::size_t matrix_size = 1500;  //some odd number, not too large
  std::size_t rhs_num = 153;

  cuarma::matrix<NumericT, F_A> cuarma_A(matrix_size, matrix_size);
  cuarma::matrix<NumericT, F_B> cuarma_B(matrix_size, rhs_num);
  cuarma::matrix<NumericT, F_B> result(matrix_size, rhs_num);

  cuarma::vector<NumericT> cuarma_vec_B(matrix_size);
  cuarma::vector<NumericT> cuarma_vec_result(matrix_size);

  fill_matrix(cuarma_A);
  fill_matrix(cuarma_B);

  fill_vector(cuarma_vec_B);
  std::cout << "------- Solve Matrix-Matrix: ----------\n" << std::endl;
  run_solver_matrix<NumericT>(cuarma_A,cuarma_B,result,cuarma::blas::lower_tag());
  run_solver_matrix<NumericT>(cuarma_A,cuarma_B,result,cuarma::blas::unit_lower_tag());
  run_solver_matrix<NumericT>(cuarma_A,cuarma_B,result,cuarma::blas::upper_tag());
  run_solver_matrix<NumericT>(cuarma_A,cuarma_B,result,cuarma::blas::unit_upper_tag());
  std::cout << "------- End Matrix-Matrix: ----------\n" << std::endl;

  std::cout << "------- Solve Matrix-Vector: ----------\n" << std::endl;
  run_solver_vector<NumericT>(cuarma_A,cuarma_vec_B,cuarma_vec_result,cuarma::blas::lower_tag());
  run_solver_vector<NumericT>(cuarma_A,cuarma_vec_B,cuarma_vec_result,cuarma::blas::unit_lower_tag());
  run_solver_vector<NumericT>(cuarma_A,cuarma_vec_B,cuarma_vec_result,cuarma::blas::upper_tag());
  run_solver_vector<NumericT>(cuarma_A,cuarma_vec_B,cuarma_vec_result,cuarma::blas::unit_upper_tag());
  std::cout << "------- End Matrix-Vector: ----------\n" << std::endl;
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
  std::cout << "## Benchmark :: Direct solve" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  run_benchmark<float,cuarma::row_major,cuarma::row_major>();

  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    run_benchmark<double,cuarma::row_major,cuarma::row_major>();
  }
  return 0;
}
