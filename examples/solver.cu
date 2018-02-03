/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/* Iterative solver tests. */

#ifndef BOOST_UBLAS_NDEBUG
 #define BOOST_UBLAS_NDEBUG
#endif

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>

#define VIENNACL_WITH_UBLAS 1

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/coordinate_matrix.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/ell_matrix.hpp"
#include "cuarma/sliced_ell_matrix.hpp"
#include "cuarma/hyb_matrix.hpp"
#include "cuarma/context.hpp"

#include "cuarma/blas/cg.hpp"
#include "cuarma/blas/bicgstab.hpp"
#include "cuarma/blas/gmres.hpp"
#include "cuarma/blas/mixed_precision_cg.hpp"

#include "cuarma/blas/ilu.hpp"
#include "cuarma/blas/ichol.hpp"
#include "cuarma/blas/jacobi_precond.hpp"
#include "cuarma/blas/row_scaling.hpp"

#include "cuarma/io/matrix_market.hpp"
#include "cuarma/tools/timer.hpp"


#include <iostream>
#include <vector>


using namespace boost::numeric;

#define BENCHMARK_RUNS          1


inline void printOps(double num_ops, double exec_time)
{
  std::cout << "GFLOPs: " << num_ops / (1000000 * exec_time * 1000) << std::endl;
}


template<typename ScalarType>
ScalarType diff_inf(ublas::vector<ScalarType> & v1, cuarma::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   cuarma::copy(v2.begin(), v2.end(), v2_cpu.begin());

   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( fabs(v2_cpu[i]), fabs(v1[i]) ) > 0 )
         v2_cpu[i] = fabs(v2_cpu[i] - v1[i]) / std::max( fabs(v2_cpu[i]), fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;
   }

   return norm_inf(v2_cpu);
}

template<typename ScalarType>
ScalarType diff_2(ublas::vector<ScalarType> & v1, cuarma::vector<ScalarType> & v2)
{
   ublas::vector<ScalarType> v2_cpu(v2.size());
   cuarma::copy(v2.begin(), v2.end(), v2_cpu.begin());

   return norm_2(v1 - v2_cpu) / norm_2(v1);
}


template<typename MatrixType, typename VectorType, typename SolverTag, typename PrecondTag>
void run_solver(MatrixType const & matrix, VectorType const & rhs, VectorType const & ref_result, SolverTag const & solver, PrecondTag const & precond, long ops)
{
  cuarma::tools::timer timer;
  VectorType result(rhs);
  VectorType residual(rhs);
  cuarma::backend::finish();

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
  {
    result = cuarma::blas::solve(matrix, rhs, solver, precond);
  }
  cuarma::backend::finish();
  double exec_time = timer.get();
  std::cout << "Exec. time: " << exec_time << std::endl;
  std::cout << "Est. "; printOps(static_cast<double>(ops), exec_time / BENCHMARK_RUNS);
  residual -= cuarma::blas::prod(matrix, result);
  std::cout << "Relative residual: " << cuarma::blas::norm_2(residual) / cuarma::blas::norm_2(rhs) << std::endl;
  std::cout << "Estimated rel. residual: " << solver.error() << std::endl;
  std::cout << "Iterations: " << solver.iters() << std::endl;
  result -= ref_result;
  std::cout << "Relative deviation from result: " << cuarma::blas::norm_2(result) / cuarma::blas::norm_2(ref_result) << std::endl;
}


template<typename ScalarType>
int run_benchmark(cuarma::context ctx)
{
  cuarma::tools::timer timer;
  double exec_time;

  ublas::vector<ScalarType> ublas_vec1;
  ublas::vector<ScalarType> ublas_vec2;
  ublas::vector<ScalarType> ublas_result;
  unsigned int solver_iters = 100;
  unsigned int solver_krylov_dim = 20;
  double solver_tolerance = 1e-6;

  ublas::compressed_matrix<ScalarType> ublas_matrix;
  if (!cuarma::io::read_matrix_market_file(ublas_matrix, "../examples/data/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "done reading matrix" << std::endl;

  ublas_result = ublas::scalar_vector<ScalarType>(ublas_matrix.size1(), ScalarType(1.0));
  ublas_vec1 = ublas::prod(ublas_matrix, ublas_result);
  ublas_vec2 = ublas_vec1;

  cuarma::compressed_matrix<ScalarType> cuarma_compressed_matrix(ublas_vec1.size(), ublas_vec1.size(), ctx);
  cuarma::coordinate_matrix<ScalarType> cuarma_coordinate_matrix(ublas_vec1.size(), ublas_vec1.size(), ctx);
  cuarma::ell_matrix<ScalarType> cuarma_ell_matrix(ctx);
  cuarma::sliced_ell_matrix<ScalarType> cuarma_sliced_ell_matrix(ctx);
  cuarma::hyb_matrix<ScalarType> cuarma_hyb_matrix(ctx);

  cuarma::vector<ScalarType> cuarma_vec1(ublas_vec1.size(), ctx);
  cuarma::vector<ScalarType> cuarma_vec2(ublas_vec1.size(), ctx);
  cuarma::vector<ScalarType> cuarma_result(ublas_vec1.size(), ctx);


  //cpu to gpu:
  cuarma::copy(ublas_matrix, cuarma_compressed_matrix);
  cuarma::copy(ublas_matrix, cuarma_coordinate_matrix);
  cuarma::copy(ublas_matrix, cuarma_ell_matrix);
  cuarma::copy(ublas_matrix, cuarma_sliced_ell_matrix);
  cuarma::copy(ublas_matrix, cuarma_hyb_matrix);
  cuarma::copy(ublas_vec1, cuarma_vec1);
  cuarma::copy(ublas_vec2, cuarma_vec2);
  cuarma::copy(ublas_result, cuarma_result);


  std::cout << "------- Jacobi preconditioner ----------" << std::endl;
  cuarma::blas::jacobi_precond< ublas::compressed_matrix<ScalarType> >    ublas_jacobi(ublas_matrix, cuarma::blas::jacobi_tag());
  cuarma::blas::jacobi_precond< cuarma::compressed_matrix<ScalarType> > cuarma_jacobi_csr(cuarma_compressed_matrix, cuarma::blas::jacobi_tag());
  cuarma::blas::jacobi_precond< cuarma::coordinate_matrix<ScalarType> > cuarma_jacobi_coo(cuarma_coordinate_matrix, cuarma::blas::jacobi_tag());

  std::cout << "------- Row-Scaling preconditioner ----------" << std::endl;
  cuarma::blas::row_scaling< ublas::compressed_matrix<ScalarType> >    ublas_row_scaling(ublas_matrix, cuarma::blas::row_scaling_tag(1));
  cuarma::blas::row_scaling< cuarma::compressed_matrix<ScalarType> > cuarma_row_scaling_csr(cuarma_compressed_matrix, cuarma::blas::row_scaling_tag(1));
  cuarma::blas::row_scaling< cuarma::coordinate_matrix<ScalarType> > cuarma_row_scaling_coo(cuarma_coordinate_matrix, cuarma::blas::row_scaling_tag(1));

  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////  Incomplete Cholesky preconditioner   //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  std::cout << "------- ICHOL0 on CPU (ublas) ----------" << std::endl;

  timer.start();
  cuarma::blas::ichol0_precond< ublas::compressed_matrix<ScalarType> >    ublas_ichol0(ublas_matrix, cuarma::blas::ichol0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_ichol0.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas time: " << exec_time << std::endl;

  std::cout << "------- ICHOL0 with ViennaCL ----------" << std::endl;

  timer.start();
  cuarma::blas::ichol0_precond< cuarma::compressed_matrix<ScalarType> > cuarma_ichol0(cuarma_compressed_matrix, cuarma::blas::ichol0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    cuarma_ichol0.apply(cuarma_vec1);
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;

  std::cout << "------- Chow-Patel parallel ICC with ViennaCL ----------" << std::endl;

  timer.start();
  cuarma::blas::chow_patel_icc_precond< cuarma::compressed_matrix<ScalarType> > cuarma_chow_patel_icc(cuarma_compressed_matrix, cuarma::blas::chow_patel_tag());
  cuarma::backend::finish();
  std::cout << "Setup time: " << timer.get() << std::endl;

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    cuarma_chow_patel_icc.apply(cuarma_vec1);
  cuarma::backend::finish();
  std::cout << "ViennaCL Chow-Patel-ICC substitution time: " << timer.get() << std::endl;


  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////           ILU preconditioner         //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  std::cout << "------- ILU0 on with ublas ----------" << std::endl;

  timer.start();
  cuarma::blas::ilu0_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilu0(ublas_matrix, cuarma::blas::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time (no level scheduling): " << exec_time << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_ilu0.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas ILU0 substitution time (no level scheduling): " << exec_time << std::endl;

  std::cout << "------- ILU0 with ViennaCL ----------" << std::endl;

  timer.start();
  cuarma::blas::ilu0_precond< cuarma::compressed_matrix<ScalarType> > cuarma_ilu0(cuarma_compressed_matrix, cuarma::blas::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time (no level scheduling): " << exec_time << std::endl;

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    cuarma_ilu0.apply(cuarma_vec1);
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL ILU0 substitution time (no level scheduling): " << exec_time << std::endl;

  timer.start();
  cuarma::blas::ilu0_tag ilu0_with_level_scheduling; ilu0_with_level_scheduling.use_level_scheduling(true);
  cuarma::blas::ilu0_precond< cuarma::compressed_matrix<ScalarType> > cuarma_ilu0_level_scheduling(cuarma_compressed_matrix, ilu0_with_level_scheduling);
  exec_time = timer.get();
  std::cout << "Setup time (with level scheduling): " << exec_time << std::endl;

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    cuarma_ilu0_level_scheduling.apply(cuarma_vec1);
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL ILU0 substitution time (with level scheduling): " << exec_time << std::endl;



  ////////////////////////////////////////////

  std::cout << "------- Block-ILU0 with ublas ----------" << std::endl;

  ublas_vec1 = ublas_vec2;
  cuarma::copy(ublas_vec1, cuarma_vec1);

  timer.start();
  cuarma::blas::block_ilu_precond< ublas::compressed_matrix<ScalarType>,
                                       cuarma::blas::ilu0_tag>          ublas_block_ilu0(ublas_matrix, cuarma::blas::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_block_ilu0.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas time: " << exec_time << std::endl;

  std::cout << "------- Block-ILU0 with ViennaCL ----------" << std::endl;

  timer.start();
  cuarma::blas::block_ilu_precond< cuarma::compressed_matrix<ScalarType>,
                                       cuarma::blas::ilu0_tag>          cuarma_block_ilu0(cuarma_compressed_matrix, cuarma::blas::ilu0_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  //cuarma_block_ilu0.apply(cuarma_vec1);  //warm-up
  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    cuarma_block_ilu0.apply(cuarma_vec1);
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;

  ////////////////////////////////////////////

  std::cout << "------- ILUT with ublas ----------" << std::endl;

  ublas_vec1 = ublas_vec2;
  cuarma::copy(ublas_vec1, cuarma_vec1);

  timer.start();
  cuarma::blas::ilut_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilut(ublas_matrix, cuarma::blas::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time (no level scheduling): " << exec_time << std::endl;
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_ilut.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas ILUT substitution time (no level scheduling): " << exec_time << std::endl;


  std::cout << "------- ILUT with ViennaCL ----------" << std::endl;

  timer.start();
  cuarma::blas::ilut_precond< cuarma::compressed_matrix<ScalarType> > cuarma_ilut(cuarma_compressed_matrix, cuarma::blas::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time (no level scheduling): " << exec_time << std::endl;

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    cuarma_ilut.apply(cuarma_vec1);
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL ILUT substitution time (no level scheduling): " << exec_time << std::endl;

  timer.start();
  cuarma::blas::ilut_tag ilut_with_level_scheduling; ilut_with_level_scheduling.use_level_scheduling(true);
  cuarma::blas::ilut_precond< cuarma::compressed_matrix<ScalarType> > cuarma_ilut_level_scheduling(cuarma_compressed_matrix, ilut_with_level_scheduling);
  exec_time = timer.get();
  std::cout << "Setup time (with level scheduling): " << exec_time << std::endl;

  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    cuarma_ilut_level_scheduling.apply(cuarma_vec1);
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL ILUT substitution time (with level scheduling): " << exec_time << std::endl;


  ////////////////////////////////////////////

  std::cout << "------- Block-ILUT with ublas ----------" << std::endl;

  ublas_vec1 = ublas_vec2;
  cuarma::copy(ublas_vec1, cuarma_vec1);

  timer.start();
  cuarma::blas::block_ilu_precond< ublas::compressed_matrix<ScalarType>,
                                       cuarma::blas::ilut_tag>          ublas_block_ilut(ublas_matrix, cuarma::blas::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  //ublas_block_ilut.apply(ublas_vec1);
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    ublas_block_ilut.apply(ublas_vec1);
  exec_time = timer.get();
  std::cout << "ublas time: " << exec_time << std::endl;

  std::cout << "------- Block-ILUT with ViennaCL ----------" << std::endl;

  timer.start();
  cuarma::blas::block_ilu_precond< cuarma::compressed_matrix<ScalarType>,
                                       cuarma::blas::ilut_tag>          cuarma_block_ilut(cuarma_compressed_matrix, cuarma::blas::ilut_tag());
  exec_time = timer.get();
  std::cout << "Setup time: " << exec_time << std::endl;

  //cuarma_block_ilut.apply(cuarma_vec1);  //warm-up
  cuarma::backend::finish();
  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    cuarma_block_ilut.apply(cuarma_vec1);
  cuarma::backend::finish();
  exec_time = timer.get();
  std::cout << "ViennaCL time: " << exec_time << std::endl;

  std::cout << "------- Chow-Patel parallel ILU with ViennaCL ----------" << std::endl;

  timer.start();
  cuarma::blas::chow_patel_ilu_precond< cuarma::compressed_matrix<ScalarType> > cuarma_chow_patel_ilu(cuarma_compressed_matrix, cuarma::blas::chow_patel_tag());
  cuarma::backend::finish();
  std::cout << "Setup time: " << timer.get() << std::endl;

  timer.start();
  for (int runs=0; runs<BENCHMARK_RUNS; ++runs)
    cuarma_chow_patel_ilu.apply(cuarma_vec1);
  cuarma::backend::finish();
  std::cout << "ViennaCL Chow-Patel-ILU substitution time: " << timer.get() << std::endl;

  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////              CG solver                //////////////////
  ///////////////////////////////////////////////////////////////////////////////
  long cg_ops = static_cast<long>(solver_iters * (ublas_matrix.nnz() + 6 * ublas_vec2.size()));

  cuarma::blas::cg_tag cg_solver(solver_tolerance, solver_iters);

  std::cout << "------- CG solver (no preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, cuarma::blas::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma::blas::no_precond(), cg_ops);

  bool is_double = (sizeof(ScalarType) == sizeof(double));
  if (is_double)
  {
    std::cout << "------- CG solver, mixed precision (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
    cuarma::blas::mixed_precision_cg_tag mixed_precision_cg_solver(solver_tolerance, solver_iters);

    run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, mixed_precision_cg_solver, cuarma::blas::no_precond(), cg_ops);
  }

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma::blas::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, ell_matrix ----------" << std::endl;
  run_solver(cuarma_ell_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma::blas::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, sliced_ell_matrix ----------" << std::endl;
  run_solver(cuarma_sliced_ell_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma::blas::no_precond(), cg_ops);

  std::cout << "------- CG solver (no preconditioner) via ViennaCL, hyb_matrix ----------" << std::endl;
  run_solver(cuarma_hyb_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma::blas::no_precond(), cg_ops);

  std::cout << "------- CG solver (ICHOL0 preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_ichol0, cg_ops);

  std::cout << "------- CG solver (ICHOL0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_ichol0, cg_ops);

  std::cout << "------- CG solver (Chow-Patel ICHOL0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_chow_patel_icc, cg_ops);

  std::cout << "------- CG solver (ILU0 preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_ilu0, cg_ops);

  std::cout << "------- CG solver (ILU0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_ilu0, cg_ops);


  std::cout << "------- CG solver (Block-ILU0 preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_block_ilu0, cg_ops);

  std::cout << "------- CG solver (Block-ILU0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_block_ilu0, cg_ops);

  std::cout << "------- CG solver (ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_ilut, cg_ops);

  std::cout << "------- CG solver (ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_ilut, cg_ops);

  std::cout << "------- CG solver (ILUT preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_ilut, cg_ops);

  std::cout << "------- CG solver (Block-ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_block_ilut, cg_ops);

  std::cout << "------- CG solver (Block-ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_block_ilut, cg_ops);

  std::cout << "------- CG solver (Jacobi preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_jacobi, cg_ops);

  std::cout << "------- CG solver (Jacobi preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_jacobi_csr, cg_ops);

  std::cout << "------- CG solver (Jacobi preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_jacobi_coo, cg_ops);


  std::cout << "------- CG solver (row scaling preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, cg_solver, ublas_row_scaling, cg_ops);

  std::cout << "------- CG solver (row scaling preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_row_scaling_csr, cg_ops);

  std::cout << "------- CG solver (row scaling preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, cg_solver, cuarma_row_scaling_coo, cg_ops);

  ///////////////////////////////////////////////////////////////////////////////
  //////////////////////           BiCGStab solver             //////////////////
  ///////////////////////////////////////////////////////////////////////////////

  long bicgstab_ops = static_cast<long>(solver_iters * (2 * ublas_matrix.nnz() + 13 * ublas_vec2.size()));

  cuarma::blas::bicgstab_tag bicgstab_solver(solver_tolerance, solver_iters);

  std::cout << "------- BiCGStab solver (no preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, cuarma::blas::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma::blas::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (ILU0 preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma_ilu0, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Chow-Patel-ILU preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma_chow_patel_ilu, bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma::blas::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, ell_matrix ----------" << std::endl;
  run_solver(cuarma_ell_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma::blas::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, sliced_ell_matrix ----------" << std::endl;
  run_solver(cuarma_sliced_ell_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma::blas::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (no preconditioner) via ViennaCL, hyb_matrix ----------" << std::endl;
  run_solver(cuarma_hyb_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma::blas::no_precond(), bicgstab_ops);

  std::cout << "------- BiCGStab solver (ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_ilut, bicgstab_ops);

  std::cout << "------- BiCGStab solver (ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma_ilut, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Block-ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_block_ilut, bicgstab_ops);


//  std::cout << "------- BiCGStab solver (ILUT preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
//  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma_ilut, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Jacobi preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_jacobi, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Jacobi preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma_jacobi_csr, bicgstab_ops);

  std::cout << "------- BiCGStab solver (Jacobi preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma_jacobi_coo, bicgstab_ops);


  std::cout << "------- BiCGStab solver (row scaling preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, bicgstab_solver, ublas_row_scaling, bicgstab_ops);

  std::cout << "------- BiCGStab solver (row scaling preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma_row_scaling_csr, bicgstab_ops);

  std::cout << "------- BiCGStab solver (row scaling preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, bicgstab_solver, cuarma_row_scaling_coo, bicgstab_ops);


  ///////////////////////////////////////////////////////////////////////////////
  ///////////////////////            GMRES solver             ///////////////////
  ///////////////////////////////////////////////////////////////////////////////

  long gmres_ops = static_cast<long>(solver_iters * (ublas_matrix.nnz() + (solver_iters * 2 + 7) * ublas_vec2.size()));

  cuarma::blas::gmres_tag gmres_solver(solver_tolerance, solver_iters, solver_krylov_dim);

  std::cout << "------- GMRES solver (no preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, cuarma::blas::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma::blas::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) on GPU, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma::blas::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) on GPU, ell_matrix ----------" << std::endl;
  run_solver(cuarma_ell_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma::blas::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) on GPU, sliced_ell_matrix ----------" << std::endl;
  run_solver(cuarma_sliced_ell_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma::blas::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (no preconditioner) on GPU, hyb_matrix ----------" << std::endl;
  run_solver(cuarma_hyb_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma::blas::no_precond(), gmres_ops);

  std::cout << "------- GMRES solver (ILUT preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, ublas_ilut, gmres_ops);

  std::cout << "------- GMRES solver (ILUT preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma_ilut, gmres_ops);

  std::cout << "------- GMRES solver (ILUT preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma_ilut, gmres_ops);


  std::cout << "------- GMRES solver (Jacobi preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, ublas_jacobi, gmres_ops);

  std::cout << "------- GMRES solver (Jacobi preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma_jacobi_csr, gmres_ops);

  std::cout << "------- GMRES solver (Jacobi preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma_jacobi_coo, gmres_ops);


  std::cout << "------- GMRES solver (row scaling preconditioner) using ublas ----------" << std::endl;
  run_solver(ublas_matrix, ublas_vec2, ublas_result, gmres_solver, ublas_row_scaling, gmres_ops);

  std::cout << "------- GMRES solver (row scaling preconditioner) via ViennaCL, compressed_matrix ----------" << std::endl;
  run_solver(cuarma_compressed_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma_row_scaling_csr, gmres_ops);

  std::cout << "------- GMRES solver (row scaling preconditioner) via ViennaCL, coordinate_matrix ----------" << std::endl;
  run_solver(cuarma_coordinate_matrix, cuarma_vec2, cuarma_result, gmres_solver, cuarma_row_scaling_coo, gmres_ops);

  return EXIT_SUCCESS;
}

int main()
{
  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "               Device Info" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

  cuarma::context ctx;

  std::cout << "---------------------------------------------------------------------------" << std::endl;
  std::cout << "---------------------------------------------------------------------------" << std::endl;
  std::cout << " Benchmark for Execution Times of Iterative Solvers provided with ViennaCL " << std::endl;
  std::cout << "---------------------------------------------------------------------------" << std::endl;
  std::cout << " Note that the purpose of this benchmark is not to run solvers until" << std::endl;
  std::cout << " convergence. Instead, only the execution times of a few iterations are" << std::endl;
  std::cout << " recorded. Residual errors are only printed for information." << std::endl << std::endl;


  std::cout << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "## Benchmark :: Solver" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  std::cout << "   # benchmarking single-precision" << std::endl;
  std::cout << "   -------------------------------" << std::endl;
  run_benchmark<float>(ctx);
  {
    std::cout << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    std::cout << "   # benchmarking double-precision" << std::endl;
    std::cout << "   -------------------------------" << std::endl;
    run_benchmark<double>(ctx);
  }
  return 0;
}

