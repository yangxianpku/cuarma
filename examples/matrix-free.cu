/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   matrix-free.cu
 *   @coding UTF-8
 *   @brief  This tutorial explains how to use the iterative solvers in cuarma in a matrix-free manner, i.e. without explicitly assembling a matrix.
 *           Operator overloading in C++ is used extensively to provide an intuitive syntax.
 *   @brief  测试：BLAS2 运算示例程序
 */
 
/** \example matrix-free.cpp
*
*   This tutorial explains how to use the iterative solvers in cuarma in a matrix-free manner, i.e. without explicitly assembling a matrix.
*
*   We consider the solution of the Poisson equation \f$ \Delta \varphi = -1 \f$ on the unit square \f$ [0,1] \times [0,1] \f$ with homogeneous Dirichlet boundary conditions using a finite-difference discretization.
*   A \f$ N \times N \f$ grid is used, where the first and the last points per dimensions represent the boundary.
*   For simplicity we only consider the host-backend here. Have a look at custom-kernels.hpp and custom-cuda.cu on how to use custom kernels in such a matrix-free setting.
**/

#include <iostream>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/cg.hpp"
#include "cuarma/blas/bicgstab.hpp"
#include "cuarma/blas/gmres.hpp"

/**
  * cuarma imposes two type requirements on a user-provided operator to compute `y = prod(A, x)` for the iterative solvers:
  *   - A member function `apply()`, taking two cuarma base vectors `x` and `y` as arguments. This member function carries out the action of the matrix to the vector.
  *   - A member function `size1()` returning the length of the result vectors.
  * Keep in mind that you can always wrap your existing classes accordingly to fit cuarma's interface requirements.
  *
  * We define a simple class for dealing with the \f$ N \times N \f$ grid for solving Poisson's equation.
  * It only holds the number of grid points per coordinate direction and implements the `apply()` and `size1()` routines.
  * Depending on whether the host, OpenCL, or CUDA is used for the computation, the respective implementation is called.
  * We skip the details for now and discuss (and implement) them at the end of this tutorial.
  **/
template<typename NumericT>
class MyOperator
{
public:
  MyOperator(std::size_t N) : N_(N) {}

  // Dispatcher for y = Ax
  void apply(cuarma::vector_base<NumericT> const & x, cuarma::vector_base<NumericT> & y) const
  {
#if defined(CUARMA_WITH_CUDA)
    if (cuarma::traits::active_handle_id(x) == cuarma::CUDA_MEMORY)
      apply_cuda(x, y);
#endif
    if (cuarma::traits::active_handle_id(x) == cuarma::MAIN_MEMORY)
      apply_host(x, y);
  }

  std::size_t size1() const { return N_ * N_; }

private:

#if defined(CUARMA_WITH_CUDA)
  void apply_cuda(cuarma::vector_base<NumericT> const & x, cuarma::vector_base<NumericT> & y) const;
#endif
  void apply_host(cuarma::vector_base<NumericT> const & x, cuarma::vector_base<NumericT> & y) const;

  std::size_t N_;
};


/**
* <h2>Main Program</h2>
*
*  In the `main()` routine we create the right hand side vector, instantiate the operator, and then call the solver.
**/
int main()
{
  typedef float       ScalarType;  // feel free to change to double (and change OpenCL kernel argument types accordingly)

  std::size_t N = 10;
  cuarma::vector<ScalarType> rhs = cuarma::scalar_vector<ScalarType>(N*N, ScalarType(-1));
  MyOperator<ScalarType> op(N);

  /**
  * Run the CG method with our on-the-fly operator.
  * Use `cuarma::blas::bicgstab_tag()` or `cuarma::blas::gmres_tag()` instead of `cuarma::blas::cg_tag()` to solve using BiCGStab or GMRES, respectively.
  **/
  cuarma::vector<ScalarType> result = cuarma::blas::solve(op, rhs, cuarma::blas::cg_tag());

  /**
   * Pretty-Print solution vector to verify solution.
   * (We use a slow direct element-access via `operator[]` here for convenience.)
   **/
  std::cout.precision(3);
  std::cout << std::fixed;
  std::cout << "Result value map: " << std::endl;
  std::cout << std::endl << "^ y " << std::endl;
  for (std::size_t i=0; i<N; ++i)
  {
    std::cout << "|  ";
    for (std::size_t j=0; j<N; ++j)
      std::cout << result[i * N + j] << "  ";
    std::cout << std::endl;
  }
  std::cout << "*---------------------------------------------> x" << std::endl;

  /**
  *  That's it, print a completion message. Read on for details on how to implement the actual compute kernels.
  **/
  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}




/**
  * <h2> Implementation Details </h2>
  *
  *  So far we have only looked at the code for the control logic.
  *  In the following we define the actual 'worker' code for the matrix-free implementations.
  *
  *  <h3> Execution on Host </h3>
  *
  *  Since the execution on the host has the smallest amount of boilerplate code surrounding it, we use this case as a starting point.
  **/
template<typename NumericT>
void MyOperator<NumericT>::apply_host(cuarma::vector_base<NumericT> const & x, cuarma::vector_base<NumericT> & y) const
{
  NumericT const * values_x = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(x.handle());
  NumericT       * values_y = cuarma::blas::host_based::detail::extract_raw_pointer<NumericT>(y.handle());

  NumericT dx = NumericT(1) / NumericT(N_ + 1);
  NumericT dy = NumericT(1) / NumericT(N_ + 1);

/**
  *  In the following we iterate over all \f$ N \times N \f$ points and apply the five-point stencil directly.
  *  This is done in a straightforward manner for illustration purposes.
  *  Multi-threaded execution via OpenMP can be obtained by uncommenting the pragma below.
  *
  *  Feel free to apply additional optimizations with respect to data access patterns and the like.
  **/

  // feel free to use
  //  #pragma omp parallel for
  // here
  for (std::size_t i=0; i<N_; ++i)
    for (std::size_t j=0; j<N_; ++j)
    {
      NumericT value_right  = (j < N_ - 1) ? values_x[ i   *N_ + j + 1] : 0;
      NumericT value_left   = (j > 0     ) ? values_x[ i   *N_ + j - 1] : 0;
      NumericT value_top    = (i < N_ - 1) ? values_x[(i+1)*N_ + j    ] : 0;
      NumericT value_bottom = (i > 0     ) ? values_x[(i-1)*N_ + j    ] : 0;
      NumericT value_center = values_x[i*N_ + j];

      values_y[i*N_ + j] =   ((value_right - value_center) / dx - (value_center - value_left)   / dx) / dx
                           + ((value_top   - value_center) / dy - (value_center - value_bottom) / dy) / dy;
    }
}


/**
  * <h3> Execution via CUDA </h3>
  *
  *  The host-based kernel code serves as a basis for the CUDA kernel.
  *  The only thing we have to adjust are the array bounds:
  *  We assign one CUDA threadblock per index `i`.
  *  For a fixed `i`, parallelization across all threads in the block is obtained with respect to `j`.
  *
  *  Again, feel free to apply additional optimizations with respect to data access patterns and the like...
  **/

#if defined(CUARMA_WITH_CUDA)
template<typename NumericT>
__global__ void apply_cuda_kernel(NumericT const * values_x,
                                  NumericT       * values_y,
                                  std::size_t N)
{
  NumericT dx = NumericT(1) / (N + 1);
  NumericT dy = NumericT(1) / (N + 1);

  for (std::size_t i = blockIdx.x; i < N; i += gridDim.x)
    for (std::size_t j = threadIdx.x; j < N; j += blockDim.x)
    {
      NumericT value_right  = (j < N - 1) ? values_x[ i   *N + j + 1] : 0;
      NumericT value_left   = (j > 0    ) ? values_x[ i   *N + j - 1] : 0;
      NumericT value_top    = (i < N - 1) ? values_x[(i+1)*N + j    ] : 0;
      NumericT value_bottom = (i > 0    ) ? values_x[(i-1)*N + j    ] : 0;
      NumericT value_center = values_x[i*N + j];

      values_y[i*N + j] =   ((value_right - value_center) / dx - (value_center - value_left)   / dx) / dx
                          + ((value_top   - value_center) / dy - (value_center - value_bottom) / dy) / dy;
    }
}
#endif

#if defined(CUARMA_WITH_CUDA)
template<typename NumericT>
void MyOperator<NumericT>::apply_cuda(cuarma::vector_base<NumericT> const & x, cuarma::vector_base<NumericT> & y) const
{
  apply_cuda_kernel<<<128, 128>>>(cuarma::cuda_arg(x), cuarma::cuda_arg(y), N_);
}
#endif