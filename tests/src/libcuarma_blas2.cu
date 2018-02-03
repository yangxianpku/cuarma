/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */


#include <iostream>
#include <vector>
#include "head_define.h"
#include "cuarma.hpp"
#include "cuarma/vector.hpp"

template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
  if (s1 > s2 || s1 < s2)
    return (s1 - s2) / std::max(std::fabs(s1), std::fabs(s2));
  return ScalarType(0);
}

template<typename ScalarType, typename cuarmaVectorType>
ScalarType diff(std::vector<ScalarType> const & v1, cuarmaVectorType const & arma_vec)
{
   std::vector<ScalarType> v2_cpu(arma_vec.size());
   cuarma::backend::finish();
   cuarma::copy(arma_vec, v2_cpu);

   ScalarType inf_norm = 0;
   for (unsigned int i=0;i<v1.size(); ++i)
   {
      if ( std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) ) > 0 )
         v2_cpu[i] = std::fabs(v2_cpu[i] - v1[i]) / std::max( std::fabs(v2_cpu[i]), std::fabs(v1[i]) );
      else
         v2_cpu[i] = 0.0;

      if (v2_cpu[i] > inf_norm)
        inf_norm = v2_cpu[i];
   }

   return inf_norm;
}

template<typename T, typename U, typename EpsilonT>
void check(T const & t, U const & u, EpsilonT eps)
{
  EpsilonT rel_error = std::fabs(static_cast<EpsilonT>(diff(t,u)));
  if (rel_error > eps)
  {
    std::cerr << "Relative error: " << rel_error << std::endl;
    std::cerr << "Aborting!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "SUCCESS ";
}

int main()
{
  std::size_t size1  = 13; // at least 7
  std::size_t size2  = 11; // at least 7
  float  eps_float  = 1e-5f;
  double eps_double = 1e-12;

  cuarmaBackend my_backend;
  cuarmaBackendCreate(&my_backend);

  std::vector<float> ref_float_x(size1); for (std::size_t i=0; i<size1; ++i) ref_float_x[i] = static_cast<float>(i);
  std::vector<float> ref_float_y(size2); for (std::size_t i=0; i<size2; ++i) ref_float_y[i] = static_cast<float>(size2 - i);
  std::vector<float> ref_float_A(size1*size2); for (std::size_t i=0; i<size1*size2; ++i) ref_float_A[i] = static_cast<float>(3*i);
  std::vector<float> ref_float_B(size1*size2); for (std::size_t i=0; i<size1*size2; ++i) ref_float_B[i] = static_cast<float>(2*i);

  std::vector<double> ref_double_x(size1, 1.0); for (std::size_t i=0; i<size1; ++i) ref_double_x[i] = static_cast<double>(i);
  std::vector<double> ref_double_y(size2, 2.0); for (std::size_t i=0; i<size2; ++i) ref_double_y[i] = static_cast<double>(size2 - i);
  std::vector<double> ref_double_A(size1*size2, 3.0); for (std::size_t i=0; i<size1*size2; ++i) ref_double_A[i] = static_cast<double>(3*i);
  std::vector<double> ref_double_B(size1*size2, 4.0); for (std::size_t i=0; i<size1*size2; ++i) ref_double_B[i] = static_cast<double>(2*i);

  // Host setup
  cuarma::vector<float> host_float_x = cuarma::scalar_vector<float>(size1, 1.0f, cuarma::context(cuarma::MAIN_MEMORY)); for (std::size_t i=0; i<size1; ++i) host_float_x[i] = float(i);
  cuarma::vector<float> host_float_y = cuarma::scalar_vector<float>(size2, 2.0f, cuarma::context(cuarma::MAIN_MEMORY)); for (std::size_t i=0; i<size2; ++i) host_float_y[i] = float(size2 - i);
  cuarma::vector<float> host_float_A = cuarma::scalar_vector<float>(size1*size2, 3.0f, cuarma::context(cuarma::MAIN_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) host_float_A[i] = float(3*i);
  cuarma::vector<float> host_float_B = cuarma::scalar_vector<float>(size1*size2, 4.0f, cuarma::context(cuarma::MAIN_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) host_float_B[i] = float(2*i);

  cuarma::vector<double> host_double_x = cuarma::scalar_vector<double>(size1, 1.0, cuarma::context(cuarma::MAIN_MEMORY)); for (std::size_t i=0; i<size1; ++i) host_double_x[i] = double(i);
  cuarma::vector<double> host_double_y = cuarma::scalar_vector<double>(size2, 2.0, cuarma::context(cuarma::MAIN_MEMORY)); for (std::size_t i=0; i<size2; ++i) host_double_y[i] = double(size2 - i);
  cuarma::vector<double> host_double_A = cuarma::scalar_vector<double>(size1*size2, 3.0, cuarma::context(cuarma::MAIN_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) host_double_A[i] = double(3*i);
  cuarma::vector<double> host_double_B = cuarma::scalar_vector<double>(size1*size2, 4.0, cuarma::context(cuarma::MAIN_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) host_double_B[i] = double(2*i);

  // CUDA setup
#ifdef CUARMA_WITH_CUDA
  cuarma::vector<float> cuda_float_x = cuarma::scalar_vector<float>(size1, 1.0f, cuarma::context(cuarma::CUDA_MEMORY)); for (std::size_t i=0; i<size1; ++i) cuda_float_x[i] = float(i);
  cuarma::vector<float> cuda_float_y = cuarma::scalar_vector<float>(size2, 2.0f, cuarma::context(cuarma::CUDA_MEMORY)); for (std::size_t i=0; i<size2; ++i) cuda_float_y[i] = float(size2 - i);
  cuarma::vector<float> cuda_float_A = cuarma::scalar_vector<float>(size1*size2, 3.0f, cuarma::context(cuarma::CUDA_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) cuda_float_A[i] = float(3*i);
  cuarma::vector<float> cuda_float_B = cuarma::scalar_vector<float>(size1*size2, 4.0f, cuarma::context(cuarma::CUDA_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) cuda_float_B[i] = float(2*i);

  cuarma::vector<double> cuda_double_x = cuarma::scalar_vector<double>(size1, 1.0, cuarma::context(cuarma::CUDA_MEMORY)); for (std::size_t i=0; i<size1; ++i) cuda_double_x[i] = double(i);
  cuarma::vector<double> cuda_double_y = cuarma::scalar_vector<double>(size2, 2.0, cuarma::context(cuarma::CUDA_MEMORY)); for (std::size_t i=0; i<size2; ++i) cuda_double_y[i] = double(size2 - i);
  cuarma::vector<double> cuda_double_A = cuarma::scalar_vector<double>(size1*size2, 3.0, cuarma::context(cuarma::CUDA_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) cuda_double_A[i] = double(3*i);
  cuarma::vector<double> cuda_double_B = cuarma::scalar_vector<double>(size1*size2, 4.0, cuarma::context(cuarma::CUDA_MEMORY)); for (std::size_t i=0; i<size1*size2; ++i) cuda_double_B[i] = double(2*i);
#endif

  // consistency checks:
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  check(ref_float_A, host_float_A, eps_float);
  check(ref_float_B, host_float_B, eps_float);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);
  check(ref_double_A, host_double_A, eps_double);
  check(ref_double_B, host_double_B, eps_double);

#ifdef CUARMA_WITH_CUDA
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  check(ref_float_A, cuda_float_A, eps_float);
  check(ref_float_B, cuda_float_B, eps_float);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
  check(ref_double_A, cuda_double_A, eps_double);
  check(ref_double_B, cuda_double_B, eps_double);
#endif

  // GEMV
  std::cout << std::endl << "-- Testing xGEMV...";
  for (std::size_t i=0; i<size1/3; ++i)
  {
    ref_float_x[i * 2 + 1] *= 0.1234f;
    ref_double_x[i * 2 + 1] *= 0.1234;
    for (std::size_t j=0; j<size2/4; ++j)
    {
      ref_float_x[i * 2 + 1]  += 3.1415f * ref_float_A[(2*i+2) * size2 + 3 * j + 1] * ref_float_y[j * 3 + 1];
      ref_double_x[i * 2 + 1] += 3.1415  * ref_double_A[(2*i+2) * size2 + 3 * j + 1] * ref_double_y[j * 3 + 1];
    }
  }

  std::cout << std::endl << "Host: ";
  cuarmaHostSgemv(my_backend,
                    cuarmaRowMajor, cuarmaNoTrans,
                    cuarmaInt(size1/3), cuarmaInt(size2/4), 3.1415f, cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_A), 2, 1, 2, 3, cuarmaInt(size2),
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_y), 1, 3,
                    0.1234f,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 1, 2);
  check(ref_float_x, host_float_x, eps_float);
  cuarmaHostDgemv(my_backend,
                    cuarmaRowMajor, cuarmaNoTrans,
                    cuarmaInt(size1/3), cuarmaInt(size2/4), 3.1415, cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_A), 2, 1, 2, 3, cuarmaInt(size2),
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_y), 1, 3,
                    0.1234,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 1, 2);
  check(ref_double_x, host_double_x, eps_double);


#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  cuarmaCUDASgemv(my_backend,
                    cuarmaRowMajor, cuarmaNoTrans,
                    cuarmaInt(size1/3), cuarmaInt(size2/4), 3.1415f, cuarma::cuda_arg(cuda_float_A), 2, 1, 2, 3, size2,
                    cuarma::cuda_arg(cuda_float_y), 1, 3,
                    0.1234f,
                    cuarma::cuda_arg(cuda_float_x), 1, 2);
  check(ref_float_x, cuda_float_x, eps_float);
  cuarmaCUDADgemv(my_backend,
                    cuarmaRowMajor, cuarmaNoTrans,
                    cuarmaInt(size1/3), cuarmaInt(size2/4), 3.1415, cuarma::cuda_arg(cuda_double_A), 2, 1, 2, 3, size2,
                    cuarma::cuda_arg(cuda_double_y), 1, 3,
                    0.1234,
                    cuarma::cuda_arg(cuda_double_x), 1, 2);
  check(ref_double_x, cuda_double_x, eps_double);
#endif

  cuarmaBackendDestroy(&my_backend);

  //
  //  That's it.
  //
  std::cout << std::endl << "!!!! TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

