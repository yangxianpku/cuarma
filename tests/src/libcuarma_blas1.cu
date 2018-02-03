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
#include <cmath>

#include "cuarma.hpp"
#include "cuarma/vector.hpp"

template<typename ScalarType>
ScalarType diff(ScalarType const & s1, ScalarType const & s2)
{
   if (s1 > s2 || s1 < s2)
      return (s1 - s2) / std::max(static_cast<ScalarType>(std::fabs(static_cast<double>(s1))),static_cast<ScalarType>(std::fabs(static_cast<double>(s2))));
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
  std::size_t size  = 10; // at least 7
  float  eps_float  = 1e-5f;
  double eps_double = 1e-12;

  float  ref_float_alpha;
  double ref_double_alpha;

  std::vector<float> ref_float_x(size, 1.0f);
  std::vector<float> ref_float_y(size, 2.0f);

  std::vector<double> ref_double_x(size, 1.0);
  std::vector<double> ref_double_y(size, 2.0);

  cuarmaBackend my_backend;
  cuarmaBackendCreate(&my_backend);

  // Host setup
  float host_float_alpha = 0;
  cuarma::vector<float> host_float_x = cuarma::scalar_vector<float>(size, 1.0f, cuarma::context(cuarma::MAIN_MEMORY));
  cuarma::vector<float> host_float_y = cuarma::scalar_vector<float>(size, 2.0f, cuarma::context(cuarma::MAIN_MEMORY));

  double host_double_alpha = 0;
  cuarma::vector<double> host_double_x = cuarma::scalar_vector<double>(size, 1.0, cuarma::context(cuarma::MAIN_MEMORY));
  cuarma::vector<double> host_double_y = cuarma::scalar_vector<double>(size, 2.0, cuarma::context(cuarma::MAIN_MEMORY));

  // CUDA setup
#ifdef CUARMA_WITH_CUDA
  float cuda_float_alpha = 0;
  cuarma::vector<float> cuda_float_x = cuarma::scalar_vector<float>(size, 1.0f, cuarma::context(cuarma::CUDA_MEMORY));
  cuarma::vector<float> cuda_float_y = cuarma::scalar_vector<float>(size, 2.0f, cuarma::context(cuarma::CUDA_MEMORY));

  double cuda_double_alpha = 0;
  cuarma::vector<double> cuda_double_x = cuarma::scalar_vector<double>(size, 1.0, cuarma::context(cuarma::CUDA_MEMORY));
  cuarma::vector<double> cuda_double_y = cuarma::scalar_vector<double>(size, 2.0, cuarma::context(cuarma::CUDA_MEMORY));
#endif

  // consistency checks:
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);

#ifdef CUARMA_WITH_CUDA
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
#endif


  // ASUM
  std::cout << std::endl << "-- Testing xASUM...";
  ref_float_alpha  = 0;
  ref_double_alpha = 0;
  for (std::size_t i=0; i<size/4; ++i)
  {
    ref_float_alpha  += std::fabs(ref_float_x[2 + 3*i]);
    ref_double_alpha += std::fabs(ref_double_x[2 + 3*i]);
  }

  std::cout << std::endl << "Host: ";
  cuarmaHostSasum(my_backend, cuarmaInt(size/4),
                    &host_float_alpha,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 2, 3);
  check(ref_float_alpha, host_float_alpha, eps_float);
  cuarmaHostDasum(my_backend, cuarmaInt(size/4),
                    &host_double_alpha,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 2, 3);
  check(ref_double_alpha, host_double_alpha, eps_double);


#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  cuarmaCUDASasum(my_backend, cuarmaInt(size/4),
                    &cuda_float_alpha,
                    cuarma::cuda_arg(cuda_float_x), 2, 3);
  check(ref_float_alpha, cuda_float_alpha, eps_float);
  cuarmaCUDADasum(my_backend, cuarmaInt(size/4),
                    &cuda_double_alpha,
                    cuarma::cuda_arg(cuda_double_x), 2, 3);
  check(ref_double_alpha, cuda_double_alpha, eps_double);
#endif

  // AXPY
  std::cout << std::endl << "-- Testing xAXPY...";
  for (std::size_t i=0; i<size/3; ++i)
  {
    ref_float_y[1 + 2*i]  += 2.0f * ref_float_x[0 + 2*i];
    ref_double_y[1 + 2*i] += 2.0  * ref_double_x[0 + 2*i];
  }

  std::cout << std::endl << "Host: ";
  cuarmaHostSaxpy(my_backend, cuarmaInt(size/3),
                    2.0f,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 0, 2,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_y), 1, 2);
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  cuarmaHostDaxpy(my_backend, cuarmaInt(size/3),
                    2.0,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 0, 2,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_y), 1, 2);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);


#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  cuarmaCUDASaxpy(my_backend, cuarmaInt(size/3),
                    2.0f,
                    cuarma::cuda_arg(cuda_float_x), 0, 2,
                    cuarma::cuda_arg(cuda_float_y), 1, 2);
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  cuarmaCUDADaxpy(my_backend, cuarmaInt(size/3),
                    2.0,
                    cuarma::cuda_arg(cuda_double_x), 0, 2,
                    cuarma::cuda_arg(cuda_double_y), 1, 2);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
#endif

#ifdef CUARMA_WITH_OPENCL
  std::cout << std::endl << "OpenCL: ";
  cuarmaOpenCLSaxpy(my_backend, cuarmaInt(size/3),
                      2.0f,
                      cuarma::traits::opencl_handle(opencl_float_x).get(), 0, 2,
                      cuarma::traits::opencl_handle(opencl_float_y).get(), 1, 2);
  check(ref_float_x, opencl_float_x, eps_float);
  check(ref_float_y, opencl_float_y, eps_float);
  if ( cuarma::ocl::current_device().double_support() )
  {
    cuarmaOpenCLDaxpy(my_backend, cuarmaInt(size/3),
                        2.0,
                        cuarma::traits::opencl_handle(*opencl_double_x).get(), 0, 2,
                        cuarma::traits::opencl_handle(*opencl_double_y).get(), 1, 2);
    check(ref_double_x, *opencl_double_x, eps_double);
    check(ref_double_y, *opencl_double_y, eps_double);
  }
#endif



  // COPY
  std::cout << std::endl << "-- Testing xCOPY...";
  for (std::size_t i=0; i<size/3; ++i)
  {
    ref_float_y[0 + 2*i]  = ref_float_x[1 + 2*i];
    ref_double_y[0 + 2*i] = ref_double_x[1 + 2*i];
  }

  std::cout << std::endl << "Host: ";
  cuarmaHostScopy(my_backend, cuarmaInt(size/3),
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 1, 2,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_y), 0, 2);
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  cuarmaHostDcopy(my_backend, cuarmaInt(size/3),
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 1, 2,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_y), 0, 2);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);


#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  cuarmaCUDAScopy(my_backend, cuarmaInt(size/3),
                    cuarma::cuda_arg(cuda_float_x), 1, 2,
                    cuarma::cuda_arg(cuda_float_y), 0, 2);
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  cuarmaCUDADcopy(my_backend, cuarmaInt(size/3),
                    cuarma::cuda_arg(cuda_double_x), 1, 2,
                    cuarma::cuda_arg(cuda_double_y), 0, 2);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
#endif

  // DOT
  std::cout << std::endl << "-- Testing xDOT...";
  ref_float_alpha  = 0;
  ref_double_alpha = 0;
  for (std::size_t i=0; i<size/2; ++i)
  {
    ref_float_alpha  += ref_float_y[3 + i]  * ref_float_x[2 + i];
    ref_double_alpha += ref_double_y[3 + i] * ref_double_x[2 + i];
  }

  std::cout << std::endl << "Host: ";
  cuarmaHostSdot(my_backend, cuarmaInt(size/2),
                   &host_float_alpha,
                   cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 2, 1,
                   cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_y), 3, 1);
  check(ref_float_alpha, host_float_alpha, eps_float);
  cuarmaHostDdot(my_backend, cuarmaInt(size/2),
                   &host_double_alpha,
                   cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 2, 1,
                   cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_y), 3, 1);
  check(ref_double_alpha, host_double_alpha, eps_double);


#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  cuarmaCUDASdot(my_backend, cuarmaInt(size/2),
                   &cuda_float_alpha,
                   cuarma::cuda_arg(cuda_float_x), 2, 1,
                   cuarma::cuda_arg(cuda_float_y), 3, 1);
  check(ref_float_alpha, cuda_float_alpha, eps_float);
  cuarmaCUDADdot(my_backend, cuarmaInt(size/2),
                   &cuda_double_alpha,
                   cuarma::cuda_arg(cuda_double_x), 2, 1,
                   cuarma::cuda_arg(cuda_double_y), 3, 1);
  check(ref_double_alpha, cuda_double_alpha, eps_double);
#endif

  // NRM2
  std::cout << std::endl << "-- Testing xNRM2...";
  ref_float_alpha  = 0;
  ref_double_alpha = 0;
  for (std::size_t i=0; i<size/3; ++i)
  {
    ref_float_alpha  += ref_float_x[1 + 2*i]  * ref_float_x[1 + 2*i];
    ref_double_alpha += ref_double_x[1 + 2*i] * ref_double_x[1 + 2*i];
  }
  ref_float_alpha = std::sqrt(ref_float_alpha);
  ref_double_alpha = std::sqrt(ref_double_alpha);

  std::cout << std::endl << "Host: ";
  cuarmaHostSnrm2(my_backend, cuarmaInt(size/3),
                    &host_float_alpha,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 1, 2);
  check(ref_float_alpha, host_float_alpha, eps_float);
  cuarmaHostDnrm2(my_backend, cuarmaInt(size/3),
                    &host_double_alpha,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 1, 2);
  check(ref_double_alpha, host_double_alpha, eps_double);


#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  cuarmaCUDASnrm2(my_backend, cuarmaInt(size/3),
                    &cuda_float_alpha,
                    cuarma::cuda_arg(cuda_float_x), 1, 2);
  check(ref_float_alpha, cuda_float_alpha, eps_float);
  cuarmaCUDADnrm2(my_backend, cuarmaInt(size/3),
                    &cuda_double_alpha,
                    cuarma::cuda_arg(cuda_double_x), 1, 2);
  check(ref_double_alpha, cuda_double_alpha, eps_double);
#endif

  // ROT
  std::cout << std::endl << "-- Testing xROT...";
  for (std::size_t i=0; i<size/4; ++i)
  {
    float tmp            =  0.6f * ref_float_x[2 + 3*i] + 0.8f * ref_float_y[1 + 2*i];
    ref_float_y[1 + 2*i] = -0.8f * ref_float_x[2 + 3*i] + 0.6f * ref_float_y[1 + 2*i];;
    ref_float_x[2 + 3*i] = tmp;

    double tmp2           =  0.6 * ref_double_x[2 + 3*i] + 0.8 * ref_double_y[1 + 2*i];
    ref_double_y[1 + 2*i] = -0.8 * ref_double_x[2 + 3*i] + 0.6 * ref_double_y[1 + 2*i];;
    ref_double_x[2 + 3*i] = tmp2;
  }

  std::cout << std::endl << "Host: ";
  cuarmaHostSrot(my_backend, cuarmaInt(size/4),
                   cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 2, 3,
                   cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_y), 1, 2,
                   0.6f, 0.8f);
  check(ref_float_x, host_float_x, eps_float);
  check(ref_float_y, host_float_y, eps_float);
  cuarmaHostDrot(my_backend, cuarmaInt(size/4),
                   cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 2, 3,
                   cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_y), 1, 2,
                   0.6, 0.8);
  check(ref_double_x, host_double_x, eps_double);
  check(ref_double_y, host_double_y, eps_double);


#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  cuarmaCUDASrot(my_backend, cuarmaInt(size/4),
                   cuarma::cuda_arg(cuda_float_x), 2, 3,
                   cuarma::cuda_arg(cuda_float_y), 1, 2,
                   0.6f, 0.8f);
  check(ref_float_x, cuda_float_x, eps_float);
  check(ref_float_y, cuda_float_y, eps_float);
  cuarmaCUDADrot(my_backend, cuarmaInt(size/4),
                   cuarma::cuda_arg(cuda_double_x), 2, 3,
                   cuarma::cuda_arg(cuda_double_y), 1, 2,
                   0.6, 0.8);
  check(ref_double_x, cuda_double_x, eps_double);
  check(ref_double_y, cuda_double_y, eps_double);
#endif

  // SCAL
  std::cout << std::endl << "-- Testing xSCAL...";
  for (std::size_t i=0; i<size/4; ++i)
  {
    ref_float_x[1 + 3*i]  *= 2.0f;
    ref_double_x[1 + 3*i] *= 2.0;
  }

  std::cout << std::endl << "Host: ";
  cuarmaHostSscal(my_backend, cuarmaInt(size/4),
                    2.0f,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 1, 3);
  check(ref_float_x, host_float_x, eps_float);
  cuarmaHostDscal(my_backend, cuarmaInt(size/4),
                    2.0,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 1, 3);
  check(ref_double_x, host_double_x, eps_double);

#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  cuarmaCUDASscal(my_backend, cuarmaInt(size/4),
                    2.0f,
                    cuarma::cuda_arg(cuda_float_x), 1, 3);
  check(ref_float_x, cuda_float_x, eps_float);
  cuarmaCUDADscal(my_backend, cuarmaInt(size/4),
                    2.0,
                    cuarma::cuda_arg(cuda_double_x), 1, 3);
  check(ref_double_x, cuda_double_x, eps_double);
#endif

  // SWAP
  std::cout << std::endl << "-- Testing xSWAP...";
  for (std::size_t i=0; i<size/3; ++i)
  {
    float tmp = ref_float_x[2 + 2*i];
    ref_float_x[2 + 2*i] = ref_float_y[1 + 2*i];
    ref_float_y[1 + 2*i] = tmp;

    double tmp2 = ref_double_x[2 + 2*i];
    ref_double_x[2 + 2*i] = ref_double_y[1 + 2*i];
    ref_double_y[1 + 2*i] = tmp2;
  }

  std::cout << std::endl << "Host: ";
  cuarmaHostSswap(my_backend, cuarmaInt(size/3),
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 2, 2,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_y), 1, 2);
  check(ref_float_y, host_float_y, eps_float);
  cuarmaHostDswap(my_backend, cuarmaInt(size/3),
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 2, 2,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_y), 1, 2);
  check(ref_double_y, host_double_y, eps_double);


#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  cuarmaCUDASswap(my_backend, cuarmaInt(size/3),
                    cuarma::cuda_arg(cuda_float_x), 2, 2,
                    cuarma::cuda_arg(cuda_float_y), 1, 2);
  check(ref_float_y, cuda_float_y, eps_float);
  cuarmaCUDADswap(my_backend, cuarmaInt(size/3),
                    cuarma::cuda_arg(cuda_double_x), 2, 2,
                    cuarma::cuda_arg(cuda_double_y), 1, 2);
  check(ref_double_y, cuda_double_y, eps_double);
#endif

  // IAMAX
  std::cout << std::endl << "-- Testing IxASUM...";
  cuarmaInt ref_index = 0;
  ref_float_alpha = 0;
  for (std::size_t i=0; i<size/3; ++i)
  {
    if (ref_float_x[0 + 2*i] > std::fabs(ref_float_alpha))
    {
      ref_index = cuarmaInt(i);
      ref_float_alpha = std::fabs(ref_float_x[0 + 2*i]);
    }
  }

  std::cout << std::endl << "Host: ";
  cuarmaInt idx = 0;
  cuarmaHostiSamax(my_backend, cuarmaInt(size/3),
                     &idx,
                     cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_float_x), 0, 2);
  check(static_cast<float>(ref_index), static_cast<float>(idx), eps_float);
  idx = 0;
  cuarmaHostiDamax(my_backend, cuarmaInt(size/3),
                     &idx,
                     cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_double_x), 0, 2);
  check(ref_index, idx, eps_double);

#ifdef CUARMA_WITH_CUDA
  std::cout << std::endl << "CUDA: ";
  idx = 0;
  cuarmaCUDAiSamax(my_backend, cuarmaInt(size/3),
                     &idx,
                     cuarma::cuda_arg(cuda_float_x), 0, 2);
  check(ref_float_x[2*ref_index], ref_float_x[2*idx], eps_float);
  idx = 0;
  cuarmaCUDAiDamax(my_backend, cuarmaInt(size/3),
                     &idx,
                     cuarma::cuda_arg(cuda_double_x), 0, 2);
  check(ref_double_x[2*ref_index], ref_double_x[2*idx], eps_double);
#endif

  cuarmaBackendDestroy(&my_backend);

  std::cout << std::endl << "!!!! TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

