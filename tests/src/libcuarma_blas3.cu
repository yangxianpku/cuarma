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
#include "cuarma.hpp"
#include "cuarma/tools/random.hpp"
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


template<typename T>
T get_value(std::vector<T> & array, cuarmaInt i, cuarmaInt j,
            cuarmaInt start1, cuarmaInt start2,
            cuarmaInt stride1, cuarmaInt stride2,
            cuarmaInt rows, cuarmaInt cols,
            cuarmaOrder order, cuarmaTranspose trans)
{
  // row-major
  if (order == cuarmaRowMajor && trans == cuarmaTrans)
    return array[static_cast<std::size_t>((j*stride1 + start1) * cols + (i*stride2 + start2))];
  else if (order == cuarmaRowMajor && trans != cuarmaTrans)
    return array[static_cast<std::size_t>((i*stride1 + start1) * cols + (j*stride2 + start2))];

  // column-major
  else if (order != cuarmaRowMajor && trans == cuarmaTrans)
    return array[static_cast<std::size_t>((j*stride1 + start1) + (i*stride2 + start2) * rows)];
  return array[static_cast<std::size_t>((i*stride1 + start1) + (j*stride2 + start2) * rows)];
}


void test_blas(cuarmaBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               cuarmaOrder order_C, cuarmaOrder order_A, cuarmaOrder order_B,
               cuarmaTranspose trans_A, cuarmaTranspose trans_B,
               cuarma::vector<float> & host_C_float, cuarma::vector<double> & host_C_double,
               cuarma::vector<float> & host_A_float, cuarma::vector<double> & host_A_double,
               cuarma::vector<float> & host_B_float, cuarma::vector<double> & host_B_double
#ifdef CUARMA_WITH_CUDA
               , cuarma::vector<float> & cuda_C_float, cuarma::vector<double> & cuda_C_double
               , cuarma::vector<float> & cuda_A_float, cuarma::vector<double> & cuda_A_double
               , cuarma::vector<float> & cuda_B_float, cuarma::vector<double> & cuda_B_double
#endif
               );

void test_blas(cuarmaBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               cuarmaOrder order_C, cuarmaOrder order_A, cuarmaOrder order_B,
               cuarmaTranspose trans_A, cuarmaTranspose trans_B,
               cuarma::vector<float> & host_C_float, cuarma::vector<double> & host_C_double,
               cuarma::vector<float> & host_A_float, cuarma::vector<double> & host_A_double,
               cuarma::vector<float> & host_B_float, cuarma::vector<double> & host_B_double
#ifdef CUARMA_WITH_CUDA
               , cuarma::vector<float> & cuda_C_float, cuarma::vector<double> & cuda_C_double
               , cuarma::vector<float> & cuda_A_float, cuarma::vector<double> & cuda_A_double
               , cuarma::vector<float> & cuda_B_float, cuarma::vector<double> & cuda_B_double
#endif
               )
{
  cuarmaInt C_size1   = 42;
  cuarmaInt C_size2   = 43;
  cuarmaInt C_start1  = 10;
  cuarmaInt C_start2  = 11;
  cuarmaInt C_stride1 = 2;
  cuarmaInt C_stride2 = 3;
  cuarmaInt C_rows    = C_size1 * C_stride1 + C_start1 + 5;
  cuarmaInt C_columns = C_size2 * C_stride2 + C_start2 + 5;

  cuarmaInt A_size1   = trans_A ? 44 : 42;
  cuarmaInt A_size2   = trans_A ? 42 : 44;
  cuarmaInt A_start1  = 12;
  cuarmaInt A_start2  = 13;
  cuarmaInt A_stride1 = 4;
  cuarmaInt A_stride2 = 5;
  cuarmaInt A_rows    = A_size1 * A_stride1 + A_start1 + 5;
  cuarmaInt A_columns = A_size2 * A_stride2 + A_start2 + 5;

  cuarmaInt B_size1   = trans_B ? 43 : 44;
  cuarmaInt B_size2   = trans_B ? 44 : 43;
  cuarmaInt B_start1  = 14;
  cuarmaInt B_start2  = 15;
  cuarmaInt B_stride1 = 6;
  cuarmaInt B_stride2 = 7;
  cuarmaInt B_rows    = B_size1 * B_stride1 + B_start1 + 5;
  cuarmaInt B_columns = B_size2 * B_stride2 + B_start2 + 5;

  // Compute reference:
  cuarmaInt size_k = trans_A ? A_size1 : A_size2;
  for (cuarmaInt i=0; i<C_size1; ++i)
    for (cuarmaInt j=0; j<C_size2; ++j)
    {
      float val_float = 0;
      double val_double = 0;
      for (cuarmaInt k=0; k<size_k; ++k)
      {
        float  val_A_float  = get_value(A_float,  i, k, A_start1, A_start2, A_stride1, A_stride2, A_rows, A_columns, order_A, trans_A);
        double val_A_double = get_value(A_double, i, k, A_start1, A_start2, A_stride1, A_stride2, A_rows, A_columns, order_A, trans_A);

        float  val_B_float  = get_value(B_float,  k, j, B_start1, B_start2, B_stride1, B_stride2, B_rows, B_columns, order_B, trans_B);
        double val_B_double = get_value(B_double, k, j, B_start1, B_start2, B_stride1, B_stride2, B_rows, B_columns, order_B, trans_B);

        val_float  += val_A_float  * val_B_float;
        val_double += val_A_double * val_B_double;
      }

      // write result
      if (order_C == cuarmaRowMajor)
      {
        C_float [static_cast<std::size_t>((i*C_stride1 + C_start1) * C_columns + (j*C_stride2 + C_start2))] = val_float;
        C_double[static_cast<std::size_t>((i*C_stride1 + C_start1) * C_columns + (j*C_stride2 + C_start2))] = val_double;
      }
      else
      {
        C_float [static_cast<std::size_t>((i*C_stride1 + C_start1) + (j*C_stride2 + C_start2) * C_rows)] = val_float;
        C_double[static_cast<std::size_t>((i*C_stride1 + C_start1) + (j*C_stride2 + C_start2) * C_rows)] = val_double;
      }
    }

  // Run GEMM and compare results:
  cuarmaHostSgemm(my_backend,
                    order_A, trans_A, order_B, trans_B, order_C,
                    C_size1, C_size2, size_k,
                    1.0f,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_A_float), A_start1, A_start2, A_stride1, A_stride2, (order_A == cuarmaRowMajor) ? A_columns : A_rows,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_B_float), B_start1, B_start2, B_stride1, B_stride2, (order_B == cuarmaRowMajor) ? B_columns : B_rows,
                    0.0f,
                    cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_C_float), C_start1, C_start2, C_stride1, C_stride2, (order_C == cuarmaRowMajor) ? C_columns : C_rows);
  check(C_float, host_C_float, eps_float);

  cuarmaHostDgemm(my_backend,
                    order_A, trans_A, order_B, trans_B, order_C,
                    C_size1, C_size2, size_k,
                    1.0,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_A_double), A_start1, A_start2, A_stride1, A_stride2, (order_A == cuarmaRowMajor) ? A_columns : A_rows,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_B_double), B_start1, B_start2, B_stride1, B_stride2, (order_B == cuarmaRowMajor) ? B_columns : B_rows,
                    0.0,
                    cuarma::blas::host_based::detail::extract_raw_pointer<double>(host_C_double), C_start1, C_start2, C_stride1, C_stride2, (order_C == cuarmaRowMajor) ? C_columns : C_rows);
  check(C_double, host_C_double, eps_double);

#ifdef CUARMA_WITH_CUDA
  cuarmaCUDASgemm(my_backend,
                    order_A, trans_A, order_B, trans_B, order_C,
                    C_size1, C_size2, size_k,
                    1.0f,
                    cuarma::cuda_arg(cuda_A_float), A_start1, A_start2, A_stride1, A_stride2, (order_A == cuarmaRowMajor) ? A_columns : A_rows,
                    cuarma::cuda_arg(cuda_B_float), B_start1, B_start2, B_stride1, B_stride2, (order_B == cuarmaRowMajor) ? B_columns : B_rows,
                    0.0f,
                    cuarma::cuda_arg(cuda_C_float), C_start1, C_start2, C_stride1, C_stride2, (order_C == cuarmaRowMajor) ? C_columns : C_rows);
  check(C_float, cuda_C_float, eps_float);

  cuarmaCUDADgemm(my_backend,
                    order_A, trans_A, order_B, trans_B, order_C,
                    C_size1, C_size2, size_k,
                    1.0,
                    cuarma::cuda_arg(cuda_A_double), A_start1, A_start2, A_stride1, A_stride2, (order_A == cuarmaRowMajor) ? A_columns : A_rows,
                    cuarma::cuda_arg(cuda_B_double), B_start1, B_start2, B_stride1, B_stride2, (order_B == cuarmaRowMajor) ? B_columns : B_rows,
                    0.0,
                    cuarma::cuda_arg(cuda_C_double), C_start1, C_start2, C_stride1, C_stride2, (order_C == cuarmaRowMajor) ? C_columns : C_rows);
  check(C_double, cuda_C_double, eps_double);
#endif


  std::cout << std::endl;
}

void test_blas(cuarmaBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               cuarmaOrder order_C, cuarmaOrder order_A, cuarmaOrder order_B,
               cuarma::vector<float> & host_C_float, cuarma::vector<double> & host_C_double,
               cuarma::vector<float> & host_A_float, cuarma::vector<double> & host_A_double,
               cuarma::vector<float> & host_B_float, cuarma::vector<double> & host_B_double
#ifdef CUARMA_WITH_CUDA
               , cuarma::vector<float> & cuda_C_float, cuarma::vector<double> & cuda_C_double
               , cuarma::vector<float> & cuda_A_float, cuarma::vector<double> & cuda_A_double
               , cuarma::vector<float> & cuda_B_float, cuarma::vector<double> & cuda_B_double
#endif
               );

void test_blas(cuarmaBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               cuarmaOrder order_C, cuarmaOrder order_A, cuarmaOrder order_B,
               cuarma::vector<float> & host_C_float, cuarma::vector<double> & host_C_double,
               cuarma::vector<float> & host_A_float, cuarma::vector<double> & host_A_double,
               cuarma::vector<float> & host_B_float, cuarma::vector<double> & host_B_double
#ifdef CUARMA_WITH_CUDA
               , cuarma::vector<float> & cuda_C_float, cuarma::vector<double> & cuda_C_double
               , cuarma::vector<float> & cuda_A_float, cuarma::vector<double> & cuda_A_double
               , cuarma::vector<float> & cuda_B_float, cuarma::vector<double> & cuda_B_double
#endif
               )
{
  std::cout << "    -> trans-trans: ";
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            order_C, order_A, order_B,
            cuarmaTrans, cuarmaTrans,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

  std::cout << "    -> trans-no:    ";
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            order_C, order_A, order_B,
            cuarmaTrans, cuarmaNoTrans,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

  std::cout << "    -> no-trans:    ";
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            order_C, order_A, order_B,
            cuarmaNoTrans, cuarmaTrans,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

  std::cout << "    -> no-no:       ";
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            order_C, order_A, order_B,
            cuarmaNoTrans, cuarmaNoTrans,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

}


void test_blas(cuarmaBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               cuarma::vector<float> & host_C_float, cuarma::vector<double> & host_C_double,
               cuarma::vector<float> & host_A_float, cuarma::vector<double> & host_A_double,
               cuarma::vector<float> & host_B_float, cuarma::vector<double> & host_B_double
#ifdef CUARMA_WITH_CUDA
               , cuarma::vector<float> & cuda_C_float, cuarma::vector<double> & cuda_C_double
               , cuarma::vector<float> & cuda_A_float, cuarma::vector<double> & cuda_A_double
               , cuarma::vector<float> & cuda_B_float, cuarma::vector<double> & cuda_B_double
#endif
               );

void test_blas(cuarmaBackend my_backend,
               float eps_float, double eps_double,
               std::vector<float> & C_float, std::vector<double> & C_double,
               std::vector<float> & A_float, std::vector<double> & A_double,
               std::vector<float> & B_float, std::vector<double> & B_double,
               cuarma::vector<float> & host_C_float, cuarma::vector<double> & host_C_double,
               cuarma::vector<float> & host_A_float, cuarma::vector<double> & host_A_double,
               cuarma::vector<float> & host_B_float, cuarma::vector<double> & host_B_double
#ifdef CUARMA_WITH_CUDA
               , cuarma::vector<float> & cuda_C_float, cuarma::vector<double> & cuda_C_double
               , cuarma::vector<float> & cuda_A_float, cuarma::vector<double> & cuda_A_double
               , cuarma::vector<float> & cuda_B_float, cuarma::vector<double> & cuda_B_double
#endif
               )
{
  std::cout << "  -> C: row, A: row, B: row" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            cuarmaRowMajor, cuarmaRowMajor, cuarmaRowMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

  std::cout << "  -> C: row, A: row, B: col" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            cuarmaRowMajor, cuarmaRowMajor, cuarmaColumnMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

  std::cout << "  -> C: row, A: col, B: row" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            cuarmaRowMajor, cuarmaColumnMajor, cuarmaRowMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

  std::cout << "  -> C: row, A: col, B: col" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            cuarmaRowMajor, cuarmaColumnMajor, cuarmaColumnMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );


  std::cout << "  -> C: col, A: row, B: row" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            cuarmaColumnMajor, cuarmaRowMajor, cuarmaRowMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

  std::cout << "  -> C: col, A: row, B: col" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            cuarmaColumnMajor, cuarmaRowMajor, cuarmaColumnMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

  std::cout << "  -> C: col, A: col, B: row" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            cuarmaColumnMajor, cuarmaColumnMajor, cuarmaRowMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

  std::cout << "  -> C: col, A: col, B: col" << std::endl;
  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double, A_float, A_double, B_float, B_double,
            cuarmaColumnMajor, cuarmaColumnMajor, cuarmaColumnMajor,
            host_C_float, host_C_double, host_A_float, host_A_double, host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double, cuda_A_float, cuda_A_double, cuda_B_float, cuda_B_double
#endif
            );

}


int main()
{
  cuarma::tools::uniform_random_numbers<float>  randomFloat;
  cuarma::tools::uniform_random_numbers<double> randomDouble;

  std::size_t size  = 500*500;
  float  eps_float  = 1e-5f;
  double eps_double = 1e-12;

  std::vector<float> C_float(size);
  std::vector<float> A_float(size);
  std::vector<float> B_float(size);

  std::vector<double> C_double(size);
  std::vector<double> A_double(size);
  std::vector<double> B_double(size);

  // fill with random data:

  for (std::size_t i = 0; i < size; ++i)
  {
    C_float[i] = 0.5f + 0.1f * randomFloat();
    A_float[i] = 0.5f + 0.1f * randomFloat();
    B_float[i] = 0.5f + 0.1f * randomFloat();

    C_double[i] = 0.5 + 0.2 * randomDouble();
    A_double[i] = 0.5 + 0.2 * randomDouble();
    B_double[i] = 0.5 + 0.2 * randomDouble();
  }


  // Host setup
  cuarmaBackend my_backend;
  cuarmaBackendCreate(&my_backend);

  cuarma::vector<float> host_C_float(size, cuarma::context(cuarma::MAIN_MEMORY));  cuarma::copy(C_float, host_C_float);
  cuarma::vector<float> host_A_float(size, cuarma::context(cuarma::MAIN_MEMORY));  cuarma::copy(A_float, host_A_float);
  cuarma::vector<float> host_B_float(size, cuarma::context(cuarma::MAIN_MEMORY));  cuarma::copy(B_float, host_B_float);

  cuarma::vector<double> host_C_double(size, cuarma::context(cuarma::MAIN_MEMORY));  cuarma::copy(C_double, host_C_double);
  cuarma::vector<double> host_A_double(size, cuarma::context(cuarma::MAIN_MEMORY));  cuarma::copy(A_double, host_A_double);
  cuarma::vector<double> host_B_double(size, cuarma::context(cuarma::MAIN_MEMORY));  cuarma::copy(B_double, host_B_double);

  // CUDA setup
#ifdef CUARMA_WITH_CUDA
  cuarma::vector<float> cuda_C_float(size, cuarma::context(cuarma::CUDA_MEMORY));  cuarma::copy(C_float, cuda_C_float);
  cuarma::vector<float> cuda_A_float(size, cuarma::context(cuarma::CUDA_MEMORY));  cuarma::copy(A_float, cuda_A_float);
  cuarma::vector<float> cuda_B_float(size, cuarma::context(cuarma::CUDA_MEMORY));  cuarma::copy(B_float, cuda_B_float);

  cuarma::vector<double> cuda_C_double(size, cuarma::context(cuarma::CUDA_MEMORY));  cuarma::copy(C_double, cuda_C_double);
  cuarma::vector<double> cuda_A_double(size, cuarma::context(cuarma::CUDA_MEMORY));  cuarma::copy(A_double, cuda_A_double);
  cuarma::vector<double> cuda_B_double(size, cuarma::context(cuarma::CUDA_MEMORY));  cuarma::copy(B_double, cuda_B_double);
#endif

  // consistency checks:
  check(C_float, host_C_float, eps_float);
  check(A_float, host_A_float, eps_float);
  check(B_float, host_B_float, eps_float);

  check(C_double, host_C_double, eps_double);
  check(A_double, host_A_double, eps_double);
  check(B_double, host_B_double, eps_double);

#ifdef CUARMA_WITH_CUDA
  check(C_float, cuda_C_float, eps_float);
  check(A_float, cuda_A_float, eps_float);
  check(B_float, cuda_B_float, eps_float);

  check(C_double, cuda_C_double, eps_double);
  check(A_double, cuda_A_double, eps_double);
  check(B_double, cuda_B_double, eps_double);
#endif

  std::cout << std::endl;

  test_blas(my_backend,
            eps_float, eps_double,
            C_float, C_double,
            A_float, A_double,
            B_float, B_double,
            host_C_float, host_C_double,
            host_A_float, host_A_double,
            host_B_float, host_B_double
#ifdef CUARMA_WITH_CUDA
            , cuda_C_float, cuda_C_double
            , cuda_A_float, cuda_A_double
            , cuda_B_float, cuda_B_double
#endif
            );

  cuarmaBackendDestroy(&my_backend);

  std::cout << std::endl << "!!!! TEST COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

