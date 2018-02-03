/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   libcuarma.cu
 *   @coding UTF-8
 *   @brief  In this example we show how one can directly interface the cuarma BLAS-like shared library.
 *   For simplicity, C++ is used as a host language, but one may also use C or any other language which is able to call C functions.
 *   @brief  测试：BLAS like 测试
 */


#include <iostream>
#include <vector>

#include "head_define.h"
#include "cuarma.hpp"
#include "cuarma/vector.hpp"


int main()
{
  std::size_t size = 10;

  cuarmaInt half_size = static_cast<cuarmaInt>(size / 2);


  /**
  * Before we start we need to create a backend.
  * This allows one later to specify OpenCL command queues, CPU threads, or CUDA streams while preserving common interfaces.
  **/
  cuarmaBackend my_backend;
  cuarmaBackendCreate(&my_backend);


  /**
  *  <h2>Host-based Execution</h2>
  *  We use the host to swap all odd entries of x (all ones) with all even entries in y (all twos):
  **/
  cuarma::vector<float> host_x = cuarma::scalar_vector<float>(size, 1.0, cuarma::context(cuarma::MAIN_MEMORY));
  cuarma::vector<float> host_y = cuarma::scalar_vector<float>(size, 2.0, cuarma::context(cuarma::MAIN_MEMORY));

  cuarmaHostSswap(my_backend, half_size,cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_x), 1, 2,
                  cuarma::blas::host_based::detail::extract_raw_pointer<float>(host_y), 0, 2);

  std::cout << " --- Host ---" << std::endl;
  std::cout << "host_x: " << host_x << std::endl;
  std::cout << "host_y: " << host_y << std::endl;

  /**
  *  <h2>CUDA-based Execution</h2>
  *  We use CUDA to swap all even entries in x (all ones) with all odd entries in y (all twos)
  **/
#ifdef CUARMA_WITH_CUDA
  cuarma::vector<float> cuda_x = cuarma::scalar_vector<float>(size, 1.0, cuarma::context(cuarma::CUDA_MEMORY));
  cuarma::vector<float> cuda_y = cuarma::scalar_vector<float>(size, 2.0, cuarma::context(cuarma::CUDA_MEMORY));

  cuarmaCUDASswap(my_backend, half_size, cuarma::cuda_arg(cuda_x), 0, 2,cuarma::cuda_arg(cuda_y), 1, 2);

  std::cout << " --- CUDA ---" << std::endl;
  std::cout << "cuda_x: " << cuda_x << std::endl;
  std::cout << "cuda_y: " << cuda_y << std::endl;
#endif

  /** The last step is to clean up by destroying the backend: **/
  cuarmaBackendDestroy(&my_backend);

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

