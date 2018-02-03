/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   wrap-cuda-buffer.cu
 *   @coding UTF-8
 *   @brief  Use cuarma with user-provided CUDA buffers
 *   @brief  测试：用户提供数据
 */
 
#include <iostream>
#include <cstdlib>
#include <string>

#include <cuda.h>

#include "head_define.h"

#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/matrix_operations.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/prod.hpp"


/** CUDA 核：向量加法 **/
template<typename T>
__global__ void my_inplace_add_kernel(T * vec1, T * vec2, unsigned int size)
{
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += gridDim.x * blockDim.x)
      vec1[i] += vec2[i];
}

int main()
{

  // Part 1: Allocate some CUDA memory
  std::size_t size = 10;
  ScalarType *cuda_x;
  ScalarType *cuda_y;

  cudaMalloc(&cuda_x, size * sizeof(ScalarType));
  cudaMalloc(&cuda_y, size * sizeof(ScalarType));

  // Initialize with data
  std::vector<ScalarType> host_x(size, 1.0);
  std::vector<ScalarType> host_y(size, 2.0);

  cudaMemcpy(cuda_x, &(host_x[0]), size * sizeof(ScalarType), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_y, &(host_y[0]), size * sizeof(ScalarType), cudaMemcpyHostToDevice);

  // run kernel
  my_inplace_add_kernel<<<128, 128>>>(cuda_x, cuda_y, static_cast<unsigned int>(1000));

  // copy result back
  std::vector<ScalarType> result_cuda(size);
  cudaMemcpy(&(result_cuda[0]), cuda_x, size * sizeof(ScalarType), cudaMemcpyDeviceToHost);

  std::cout << "Result with CUDA (native): ";
  for (std::size_t i=0; i<size; ++i)
    std::cout << result_cuda[i] << " ";
  std::cout << std::endl;

  // Part 2: Now do the same within cuarma
  // wrap the existing CUDA buffers inside cuarma vectors
  cuarma::vector<ScalarType> arma_vec1(cuda_x, cuarma::CUDA_MEMORY, size); // Second parameter specifies that this is CUDA memory rather than host memory
  cuarma::vector<ScalarType> arma_vec2(cuda_y, cuarma::CUDA_MEMORY, size); // Second parameter specifies that this is CUDA memory rather than host memory

  // reset values to 0 and 1, respectively
  arma_vec1 = cuarma::scalar_vector<ScalarType>(size, ScalarType(1.0));
  arma_vec2 = cuarma::scalar_vector<ScalarType>(size, ScalarType(2.0));

  arma_vec1 += arma_vec2;

  std::cout << "Result with cuarma: " << arma_vec1 << std::endl;

  // cuarma does not automatically free your buffers (you're still the owner), so don't forget to clean up :-)
  cudaFree(cuda_x);
  cudaFree(cuda_y);

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

