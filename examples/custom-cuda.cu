/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   custom-cuda.cu
 *   @coding UTF-8
 *   @brief  This tutorial shows how you can use your own CUDA buffers and CUDA kernels with cuarma.
 *           We demonstrate this for simple vector and matrix-vector operations.
 *   @brief  测试：CUDA 矩阵-向量操作
 */
 
#include <iostream>
#include <string>

#ifndef CUARMA_WITH_CUDA
  #define CUARMA_WITH_CUDA
#endif

#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/matrix_operations.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/prod.hpp"


// MATLAB notation this is 'result = v1 .* v2
template<typename NumericT>
__global__ void elementwise_multiplication(const NumericT * vec1,const NumericT * vec2, NumericT * result,unsigned int size)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;i < size; i += gridDim.x * blockDim.x)
    result[i] = vec1[i] * vec2[i];
}


int main()
{
  typedef float       NumericType;

  std::size_t N = 5;

  NumericType * device_vec1;
  NumericType * device_vec2;
  NumericType * device_result;
  NumericType * device_A;

  cudaMalloc(&device_vec1,   N * sizeof(NumericType));
  cudaMalloc(&device_vec2,   N * sizeof(NumericType));
  cudaMalloc(&device_result, N * sizeof(NumericType));
  cudaMalloc(&device_A,  N * N * sizeof(NumericType));

  // fill vectors and matrix with data:
  std::vector<NumericType> temp(N);
  for (std::size_t i=0; i<temp.size(); ++i)
    temp[i] = NumericType(i);
  cudaMemcpy(device_vec1, &(temp[0]), temp.size() * sizeof(NumericType), cudaMemcpyHostToDevice);

  for (std::size_t i=0; i<temp.size(); ++i)
    temp[i] = NumericType(2*i);
  cudaMemcpy(device_vec2, &(temp[0]), temp.size() * sizeof(NumericType), cudaMemcpyHostToDevice);

  temp.resize(N*N);
  for (std::size_t i=0; i<temp.size(); ++i)
    temp[i] = NumericType(i);
  cudaMemcpy(device_A, &(temp[0]), temp.size() * sizeof(NumericType), cudaMemcpyHostToDevice);


  // Part 2: Reuse Custom CUDA buffers with cuarma
  cuarma::vector<NumericType> arma_vec1(device_vec1, cuarma::CUDA_MEMORY, N);
  cuarma::vector<NumericType> arma_vec2(device_vec2, cuarma::CUDA_MEMORY, N);
  cuarma::vector<NumericType> arma_result(device_result, cuarma::CUDA_MEMORY, N);

  std::cout << "Standard vector operations within cuarma:" << std::endl;
  arma_result = NumericType(3.1415) * arma_vec1 + arma_vec2;

  std::cout << "vec1  : " << arma_vec1 << std::endl;
  std::cout << "vec2  : " << arma_vec2 << std::endl;
  std::cout << "result: " << arma_result << std::endl;

  /**
  * We can also reuse the existing elementwise_prod kernel.
  * Therefore, we first have to make the existing program known to cuarma
  * For more details on the three lines, see tutorial 'custom-kernels'
  **/
  std::cout << "Using existing kernel within cuarma:" << std::endl;
  elementwise_multiplication<<<128, 128>>>(cuarma::cuda_arg(arma_vec1),cuarma::cuda_arg(arma_vec2),cuarma::cuda_arg(arma_result), N);

  std::cout << "vec1  : " << arma_vec1 << std::endl;
  std::cout << "vec2  : " << arma_vec2 << std::endl;
  std::cout << "result: " << arma_result << std::endl;


  /**
  * Since a linear piece of memory can be interpreted in several ways,
  * we will now create a 5x5 row-major matrix out of the linear memory in device_A
  * The entries in arma_vec2 and arma_result are used to carry out matrix-vector products:
  **/
  cuarma::matrix<NumericType> arma_matrix(device_A, 
	                                       cuarma::CUDA_MEMORY,
                                           N, // number of rows.
                                           N);// number of colums.

  arma_result = cuarma::blas::prod(arma_matrix, arma_vec2);

  std::cout << "result of matrix-vector product: ";
  std::cout << arma_result << std::endl;

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

