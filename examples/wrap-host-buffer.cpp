/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   wrap-host-buffer.cu
 *   @coding UTF-8
 *   @brief  This tutorial shows how cuarma can be used to wrap user-provided memory buffers allocated on the host.
 *           The benefit of such a wrapper is that the algorithms in cuarma can directly be run without pre- or postprocessing the data.
 *   @brief  测试：CPU数据包装
 */


#include <iostream>
#include <cstdlib>
#include <string>

#include "cuarma/vector.hpp"


int main()
{
  typedef double ScalarType;
  std::size_t size = 10;

  /** host 分配内存 **/
  std::vector<ScalarType> host_x(size, 1.0);
  std::vector<ScalarType> host_y(size, 2.0);

  std::cout << "Result on host: ";
  for (std::size_t i=0; i<size; ++i)
    std::cout << host_x[i] + host_y[i] << " ";
  std::cout << std::endl;

  /**
  *   <h2>Part 2: Now do the same computations within cuarma</h2>
  **/

  // wrap host buffer within cuarma vectors:
  cuarma::vector<ScalarType> arma_vec1(&(host_x[0]), cuarma::MAIN_MEMORY, size); // Second parameter specifies that this is host memory rather than CUDA memory
  cuarma::vector<ScalarType> arma_vec2(&(host_y[0]), cuarma::MAIN_MEMORY, size); // Second parameter specifies that this is host memory rather than CUDA memory

  arma_vec1 += arma_vec2;

  std::cout << "Result with cuarma: " << arma_vec1 << std::endl;

  std::cout << "Data in STL-vector: ";
  for (std::size_t i=0; i<size; ++i)
    std::cout << host_x[i] << " ";
  std::cout << std::endl;

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

