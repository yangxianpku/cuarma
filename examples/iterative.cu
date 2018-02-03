/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   iterative.cu
 *   @coding UTF-8
 *   @brief  This tutorial explains the use of iterative solvers in cuarma.
 *   @brief  测试：迭代法求解线性代数方程
 */
 
#include <iostream>

#ifndef NDEBUG
 #define BOOST_UBLAS_NDEBUG
#endif

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

#define CUARMA_WITH_UBLAS 1

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/compressed_matrix.hpp"
#include "cuarma/coordinate_matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/ilu.hpp"
#include "cuarma/blas/jacobi_precond.hpp"
#include "cuarma/blas/cg.hpp"
#include "cuarma/blas/bicgstab.hpp"
#include "cuarma/blas/gmres.hpp"
#include "cuarma/io/matrix_market.hpp"
#include "vector-io.hpp"

using namespace boost::numeric;

int main()
{
  // 新建ublas对象
  ublas::vector<ScalarType> rhs;
  ublas::vector<ScalarType> ref_result;
  ublas::vector<ScalarType> result;
  ublas::compressed_matrix<ScalarType> ublas_matrix;

  // 从文件读取矩阵值
  if (!cuarma::io::read_matrix_market_file(ublas_matrix, "data/mat65k.mtx"))
  {
    std::cout << "Error reading Matrix file" << std::endl;
    return EXIT_FAILURE;
  }

  // 读取右端向量
  if (!readVectorFromFile("data/rhs65025.txt", rhs))
  {
    std::cout << "Error reading RHS file" << std::endl;
    return EXIT_FAILURE;
  }

  // 读取结果
  if (!readVectorFromFile("data/result65025.txt", ref_result))
  {
    std::cout << "Error reading Result file" << std::endl;
    return EXIT_FAILURE;
  }

  // 新建cuarma对象
  std::size_t arma_size = rhs.size();
  cuarma::compressed_matrix<ScalarType> arma_compressed_matrix;
  cuarma::coordinate_matrix<ScalarType> arma_coordinate_matrix;
  cuarma::vector<ScalarType> arma_rhs(arma_size);
  cuarma::vector<ScalarType> arma_result(arma_size);
  cuarma::vector<ScalarType> arma_ref_result(arma_size);

  // 数据拷贝
  cuarma::copy(rhs.begin(), rhs.end(), arma_rhs.begin());
  cuarma::copy(ref_result.begin(), ref_result.end(), arma_ref_result.begin());

  // ublas-matrix to GPU CPU端数据拷贝到GPU
  cuarma::copy(ublas_matrix, arma_compressed_matrix);

  // 使用std::vector< std::map< unsigned int, ScalarType> > 创建STL矩阵，然后拷贝到uBLAS矩阵和cuarma矩阵中
  std::vector< std::map< unsigned int, ScalarType> > stl_matrix(rhs.size());
  for (ublas::compressed_matrix<ScalarType>::iterator1 iter1 = ublas_matrix.begin1(); iter1 != ublas_matrix.end1();++iter1)
  {
    for (ublas::compressed_matrix<ScalarType>::iterator2 iter2 = iter1.begin();iter2 != iter1.end();++iter2)
         stl_matrix[iter2.index1()][static_cast<unsigned int>(iter2.index2())] = *iter2;
  }

  // 声明stl_rhs和stl_result右端向量
  std::vector<ScalarType> stl_rhs(rhs.size()), stl_result(result.size());

  std::copy(rhs.begin(), rhs.end(), stl_rhs.begin());
  cuarma::copy(stl_matrix, arma_coordinate_matrix);
  cuarma::copy(arma_coordinate_matrix, stl_matrix);

  // 为uBLAS对象设置ILUT预处理器
  std::cout << "Setting up preconditioners for uBLAS-matrix..." << std::endl;
  cuarma::blas::ilut_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilut(ublas_matrix, cuarma::blas::ilut_tag());
  cuarma::blas::ilu0_precond< ublas::compressed_matrix<ScalarType> >    ublas_ilu0(ublas_matrix, cuarma::blas::ilu0_tag());
  cuarma::blas::block_ilu_precond< ublas::compressed_matrix<ScalarType>, cuarma::blas::ilu0_tag>  ublas_block_ilu0(ublas_matrix, cuarma::blas::ilu0_tag());

  // 为cuarma对象设置ILUT预处理器
  std::cout << "Setting up preconditioners for cuarma-matrix..." << std::endl;
  cuarma::blas::ilut_precond< cuarma::compressed_matrix<ScalarType> > arma_ilut(arma_compressed_matrix, cuarma::blas::ilut_tag());
  cuarma::blas::ilu0_precond< cuarma::compressed_matrix<ScalarType> > arma_ilu0(arma_compressed_matrix, cuarma::blas::ilu0_tag());
  cuarma::blas::block_ilu_precond< cuarma::compressed_matrix<ScalarType>, cuarma::blas::ilu0_tag> arma_block_ilu0(arma_compressed_matrix, cuarma::blas::ilu0_tag());

  // 设置Jacobi预处理器
  cuarma::blas::jacobi_precond< ublas::compressed_matrix<ScalarType> >    ublas_jacobi(ublas_matrix, cuarma::blas::jacobi_tag());
  cuarma::blas::jacobi_precond< cuarma::compressed_matrix<ScalarType> > arma_jacobi(arma_compressed_matrix, cuarma::blas::jacobi_tag());

  // 共轭梯度法求解器
  std::cout << "----- CG Method -----" << std::endl;

  // 对uBLAS对象使用CG法进行求解，误差容限为1e-6，最大迭代次数为20
  result = cuarma::blas::solve(ublas_matrix, rhs, cuarma::blas::cg_tag());
  result = cuarma::blas::solve(ublas_matrix, rhs, cuarma::blas::cg_tag(1e-6, 20), ublas_ilut);
  result = cuarma::blas::solve(ublas_matrix, rhs, cuarma::blas::cg_tag(1e-6, 20), ublas_jacobi);

  arma_result = cuarma::blas::solve(arma_compressed_matrix, arma_rhs, cuarma::blas::cg_tag());
  arma_result = cuarma::blas::solve(arma_compressed_matrix, arma_rhs, cuarma::blas::cg_tag(1e-6, 20), arma_ilut);
  arma_result = cuarma::blas::solve(arma_compressed_matrix, arma_rhs, cuarma::blas::cg_tag(1e-6, 20), arma_jacobi);

  stl_result = cuarma::blas::solve(stl_matrix, stl_rhs, cuarma::blas::cg_tag());
  stl_result = cuarma::blas::solve(stl_matrix, stl_rhs, cuarma::blas::cg_tag(1e-6, 20), arma_ilut);
  stl_result = cuarma::blas::solve(stl_matrix, stl_rhs, cuarma::blas::cg_tag(1e-6, 20), arma_jacobi);


  // 稳定双共轭梯度法求解器
  std::cout << "----- BiCGStab Method -----" << std::endl;
  result = cuarma::blas::solve(ublas_matrix, rhs, cuarma::blas::bicgstab_tag());         
  result = cuarma::blas::solve(ublas_matrix, rhs, cuarma::blas::bicgstab_tag(1e-6, 20), ublas_ilut); 
  result = cuarma::blas::solve(ublas_matrix, rhs, cuarma::blas::bicgstab_tag(1e-6, 20), ublas_jacobi); 

  arma_result = cuarma::blas::solve(arma_compressed_matrix, arma_rhs, cuarma::blas::bicgstab_tag());   
  arma_result = cuarma::blas::solve(arma_compressed_matrix, arma_rhs, cuarma::blas::bicgstab_tag(1e-6, 20), arma_ilut); 
  arma_result = cuarma::blas::solve(arma_compressed_matrix, arma_rhs, cuarma::blas::bicgstab_tag(1e-6, 20), arma_jacobi); 


  stl_result = cuarma::blas::solve(stl_matrix, stl_rhs, cuarma::blas::bicgstab_tag());
  stl_result = cuarma::blas::solve(stl_matrix, stl_rhs, cuarma::blas::bicgstab_tag(1e-6, 20), arma_ilut);
  stl_result = cuarma::blas::solve(stl_matrix, stl_rhs, cuarma::blas::bicgstab_tag(1e-6, 20), arma_jacobi);

  // GMRES 求解器
  std::cout << "----- GMRES Method -----" << std::endl;
  result = cuarma::blas::solve(ublas_matrix, rhs, cuarma::blas::gmres_tag());   
  result = cuarma::blas::solve(ublas_matrix, rhs, cuarma::blas::gmres_tag(1e-6, 20), ublas_ilut);
  result = cuarma::blas::solve(ublas_matrix, rhs, cuarma::blas::gmres_tag(1e-6, 20), ublas_jacobi);

  arma_result = cuarma::blas::solve(arma_compressed_matrix, arma_rhs, cuarma::blas::gmres_tag());   
  arma_result = cuarma::blas::solve(arma_compressed_matrix, arma_rhs, cuarma::blas::gmres_tag(1e-6, 20), arma_ilut);
  arma_result = cuarma::blas::solve(arma_compressed_matrix, arma_rhs, cuarma::blas::gmres_tag(1e-6, 20), arma_jacobi);

  stl_result = cuarma::blas::solve(stl_matrix, stl_rhs, cuarma::blas::gmres_tag());
  stl_result = cuarma::blas::solve(stl_matrix, stl_rhs, cuarma::blas::gmres_tag(1e-6, 20), arma_ilut);
  stl_result = cuarma::blas::solve(stl_matrix, stl_rhs, cuarma::blas::gmres_tag(1e-6, 20), arma_jacobi);

  std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;

  return EXIT_SUCCESS;
}

