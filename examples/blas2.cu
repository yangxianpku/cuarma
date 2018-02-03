/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   blas2.cu
 *   @coding UTF-8
 *   @brief  In this tutorial the BLAS level 2 functionality in cuarma is demonstrated.
 *           Operator overloading in C++ is used extensively to provide an intuitive syntax.
 *   @brief  测试：BLAS2 运算示例程序
 */

#include <iostream>

#include "head_define.h"

#include <boost/serialization/array_wrapper.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/lu.hpp"

// Must be set if you want to use cuarma algorithms on ublas objects
#define CUARMA_WITH_UBLAS 1

#include "host_defines.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/direct_solve.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/lu.hpp"
#include "cuarma/tools/random.hpp"

using namespace boost::numeric;

int main()
{
	typedef float ScalarType;
	cuarma::tools::uniform_random_numbers<ScalarType> randomNumer;

	// 使用uBLAS产生一些数据，以便后边给cuarmacuarma象使用
	ublas::vector<ScalarType> rhs(12);     // CPU向量
	for (unsigned int i = 0; i < rhs.size();i++)
	{
		rhs(i) = randomNumer();
	}
	ublas::vector<ScalarType> rhs2 = rhs;   // 可以直接赋值
	ublas::vector<ScalarType> rhs_trans = rhs;

	ublas::vector<ScalarType> result = ublas::zero_vector<ScalarType>(10);
	ublas::vector<ScalarType> result2 = result;

	rhs_trans.resize(result.size(),true);

	ublas::vector<ScalarType> result_trans = ublas::zero_vector<ScalarType>(rhs.size());

	ublas::matrix<ScalarType> matrix(result.size(), rhs.size());  // 声明uBLAS矩阵

	// 矩阵填充值
	for (unsigned int i = 0; i < matrix.size1(); ++i)
	for (unsigned int j = 0; j < matrix.size2(); ++j)
		matrix(i, j) = randomNumer();

	// STL类型
	std::vector< ScalarType > stl_result(result.size());
	std::vector< ScalarType > stl_rhs(rhs.size());
	std::vector< std::vector<ScalarType> > stl_matrix(result.size());   // 两重vector表示矩阵
	for (unsigned int i = 0; i < result.size(); ++i)
	{
		stl_matrix[i].resize(rhs.size());
		for (unsigned int j = 0; j < matrix.size2(); ++j)
		{
			stl_rhs[j] = rhs[j];
			stl_matrix[i][j] = matrix(i, j);
		}
	}

	// 新建cuarma对象
	cuarma::vector<ScalarType> arma_rhs(rhs.size());
	cuarma::vector<ScalarType> arma_result(result.size());
	cuarma::matrix<ScalarType> arma_matrix(result.size(),rhs.size());
	cuarma::matrix<ScalarType> arma_matrix2(result.size(), rhs.size());

	cuarma::copy(rhs.begin(), rhs.end(),arma_rhs.begin());
	cuarma::copy(matrix, arma_matrix);

	// 一些基本操作
	arma_matrix2 = arma_matrix;
	arma_matrix2 += arma_matrix;
	arma_matrix2 -= arma_matrix;
	arma_matrix2 = arma_matrix2 + arma_matrix;
	arma_matrix2 = arma_matrix2 - arma_matrix;

	cuarma::scalar<ScalarType> arma_3(3.0);
	arma_matrix2 *= ScalarType(3.0);   // CPU标量
	arma_matrix2 /= ScalarType(3.0);

	arma_matrix2 *= arma_3;           // GPU标量
	arma_matrix2 /= arma_3;

	// 清掉矩阵值
	arma_matrix.clear();

	// 一些别的值拷贝方式
	cuarma::copy(stl_matrix, arma_matrix);  //  STL vector< vector<> > to cuarma::matrix
	cuarma::copy(arma_matrix, matrix);      //  GPU to CPU matrix
	cuarma::copy(arma_matrix, stl_matrix);  //  GPU to CPU matrix

	// 矩阵-向量操作
	std::cout << "----- Matrix-Vector product -----" << std::endl;
	result = ublas::prod(matrix,rhs);
	stl_result = cuarma::blas::prod(stl_matrix, stl_rhs);      // 自动选择计算场所：CPU端计算
	arma_result = cuarma::blas::prod(arma_matrix, arma_rhs);   // 自动选择计算场所：GPU端计算

	// 矩阵转置-向量操作
	std::cout << "----- Transposed Matrix-Vector product -----" << std::endl;
	result_trans = prod(trans(matrix),rhs_trans);

	cuarma::vector<ScalarType> arma_rhs_trans(rhs_trans.size());
	cuarma::vector<ScalarType> arma_result_trans(result_trans.size());
	cuarma::copy(rhs_trans, arma_rhs_trans);
	arma_result_trans = cuarma::blas::prod(trans(arma_matrix), arma_rhs_trans);   // 直接使用trans进行转置

	// 直接求解器
	/**
	 * In order to demonstrate the direct solvers, we first need to setup suitable square matrices.
	 * This is again achieved by running the setup on the CPU and then copy the data over to cuarma types:
	 **/

	ublas::matrix<ScalarType> tri_matrix(10, 10);             // CPU端生成三角矩阵
	for (std::size_t i = 0; i < tri_matrix.size1(); ++i)      //  size1()表示行
	{
		for (std::size_t j = 0; j < i; ++j)
			tri_matrix(i, j) = 0.0;

		for (std::size_t j = i; j < tri_matrix.size2(); ++j)  //  size2() 表示列
			tri_matrix(i, j) = matrix(i, j);
	}

	cuarma::matrix<ScalarType> arma_tri_matrix = cuarma::identity_matrix<ScalarType>(tri_matrix.size1());
	cuarma::copy(tri_matrix, arma_tri_matrix);               // 值拷贝

	rhs.resize(tri_matrix.size1(),true);
	rhs2.resize(tri_matrix.size1(),true);
	arma_rhs.resize(tri_matrix.size1(), true);

	cuarma::copy(rhs.begin(), rhs.end(), arma_rhs.begin());
	arma_result.resize(10);

	/** Run a triangular solver on the upper triangular part of the matrix: **/
	std::cout << "----- Upper Triangular solve -----" << std::endl;
	result = ublas::solve(tri_matrix, rhs, ublas::upper_tag());    // 上三角
	arma_result = cuarma::blas::solve(arma_tri_matrix, arma_rhs, cuarma::blas::upper_tag());    // GPU求解

	// 原位求解
	ublas::inplace_solve(tri_matrix, rhs, ublas::upper_tag());  // 不要新的变量来存储值
	cuarma::blas::inplace_solve(arma_tri_matrix, arma_rhs, cuarma::blas::upper_tag());

	// LU分解，进行满系统求解
	std::size_t lu_dim = 300;
	ublas::matrix<ScalarType> square_matrix(lu_dim,lu_dim);
	ublas::vector<ScalarType> lu_rhs(lu_dim);
	cuarma::matrix<ScalarType> arma_square_matrix(lu_dim,lu_dim);
	cuarma::vector<ScalarType> arma_lu_rhs(lu_dim);

	// CPU方阵赋予初值
	for (std::size_t i = 0; i < lu_dim; ++i)
	for (std::size_t j = 0; j < lu_dim; ++j)
		square_matrix(i, j) = randomNumer();

	// 让矩阵对角占优
	for (std::size_t j = 0; j < lu_dim; ++j)
	{
		square_matrix(j, j) += ScalarType(10.0);
		lu_rhs(j) = randomNumer();
	}


	cuarma::copy(square_matrix,arma_square_matrix);
	cuarma::copy(lu_rhs, arma_lu_rhs);
	cuarma::blas::lu_factorize(arma_square_matrix);
	cuarma::blas::lu_substitute(arma_square_matrix, arma_lu_rhs);
	cuarma::copy(square_matrix, arma_square_matrix);
	cuarma::copy(lu_rhs, arma_lu_rhs);

	// uBLAS求解
	ublas::lu_factorize(square_matrix);
	ublas::inplace_solve(square_matrix, lu_rhs, ublas::unit_lower_tag());
	ublas::inplace_solve(square_matrix, lu_rhs, ublas::upper_tag());

	// cuarma求解
	cuarma::blas::lu_factorize(arma_square_matrix);
	cuarma::blas::lu_substitute(arma_square_matrix, arma_lu_rhs);


	std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
	return EXIT_SUCCESS;
}