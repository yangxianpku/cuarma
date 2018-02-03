/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   blas3.cu
 *   @coding UTF-8
 *   @brief  In this tutorial the BLAS level 2 functionality in cuarma is demonstrated.
 *           Operator overloading in C++ is used extensively to provide an intuitive syntax.
 *   @brief  测试：BLAS3 运算示例程序
 */

#ifndef NDEBUG
#define NDEBUG
#endif

#include <iostream>

#include "head_define.h"

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>

#define CUARMA_WITH_UBLAS 1

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/blas/prod.hpp"
#include "cuarma/tools/random.hpp"
#include "cuarma/tools/timer.hpp"

#define MATRIX_SIZE   400

using namespace boost::numeric;

int main()
{
	typedef float ScalarType;
	cuarma::tools::timer timer;
	double exec_time;   // 程序运行时间

	cuarma::tools::uniform_random_numbers<ScalarType> randomNumber;

	// ublas对象，数值初始化
	ublas::matrix<ScalarType>  ublasA(MATRIX_SIZE, MATRIX_SIZE);
	ublas::matrix<ScalarType,  ublas::column_major> ublasB(MATRIX_SIZE,MATRIX_SIZE);   // 列存储矩阵
	ublas::matrix<ScalarType>  ublasC(MATRIX_SIZE, MATRIX_SIZE);
	ublas::matrix<ScalarType>  ublasC1(MATRIX_SIZE, MATRIX_SIZE);

	// 随机数初始化ublasA
	for (unsigned int i = 0; i < ublasA.size1();i++)
	{
		for (unsigned int j = 0; j < ublasA.size2();j++)
		{
			ublasA(i, j) = randomNumber();
		}
	}

	// 随机数初始化ublasB
	for (unsigned int i = 0; i < ublasB.size1(); i++)
	{
		for (unsigned int j = 0; j < ublasB.size2(); j++)
		{
			ublasB(i, j) = randomNumber();
		}
	}

	// 声明cuarma矩阵对象
	cuarma::matrix<ScalarType> armaA(MATRIX_SIZE,MATRIX_SIZE);
	cuarma::matrix<ScalarType, cuarma::column_major> armaB(MATRIX_SIZE,MATRIX_SIZE);
	cuarma::matrix<ScalarType> armaC(MATRIX_SIZE,MATRIX_SIZE);

	// uBLAS计算量矩阵乘积
	std::cout << "--- Computing matrix-matrix product using ublas ---" << std::endl;
	timer.start();
	ublasC = ublas::prod(ublasA, ublasB);
	exec_time = timer.get();
	std::cout << " - Execution time: " << exec_time << std::endl;

	std::cout << std::endl << "--- Computing matrix-matrix product on each available compute device using cuarma ---" << std::endl;

	cuarma::copy(ublasA, armaA);
	cuarma::copy(ublasB, armaB);
	armaC = cuarma::blas::prod(armaA, armaB);
	cuarma::backend::finish();

	// computing on GPU
	timer.start();
	armaC = cuarma::blas::prod(armaA, armaB);
	cuarma::backend::finish();
	exec_time = timer.get();
	std::cout << " - Execution time on device (no setup time included): " << exec_time << std::endl;

	// GPU to CPU
	cuarma::copy(armaC, ublasC1);

	// verify the result
	std::cout << " - Checking result... ";
	bool check_ok = true;
	for (std::size_t i = 0; i < ublasA.size1();i++)
	{
		for (std::size_t j = 0; j < ublasA.size2(); j++)
		{
			if (std::fabs(ublasC(i,j)-ublasC1(i,j))/ublasC(i,j) > 1e-4)
			{
				check_ok = false;
				break;
			}
		}
		if (!check_ok)
		{
			break;
		}
	}

	if (check_ok)
		std::cout << "[OK]" << std::endl << std::endl;
	else
		std::cout << "[FAILED]" << std::endl << std::endl;


	std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;
	return EXIT_SUCCESS;
}