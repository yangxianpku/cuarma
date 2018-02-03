/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   fft.cu
 *   @coding UTF-8
 *   @brief  This tutorial showcasts FFT functionality.
 *   @brief  测试：快速傅里叶变换运算示例程序
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fstream>

#include "head_define.h"

#include "cuarma/vector.hpp"
#include "cuarma/matrix.hpp"
#include "cuarma/fft.hpp"
#include "cuarma/blas/fft_operations.hpp"


int main()
{
	typedef float ScalarType;

	cuarma::vector<ScalarType> input_vec(16);
	cuarma::vector<ScalarType> output_vec(16);
	cuarma::vector<ScalarType> input_vec2(16);

	cuarma::matrix<ScalarType> m(4, 8);
	cuarma::matrix<ScalarType> o(4, 8);

	// m矩阵填充
	for (std::size_t i = 0; i < m.size1(); i++)
	for (std::size_t s = 0; s < m.size2(); s++)
		m(i, s) = ScalarType((i + s) / 2);

	// 向量填充
	for (std::size_t i = 0; i < input_vec.size(); ++i)
	{
		if (i % 2 == 0)
		{
			input_vec(i) = ScalarType(i / 2); // even indices represent real part
			input_vec2(i) = ScalarType(i / 2);
		}
		else
			input_vec(i) = 0;                // odd indices represent imaginary part
	}


	std::cout << "Computing FFT Matrix" << std::endl;
	std::cout << "m: " << m << std::endl;
	std::cout << "o: " << o << std::endl;
	cuarma::fft(m, o);  // 变换过后的值存到o矩阵

	std::cout << "Done" << std::endl;
	std::cout << "m: " << m << std::endl;
	std::cout << "o: " << o << std::endl;

	// 复数矩阵转置
	std::cout << "Transpose" << std::endl;
	cuarma::blas::transpose(m, o);
	std::cout << "m: " << m << std::endl;
	std::cout << "o: " << o << std::endl;

	std::cout << "---------------------" << std::endl;

	// Bluestein算法计算FFT,通常更快，但是也相应消耗等多内存
	std::cout << "Computing FFT bluestein" << std::endl;
	cuarma::blas::bluestein(input_vec, output_vec, 0);
	std::cout << "input_vec: "  << input_vec << std::endl;
	std::cout << "output_vec: " << output_vec << std::endl;
	std::cout << "---------------------" << std::endl;


	// 标准radix-FFT方法计算向量FFT 
	std::cout << "Computing FFT " << std::endl;
	cuarma::fft(input_vec, output_vec);
	std::cout << "input_vec: " << input_vec << std::endl;
	std::cout << "output_vec: " << output_vec << std::endl;
	std::cout << "---------------------" << std::endl;


	// 标准radix-FFT方法计算向量FFT逆变换 
	std::cout << "Computing inverse FFT..." << std::endl;
	cuarma::inplace_ifft(output_vec);     // or compute in-place
	std::cout << "input_vec: "  << input_vec << std::endl;
	std::cout << "output_vec: " << output_vec << std::endl;
	std::cout << "---------------------" << std::endl;


	// 奇数项代表实部，偶数项表示虚部
	std::cout << "Computing real to complex..." << std::endl;
	std::cout << "input_vec: " << input_vec << std::endl;
	cuarma::blas::real_to_complex(input_vec, output_vec, input_vec.size() / 2); // or compute in-place
	std::cout << "output_vec: " << output_vec << std::endl;
	std::cout << "---------------------" << std::endl;

	std::cout << "Computing complex to real..." << std::endl;
	std::cout << "input_vec: " << input_vec << std::endl;
	cuarma::blas::complex_to_real(input_vec, output_vec, input_vec.size() / 2); // or compute in-place
	std::cout << "output_vec: " << output_vec << std::endl;
	std::cout << "---------------------" << std::endl;


	//  两个复向量按元素乘积
	std::cout << "Computing multiply complex" << std::endl;
	std::cout << "input_vec: " << input_vec << std::endl;
	std::cout << "input_vec2: " << input_vec2 << std::endl;
	cuarma::blas::multiply_complex(input_vec, input_vec2, output_vec);
	std::cout << "Done" << std::endl;
	std::cout << "output_vec: " << output_vec << std::endl;
	std::cout << "---------------------" << std::endl;

	std::cout << "!!!! TUTORIAL COMPLETED SUCCESSFULLY !!!!" << std::endl;


}
