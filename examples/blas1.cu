/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   blas1.cu
 *   @coding UTF-8
 *   @brief  This tutorial shows how the BLAS level 1 functionality available in cuarma can be used.
 *           Operator overloading in C++ is used extensively to provide an intuitive syntax.
 *   @brief  ���ԣ�BLAS1 ����ʾ������
 */

#include <iostream>

#include "head_define.h"

#include "cuarma/scalar.hpp"
#include "cuarma/vector.hpp"
#include "cuarma/blas/inner_prod.hpp"
#include "cuarma/blas/norm_1.hpp"
#include "cuarma/blas/norm_2.hpp"
#include "cuarma/blas/norm_inf.hpp"
#include "cuarma/tools/random.hpp"

int main()
{

	typedef float ScalarType;

	cuarma::tools::uniform_random_numbers<ScalarType> randomNumber;

	ScalarType s1 = ScalarType(3.1415926);      // CPU scalar
	ScalarType s2 = ScalarType(2.71763);
	ScalarType s3 = ScalarType(42.0);

	cuarma::scalar<ScalarType> arma_s1;         // GPU scalar
	cuarma::scalar<ScalarType> arma_s2 = ScalarType(1.0);
	cuarma::scalar<ScalarType> arma_s3 = ScalarType(1.0);

	// CPU scalars can be transparently assigned to GPU scalars and vice versa:
	// CPU ����ͨ����������ؿ���ֱ�Ӹ�ֵ��GPU��������֮��Ȼ
	std::cout << "Copy a few Scalars ... "<< std::endl;

	arma_s1 = s1;
	arma_s2 = s2;
	arma_s3 = s3;

	/**
	* Operations between GPU scalars work just as for CPU scalars:
	* (Note that such single compute kernels on the GPU are considerably slower than on the CPU)
	**/
	std::cout << "Manipulating a few scalars..." << std::endl;
	std::cout << "operator +=" << std::endl;

	s1 += s2;
	arma_s1 += arma_s2; 

	std::cout << "operator +=" << std::endl;

	s1 -= s2;
	arma_s1 -= arma_s2;

	std::cout << "operator +=" << std::endl;

	s1 *= s2;
	arma_s1 *= arma_s2;

	std::cout << "operator +=" << std::endl;

	s1 /= s2;
	arma_s1 /= arma_s2;

	std::cout << "operator +" << std::endl;

	s1 = s2 + s3;
	arma_s1 = arma_s2 + arma_s3;

	std::cout << "multiple operators" << std::endl;    // ������

	s1 = s2 + s3 * s2 - s3 / s1;
	arma_s1 = arma_s2 + arma_s3 * arma_s2 - arma_s3 / arma_s1;

	std::cout << "mixed operators" << std::endl;       // �������
	arma_s1 = s1 * arma_s2 + s3 - arma_s3;

	std::cout << "CPU scalar s3: " << s3 << std::endl;
	std::cout << "GPU scalar arma_s3: " << arma_s3 << std::endl;

	//==================================================================================
	// ��������

	std::vector<ScalarType> std_vec1(10);
	std::vector<ScalarType> std_vec2(10);
	ScalarType            plain_vec3[10];   // array

	cuarma::vector<ScalarType> arma_vec1(10);
	cuarma::vector<ScalarType> arma_vec2(10);
	cuarma::vector<ScalarType> arma_vec3(10);

	// ���������CPU����
	for (unsigned int i = 0; i < 10;i++)
	{
		std_vec1[i] = randomNumber();
		arma_vec2(i) = randomNumber();    // GPU�ϲ�������������Ǳ�CPU�����ܶ�
		plain_vec3[i] = randomNumber();
	}

	// ��CPU����ֵ���Ƶ�GPU������֮��Ȼ
	cuarma::copy(std_vec1.begin(), std_vec1.end(), arma_vec1.begin());
	cuarma::copy(arma_vec2.begin(), arma_vec2.end(), std_vec2.begin());
	cuarma::copy(arma_vec2, std_vec2);                           // ֱ�Ӹ���
	cuarma::copy(arma_vec2.begin(),arma_vec2.end(),plain_vec3);  // ��������ͨ����

	//std::cout << arma_vec1 << std::endl;
	//std::cout << arma_vec2 << std::endl;
	//std::cout << arma_vec3 << std::endl;

	//  ���ֿ���
	cuarma::copy(std_vec1.begin() + 4,  std_vec1.begin() + 8, arma_vec1.begin() + 4);    // cpu to gpu
	cuarma::copy(arma_vec1.begin() + 4, arma_vec1.begin() + 8, arma_vec2.begin() + 1);   // gpu to gpu
	cuarma::copy(arma_vec1.begin() + 4, arma_vec1.begin() + 8, std_vec1.begin() + 1);    // gpu to cpu

	//std::cout << arma_vec1 << std::endl;
	//std::cout << arma_vec2 << std::endl;
	//std::cout << arma_vec3 << std::endl;

	// �ڻ�����,�����Զ�����
	arma_s1 = cuarma::blas::inner_prod(arma_vec1, arma_vec2);
	s1 = cuarma::blas::inner_prod(arma_vec1, arma_vec2);
	s2 = cuarma::blas::inner_prod(std_vec1, std_vec2);

	// ��������,�����Զ�����
	s1 = cuarma::blas::norm_1(arma_vec1);
	arma_s2 = cuarma::blas::norm_2(arma_vec2);
	s3 = cuarma::blas::norm_inf(arma_vec3);


	// ������֮���ƽ����ת
	cuarma::blas::plane_rotation(arma_vec1, arma_vec2, 1.1f, 2.3f);

	// ��������
	arma_vec1 = arma_vec2 + arma_vec3;   // ֱ�ӽ������
	arma_vec1 = arma_s1 * arma_vec2 / arma_s3;
	arma_vec1 = arma_vec2 / arma_s3 + arma_s2*(arma_vec1-arma_vec2*arma_s2);

	// ��������������ֵ
	cuarma::swap(arma_vec1, arma_vec2);
	cuarma::fast_swap(arma_vec1, arma_vec2);

	arma_vec1.clear();
	arma_vec2.clear();
	arma_vec3.clear();

	return EXIT_SUCCESS;
}