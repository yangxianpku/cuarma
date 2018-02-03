/* =========================================================================
      Copyright (c) 2015-2017, COE of Peking University, Shaoqiang Tang.

                         -----------------
            cuarma - COE of Peking University, Shaoqiang Tang.
                         -----------------

                  Author Email    yangxianpku@pku.edu.cn

         Code Repo   https://github.com/yangxianpku/cuarma

                      License:    MIT (X11) License
============================================================================= */

/**  @file   fft_1d.cu
 *   @coding UTF-8
 *   @brief  Tests the one-dimensional FFT routines.
 *   @brief  ≤‚ ‘£∫“ªŒ¨FFt≤‚ ‘
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>

#include "head_define.h"
#include "head_data.h"

#include "cuarma/blas/host_based/fft_operations.hpp"

#ifdef CUARMA_WITH_CUDA
#include "cuarma/blas/cuda/fft_operations.hpp"
#endif
#include "cuarma/blas/fft_operations.hpp"
#include "cuarma/fft.hpp"

typedef float ScalarType;

const ScalarType EPS = ScalarType(0.06f); //use smaller values in double precision

typedef ScalarType(*test_function_ptr)(std::vector<ScalarType>&, std::vector<ScalarType>&,
	unsigned int, unsigned int, unsigned int);
typedef void(*input_function_ptr)(std::vector<ScalarType>&, std::vector<ScalarType>&,
	unsigned int&, unsigned int&, unsigned int&, const std::string&);

void set_values_struct(std::vector<ScalarType>& input, std::vector<ScalarType>& output,
	unsigned int& rows, unsigned int& cols, unsigned int& batch_size, TestData1D& data);

void set_values_struct(std::vector<ScalarType>& input, std::vector<ScalarType>& output,
	unsigned int& rows, unsigned int& cols, unsigned int& batch_size, TestData1D& data)
{
	unsigned int size = data.col_num * data.batch_num * 2;
	input.resize(size);
	output.resize(size);
	rows = data.row_num;
	cols = data.col_num;
	batch_size = data.batch_num;
	for (unsigned int i = 0; i < size; i++)
	{
		input[i] = data.input[i];
		output[i] = data.output[i];
	}

}

void set_values_struct(std::vector<ScalarType>& input, std::vector<ScalarType>&input2,
	std::vector<ScalarType>& output, unsigned int& rows, unsigned int& cols,
	unsigned int& batch_size, TestData1D& data);

void set_values_struct(std::vector<ScalarType>& input, std::vector<ScalarType>&input2,
	std::vector<ScalarType>& output, unsigned int& rows, unsigned int& cols,
	unsigned int& batch_size, TestData1D& data)
{
	unsigned int size = data.col_num * data.batch_num * 2;
	input.resize(size);
	input2.resize(size);
	output.resize(size);
	rows = data.row_num;
	cols = data.col_num;
	batch_size = data.batch_num;
	for (unsigned int i = 0; i < size; i++)
	{

		input[i] = data.input[i];
		input2[i] = data.output[i];
		output[i] = data.result_multiply[i];
	}
}

void read_vectors_pair(std::vector<ScalarType>& input, std::vector<ScalarType>& output,
	unsigned int& rows, unsigned int& cols, unsigned int& batch_size, const std::string& log_tag);

void read_vectors_pair(std::vector<ScalarType>& input, std::vector<ScalarType>& output,
	unsigned int& rows, unsigned int& cols, unsigned int& batch_size, const std::string& log_tag)
{

	if (log_tag == "fft::direct" || log_tag == "fft::convolve::1" || log_tag == "fft::bluestein::1"
		|| log_tag == "fft::fft_reverse_direct")
		set_values_struct(input, output, rows, cols, batch_size, cufft);

	if (log_tag == "fft:real_to_complex")
		set_values_struct(input, output, rows, cols, batch_size, real_to_complex_data);

	if (log_tag == "fft:complex_to_real")
		set_values_struct(input, output, rows, cols, batch_size, complex_to_real_data);

	if (log_tag == "fft::batch::direct" || log_tag == "fft::batch::radix2")
		set_values_struct(input, output, rows, cols, batch_size, batch_radix);

	if (log_tag == "fft::radix2" || log_tag == "fft::convolve::2" || log_tag == "fft::bluestein::2"
		|| log_tag == "fft::fft_ifft_radix2" || log_tag == "fft::ifft_fft_radix2")
		set_values_struct(input, output, rows, cols, batch_size, radix2_data);

}

void read_vectors_three(std::vector<ScalarType>& input, std::vector<ScalarType>&input2,
	std::vector<ScalarType>& output, unsigned int& rows, unsigned int& cols,
	unsigned int& batch_size, const std::string& log_tag);

void read_vectors_three(std::vector<ScalarType>& input, std::vector<ScalarType>&input2,
	std::vector<ScalarType>& output, unsigned int& rows, unsigned int& cols,
	unsigned int& batch_size, const std::string& log_tag)
{
	if (log_tag == "fft::multiplt::complex")
		set_values_struct(input, input2, output, rows, cols, batch_size, cufft);

}

template<typename ScalarType>
ScalarType diff(std::vector<ScalarType>& vec, std::vector<ScalarType>& ref)
{
	ScalarType df = 0.0;
	ScalarType norm_ref = 0;

	for (std::size_t i = 0; i < vec.size(); i++)
	{
		df = df + pow(vec[i] - ref[i], 2);
		norm_ref += ref[i] * ref[i];
	}

	return sqrt(df / norm_ref);
}

template<typename ScalarType>
ScalarType diff_max(std::vector<ScalarType>& vec, std::vector<ScalarType>& ref)
{
	ScalarType df = 0.0;
	ScalarType mx = 0.0;
	ScalarType norm_max = 0;

	for (std::size_t i = 0; i < vec.size(); i++)
	{
		df = std::max<ScalarType>(std::fabs(ScalarType(vec[i] - ref[i])), df);
		mx = std::max<ScalarType>(std::fabs(ScalarType(vec[i])), mx);

		if (mx > 0)
		{
			if (norm_max < df / mx)
				norm_max = df / mx;
		}
	}

	return norm_max;
}

template<class SCALARTYPE, unsigned int ALIGNMENT>
void copy_to_vector(std::complex<SCALARTYPE> * input_complex,
	cuarma::vector<SCALARTYPE, ALIGNMENT>& in, unsigned int size)
{
	for (unsigned int i = 0; i < size; i += 1)
	{
		in(i * 2) = (SCALARTYPE)std::real(input_complex[i]);
		in(i * 2 + 1) = std::imag(input_complex[i]);
	}
}
template<class SCALARTYPE>
void copy_to_vector(std::complex<SCALARTYPE> * input_complex, SCALARTYPE * in, unsigned int size)
{
#ifdef CUARMA_WITH_OPENMP
#pragma omp parallel for if (size > CUARMA_OPENMP_VECTOR_MIN_SIZE)
#endif
	for (unsigned int i = 0; i < size; i += 1)
	{
		in[i * 2] = (SCALARTYPE)std::real(input_complex[i]);
		in[i * 2 + 1] = std::imag(input_complex[i]);
	}
}
template<class SCALARTYPE>
void copy_to_complex_array(std::complex<SCALARTYPE> * input_complex, const SCALARTYPE * in, int size)
{
#ifdef CUARMA_WITH_OPENMP
#pragma omp parallel for if (size > CUARMA_OPENMP_VECTOR_MIN_SIZE)
#endif
	for (int i = 0; i < size * 2; i += 2)
	{ //change array to complex array
		input_complex[i / 2] = std::complex<SCALARTYPE>(in[i], in[i + 1]);
	}
}

void convolve_ref(std::vector<ScalarType>& in1, std::vector<ScalarType>& in2, std::vector<ScalarType>& out);

void convolve_ref(std::vector<ScalarType>& in1, std::vector<ScalarType>& in2, std::vector<ScalarType>& out)
{
	out.resize(in1.size());
	unsigned int data_size = static_cast<unsigned int>(in1.size()) >> 1;

	for (unsigned int n = 0; n < data_size; n++)
	{
		std::complex<ScalarType> el;
		for (unsigned int k = 0; k < data_size; k++)
		{
			int offset = int(n) - int(k);
			if (offset < 0)
				offset += data_size;
			std::complex<ScalarType> m1(in1[2 * k], in1[2 * k + 1]);
			std::complex<ScalarType> m2(in2[2 * std::size_t(offset)], in2[2 * std::size_t(offset) + 1]);
			el = el + m1 * m2;
		}
		out[2 * n] = el.real();
		out[2 * n + 1] = el.imag();
	}
}

ScalarType fft(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int /*row*/,
	unsigned int /*col*/, unsigned int batch_size);

ScalarType fft(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int /*row*/,
	unsigned int /*col*/, unsigned int batch_size)
{
	cuarma::vector<ScalarType> input(in.size());
	cuarma::vector<ScalarType> output(in.size());

	std::vector<ScalarType> res(in.size());

	cuarma::fast_copy(in, input);

	cuarma::fft(input, output, batch_size);

	cuarma::backend::finish();
	cuarma::fast_copy(output, res);

	return diff_max(res, out);
}

ScalarType direct(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int /*row*/,
	unsigned int /*col*/, unsigned int batch_num);

ScalarType direct(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int /*row*/,
	unsigned int /*col*/, unsigned int batch_num)
{
	cuarma::vector<ScalarType> input(in.size());
	cuarma::vector<ScalarType> output(in.size());

	std::vector<ScalarType> res(in.size());

	cuarma::fast_copy(in, input);

	unsigned int size = (static_cast<unsigned int>(input.size()) >> 1) / batch_num;

	cuarma::blas::direct(input, output, size, size, batch_num);
	cuarma::backend::finish();
	cuarma::fast_copy(output, res);

	return diff_max(res, out);
}

ScalarType bluestein(std::vector<ScalarType>& in, std::vector<ScalarType>& out,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_size);

ScalarType bluestein(std::vector<ScalarType>& in, std::vector<ScalarType>& out,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_size)
{
	cuarma::vector<ScalarType> input(in.size());
	cuarma::vector<ScalarType> output(in.size());

	std::vector<ScalarType> res(in.size());

	cuarma::fast_copy(in, input);

	cuarma::blas::bluestein(input, output, batch_size);

	cuarma::backend::finish();
	cuarma::fast_copy(output, res);

	return diff_max(res, out);
}

ScalarType radix2(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int /*row*/,
	unsigned int /*col*/, unsigned int batch_num);

ScalarType radix2(std::vector<ScalarType>& in, std::vector<ScalarType>& out, unsigned int /*row*/,
	unsigned int /*col*/, unsigned int batch_num)
{
	cuarma::vector<ScalarType> input(in.size());
	cuarma::vector<ScalarType> output(in.size());

	std::vector<ScalarType> res(in.size());

	cuarma::fast_copy(in, input);

	unsigned int size = (static_cast<unsigned int>(input.size()) >> 1) / batch_num;
	cuarma::blas::radix2(input, size, size, batch_num);

	cuarma::backend::finish();
	cuarma::fast_copy(input, res);

	return diff_max(res, out);
}

ScalarType fft_ifft_radix2(std::vector<ScalarType>& in, std::vector<ScalarType>& /*out*/,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_num);

ScalarType fft_ifft_radix2(std::vector<ScalarType>& in, std::vector<ScalarType>& /*out*/,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_num)
{
	cuarma::vector<ScalarType> input(in.size());

	std::vector<ScalarType> res(in.size());

	cuarma::fast_copy(in, input);

	unsigned int size = (static_cast<unsigned int>(input.size()) >> 1) / batch_num;
	cuarma::blas::radix2(input, size, size, batch_num);

	cuarma::inplace_ifft(input);

	cuarma::backend::finish();
	cuarma::fast_copy(input, res);

	return diff_max(res, in);
}

ScalarType ifft_fft_radix2(std::vector<ScalarType>& in, std::vector<ScalarType>& /*out*/,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_num);

ScalarType ifft_fft_radix2(std::vector<ScalarType>& in, std::vector<ScalarType>& /*out*/,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int batch_num)
{
	cuarma::vector<ScalarType> input(in.size());

	std::vector<ScalarType> res(in.size());

	cuarma::fast_copy(in, input);

	cuarma::inplace_ifft(input);

	unsigned int size = (static_cast<unsigned int>(input.size()) >> 1) / batch_num;
	cuarma::blas::radix2(input, size, size, batch_num);

	cuarma::backend::finish();
	cuarma::fast_copy(input, res);

	return diff_max(res, in);
}

ScalarType fft_reverse_direct(std::vector<ScalarType>& in, std::vector<ScalarType>& out,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int /*batch_num*/);

ScalarType fft_reverse_direct(std::vector<ScalarType>& in, std::vector<ScalarType>& out,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int /*batch_num*/)
{
	cuarma::vector<ScalarType> input(in.size());

	cuarma::fast_copy(in, input);

	cuarma::vector<ScalarType> tmp(out.size());
	cuarma::fast_copy(in, tmp);
	cuarma::blas::reverse(tmp);
	cuarma::blas::reverse(tmp);

	cuarma::fast_copy(tmp, out);

	cuarma::backend::finish();

	return diff_max(out, in);
}

ScalarType real_to_complex(std::vector<ScalarType>& in, std::vector<ScalarType>& out,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int /*batch_num*/);

ScalarType real_to_complex(std::vector<ScalarType>& in, std::vector<ScalarType>& out,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int /*batch_num*/)
{

	cuarma::vector<ScalarType> input(in.size());
	cuarma::vector<ScalarType> output(in.size());

	std::vector<ScalarType> res(in.size());
	cuarma::fast_copy(in, input);

	cuarma::blas::real_to_complex(input, output, input.size() / 2);

	cuarma::backend::finish();
	cuarma::fast_copy(output, res);

	return diff_max(res, out);

}


ScalarType complex_to_real(std::vector<ScalarType>& in, std::vector<ScalarType>& out,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int /*batch_num*/);

ScalarType complex_to_real(std::vector<ScalarType>& in, std::vector<ScalarType>& out,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int /*batch_num*/)
{

	cuarma::vector<ScalarType> input(in.size());
	cuarma::vector<ScalarType> output(in.size());

	std::vector<ScalarType> res(in.size());
	cuarma::fast_copy(in, input);

	cuarma::blas::complex_to_real(input, output, input.size() / 2);

	cuarma::backend::finish();
	cuarma::fast_copy(output, res);

	return diff_max(res, out);

}

ScalarType multiply_complex(std::vector<ScalarType>& in, std::vector<ScalarType>& in2,
	std::vector<ScalarType>& out);

ScalarType multiply_complex(std::vector<ScalarType>& in, std::vector<ScalarType>& in2,
	std::vector<ScalarType>& out)
{

	std::vector<ScalarType> res(out.size());

	cuarma::vector<ScalarType> input1(in.size());
	cuarma::vector<ScalarType> input2(out.size());
	cuarma::vector<ScalarType> output(out.size());

	cuarma::fast_copy(in, input1);
	cuarma::fast_copy(in2, input2);
	cuarma::blas::multiply_complex(input1, input2, output);

	std::cout << std::endl;

	cuarma::vector<ScalarType> tmp(out.size());

	cuarma::fast_copy(output, res);

	return diff_max(res, out);
}

ScalarType convolve(std::vector<ScalarType>& in1, std::vector<ScalarType>& in2,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int /*batch_size*/);

ScalarType convolve(std::vector<ScalarType>& in1, std::vector<ScalarType>& in2,
	unsigned int /*row*/, unsigned int /*col*/, unsigned int /*batch_size*/)
{
	//if (in1.size() > 2048) return -1;
	cuarma::vector<ScalarType> input1(in1.size());
	cuarma::vector<ScalarType> input2(in2.size());
	cuarma::vector<ScalarType> output(in1.size());

	cuarma::fast_copy(in1, input1);
	cuarma::fast_copy(in2, input2);

	cuarma::blas::convolve(input1, input2, output);

	cuarma::backend::finish();
	std::vector<ScalarType> res(in1.size());
	cuarma::fast_copy(output, res);

	std::vector<ScalarType> ref(in1.size());
	convolve_ref(in1, in2, ref);

	return diff_max(res, ref);
}

int test_correctness(const std::string& log_tag, input_function_ptr input_function,
	test_function_ptr func);

int test_correctness(const std::string& log_tag, input_function_ptr input_function,
	test_function_ptr func)
{

	std::vector<ScalarType> input;
	std::vector<ScalarType> output;

	std::cout << std::endl;
	std::cout << "*****************" << log_tag << "***************************\n";

	unsigned int batch_size;
	unsigned int rows_num, cols_num;

	input_function(input, output, rows_num, cols_num, batch_size, log_tag);
	ScalarType df = func(input, output, rows_num, cols_num, batch_size);
	printf("%7s ROWS=%6d COLS=%6d; BATCH=%3d; DIFF=%3.15f;\n", ((fabs(df) < EPS) ? "[Ok]" : "[Fail]"),
		rows_num, cols_num, batch_size, df);
	std::cout << std::endl;

	if (df > EPS)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}

int testcorrectnes_multiply(const std::string& log_tag);

int testcorrectnes_multiply(const std::string& log_tag)
{
	std::vector<ScalarType> input;
	std::vector<ScalarType> input2;
	std::vector<ScalarType> output;
	unsigned int batch_size;
	unsigned int rows_num, cols_num;
	std::cout << std::endl;
	std::cout << "*****************" << log_tag << "***************************\n";
	read_vectors_three(input, input2, output, rows_num, cols_num, batch_size, log_tag);
	ScalarType df = multiply_complex(input, input2, output);
	printf("%7s ROWS=%6d COLS=%6d; BATCH=%3d; DIFF=%3.15f;\n", ((fabs(df) < EPS) ? "[Ok]" : "[Fail]"),
		rows_num, cols_num, batch_size, df);
	std::cout << std::endl;

	if (df > EPS)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}

int main()
{
	std::cout << "*" << std::endl;
	std::cout << "* cuarma test: FFT" << std::endl;
	std::cout << "*" << std::endl;

	//1D FFT tests
	if (test_correctness("fft::direct", read_vectors_pair, &direct) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::batch::direct", read_vectors_pair, &direct) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::radix2", read_vectors_pair, &radix2) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::fft_ifft_radix2", read_vectors_pair, &fft_ifft_radix2) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::ifft_fft_radix2", read_vectors_pair, &ifft_fft_radix2) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::batch::radix2", read_vectors_pair, &radix2) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::convolve::1", read_vectors_pair, &convolve) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::convolve::2", read_vectors_pair, &convolve) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::bluestein::1", read_vectors_pair, &bluestein) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::bluestein::2", read_vectors_pair, &bluestein) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (testcorrectnes_multiply("fft::multiplt::complex") == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft:real_to_complex", read_vectors_pair, &real_to_complex) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft:complex_to_real", read_vectors_pair, &complex_to_real) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (test_correctness("fft::fft_reverse_direct", read_vectors_pair,
		&fft_reverse_direct) == EXIT_FAILURE)
		return EXIT_FAILURE;

	std::cout << std::endl;
	std::cout << "------- Test completed --------" << std::endl;
	std::cout << std::endl;

	return EXIT_SUCCESS;
}
